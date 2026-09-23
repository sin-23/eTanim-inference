"""
e-Tanim Mini PC harvest-detection service.

Current pipeline:
  Camera -> Orchestrator -> Detection -> Firebase test upload

Evaluator models are not available yet, so maturity fields remain 0.

Run:
  python detect_and_upload.py
  python detect_and_upload.py --once --dry-run
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("etanim")

# ── Configuration ────────────────────────────────────────────────────────────

RTDB_URL = os.getenv("FIREBASE_RTDB_URL")
CRED_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
ORCH_MODEL = os.getenv("ORCHESTRATOR_MODEL", "models/best (7).pt")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROUTE_CONF = float(os.getenv("ROUTE_CONF", "0.70"))
USE_CLAHE = os.getenv("USE_CLAHE", "1") == "1"
FIREBASE_TEST_INTERVAL = int(os.getenv("FIREBASE_TEST_INTERVAL", "30"))

CROPS = {"tomato", "eggplant", "bell_pepper"}
MIN_BOX_PX = 16


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def apply_clahe(frame):
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )


# ── Model ────────────────────────────────────────────────────────────────────

class Models:

    def __init__(self):
        from ultralytics import YOLO

        if not os.path.exists(ORCH_MODEL):
            raise FileNotFoundError(
                f"Orchestrator model not found: {ORCH_MODEL}"
            )

        self.orch = YOLO(ORCH_MODEL)

        log.info(
            "Orchestrator classes: %s",
            self.orch.names
        )


# ── Detection ────────────────────────────────────────────────────────────────

def detect(models, frame):

    if USE_CLAHE:
        frame = apply_clahe(frame)

    h, w = frame.shape[:2]

    result = models.orch(
        frame,
        conf=0.25,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:

        conf = float(box.conf[0])

        if conf < ROUTE_CONF:
            continue

        crop = normalize(
            models.orch.names[int(box.cls[0])]
        )

        if crop not in CROPS:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            continue

        detections.append(
            (crop, conf, (x1, y1, x2, y2))
        )

    return detections


def draw_detections(frame, detections):

    import cv2

    for crop, conf, (x1, y1, x2, y2) in detections:

        label = f"{crop} {conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame


# ── Firebase ─────────────────────────────────────────────────────────────────

def init_firebase():

    import firebase_admin
    from firebase_admin import credentials, db

    if not RTDB_URL:
        raise RuntimeError(
            "FIREBASE_RTDB_URL is not set in .env"
        )

    if not os.path.exists(CRED_PATH):
        raise RuntimeError(
            f"Firebase credentials not found: {CRED_PATH}"
        )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(CRED_PATH),
            {"databaseURL": RTDB_URL}
        )

    log.info("Firebase initialized.")

    return db


def upload_test(fb_db, detections):

    if not detections:
        return

    by_crop = {}

    for crop, conf, _ in detections:
        by_crop.setdefault(crop, []).append(conf)

    for crop, confidences in by_crop.items():

        payload = {
            "underripe": 0,
            "ripe": 0,
            "damaged": 0,
            "confidence": round(max(confidences), 2),
            "updatedAt": int(time.time() * 1000)
        }

        fb_db.reference(
            f"detections/{crop}"
        ).set(payload)

        log.info(
            "[Firebase TEST] detections/%s <- %s",
            crop,
            payload
        )


# ── Live mode ────────────────────────────────────────────────────────────────

def live(models, fb_db):

    import cv2

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}"
        )

    log.info(
        "Live camera started. Press Q or ESC to quit."
    )

    last_upload = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                log.warning("Camera returned no frame.")
                continue

            detections = detect(
                models,
                frame
            )

            for crop, conf, box in detections:
                log.info(
                    "Detected %s %.2f at %s",
                    crop,
                    conf,
                    box
                )

            display = draw_detections(
                frame,
                detections
            )

            cv2.imshow(
                "e-Tanim - Live Detection",
                display
            )

            now = time.time()

            if (
                fb_db
                and detections
                and now - last_upload >= FIREBASE_TEST_INTERVAL
            ):
                try:
                    upload_test(
                        fb_db,
                        detections
                    )
                    last_upload = now
                except Exception:
                    log.exception(
                        "Firebase upload failed."
                    )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Camera stopped.")


# ── Single-frame mode ────────────────────────────────────────────────────────

def once(models, fb_db):

    import cv2

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}"
        )

    try:

        ok, frame = cap.read()

        if not ok:
            log.error("Camera returned no frame.")
            return

        detections = detect(
            models,
            frame
        )

        for crop, conf, box in detections:
            log.info(
                "Detected %s %.2f at %s",
                crop,
                conf,
                box
            )

        log.info(
            "Detection cycle complete: %d detection(s)",
            len(detections)
        )

        if fb_db and detections:
            upload_test(
                fb_db,
                detections
            )

    finally:
        cap.release()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    models = Models()

    fb_db = None

    if not args.dry_run:
        fb_db = init_firebase()

    if args.once:
        once(models, fb_db)
    else:
        live(models, fb_db)


if __name__ == "__main__":
    main()