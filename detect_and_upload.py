"""
e-Tanim Mini PC harvest-detection service.

Current pipeline (tomato):
  Camera -> Orchestrator (with tracker ID) -> 224x224 crop -> Tomato evaluator
  -> ripeness counters -> Firebase upload

Each detected tomato gets a tracker ID. The 224x224 crop around it is sent to
the evaluator, and the label is remembered per ID so the same fruit is not
re-classified on every frame. Other crops are detected but have no evaluator
yet, so their counters remain 0.

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
ORCH_MODEL = os.getenv("ORCHESTRATOR_MODEL", "models/best.pt")
TOMATO_CLS_MODEL = os.getenv("TOMATO_CLS_MODEL", "models/tomato.pt")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROUTE_CONF = float(os.getenv("ROUTE_CONF", "0.70"))
USE_CLAHE = os.getenv("USE_CLAHE", "1") == "1"
FIREBASE_TEST_INTERVAL = int(os.getenv("FIREBASE_TEST_INTERVAL", "30"))

CROPS = {"tomato", "eggplant", "bell_pepper"}
MIN_BOX_PX = 16

CROP_SIZE = 224
CLS_REFRESH = int(os.getenv("CLS_REFRESH", "600"))  # seconds before an ID is re-classified

# Evaluator class name -> counter field the website reads.
LABEL_KEYS = {
    "unripe": "underripe",
    "underripe": "underripe",
    "ripe": "ripe",
    "rotten": "damaged",
    "damaged": "damaged",
}


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


def crop_224(frame, box):
    """224x224 window centred on the box, shifted to stay inside the frame."""

    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    half = CROP_SIZE // 2

    x0 = min(max(0, cx - half), max(0, w - CROP_SIZE))
    y0 = min(max(0, cy - half), max(0, h - CROP_SIZE))

    patch = frame[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

    # Only happens if the frame itself is smaller than 224 px.
    if patch.shape[0] != CROP_SIZE or patch.shape[1] != CROP_SIZE:
        patch = cv2.resize(patch, (CROP_SIZE, CROP_SIZE))

    return patch


# ── Model ────────────────────────────────────────────────────────────────────

class Models:

    def __init__(self):
        from ultralytics import YOLO

        if not os.path.exists(ORCH_MODEL):
            raise FileNotFoundError(
                f"Orchestrator model not found: {ORCH_MODEL}"
            )

        if not os.path.exists(TOMATO_CLS_MODEL):
            raise FileNotFoundError(
                f"Tomato evaluator model not found: {TOMATO_CLS_MODEL}"
            )

        self.orch = YOLO(ORCH_MODEL)
        self.tomato_cls = YOLO(TOMATO_CLS_MODEL)

        log.info(
            "Orchestrator classes: %s",
            self.orch.names
        )

        log.info(
            "Tomato evaluator classes: %s",
            self.tomato_cls.names
        )

        unknown = [
            n for n in self.tomato_cls.names.values()
            if normalize(n) not in LABEL_KEYS
        ]

        if unknown:
            raise RuntimeError(
                "Tomato evaluator classes are not ripeness labels: "
                f"{unknown[:5]}. Use the trained unripe/ripe/rotten "
                "weights, or add the class names to LABEL_KEYS."
            )

        # tracker ID -> (counter field, evaluator confidence, time classified)
        self.labels = {}


# ── Detection ────────────────────────────────────────────────────────────────

def detect(models, frame):

    if USE_CLAHE:
        frame = apply_clahe(frame)

    h, w = frame.shape[:2]

    result = models.orch.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
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

        # box.id is None until the tracker has confirmed the object.
        track_id = int(box.id[0]) if box.id is not None else None

        detections.append(
            (crop, conf, (x1, y1, x2, y2), track_id)
        )

    return detections


def evaluate(models, frame, detections):
    """Return one counter field (or None) per detection, in the same order."""

    labels = []
    now = time.time()

    for crop, _, box, track_id in detections:

        if crop != "tomato":
            labels.append(None)
            continue

        cached = models.labels.get(track_id)

        if cached and now - cached[2] < CLS_REFRESH:
            labels.append(cached[0])
            continue

        result = models.tomato_cls(
            crop_224(frame, box),
            verbose=False
        )[0]

        top = int(result.probs.top1)
        cls_conf = float(result.probs.top1conf)
        label = LABEL_KEYS[normalize(result.names[top])]

        if track_id is not None:
            models.labels[track_id] = (label, cls_conf, now)

        log.info(
            "Tomato #%s -> %s (%.2f)",
            track_id,
            label,
            cls_conf
        )

        labels.append(label)

    return labels


def draw_detections(frame, detections, labels):

    import cv2

    for (crop, conf, (x1, y1, x2, y2), track_id), label in zip(
        detections,
        labels
    ):

        text = f"#{track_id} {crop} {conf:.2f}"

        if label:
            text += f" {label}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
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


def upload_test(fb_db, detections, labels):

    if not detections:
        return

    by_crop = {}

    for (crop, conf, _, _), label in zip(detections, labels):

        entry = by_crop.setdefault(
            crop,
            {"confidences": [], "underripe": 0, "ripe": 0, "damaged": 0}
        )

        entry["confidences"].append(conf)

        if label:
            entry[label] += 1

    for crop, entry in by_crop.items():

        payload = {
            "underripe": entry["underripe"],
            "ripe": entry["ripe"],
            "damaged": entry["damaged"],
            "confidence": round(max(entry["confidences"]), 2),
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

            # Crop from the untouched frame, before boxes are drawn on it.
            labels = evaluate(
                models,
                frame,
                detections
            )

            display = draw_detections(
                frame,
                detections,
                labels
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
                        detections,
                        labels
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

        labels = evaluate(
            models,
            frame,
            detections
        )

        for (crop, conf, box, track_id), label in zip(
            detections,
            labels
        ):
            log.info(
                "Detected #%s %s %.2f at %s -> %s",
                track_id,
                crop,
                conf,
                box,
                label
            )

        log.info(
            "Detection cycle complete: %d detection(s)",
            len(detections)
        )

        if fb_db and detections:
            upload_test(
                fb_db,
                detections,
                labels
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
