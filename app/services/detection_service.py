"""
Detection service — YOLO11 inference for the web app.

The trained model is loaded exactly ONCE (cached) and reused for every request;
loading weights is expensive, so we never reload per request. All expected
failures (missing weights, inference errors) surface as ``ApiError`` so the API
layer stays thin.

The app only ever *loads* an exported model from ``MODEL_PATH`` (produced by
``ml/scripts/export.py``). It never imports training code.
"""
import logging
import os
from functools import lru_cache

from flask import current_app

from app.utils.errors import ApiError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """
    Load and cache the YOLO model for ``model_path`` (called once per path).

    ``lru_cache`` only stores successful loads — if the weights are missing we
    raise ``ApiError`` and nothing is cached, so a later request can retry once
    the file exists. Ultralytics/torch are imported lazily so importing this
    module (e.g. in tests) never pulls in the ML stack.
    """
    if not os.path.isfile(model_path):
        raise ApiError(
            f"Detection model not found at '{model_path}'. "
            "Run 'python ml/scripts/export.py' to create models/best.pt.",
            status_code=500,
        )
    try:
        from ultralytics import YOLO
        logger.info("Loading YOLO model from %s", model_path)
        return YOLO(model_path)
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any load failure as ApiError
        raise ApiError("Failed to load the detection model.", status_code=500) from exc


def get_model():
    """Return the cached model instance for the app's configured ``MODEL_PATH``."""
    return _load_model(str(current_app.config["MODEL_PATH"]))


def run_detection(image_path: str) -> dict:
    """
    Run object detection on the image at ``image_path``.

    Returns a JSON-serialisable dict::

        {
          "items": [
            {"id": 0, "class_name": "PLASTIC", "display_name": "Plastic",
             "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
          ],
          "image": {"width": 1280, "height": 720}
        }

    Detections below ``CONFIDENCE_THRESHOLD`` are dropped. No detections is a
    valid result (``items: []``), not an error. Class names are mapped to
    friendly display names via ``APP_CLASS_DISPLAY_NAMES``.
    """
    if not os.path.isfile(image_path):
        raise ApiError("Uploaded image could not be found for detection.", status_code=400)

    model = get_model()
    conf_threshold = float(current_app.config["CONFIDENCE_THRESHOLD"])
    device = str(current_app.config.get("INFERENCE_DEVICE", "cpu"))
    display_names = current_app.config["APP_CLASS_DISPLAY_NAMES"]

    try:
        results = model.predict(source=image_path, conf=conf_threshold,
                                device=device, verbose=False)
    except Exception as exc:  # noqa: BLE001 - any inference failure is client-facing
        raise ApiError("Detection inference failed.", status_code=500) from exc

    result = results[0]
    height, width = int(result.orig_shape[0]), int(result.orig_shape[1])

    items = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:  # belt-and-suspenders (predict already filters)
            continue
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        items.append({
            "id": len(items),
            "class_name": class_name,
            "display_name": display_names.get(class_name, class_name),
            "confidence": round(confidence, 4),
            "bbox": [x1, y1, x2, y2],
        })

    return {"items": items, "image": {"width": width, "height": height}}
