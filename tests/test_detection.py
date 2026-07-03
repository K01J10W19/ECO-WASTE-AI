"""
Unit tests for app/services/detection_service.py.

The cached model loader is monkeypatched to return a fabricated model, so these
tests need NO real weights, NO GPU and NO network. We verify display-name
mapping, the confidence-threshold filter, and that an empty result is valid.
"""
import types

from app.services import detection_service as ds
from app.schemas.detection import DetectionResult

# LOCKED class order (CLAUDE.md §6).
_NAMES = {0: "BIODEGRADABLE", 1: "CARDBOARD", 2: "GLASS",
          3: "METAL", 4: "PAPER", 5: "PLASTIC"}


def _fake_box(class_id, conf, xyxy):
    """A stand-in for an Ultralytics Boxes row (index [0] like tensors)."""
    return types.SimpleNamespace(cls=[class_id], conf=[conf], xyxy=[xyxy])


def _fake_model(boxes, width=1280, height=720):
    """A stand-in YOLO model whose predict() returns one fabricated result."""
    result = types.SimpleNamespace(orig_shape=(height, width), boxes=boxes)
    return types.SimpleNamespace(names=_NAMES, predict=lambda *a, **k: [result])


def _dummy_image(tmp_path):
    """run_detection checks the file exists; content is irrelevant (model mocked)."""
    p = tmp_path / "upload.jpg"
    p.write_bytes(b"not-a-real-image")
    return str(p)


def test_maps_display_name_and_filters_confidence(app, monkeypatch, tmp_path):
    boxes = [
        _fake_box(5, 0.90, [10.0, 20.0, 110.0, 120.0]),   # PLASTIC, kept
        _fake_box(2, 0.10, [0.0, 0.0, 50.0, 50.0]),        # GLASS 0.10 < 0.35, dropped
    ]
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model(boxes))

    with app.app_context():
        out = ds.run_detection(_dummy_image(tmp_path))

    assert out["image"] == {"width": 1280, "height": 720}
    assert len(out["items"]) == 1                     # low-confidence box filtered out
    item = out["items"][0]
    assert item["id"] == 0
    assert item["class_name"] == "PLASTIC"
    assert item["display_name"] == "Plastic"          # mapped via APP_CLASS_DISPLAY_NAMES
    assert item["bbox"] == [10, 20, 110, 120]         # pixel ints
    assert item["confidence"] == 0.9
    DetectionResult(**out)                            # response matches the schema


def test_empty_detections_is_not_an_error(app, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model([], width=640, height=480))

    with app.app_context():
        out = ds.run_detection(_dummy_image(tmp_path))

    assert out["items"] == []
    assert out["image"] == {"width": 640, "height": 480}
    DetectionResult(**out)
