"""
Tests for POST /api/predict.

detection_service.run_detection is mocked, so no real model/GPU runs. A tiny
in-memory PNG stands in for the upload. We assert the happy path shape and the
ApiError responses for missing / disallowed files.
"""
import io
from unittest.mock import patch

from PIL import Image

from app.schemas.detection import PredictResponse

_FAKE_DETECTION = {
    "items": [
        {"id": 0, "class_name": "PLASTIC", "display_name": "Plastic",
         "confidence": 0.87, "bbox": [10, 20, 110, 120]},
    ],
    "image": {"width": 32, "height": 32},
}


def _png_upload():
    """A minimal valid PNG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 120, 120)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_ok(client, tmp_path):
    # Route uploads into a temp dir so the test leaves no artefacts behind.
    client.application.config["UPLOAD_FOLDER"] = str(tmp_path)

    with patch("app.blueprints.api.routes.run_detection", return_value=_FAKE_DETECTION):
        res = client.post(
            "/api/predict",
            data={"image": (_png_upload(), "test.png")},
            content_type="multipart/form-data",
        )

    assert res.status_code == 200
    body = res.get_json()
    PredictResponse(**body)  # shape matches the documented contract

    assert body["items"][0]["display_name"] == "Plastic"
    assert body["image"]["filename"].endswith("test.png")
    assert body["image"]["url"].startswith("/static/uploads/")
    assert body["image"]["width"] == 32


def test_predict_missing_file_is_400(client):
    res = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_disallowed_extension_is_400(client):
    res = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()
