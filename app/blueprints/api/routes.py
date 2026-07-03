"""
JSON API endpoints. These are thin controllers: they validate input,
call a service, and return JSON. No business logic lives here.

Endpoints:
  POST /api/predict           -> run YOLO on an uploaded image  (Step 4)
  POST /api/calculate-impact  -> call carbon API for weighted items  (Step 5)
  POST /api/recommend         -> generate disposal recommendations  (Step 6)
"""
import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.services.detection_service import run_detection
from app.utils.errors import ApiError

api_bp = Blueprint("api", __name__)


@api_bp.route("/health")
def health():
    """Cheap liveness check — useful for deployment + tests."""
    return jsonify(status="ok"), 200


def _is_allowed(filename: str) -> bool:
    """True if the filename has an allowed image extension (case-insensitive)."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Detect waste items in an uploaded image.

    Request : multipart/form-data with an ``image`` file field.
    Response: { "items": [...], "image": {filename, width, height, url} }

    Oversized uploads are rejected by Flask (MAX_CONTENT_LENGTH -> 413 handler).
    """
    if "image" not in request.files:
        raise ApiError("No image file provided (form field 'image').", status_code=400)

    file = request.files["image"]
    if not file or file.filename == "":
        raise ApiError("No image selected.", status_code=400)

    if not _is_allowed(file.filename):
        allowed = ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"]))
        raise ApiError(f"Unsupported file type. Allowed: {allowed}.", status_code=400)

    # Collision-proof filename: uuid prefix + sanitised original name.
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    detection = run_detection(save_path)

    # TODO (Step 5): persist a Scan row once carbon totals/location are available.
    return jsonify(
        items=detection["items"],
        image={
            "filename": filename,
            "width": detection["image"]["width"],
            "height": detection["image"]["height"],
            "url": f"/static/uploads/{filename}",
        },
    ), 200
