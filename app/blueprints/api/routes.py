"""
JSON API endpoints. These are thin controllers: they validate input,
call a service, and return JSON. No business logic lives here.

Endpoints (filled in over the coming steps):
  POST /api/predict           -> run YOLO on an uploaded image
  POST /api/calculate-impact  -> call carbon API for weighted items
  POST /api/recommend         -> generate disposal recommendations
"""
from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)


@api_bp.route("/health")
def health():
    """Cheap liveness check — useful for deployment + tests."""
    return jsonify(status="ok"), 200
