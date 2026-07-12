"""
JSON API endpoints. These are thin controllers: they validate input,
call a service, and return JSON. No business logic lives here.

Endpoints:
  POST /api/predict           -> dual-tower analysis (waste detector + TrashNet ViT)
  POST /api/calculate-impact  -> CO2e for user-weighted items (Climatiq or local fallback)
  POST /api/recommend         -> DMM: ranked 3-path disposal recommendations (Step 6)
"""
import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError
from werkzeug.utils import secure_filename

from app.schemas.carbon import CalculateImpactRequest
from app.schemas.recommendation import RecommendRequest
from app.services.carbon_service import calculate_impact as calculate_impact_service
from app.services.detection_service import analyze_waste_pipeline
from app.services.recommendation_service import recommend_for_items
from app.utils.errors import ApiError

api_bp = Blueprint("api", __name__)


def _validated_json(schema_cls):
    """Parse the request body against a pydantic schema; 400 on any problem."""
    payload = request.get_json(silent=True)
    if payload is None:
        raise ApiError("Request body must be JSON.", status_code=400)
    try:
        return schema_cls(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise ApiError(f"Invalid request: {loc}: {first.get('msg', 'invalid')}",
                       status_code=400)


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


def _parse_conf(raw):
    """Optional per-request confidence-threshold override; None when not supplied."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ApiError("Invalid 'conf' value; must be a number between 0 and 1.", status_code=400)


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Analyse waste in an uploaded image via the two-stage pipeline
    (YOLO-World localization → crop → CLIP material classification).

    Request : multipart/form-data with an ``image`` file field; optional
              ``conf`` field overrides the Stage-1 threshold for this call.
    Response: { "items": [...], "image": {filename, width, height, url} }
              Each item carries the material verdict, both stages' scores and
              a placeholder ``carbon_factor_kg_per_kg``.

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

    detection = analyze_waste_pipeline(save_path, conf=_parse_conf(request.form.get("conf")))

    return jsonify(
        items=detection["items"],
        image={
            "filename": filename,
            "width": detection["image"]["width"],
            "height": detection["image"]["height"],
            "url": f"/static/uploads/{filename}",
        },
    ), 200


@api_bp.route("/calculate-impact", methods=["POST"])
def calculate_impact():
    """
    Stage-B precision audit: real CO2e for verified items (Step 5, v3.5 UX).

    Request : JSON { "items": [{"id": 0, "material": "plastic",
                                "weight_kg": 0.5 | "box_area_px": 233712}, ...],
                     "country": "MY" (optional ISO alpha-2 — typically the
                     frontend's IP-geolocated default; blank = global) }
              ``id`` is the client's grid row key (echoed back verbatim for
              split-screen canvas↔grid sync); items without a weight fall
              back to the box_area_px / gamma pixel proxy.
    Response: { "items": [{id, material, weight_kg, weight_source, ...}],
                "total_co2e_kg", "country", "provider" }

    Uses live Climatiq factors when CLIMATIQ_API_KEY is configured; falls back
    to the local dummy coefficients otherwise (the app never requires a key).
    """
    req = _validated_json(CalculateImpactRequest)
    result = calculate_impact_service(
        [item.model_dump() for item in req.items], req.country)
    return jsonify(result), 200


@api_bp.route("/recommend", methods=["POST"])
def recommend():
    """
    Decision Making Module (Step 6): ranked end-of-life recommendations.

    Request : JSON { "items": [{"material": "plastic", "weight_kg": 0.5},
                               {"material": "glass", "box_area_px": 16000}],
                     "country": "MY" (optional ISO alpha-2; blank = global) }
              Each item needs weight_kg (user-verified) OR box_area_px (the
              blind pixel proxy from /predict; weight = area / gamma).
    Response: { "items": [ { material, effective_weight_kg, weight_source,
                             best_method, max_saving_kg,
                             recommendations: [3 ranked paths with CO2e,
                             status_tag, verdict, pros, cons] } ],
                "summary": {...}, "country",
                "provider": "llm_enriched" | "local_knowledge_base"
                            | "local_fallback" }

    The carbon simulation and ranking are always local + deterministic; the
    optional v3.6 LLM text layer (free OpenAI-compatible endpoint) rewrites
    only the three literary fields in child-friendly, country-localized
    language and degrades seamlessly to the local grid on any failure —
    this endpoint never 502s. Live region-scoped FACTORS remain
    POST /api/calculate-impact's concern.
    """
    req = _validated_json(RecommendRequest)
    result = recommend_for_items(
        [item.model_dump() for item in req.items], req.country)
    return jsonify(result), 200
