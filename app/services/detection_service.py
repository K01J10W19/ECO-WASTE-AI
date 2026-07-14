"""
Detection service — the DUAL-TOWER HYBRID pipeline orchestrator.

ARCHITECTURE (2026-07, v3.2 — specialized waste OBJECT DETECTION + Method B):

  Stage 1 — SPECIALIZED WASTE DETECTOR (``models/yolov8n-waste-det.pt``, from
            GitHub ``gianlucasposito/YOLO-Waste-Detection``, MIT): a YOLOv8-N
            object-detection model fine-tuned on a blended universal waste
            corpus (4.1k images of wild litter + household recyclables). Its
            latent space only knows waste, so background furniture/floors/
            plants rarely fire, and the nano backbone keeps per-frame latency
            minimal on edge hardware. Detection-only by design (no masks):
            the geometric BOX AREA from ``box.xyxy`` serves as the physical
            volume/mass proxy for carbon scaling. Its 5 coarse labels
            (Glass/Metal/Paper/Plastic/Waste) are noted as ``located_as``
            diagnostics and never trusted for identity.

  Processing layer — (a) CONTEXT-AWARE SQUARE PADDING (PIL), anchored
            directly on ``box.xyxy``: +15% context margin per side, square
            pad on neutral gray (no aspect distortion), resize 224x224.
            (b) METHOD B — CLASSICAL CV PHYSICS EXTRACTOR (OpenCV): per patch,
            a heuristic Plasticity Index psi from Laplacian texture variance
            (micro-wrinkles) and Canny edge density.

  Stage 2 — TrashNet ViT (classification_service) names the material. When
            the ViT's top-2 is an AMBIGUOUS plastic-vs-glass call (score gap
            < PLASTICITY_TIEBREAK_MARGIN), psi breaks the tie; every item
            carries its physics readings and a ``tiebreak_applied`` flag.

Carbon: Final Impact = Base Coefficient x (Box Area / gamma), gamma
recalibrated to 8000 for rectangular over-coverage (carbon_service).

Conventions preserved: models are cached singletons, heavy imports stay lazy
(tests import this module freely), all expected failures raise ``ApiError``.
The legacy training workspace under ``ml/`` remains untouched and unimported.
"""
import logging
import os
import shutil
from functools import lru_cache

from flask import current_app

from app.services.carbon_service import estimate_dynamic_impact, get_carbon_factor
from app.services.classification_service import DISPLAY_NAMES, classify_crops
from app.utils.errors import ApiError

logger = logging.getLogger(__name__)

# Boxes thinner than this (pixels) are skipped — too small to classify.
_MIN_CROP_SIDE = 2
# Processing layer: context margin added around each box before squaring.
_CONTEXT_PAD_FRAC = 0.15
# Processing layer: neutral pad colour (the letterbox-gray convention).
_PAD_FILL = (114, 114, 114)
# Processing layer: the ViT's native input resolution.
_PATCH_SIZE = 224

# Stage-1 suppression — the FINAL locked configuration (2026-07-14, after a
# full investigation cycle): CLASS-AGNOSTIC NMS at a relaxed 0.60 IoU.
#   * agnostic_nms=True kills the observed DUPLICATE-FRAME bug: one physical
#     object firing under TWO coarse labels (e.g. "Paper" + "Waste",
#     IoU >= ~0.8) survived per-class suppression twice and Stage 2 named
#     both crops identically. Per-class mode was trialled and reverted the
#     same day once the trade-off was weighed.
#   * iou=0.60 (raised from 0.45) lets tightly packed DISTINCT items keep
#     their boxes — adjacent items can legitimately overlap ~50%.
#   * KNOWN LIMIT (NMS-independent): identical-item clusters (three bottles
#     → one group box) are a nano-detector CAPACITY limit — the raw
#     candidate pool held NO per-bottle boxes above 1% conf and agnostic
#     True/False were byte-identical on the probe scene. Only the A/B
#     locators (yolov8m-seg-trash, yolo26x-seg) split such scenes.
# The NMS-free A/B baselines (YOLO26, RT-DETR) accept and ignore both args.
_NMS_IOU = 0.60
_NMS_AGNOSTIC = True

# ---------------------------------------------------------------------------
# Method B calibration constants (Plasticity Index psi).
#
# psi blends two classical-CV cues, each squashed to [0, 1] against a
# reference scale, then averaged:
#   psi = 0.5*min(1, laplacian_var / _LAPLACIAN_REF)
#       + 0.5*min(1, edge_density  / _EDGE_DENSITY_REF)
# psi >= 0.5 reads "plastic-like" (crinkled, thin sharp edges);
# psi <  0.5 reads "glass-like"   (smooth, broad refractive contours).
# ---------------------------------------------------------------------------
_LAPLACIAN_REF = 500.0     # texture variance typical of crinkled plastic film
_EDGE_DENSITY_REF = 0.10   # fraction of Canny edge pixels in a busy patch
# The ViT tie-break only fires when plastic & glass are the top-2 AND their
# score gap is inside this margin — clear verdicts are never overridden.
PLASTICITY_TIEBREAK_MARGIN = 0.15

# Known specialist checkpoints fetched automatically when the local file is
# missing (keeps first-run setup one command). Each entry is either
# ("url", direct_download_url) or ("hf", repo_id, filename_in_repo).
_WEIGHT_SOURCES = {
    # ACTIVE: blended-waste object detector (GitHub, MIT).
    "yolov8n-waste-det.pt": (
        "url",
        "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt",
    ),
    # Archived v3.1 segmentation locator — kept resolvable for A/B comparisons.
    "yolov8m-seg-trash.pt": (
        "hf", "turhancan97/yolov8-segment-trash-detection", "yolov8m-seg.pt",
    ),
}


def _fetch_weights(model_path: str, source: tuple) -> None:
    """Download a registered specialist checkpoint into ``model_path``."""
    target_dir = os.path.dirname(model_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    if source[0] == "hf":
        from huggingface_hub import hf_hub_download  # ships with transformers
        _, repo_id, remote_name = source
        logger.info("Fetching %s from HF hub %s (%s)", model_path, repo_id, remote_name)
        shutil.copy2(hf_hub_download(repo_id, remote_name), model_path)
        return
    import requests
    _, url = source
    logger.info("Fetching %s from %s", model_path, url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(model_path, "wb") as fh:
        fh.write(resp.content)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """
    Load and cache the Stage-1 detector (called once per path).

    Resolution order for ``model_path``:
      1. An existing local file — loaded directly.
      2. A filename registered in ``_WEIGHT_SOURCES`` — downloaded once into
         place (the specialist waste weights).
      3. A bare official Ultralytics name (``yolo26x-seg.pt``, ...) —
         auto-downloaded by Ultralytics (kept switchable for A/B baselines).
    RT-DETR checkpoints still load through the ``RTDETR`` class for backward
    compatibility; everything else goes through ``YOLO``.

    ``lru_cache`` only stores successful loads — on failure we raise
    ``ApiError`` and nothing is cached, so a later request can retry.
    """
    base = os.path.basename(model_path)
    has_dir = base != model_path
    if not os.path.isfile(model_path) and base in _WEIGHT_SOURCES:
        try:
            _fetch_weights(model_path, _WEIGHT_SOURCES[base])
        except Exception as exc:  # noqa: BLE001
            raise ApiError(
                f"Could not download the waste detection weights '{base}'. Check "
                f"your connection or place the file at '{model_path}' manually.",
                status_code=500,
            ) from exc
    if has_dir and not os.path.isfile(model_path):
        raise ApiError(
            f"Detection model not found at '{model_path}'. Set MODEL_PATH to an "
            "existing .pt file, a known specialist name (yolov8n-waste-det.pt), "
            "or a bare official name like 'yolo26x-seg.pt'.",
            status_code=500,
        )
    try:
        if "rtdetr" in base.lower():
            from ultralytics import RTDETR  # lazy: keep the ML stack out of app import time
            logger.info("Loading RT-DETR locator from %s", model_path)
            return RTDETR(model_path)
        from ultralytics import YOLO
        logger.info("Loading YOLO locator from %s", model_path)
        return YOLO(model_path)
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any load failure as ApiError
        raise ApiError("Failed to load the detection model.", status_code=500) from exc


def get_model():
    """Return the cached locator instance for the app's configured ``MODEL_PATH``."""
    return _load_model(str(current_app.config["MODEL_PATH"]))


def _locate_objects(image_path: str, conf_threshold: float) -> tuple:
    """
    Stage 1: run the specialist waste detector and return (instances, image_size).

    Each instance is ``{"bbox": [x1, y1, x2, y2], "box_confidence": f,
    "located_as": "Plastic", "box_area_px": a}`` in original-image pixels.
    ``located_as`` is the detector's own coarse label — kept purely for
    auditing; the pipeline never branches on it. ``box_area_px`` is the
    geometric box area (x2-x1)*(y2-y1) of the CLAMPED box — the volume/mass
    proxy for carbon scaling. The predict call runs the owner-locked
    suppression configuration (``agnostic_nms=_NMS_AGNOSTIC``,
    ``iou=_NMS_IOU`` — see the constants block for the trade-off notes).
    """
    model = get_model()
    device = str(current_app.config.get("INFERENCE_DEVICE", "cpu"))

    try:
        results = model.predict(source=image_path, conf=conf_threshold,
                                iou=_NMS_IOU, agnostic_nms=_NMS_AGNOSTIC,
                                device=device, verbose=False)
    except Exception as exc:  # noqa: BLE001 - any inference failure is client-facing
        raise ApiError("Object localization failed.", status_code=500) from exc

    result = results[0]
    height, width = int(result.orig_shape[0]), int(result.orig_shape[1])

    instances = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:  # belt-and-suspenders (predict already filters)
            continue
        class_id = int(box.cls[0])
        located_as = str(model.names.get(class_id, class_id)) \
            if isinstance(model.names, dict) else str(model.names[class_id])
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        # Clamp to the image so the crop below never fails on edge boxes.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if (x2 - x1) < _MIN_CROP_SIDE or (y2 - y1) < _MIN_CROP_SIDE:
            logger.debug("Skipping degenerate box %s", [x1, y1, x2, y2])
            continue

        instances.append({
            "bbox": [x1, y1, x2, y2],
            "box_confidence": round(confidence, 4),
            "located_as": located_as,
            # Box Area = (x2 - x1) * (y2 - y1) — the volume/mass proxy.
            "box_area_px": float((x2 - x1) * (y2 - y1)),
        })

    return instances, {"width": width, "height": height}


def _prepare_patches(image_path: str, instances: list) -> list:
    """
    Processing layer (a): CONTEXT-AWARE SQUARE PADDING, anchored on box.xyxy.

    For each Stage-1 box: (1) expand outward by 15% per side to keep the
    object's immediate visual context; (2) paste the crop centred on a square
    neutral-gray canvas so the aspect ratio is preserved (a long wire/receipt
    is padded, never stretched); (3) resize to exactly 224x224 — the ViT's
    native input grid. Returns RGB patches aligned index-for-index with
    ``instances``.
    """
    try:
        from PIL import Image  # lazy for symmetry with the other heavy imports
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            img_w, img_h = rgb.size

            patches = []
            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                # 1) Context padding, clamped to the image.
                pad_x = int(round((x2 - x1) * _CONTEXT_PAD_FRAC))
                pad_y = int(round((y2 - y1) * _CONTEXT_PAD_FRAC))
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
                crop = rgb.crop((cx1, cy1, cx2, cy2))

                # 2) Square padding — neutral margins, no stretching.
                side = max(crop.size)
                canvas = Image.new("RGB", (side, side), _PAD_FILL)
                canvas.paste(crop, ((side - crop.size[0]) // 2,
                                    (side - crop.size[1]) // 2))

                # 3) Clean scale to the ViT input resolution.
                patches.append(canvas.resize((_PATCH_SIZE, _PATCH_SIZE),
                                             Image.LANCZOS))
            return patches
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ApiError("Could not prepare detected regions for classification.",
                       status_code=500) from exc


def extract_classical_physics_features(img_patch) -> dict:
    """
    Processing layer (b) — METHOD B: classical CV physics extractor.

    Computes the heuristic Plasticity Index psi for one 224x224 RGB patch:

      * Micro-wrinkle detection — ``cv2.Laplacian`` texture variance.
        Disposable plastic exhibits high-frequency crinkles and micro-folds
        (high variance); pristine glass stays structurally smooth (low).
      * Edge profile — ``cv2.Canny`` edge-pixel density. Thin plastic yields
        dense, sharp single-pixel contours; thick glass yields sparser, broad
        refractive edges.

    Both cues are squashed to [0, 1] against calibration references and
    averaged; psi >= 0.5 reads plastic-like, psi < 0.5 glass-like. Purely
    classical — no training involved.
    """
    import cv2          # lazy: opencv-python-headless
    import numpy as np

    gray = cv2.cvtColor(np.asarray(img_patch), cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    psi = (0.5 * min(1.0, laplacian_variance / _LAPLACIAN_REF)
           + 0.5 * min(1.0, edge_density / _EDGE_DENSITY_REF))
    return {
        "laplacian_variance": round(laplacian_variance, 2),
        "edge_density": round(edge_density, 4),
        "plasticity_index": round(psi, 4),
    }


def _apply_plasticity_tiebreak(scores: list, physics: dict) -> tuple:
    """
    Correct an ambiguous plastic-vs-glass ViT verdict with the physics cue.

    Fires ONLY when (a) plastic and glass are the two top-ranked materials AND
    (b) their score gap is below ``PLASTICITY_TIEBREAK_MARGIN``. The winner is
    then chosen by psi (>= 0.5 → plastic, else glass); if that disagrees with
    the ViT ranking, the two labels SWAP their scores (rank correction — total
    probability mass is untouched) and the list is re-sorted.

    Returns ``(scores, tiebreak_applied)`` — the flag is True only when the
    ranking actually changed, so the payload stays honest.
    """
    if len(scores) < 2 or {scores[0]["label"], scores[1]["label"]} != {"plastic", "glass"}:
        return scores, False
    if abs(scores[0]["score"] - scores[1]["score"]) >= PLASTICITY_TIEBREAK_MARGIN:
        return scores, False

    physics_winner = "plastic" if physics["plasticity_index"] >= 0.5 else "glass"
    if scores[0]["label"] == physics_winner:
        return scores, False   # physics agrees with the ViT — nothing to correct

    corrected = [dict(s) for s in scores]
    corrected[0]["score"], corrected[1]["score"] = corrected[1]["score"], corrected[0]["score"]
    corrected.sort(key=lambda s: s["score"], reverse=True)
    return corrected, True


def analyze_waste_pipeline(image_path: str, conf=None) -> dict:
    """
    Full hybrid analysis: detect → pad → classify (+ physics tie-break) →
    scale carbon by box area.

    Returns a JSON-serialisable dict::

        {
          "items": [
            {
              "id": 0,
              "class_name": "plastic",              # final material verdict
              "display_name": "Plastic",
              "confidence": 0.83,                    # top score after tie-break
              "box_confidence": 0.41,                # Stage-1 localization score
              "located_as": "Plastic",               # detector's own label (diagnostic)
              "bbox": [x1, y1, x2, y2],
              "box_area_px": 10000.0,                # (x2-x1)*(y2-y1) volume proxy
              "material_scores": [...],              # ViT distribution (post tie-break)
              "physics": {                           # Method B evidence
                "laplacian_variance": 812.4,
                "edge_density": 0.13,
                "plasticity_index": 0.94,
                "tiebreak_applied": false
              },
              "carbon_factor_kg_per_kg": 3.1,        # base material coefficient
              "estimated_carbon_kg": 3.875           # base x (box area / gamma)
            }
          ],
          "image": {"width": 1280, "height": 720}
        }

    ``conf`` overrides the Stage-1 threshold for this call (clamped to
    [0.01, 1.0]). The default (``CONFIDENCE_THRESHOLD``, 0.15) is deliberately
    LOW: Stage 1 is recall-first, and Stage 2's material scores carry the
    per-item certainty. No detections is a valid result, not an error.
    """
    if not os.path.isfile(image_path):
        raise ApiError("Uploaded image could not be found for detection.", status_code=400)

    default_conf = float(current_app.config["CONFIDENCE_THRESHOLD"])
    conf_threshold = default_conf if conf is None else float(conf)
    conf_threshold = max(0.01, min(conf_threshold, 1.0))  # keep within a sane range

    # Stage 1: where is the waste, and how big is each box?
    instances, image_size = _locate_objects(image_path, conf_threshold)

    # Processing layer + Stage 2: what material is each object?
    patches = _prepare_patches(image_path, instances)
    score_lists = classify_crops(patches)

    items = []
    for inst, scores, patch in zip(instances, score_lists, patches):
        physics = extract_classical_physics_features(patch)
        scores, corrected = _apply_plasticity_tiebreak(scores, physics)
        top = scores[0]  # sorted desc (classify_crops + tie-break preserve this)
        material = top["label"]
        items.append({
            "id": len(items),
            "class_name": material,
            "display_name": DISPLAY_NAMES.get(material, material.title()),
            "confidence": top["score"],
            "box_confidence": inst["box_confidence"],
            "located_as": inst["located_as"],
            "bbox": inst["bbox"],
            "box_area_px": inst["box_area_px"],
            "material_scores": scores,
            "physics": {**physics, "tiebreak_applied": corrected},
            "carbon_factor_kg_per_kg": get_carbon_factor(material),
            "estimated_carbon_kg": estimate_dynamic_impact(material, inst["box_area_px"]),
        })

    return {"items": items, "image": image_size}
