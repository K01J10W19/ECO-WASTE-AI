"""
Classification service — Stage 2 of the Dual-Tower Hybrid: a SUPERVISED
Vision Transformer fine-tuned on TrashNet-enhanced.

The zero-shot CLIP classifier is retired (archived — see CLAUDE.md §11). Stage
2 is now ``edwinpalegre/ee8225-group4-vit-trashnet-enhanced``: ViT-B/16
(``google/vit-base-patch16-224-in21k``) fine-tuned on the trashnet-enhanced
dataset (98.17% validation accuracy, Apache-2.0). Unlike CLIP's text-image
matching, this model has *seen thousands of real waste items* — its global
self-attention evaluates texture, gloss and material properties of the
normalized 224x224 patches produced by the processing layer.

The model's NATIVE labels (verified from its config.json id2label) are:

    biodegradable, cardboard, glass, metal, paper, plastic, trash

— seven classes that map 1:1 onto the system taxonomy below; only ``trash``
is renamed to ``general rubbish``. The raw material string remains the
system-wide join key (display names, carbon coefficients, tests).

Mirrors detection_service's conventions: cached singleton pipeline, lazy heavy
imports, failures surfaced as ``ApiError``.
"""
import logging
from functools import lru_cache

from flask import current_app

from app.utils.errors import ApiError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LOCKED 7-class material taxonomy (the system-wide output vocabulary).
#
# These raw strings are the join key: display names (below),
# carbon_service.DUMMY_CARBON_FACTORS + DISPOSAL_METHOD_FACTORS and the DMM's
# DISPOSAL_PATHS / EXPERT_KNOWLEDGE are all keyed on them. Unit tests enforce
# lockstep (full 7x3 coverage on the recommendation side).
# ---------------------------------------------------------------------------
MATERIAL_CLASSES = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "general rubbish",
]

# Friendly labels for the web UI (the raw material string stays in `class_name`).
DISPLAY_NAMES = {
    "biodegradable": "Biodegradable",
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "general rubbish": "General Rubbish",
}

# The ViT's native label -> system taxonomy. The fine-tuned checkpoint emits
# "trash" as its catch-all; every other label already matches the taxonomy.
# Unknown labels (a different checkpoint) pass through unchanged and fall back
# to the default carbon factor downstream.
MODEL_LABEL_TO_MATERIAL = {
    "trash": "general rubbish",
}

# The model has 7 labels; ask for generously more — the pipeline clamps top_k
# to num_labels, so this always returns the FULL distribution.
_TOP_K = 32


def _resolve_device(device: str) -> int:
    """Map the app's INFERENCE_DEVICE ("cpu" | "0" | "1"...) onto the
    transformers pipeline convention (-1 = CPU, N = CUDA device N)."""
    if str(device).lower() == "cpu":
        return -1
    try:
        return int(device)
    except (TypeError, ValueError):
        logger.warning("Unrecognised INFERENCE_DEVICE %r; falling back to CPU", device)
        return -1


@lru_cache(maxsize=1)
def _load_classifier(model_name: str, device: str):
    """
    Load and cache the ViT image-classification pipeline (once per process).

    The Hugging Face weights (~343 MB for the TrashNet ViT) download to the
    local HF cache on first load — internet needed once. ``lru_cache`` only
    stores successful loads, so a failed download can be retried.
    """
    try:
        from transformers import pipeline  # lazy: keep HF out of app import time
        logger.info("Loading ViT material classifier '%s' (device=%s)", model_name, device)
        return pipeline(
            "image-classification",
            model=model_name,
            device=_resolve_device(device),
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure as ApiError
        raise ApiError("Failed to load the ViT material classifier.", status_code=500) from exc


def get_classifier():
    """Return the cached ViT pipeline for the app's configured model + device."""
    return _load_classifier(
        str(current_app.config["VIT_MODEL_NAME"]),
        str(current_app.config.get("INFERENCE_DEVICE", "cpu")),
    )


def classify_crops(crops) -> list:
    """
    Classify a batch of normalized 224x224 PIL patches with the TrashNet ViT.

    Returns one score list per patch, labels already mapped onto the system
    taxonomy and sorted by descending score::

        [
          [{"label": "plastic", "score": 0.83}, {"label": "glass", "score": 0.07}, ...],
          ...
        ]

    An empty input is a valid no-op (returns []). The full softmax
    distribution is returned (not just the winner) so the UI can show the
    model's evidence per item.
    """
    if not crops:
        return []

    classifier = get_classifier()
    try:
        results = classifier(crops, top_k=_TOP_K)
    except Exception as exc:  # noqa: BLE001 - any inference failure is client-facing
        raise ApiError("Material classification failed.", status_code=500) from exc

    # With a single-image input the pipeline returns a flat list of dicts;
    # normalise to always be a list-of-lists aligned with `crops`.
    if results and isinstance(results[0], dict):
        results = [results]

    return [
        sorted(
            (
                {
                    "label": MODEL_LABEL_TO_MATERIAL.get(r["label"], r["label"]),
                    "score": round(float(r["score"]), 4),
                }
                for r in per_crop
            ),
            key=lambda s: s["score"], reverse=True,
        )
        for per_crop in results
    ]
