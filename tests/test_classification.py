"""
Unit tests for app/services/classification_service.py (Stage 2, TrashNet ViT).

The Hugging Face pipeline is monkeypatched — no downloads, no GPU. We pin the
normalisation contract: list-of-lists aligned with the input patches, each
list sorted by descending score, scores rounded, the model's native "trash"
label mapped onto the system taxonomy's "general rubbish", and the full
distribution requested via top_k.
"""
from app.services import classification_service as cs


def _fake_pipeline(return_value):
    """A stand-in HF image-classification pipeline: callable returning a canned result."""
    def pipeline(images, top_k=None):
        # The service must ask for the FULL distribution (top_k clamps to
        # num_labels server-side; the default of 5 would truncate 7 classes).
        assert top_k is not None and top_k >= len(cs.MATERIAL_CLASSES)
        return return_value
    return pipeline


def test_empty_input_short_circuits(app):
    with app.app_context():
        assert cs.classify_crops([]) == []   # never touches the model


def test_batch_output_is_normalised_sorted_and_mapped(app, monkeypatch):
    raw = [
        [{"label": "glass", "score": 0.2}, {"label": "plastic", "score": 0.7},
         {"label": "trash", "score": 0.1}],
        [{"label": "metal", "score": 0.9}, {"label": "glass", "score": 0.1}],
    ]
    monkeypatch.setattr(cs, "get_classifier", lambda: _fake_pipeline(raw))

    with app.app_context():
        out = cs.classify_crops(["patch1", "patch2"])

    assert len(out) == 2
    assert out[0][0] == {"label": "plastic", "score": 0.7}   # re-sorted desc
    # The model's native "trash" label is mapped onto the system taxonomy.
    assert [s["label"] for s in out[0]] == ["plastic", "glass", "general rubbish"]
    assert out[1][0]["label"] == "metal"


def test_single_image_flat_result_is_wrapped(app, monkeypatch):
    # With one input the HF pipeline returns a flat list of dicts, not a batch.
    raw = [{"label": "cardboard", "score": 0.55}, {"label": "trash", "score": 0.45}]
    monkeypatch.setattr(cs, "get_classifier", lambda: _fake_pipeline(raw))

    with app.app_context():
        out = cs.classify_crops(["only-patch"])

    assert len(out) == 1
    assert out[0][0]["label"] == "cardboard"
    assert out[0][1]["label"] == "general rubbish"           # trash -> taxonomy


def test_every_native_vit_label_lands_in_the_taxonomy():
    """The checkpoint's id2label (verified from its config.json) must resolve
    onto MATERIAL_CLASSES after mapping — otherwise carbon lookups fall back."""
    native_labels = ["biodegradable", "cardboard", "glass", "metal",
                     "paper", "plastic", "trash"]
    for label in native_labels:
        mapped = cs.MODEL_LABEL_TO_MATERIAL.get(label, label)
        assert mapped in cs.MATERIAL_CLASSES


def test_resolve_device_mapping():
    assert cs._resolve_device("cpu") == -1
    assert cs._resolve_device("CPU") == -1
    assert cs._resolve_device("0") == 0
    assert cs._resolve_device("1") == 1
    assert cs._resolve_device("nonsense") == -1   # safe fallback
