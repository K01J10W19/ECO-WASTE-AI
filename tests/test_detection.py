"""
Unit tests for app/services/detection_service.py (Dual-Tower Hybrid v3.1).

Stage 1 (specialist waste segmenter, 5 coarse labels + instance masks) and
Stage 2 (TrashNet ViT classifier) are both monkeypatched, so these tests need
NO real weights, NO GPU and NO network. The processing layer (square padding +
Method B physics extractor) runs for real on tiny generated images. We verify
the segment → pad → classify → tie-break → carbon composition, the
polygon/mask-area payload, the pixel-area dynamic carbon formula, the psi
plastic-vs-glass tie-breaker, box clamping, the box-corner fallback when a
checkpoint yields no masks, the Stage-1 conf override, and that an empty
result is valid.
"""
import types

from PIL import Image

from app.services import detection_service as ds
from app.services import classification_service as cs
from app.services.carbon_service import PIXEL_AREA_GAMMA
from app.schemas.detection import DetectionResult

# The names dict the specialist waste segmenter exposes (verified live).
_SEG_NAMES = {0: "Glass", 1: "Metal", 2: "Paper", 3: "Plastic", 4: "Waste"}


def _fake_box(conf, xyxy, cls_id=3):
    """A stand-in for an Ultralytics Boxes row (index [0] like tensors)."""
    return types.SimpleNamespace(cls=[cls_id], conf=[conf], xyxy=[xyxy])


def _fake_model(boxes, polys=None, width=1280, height=720):
    """A stand-in waste-seg locator whose predict() returns one result.
    ``polys`` mirrors result.masks.xy (one vertex array per box, same order);
    None mimics a mask-less (plain detection) checkpoint."""
    masks = types.SimpleNamespace(xy=polys) if polys is not None else None
    result = types.SimpleNamespace(orig_shape=(height, width), boxes=boxes, masks=masks)
    return types.SimpleNamespace(names=_SEG_NAMES, predict=lambda *a, **k: [result])


def _scores(winner, score=0.83):
    """A plausible sorted ViT distribution with `winner` on top."""
    rest = [m for m in cs.MATERIAL_CLASSES if m != winner]
    remaining = round((1.0 - score) / len(rest), 4)
    return ([{"label": winner, "score": score}]
            + [{"label": m, "score": remaining} for m in rest])


def _fake_classify(per_crop_winners):
    """Return a classify_crops replacement yielding one score list per patch."""
    def classify(crops):
        assert len(crops) == len(per_crop_winners), "patch/instance misalignment"
        for crop in crops:  # the processing layer must emit 224x224 squares
            assert crop.size == (224, 224)
        return [_scores(w) for w in per_crop_winners]
    return classify


def _real_image(tmp_path, width=1280, height=720):
    """The processing layer opens the file with PIL, so it must be a real image."""
    p = tmp_path / "upload.jpg"
    Image.new("RGB", (width, height), (90, 90, 90)).save(p, format="JPEG")
    return str(p)


def _rect_poly(x1, y1, x2, y2):
    """A rectangular mask polygon (shoelace area == (x2-x1)*(y2-y1))."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_dual_tower_composition(app, monkeypatch, tmp_path):
    boxes = [
        _fake_box(0.41, [10.0, 20.0, 110.0, 120.0], cls_id=4),   # segmenter says "Waste"
        _fake_box(0.90, [200.0, 200.0, 400.0, 380.0], cls_id=3),  # segmenter says "Plastic"
    ]
    polys = [
        _rect_poly(10, 20, 110, 120),     # area 100*100 = 10,000 px²
        _rect_poly(200, 200, 400, 380),   # area 200*180 = 36,000 px²
    ]
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model(boxes, polys))
    monkeypatch.setattr(ds, "classify_crops", _fake_classify(["plastic", "glass"]))

    with app.app_context():
        out = ds.analyze_waste_pipeline(_real_image(tmp_path))

    assert out["image"] == {"width": 1280, "height": 720}
    assert len(out["items"]) == 2

    first = out["items"][0]
    assert first["class_name"] == "plastic"           # Stage-2 verdict — NOT the segmenter label
    assert first["display_name"] == "Plastic"
    assert first["confidence"] == 0.83                # ViT softmax score
    assert first["box_confidence"] == 0.41            # Stage-1 localization score
    assert first["located_as"] == "Waste"             # segmenter label kept as diagnostic only
    assert first["bbox"] == [10, 20, 110, 120]
    assert first["polygon"] == _rect_poly(10, 20, 110, 120)
    assert first["mask_area_px"] == 10000.0           # shoelace area of the mask
    assert first["material_scores"][0]["label"] == "plastic"
    assert len(first["material_scores"]) == len(cs.MATERIAL_CLASSES)
    assert first["carbon_factor_kg_per_kg"] == 3.10
    # Dynamic formula: base x (area / gamma) = 3.10 x (10000 / 5000) = 6.2
    assert first["estimated_carbon_kg"] == round(3.10 * 10000 / PIXEL_AREA_GAMMA, 4)
    # Method B evidence rides along; a clear 0.83 verdict is never tie-broken.
    assert first["physics"]["tiebreak_applied"] is False
    assert 0.0 <= first["physics"]["plasticity_index"] <= 1.0

    second = out["items"][1]
    assert second["class_name"] == "glass"            # even though the segmenter said "Plastic"
    assert second["mask_area_px"] == 36000.0
    assert second["estimated_carbon_kg"] == round(0.85 * 36000 / PIXEL_AREA_GAMMA, 4)
    DetectionResult(**out)                            # response matches the schema


def test_boxes_clamped_and_degenerate_skipped(app, monkeypatch, tmp_path):
    boxes = [
        _fake_box(0.50, [-15.0, -10.0, 900.0, 800.0]),  # spills past the 640x480 image
        _fake_box(0.60, [100.0, 100.0, 101.0, 300.0]),  # 1px wide -> unclassifiable, skipped
    ]
    polys = [
        _rect_poly(-15, -10, 900, 800),   # raw shoelace area = 915 * 810
        _rect_poly(100, 100, 101, 300),   # skipped along with its box
    ]
    monkeypatch.setattr(ds, "get_model",
                        lambda: _fake_model(boxes, polys, width=640, height=480))
    monkeypatch.setattr(ds, "classify_crops", _fake_classify(["general rubbish"]))

    with app.app_context():
        out = ds.analyze_waste_pipeline(_real_image(tmp_path, 640, 480))

    assert len(out["items"]) == 1                        # degenerate instance dropped
    item = out["items"][0]
    assert item["bbox"] == [0, 0, 640, 480]              # clamped to image bounds
    assert all(0 <= x <= 640 and 0 <= y <= 480           # polygon vertices clamped too
               for x, y in item["polygon"])
    assert item["mask_area_px"] == 915.0 * 810.0         # area from the raw mask vertices
    assert item["carbon_factor_kg_per_kg"] == 1.20


def test_maskless_checkpoint_falls_back_to_box_polygon(app, monkeypatch, tmp_path):
    # A plain detection checkpoint (masks=None) must not break the contract.
    boxes = [_fake_box(0.70, [10.0, 10.0, 60.0, 40.0])]
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model(boxes, polys=None))
    monkeypatch.setattr(ds, "classify_crops", _fake_classify(["metal"]))

    with app.app_context():
        out = ds.analyze_waste_pipeline(_real_image(tmp_path))

    item = out["items"][0]
    assert item["polygon"] == _rect_poly(10, 10, 60, 40)  # box corners stand in
    assert item["mask_area_px"] == 50.0 * 30.0            # box area stands in
    DetectionResult(**out)


def test_empty_detections_is_not_an_error(app, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_model",
                        lambda: _fake_model([], polys=[], width=640, height=480))
    # No instances -> no patches -> the REAL classify_crops([]) short-circuits to [].

    with app.app_context():
        out = ds.analyze_waste_pipeline(_real_image(tmp_path, 640, 480))

    assert out["items"] == []
    assert out["image"] == {"width": 640, "height": 480}
    DetectionResult(**out)


def test_conf_override_surfaces_more_detections(app, monkeypatch, tmp_path):
    # A 0.10-confidence instance is dropped at the 0.15 default but kept at conf=0.05.
    boxes = [
        _fake_box(0.90, [10.0, 20.0, 110.0, 120.0]),
        _fake_box(0.10, [0.0, 0.0, 50.0, 50.0]),
    ]
    polys = [_rect_poly(10, 20, 110, 120), _rect_poly(0, 0, 50, 50)]
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model(boxes, polys))

    def classify(crops):  # length adapts to however many instances survived
        return [_scores("metal") for _ in crops]
    monkeypatch.setattr(ds, "classify_crops", classify)

    with app.app_context():
        img = _real_image(tmp_path)
        default_out = ds.analyze_waste_pipeline(img)              # 0.15 default
        lowered_out = ds.analyze_waste_pipeline(img, conf=0.05)

    assert len(default_out["items"]) == 1     # low-confidence instance filtered
    assert len(lowered_out["items"]) == 2     # override lets it through


def _flat_patch(color=(90, 90, 90)):
    """A structurally smooth 224x224 patch (glass-like: psi ~ 0)."""
    return Image.new("RGB", (224, 224), color)


def _noisy_patch(seed=7):
    """A high-frequency crinkled 224x224 patch (plastic-like: psi ~ 1)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype="uint8"), "RGB")


def test_physics_extractor_separates_smooth_from_crinkled():
    smooth = ds.extract_classical_physics_features(_flat_patch())
    crinkled = ds.extract_classical_physics_features(_noisy_patch())

    assert smooth["plasticity_index"] < 0.1          # no wrinkles, no edges
    assert crinkled["plasticity_index"] > 0.9        # saturated on both cues
    assert crinkled["laplacian_variance"] > smooth["laplacian_variance"]
    assert 0.0 <= smooth["edge_density"] <= 1.0


def test_tiebreak_corrects_ambiguous_plastic_to_glass():
    # ViT narrowly favours plastic, but the patch physics reads glass-like.
    scores = [{"label": "plastic", "score": 0.48}, {"label": "glass", "score": 0.42},
              {"label": "metal", "score": 0.10}]
    corrected, applied = ds._apply_plasticity_tiebreak(
        scores, {"plasticity_index": 0.05})

    assert applied is True
    assert corrected[0] == {"label": "glass", "score": 0.48}   # ranks (scores) swapped
    assert corrected[1] == {"label": "plastic", "score": 0.42}
    assert scores[0]["label"] == "plastic"                      # input not mutated


def test_tiebreak_skips_when_physics_agrees_with_vit():
    scores = [{"label": "glass", "score": 0.48}, {"label": "plastic", "score": 0.42}]
    corrected, applied = ds._apply_plasticity_tiebreak(
        scores, {"plasticity_index": 0.05})   # psi says glass; ViT already says glass
    assert applied is False
    assert corrected == scores


def test_tiebreak_skips_clear_verdicts_and_other_materials():
    clear = [{"label": "plastic", "score": 0.83}, {"label": "glass", "score": 0.05}]
    assert ds._apply_plasticity_tiebreak(clear, {"plasticity_index": 0.0}) == (clear, False)

    other = [{"label": "paper", "score": 0.45}, {"label": "cardboard", "score": 0.40}]
    assert ds._apply_plasticity_tiebreak(other, {"plasticity_index": 1.0}) == (other, False)


def test_pipeline_applies_tiebreak_on_smooth_patch(app, monkeypatch, tmp_path):
    # Ambiguous plastic-vs-glass ViT call on a flat gray image: the real
    # physics extractor reads the patch as glass-like and flips the verdict.
    boxes = [_fake_box(0.60, [10.0, 20.0, 110.0, 120.0], cls_id=0)]
    polys = [_rect_poly(10, 20, 110, 120)]
    monkeypatch.setattr(ds, "get_model", lambda: _fake_model(boxes, polys))

    ambiguous = [{"label": "plastic", "score": 0.48}, {"label": "glass", "score": 0.42},
                 {"label": "metal", "score": 0.05}, {"label": "paper", "score": 0.05}]
    monkeypatch.setattr(ds, "classify_crops", lambda crops: [list(ambiguous)])

    with app.app_context():
        out = ds.analyze_waste_pipeline(_real_image(tmp_path))

    item = out["items"][0]
    assert item["class_name"] == "glass"              # psi corrected the argmax
    assert item["confidence"] == 0.48                 # winner takes the higher score
    assert item["physics"]["tiebreak_applied"] is True
    assert item["physics"]["plasticity_index"] < 0.5
    assert item["carbon_factor_kg_per_kg"] == 0.85    # carbon follows the corrected label
    DetectionResult(**out)


def test_polygon_area_shoelace():
    # Rectangle, triangle, and degenerate cases pin the area math the carbon
    # formula depends on.
    assert ds._polygon_area([[0, 0], [10, 0], [10, 5], [0, 5]]) == 50.0
    assert ds._polygon_area([[0, 0], [10, 0], [0, 10]]) == 50.0
    assert ds._polygon_area([[0, 0], [10, 0]]) == 0.0   # < 3 vertices


def test_taxonomy_lockstep():
    """Material classes, display names and carbon factors must stay aligned —
    the raw material string is the system-wide join key."""
    from app.services.carbon_service import DUMMY_CARBON_FACTORS

    for material in cs.MATERIAL_CLASSES:
        assert material in cs.DISPLAY_NAMES
        assert material in DUMMY_CARBON_FACTORS
