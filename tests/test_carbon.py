"""
Unit tests for app/services/carbon_service.py (placeholder multipliers).

These pin the dummy-factor contract the two-stage pipeline relies on until
Step 5 swaps the internals for live Climatiq calls. No network involved.
"""
import pytest

from app.services import carbon_service as cs


def test_known_material_returns_its_factor():
    assert cs.get_carbon_factor("metal") == 4.50
    assert cs.get_carbon_factor("general rubbish") == 1.20


def test_unknown_label_falls_back_to_default():
    assert cs.get_carbon_factor("mystery item") == cs.DEFAULT_CARBON_FACTOR


def test_estimate_impact_multiplies_by_weight():
    # 0.5 kg of plastic at 3.10 kg CO2e/kg
    assert cs.estimate_impact("plastic", 0.5) == pytest.approx(1.55)


def test_estimate_impact_zero_weight_is_zero():
    assert cs.estimate_impact("glass", 0.0) == 0.0


def test_estimate_impact_rejects_negative_weight():
    with pytest.raises(ValueError):
        cs.estimate_impact("cardboard", -1.0)


def test_dynamic_impact_scales_with_box_area():
    # base x (area / gamma): a box of exactly gamma pixels scores 1x its base.
    assert cs.PIXEL_AREA_GAMMA == 8000.0   # recalibrated for rectangular over-coverage
    assert cs.estimate_dynamic_impact("plastic", cs.PIXEL_AREA_GAMMA) == 3.10
    assert cs.estimate_dynamic_impact("plastic", 16000.0) == pytest.approx(6.2)
    assert cs.estimate_dynamic_impact("glass", 0.0) == 0.0


def test_dynamic_impact_unknown_label_uses_default_factor():
    expected = round(cs.DEFAULT_CARBON_FACTOR * (2500 / cs.PIXEL_AREA_GAMMA), 4)
    assert cs.estimate_dynamic_impact("mystery item", 2500) == expected


def test_dynamic_impact_rejects_negative_area():
    with pytest.raises(ValueError):
        cs.estimate_dynamic_impact("metal", -10.0)


# --- disposal-path matrix (Module 3 factor side) -----------------------------

def test_disposal_factor_lookup_supports_negative_credits():
    assert cs.get_disposal_factor("plastic", "recycling") < 0    # net offset
    assert cs.get_disposal_factor("glass", "landfill") > 0       # net burden


def test_estimate_disposal_impact_scales_by_weight():
    # -4.10 kg CO2e/kg credit for metal recycling, 2 kg -> -8.2 net.
    assert cs.estimate_disposal_impact("metal", "recycling", 2.0) == \
        pytest.approx(-8.2)
    assert cs.estimate_disposal_impact("paper", "landfill", 0.0) == 0.0


def test_estimate_disposal_impact_rejects_negative_weight():
    with pytest.raises(ValueError):
        cs.estimate_disposal_impact("plastic", "landfill", -1.0)


def test_disposal_factor_unknown_combination_fails_loudly():
    from app.utils.errors import ApiError
    with pytest.raises(ApiError) as exc:
        cs.get_disposal_factor("plastic", "composting")   # organics-only path
    assert exc.value.status_code == 400
    with pytest.raises(ApiError):
        cs.get_disposal_factor("unobtainium", "landfill")
