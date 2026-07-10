"""
Carbon service — PLACEHOLDER emission multipliers (Step 5 wires in Climatiq).

Stage 2 of the pipeline (classification_service) resolves every detected object
to one of the 7 material classes. This module maps those material strings onto
per-kg CO2e multipliers so the rest of the stack (API response, frontend,
tests) can already carry carbon figures end-to-end.

IMPORTANT — the numbers below are DUMMY factors, order-of-magnitude realistic
(sourced from common LCA literature) but NOT authoritative. Step 5 replaces the
lookup with live Climatiq API calls (material + weight + country); the public
functions keep the same signatures so callers do not change.
"""
import logging

logger = logging.getLogger(__name__)

# kg CO2e emitted per kg of material — PLACEHOLDER values only.
# Keys MUST match classification_service.MATERIAL_CLASSES exactly (the raw
# material strings ARE the join key between classification and carbon).
DUMMY_CARBON_FACTORS = {
    "biodegradable": 0.57,     # landfill methane from organics, CO2e-adjusted
    "cardboard": 0.94,         # pulping + corrugation
    "glass": 0.85,             # energy-heavy furnaces, but inert material
    "metal": 4.50,             # blended cans/foil figure (aluminium ~9, steel ~2)
    "paper": 1.09,             # virgin-fibre paper production
    "plastic": 3.10,           # PET/HDPE production is highly carbon-intensive
    "general rubbish": 1.20,   # mixed municipal solid waste average
}

# Conservative fallback when a label has no mapping (should not happen while
# the vocabulary is hardcoded, but open-vocab labels may grow later).
DEFAULT_CARBON_FACTOR = 1.00

# Calibration constant for the pixel-area dynamic scaling (the academic core
# feature): gamma is the reference pixel density — an area of exactly gamma
# pixels scores 1x its base material coefficient. Larger areas scale up,
# smaller areas scale down, so on-screen size acts as a physical volume/mass
# proxy until real user-entered weights arrive with Climatiq (Step 5).
#
# RECALIBRATED for BOX areas (v3.2): the detector supplies rectangular
# (x2-x1)*(y2-y1) areas, and a bounding rectangle over-covers a tight object
# contour by ~1.6x on measured waste samples (mask/box fill factor ~0.6).
# The mask-era gamma of 5000 is therefore scaled to 5000 / 0.625 = 8000 so
# carbon magnitudes stay comparable across the locator generations.
PIXEL_AREA_GAMMA = 8000.0


def get_carbon_factor(label: str) -> float:
    """Return the per-kg CO2e multiplier for a detected label string."""
    factor = DUMMY_CARBON_FACTORS.get(label)
    if factor is None:
        logger.warning("No carbon factor for label '%s'; using default %.2f",
                       label, DEFAULT_CARBON_FACTOR)
        return DEFAULT_CARBON_FACTOR
    return factor


def estimate_impact(label: str, weight_kg: float) -> float:
    """
    Estimate the CO2e (kg) for ``weight_kg`` of the detected material.

    Step 5 swaps the internals for an async Climatiq call keyed on material,
    weight and ISO country code — same signature, real data.
    """
    if weight_kg < 0:
        raise ValueError("weight_kg must be non-negative")
    return round(get_carbon_factor(label) * weight_kg, 4)


def estimate_dynamic_impact(label: str, area_px: float) -> float:
    """
    Pixel-area dynamic carbon estimate (kg CO2e) for one detected instance::

        Box Area            = (x2 - x1) * (y2 - y1)
        Final Carbon Impact = Base Material Coefficient x (Box Area / gamma)

    The bounding box's geometric pixel area stands in for the item's physical
    volume/mass until Step 5 introduces user-entered weights. ``gamma``
    (PIXEL_AREA_GAMMA, recalibrated to 8000 for rectangular over-coverage) is
    the reference pixel density.
    """
    if area_px < 0:
        raise ValueError("area_px must be non-negative")
    return round(get_carbon_factor(label) * (area_px / PIXEL_AREA_GAMMA), 4)
