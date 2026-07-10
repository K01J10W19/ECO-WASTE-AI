"""
Carbon service — Climatiq-backed emission estimates with a local fallback.

Stage 2 of the pipeline (classification_service) resolves every detected object
to one of the 7 material classes. This module turns those material strings into
CO2e figures two ways:

  * LIVE (Step 5): when ``CLIMATIQ_API_KEY`` is configured, per-kg emission
    factors come from the Climatiq API (material + ISO country code), fetched
    once per (material, country) and cached for the process lifetime. Weighted
    items are then scaled locally — one upstream call per unique material, not
    per item.
  * FALLBACK: with no key (local dev, tests, offline), the DUMMY per-kg
    coefficients below keep the whole stack working end-to-end. The app MUST
    always boot and pass tests without any API key (CLAUDE.md hard rule).

Public lookup signatures (``get_carbon_factor``, ``estimate_impact``,
``estimate_dynamic_impact``) are unchanged from the placeholder era;
``calculate_impact`` is the Step-5 aggregate entry used by
``POST /api/calculate-impact``. All upstream failures surface as clean
``ApiError``s — an unreachable carbon API must never 500 silently.
"""
import logging
from functools import lru_cache

from flask import current_app, has_app_context

from app.utils.errors import ApiError

logger = logging.getLogger(__name__)

# kg CO2e emitted per kg of material — FALLBACK values, order-of-magnitude
# realistic (common LCA literature) but NOT authoritative. Keys MUST match
# classification_service.MATERIAL_CLASSES exactly (the raw material strings
# ARE the join key between classification and carbon).
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
# proxy where no user-entered weight exists.
#
# RECALIBRATED for BOX areas (v3.2): the detector supplies rectangular
# (x2-x1)*(y2-y1) areas, and a bounding rectangle over-covers a tight object
# contour by ~1.6x on measured waste samples (mask/box fill factor ~0.6).
# The mask-era gamma of 5000 is therefore scaled to 5000 / 0.625 = 8000 so
# carbon magnitudes stay comparable across the locator generations.
PIXEL_AREA_GAMMA = 8000.0

# ---------------------------------------------------------------------------
# Climatiq integration (Step 5).
#
# Each material maps to a Climatiq activity id for end-of-life treatment.
# NOTE for the operator: confirm/adjust these ids in the Climatiq Data
# Explorer (https://www.climatiq.io/data) for your data plan — activity ids
# vary by source dataset and data_version. A wrong id fails loudly with the
# API's own message (never silently).
# ---------------------------------------------------------------------------
CLIMATIQ_ESTIMATE_URL = "https://api.climatiq.io/data/v1/estimate"
CLIMATIQ_DATA_VERSION = "^21"
CLIMATIQ_TIMEOUT_S = 10
# All seven ids verified LIVE against data_version ^21 (BEIS GB dataset,
# landfill end-of-life) on 2026-07-10 with a real key — factors resolve.
MATERIAL_TO_CLIMATIQ_ACTIVITY = {
    "biodegradable": "waste-type_organic_food_and_drink-disposal_method_landfill",
    "cardboard": "waste-type_cardboard-disposal_method_landfill",
    "glass": "waste-type_glass-disposal_method_landfill",
    "metal": "waste-type_metals-disposal_method_landfill",
    "paper": "waste-type_paper-disposal_method_landfill",
    "plastic": "waste-type_plastics-disposal_method_landfill",
    "general rubbish": "waste-type_household_residual_waste-disposal_method_landfill",
}


def _climatiq_api_key() -> str:
    """The configured Climatiq key, or '' outside an app context / when unset."""
    if not has_app_context():
        return ""
    return str(current_app.config.get("CLIMATIQ_API_KEY", "") or "")


@lru_cache(maxsize=64)
def _fetch_climatiq_factor(material: str, country: str, api_key: str) -> float:
    """
    Fetch the per-kg CO2e factor for ``material`` from Climatiq (cached).

    Asks the estimate endpoint for exactly 1 kg, so the result IS the per-kg
    factor; weighted items are then scaled locally without further calls.
    ``country`` scopes the emission factor region when provided ('' = global).
    Raises ``ApiError`` on any upstream problem (auth, unknown activity id,
    timeout, network) with a user-facing message.
    """
    import requests  # local import keeps module import light for tests

    selector = {
        "activity_id": MATERIAL_TO_CLIMATIQ_ACTIVITY[material],
        "data_version": CLIMATIQ_DATA_VERSION,
    }
    if country:
        selector["region"] = country.upper()
    payload = {
        "emission_factor": selector,
        "parameters": {"weight": 1, "weight_unit": "kg"},
    }
    try:
        resp = requests.post(
            CLIMATIQ_ESTIMATE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=CLIMATIQ_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as exc:
        raise ApiError("Carbon provider (Climatiq) is unreachable — try again "
                       "later or remove the API key to use local estimates.",
                       status_code=502) from exc

    if resp.status_code != 200:
        detail = ""
        try:
            detail = str(resp.json().get("message", ""))[:200]
        except Exception:  # noqa: BLE001 - body may not be JSON
            pass
        # Region miss (e.g. no BEIS factor published for "MY"): fall back to
        # the region-unscoped factor rather than failing the whole request.
        if resp.status_code == 400 and country:
            logger.warning("Climatiq has no '%s' factor for region %s; "
                           "falling back to the global factor.", material, country)
            return _fetch_climatiq_factor(material, "", api_key)
        raise ApiError(f"Climatiq rejected the request for '{material}' "
                       f"(HTTP {resp.status_code}). {detail}".strip(),
                       status_code=502)

    try:
        co2e = float(resp.json()["co2e"])
    except Exception as exc:  # noqa: BLE001 - malformed upstream payload
        raise ApiError("Climatiq returned an unexpected response shape.",
                       status_code=502) from exc

    logger.info("Climatiq factor %s (%s): %.4f kgCO2e/kg",
                material, country or "global", co2e)
    return co2e


def get_carbon_factor(label: str) -> float:
    """Return the FALLBACK per-kg CO2e multiplier for a material string.

    Deliberately local/deterministic — this feeds the pixel-area proxy in the
    detection payload, which must work offline. Live Climatiq factors are used
    by ``estimate_impact`` / ``calculate_impact`` when a key is configured.
    """
    factor = DUMMY_CARBON_FACTORS.get(label)
    if factor is None:
        logger.warning("No carbon factor for label '%s'; using default %.2f",
                       label, DEFAULT_CARBON_FACTOR)
        return DEFAULT_CARBON_FACTOR
    return factor


def _resolve_factor(label: str, country: str) -> tuple:
    """(per-kg factor, source) — Climatiq when configured+mapped, else dummy."""
    api_key = _climatiq_api_key()
    if api_key and label in MATERIAL_TO_CLIMATIQ_ACTIVITY:
        return _fetch_climatiq_factor(label, country or "", api_key), "climatiq"
    return get_carbon_factor(label), "local_dummy"


def estimate_impact(label: str, weight_kg: float, country: str = None) -> float:
    """
    Estimate the CO2e (kg) for ``weight_kg`` of the detected material.

    Uses the live Climatiq factor when an API key is configured (optionally
    scoped by ISO ``country``), the local dummy factor otherwise. Same
    signature as the placeholder era (``country`` is additive and optional).
    """
    if weight_kg < 0:
        raise ValueError("weight_kg must be non-negative")
    factor, _ = _resolve_factor(label, country)
    return round(factor * weight_kg, 4)


def estimate_dynamic_impact(label: str, area_px: float) -> float:
    """
    Pixel-area dynamic carbon estimate (kg CO2e) for one detected instance::

        Box Area            = (x2 - x1) * (y2 - y1)
        Final Carbon Impact = Base Material Coefficient x (Box Area / gamma)

    The bounding box's geometric pixel area stands in for the item's physical
    volume/mass when no user-entered weight exists. Deliberately uses the
    LOCAL base coefficients (offline-safe; the /predict path never blocks on
    a network call). ``gamma`` = PIXEL_AREA_GAMMA.
    """
    if area_px < 0:
        raise ValueError("area_px must be non-negative")
    return round(get_carbon_factor(label) * (area_px / PIXEL_AREA_GAMMA), 4)


def calculate_impact(items: list, country: str = None) -> dict:
    """
    Step-5 aggregate: real CO2e for user-weighted items (POST /api/calculate-impact).

    ``items`` is a list of ``{"material": str, "weight_kg": float}`` dicts
    (already type-validated by the pydantic schema at the route). Returns::

        {
          "items": [ { material, weight_kg, carbon_factor_kg_per_kg,
                       co2e_kg, source } ],
          "total_co2e_kg": 4.83,
          "country": "MY" | None,
          "provider": "climatiq" | "local_dummy"
        }

    Unknown materials are a clean 400 (the frontend only ever submits the
    7-class taxonomy). One factor fetch per unique material (cached), so a
    20-item scan costs at most 7 upstream calls — effectively 'async enough'
    for a single request/response cycle.
    """
    from app.services.classification_service import MATERIAL_CLASSES

    results, total = [], 0.0
    sources = set()
    for entry in items:
        material = entry["material"]
        weight_kg = float(entry["weight_kg"])
        if material not in MATERIAL_CLASSES:
            raise ApiError(
                f"Unknown material '{material}'. Valid materials: "
                f"{', '.join(MATERIAL_CLASSES)}.",
                status_code=400,
            )
        if weight_kg <= 0:
            raise ApiError("Each item needs a positive weight_kg.", status_code=400)

        factor, source = _resolve_factor(material, country)
        co2e = round(factor * weight_kg, 4)
        total += co2e
        sources.add(source)
        results.append({
            "material": material,
            "weight_kg": weight_kg,
            "carbon_factor_kg_per_kg": round(factor, 4),
            "co2e_kg": co2e,
            "source": source,
        })

    return {
        "items": results,
        "total_co2e_kg": round(total, 4),
        "country": country.upper() if country else None,
        "provider": "climatiq" if sources == {"climatiq"} else
                    ("local_dummy" if sources == {"local_dummy"} else "mixed"),
    }
