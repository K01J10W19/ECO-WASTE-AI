"""
Carbon service — the system's carbon engine (Module 2 + Module 3's factor side).

The material tower (classification_service) resolves every detected object to
one of the 7 material classes. This module turns those material strings into
CO2e figures along a DUAL-STAGE UX pipeline (v3.5):

  * STAGE A — BLIND ESTIMATE (photo upload → /api/predict): no real weight is
    known yet, so ``estimate_dynamic_impact`` prices each instance from its
    clamped bounding-box geometry: base local factor x (box_area_px / gamma).
    Deliberately 100% local and deterministic — the predict path must never
    block on a network call.
  * STAGE B — PRECISION AUDIT (/api/calculate-impact): the user verifies real
    weights (kg) and optionally an ISO 3166-1 alpha-2 country. When
    ``CLIMATIQ_API_KEY`` is configured the per-kg factor comes LIVE from the
    Climatiq estimate endpoint as a 1-kg probe, cached via
    ``lru_cache(maxsize=64)`` on the unique (material, country, api_key)
    tuple; weighted items are then scaled locally — upstream request density
    stays minimal (one call per unique factor, not per item).
  * FALLBACK: with no key (local dev, tests, offline), the DUMMY per-kg
    coefficients below keep the whole stack working end-to-end. The app MUST
    always boot and pass tests without any API key (CLAUDE.md hard rule).

MODULE 3 SUPPORT (Decision Making Module): ``DISPOSAL_METHOD_FACTORS`` +
``estimate_disposal_impact`` price the SAME item down three alternative
end-of-life routes for recommendation_service's ranked simulation. These are
deliberately local-only and app-context-free — the DMM fans them out across
worker threads and its ranking must stay deterministic and offline-safe.

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
# Module 3 (Decision Making Module) — end-of-life DISPOSAL-PATH matrix.
#
# NET kg CO2e per kg of material for each simulated end-of-life route,
# INCLUDING avoided-burden credits: NEGATIVE values are net offsets (e.g.
# recycling metal displaces energy-hungry ore smelting). Values are
# literature-order heuristics (EPA WARM / UK BEIS flavour), NOT authoritative
# — like the dummy production factors above they exist so the decision layer
# stays deterministic, offline and fully auditable (every factor used is
# echoed back in the recommendation payload).
#
# GHG-only lens, documented honestly: landfilled plastic is biologically
# inert, so it out-scores incineration on pure CO2e — the DMM's knowledge
# base carries the microplastic caveat that this number cannot see.
#
# Keys MUST stay in lockstep with classification_service.MATERIAL_CLASSES and
# recommendation_service.DISPOSAL_PATHS — tests enforce full 7x3 coverage.
# ---------------------------------------------------------------------------
DISPOSAL_METHOD_FACTORS = {
    "plastic": {
        "recycling": -1.08,       # avoided virgin polymer (petroleum) production
        "incineration": 2.35,     # fossil carbon to atmosphere, minus energy credit
        "landfill": 0.09,         # biologically inert: collection/equipment only
    },
    "glass": {
        "recycling": -0.31,       # cullet remelt beats virgin batch calcination
        "incineration": 0.03,     # non-combustible: furnace dead-weight, no energy
        "landfill": 0.02,         # chemically stable, no gas generation
    },
    "metal": {
        "recycling": -4.10,       # smelting avoidance (Al ~ -9, steel ~ -1.8, blended)
        "incineration": 0.03,     # passes to bottom ash; no calorific contribution
        "landfill": 0.02,         # structurally stable; embodied energy forfeited
    },
    "cardboard": {
        "recycling": -0.96,       # repulping displaces virgin kraft pulping
        "incineration": 0.07,     # biogenic carbon, near-neutral after energy credit
        "landfill": 1.10,         # anaerobic fibre decomposition -> methane
    },
    "paper": {
        "recycling": -0.89,       # avoided virgin pulping (most energy-intense stage)
        "incineration": 0.09,     # biogenic carbon, near-neutral after energy credit
        "landfill": 1.29,         # most methane-productive landfill fibre
    },
    "biodegradable": {
        "composting": 0.05,       # small process CH4/N2O; soil-carbon return
        "anaerobic_digestion": -0.14,  # captured biogas displaces fossil energy
        "landfill": 0.90,         # uncontrolled anaerobic decomposition -> methane
    },
    "general rubbish": {
        "material_recovery": 0.30,  # MRF/MBT residual sorting: modest reclaim credit
        "incineration": 0.45,       # mixed-stream WtE: fossil fraction minus energy
        "landfill": 1.20,           # decades of methane + leachate management
    },
}

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
    """(per-kg factor, source) — Climatiq when configured+mapped, else dummy.

    ``country`` is normalised (upper-cased, '' when absent) BEFORE the cached
    probe so "my" and "MY" share one cache slot; an empty value omits the
    region selector entirely and Climatiq falls back to its global dataset.
    """
    api_key = _climatiq_api_key()
    if api_key and label in MATERIAL_TO_CLIMATIQ_ACTIVITY:
        return _fetch_climatiq_factor(label, (country or "").upper(), api_key), "climatiq"
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


def resolve_effective_weight(entry: dict) -> tuple:
    """
    (effective_weight_kg, weight_source) for one request item — the dual-stage
    weight substitution shared by ``/api/calculate-impact`` and the DMM.

    A user-verified ``weight_kg`` (the Stage-B audit value) always wins; a
    ``box_area_px`` falls back to the blind pixel proxy, clamped box area /
    ``PIXEL_AREA_GAMMA`` — the exact calibration the /predict payload uses.
    At least one of the two is required (schemas enforce this too; the
    service double-checks). ``weight_source`` is ``"user_weight"`` or
    ``"box_area_proxy"`` so every response stays honest about provenance.
    """
    weight_kg = entry.get("weight_kg")
    if weight_kg is not None:
        weight_kg = float(weight_kg)
        if weight_kg <= 0:
            raise ApiError("Each item needs a positive weight_kg.", status_code=400)
        return weight_kg, "user_weight"

    box_area_px = entry.get("box_area_px")
    if box_area_px is not None:
        box_area_px = float(box_area_px)
        if box_area_px <= 0:
            raise ApiError("box_area_px must be positive.", status_code=400)
        return box_area_px / PIXEL_AREA_GAMMA, "box_area_proxy"

    raise ApiError("Each item needs weight_kg (user-verified) or box_area_px "
                   "(the blind pixel proxy).", status_code=400)


def get_disposal_factor(material: str, method: str) -> float:
    """
    Net per-kg CO2e factor for sending ``material`` down one ``method``
    (end-of-life route). NEGATIVE values are net offsets (avoided-burden
    credits — e.g. recycling displacing virgin production).

    LOCAL and app-context-free by design: the Decision Making Module calls
    this from worker threads (no Flask context available) and its ranking
    must never block on the network. Unknown combinations fail loudly.
    """
    try:
        return DISPOSAL_METHOD_FACTORS[material][method]
    except KeyError as exc:
        valid = ", ".join(DISPOSAL_METHOD_FACTORS.get(material, {})) or "none"
        raise ApiError(
            f"No disposal factor for material '{material}' via method "
            f"'{method}'. Valid methods for this material: {valid}.",
            status_code=400,
        ) from exc


def estimate_disposal_impact(material: str, method: str, weight_kg: float) -> float:
    """
    Net CO2e (kg) for ``weight_kg`` of ``material`` down one end-of-life
    route: disposal factor x weight. May be NEGATIVE (a net carbon offset).
    Pure local arithmetic — safe for the DMM's parallel path fan-out.
    """
    if weight_kg < 0:
        raise ValueError("weight_kg must be non-negative")
    return round(get_disposal_factor(material, method) * weight_kg, 4)


def calculate_impact(items: list, country: str = None) -> dict:
    """
    Stage-B aggregate: real CO2e for verified items (POST /api/calculate-impact).

    ``items`` is a list of ``{"id": int?, "material": str, "weight_kg": float?,
    "box_area_px": float?}`` dicts (already type-validated by the pydantic
    schema at the route). Each item needs at least one size signal: a
    user-verified ``weight_kg`` wins, otherwise ``box_area_px / gamma`` is the
    blind pixel-proxy substitute (``resolve_effective_weight``). The optional
    ``id`` is the client's grid/canvas row key — echoed back VERBATIM per item
    so the split-screen UI can track edits bi-directionally. Returns::

        {
          "items": [ { id, material, weight_kg (effective), weight_source,
                       carbon_factor_kg_per_kg, co2e_kg, source } ],
          "total_co2e_kg": 4.83,
          "country": "MY" | None,
          "provider": "climatiq" | "local_dummy" | "mixed"
        }

    ``country`` may be omitted/blank (global factors) or an ISO alpha-2 code —
    typically the frontend's IP-geolocated default — which region-scopes the
    live factors. Unknown materials are a clean 400. One factor fetch per
    unique (material, country) via the cache, so a 20-item scan costs at most
    7 upstream calls.
    """
    from app.services.classification_service import MATERIAL_CLASSES

    results, total = [], 0.0
    sources = set()
    for entry in items:
        material = entry["material"]
        if material not in MATERIAL_CLASSES:
            raise ApiError(
                f"Unknown material '{material}'. Valid materials: "
                f"{', '.join(MATERIAL_CLASSES)}.",
                status_code=400,
            )
        weight_kg, weight_source = resolve_effective_weight(entry)

        factor, source = _resolve_factor(material, country)
        co2e = round(factor * weight_kg, 4)
        total += co2e
        sources.add(source)
        results.append({
            "id": entry.get("id"),
            "material": material,
            "weight_kg": round(weight_kg, 4),
            "weight_source": weight_source,
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
