"""
Recommendation service — Module 3: the DECISION MAKING MODULE (DMM).

The DMM converts the pipeline's quantitative carbon data into qualitative,
ORDERED prescriptions. Instead of pricing a single default disposal path
(landfill), each incoming item is forked into THREE end-of-life simulations
that run in parallel, branched by the material's taxonomy:

    dry recyclables (plastic/glass/metal/cardboard/paper)
        -> recycling | incineration | landfill
    organics (biodegradable)
        -> composting | anaerobic_digestion | landfill
    residual (general rubbish)
        -> material_recovery | incineration | landfill

Each path's net CO2e comes from the carbon engine
(``carbon_service.estimate_disposal_impact`` — local matrix, credits allowed
to go NEGATIVE). The SORTING ENGINE then ranks the three outcomes ascending
(lowest footprint / deepest offset wins): Rank 1 = Optimal green path,
Rank 3 = worst-case baseline. Every ranked choice is annotated from the
structured EXPERT KNOWLEDGE base (professional pros/cons per material+method)
plus a rank-aware encouraging verdict, so the frontend receives prescriptive,
auditable guidance — not just numbers.

Weight resolution mirrors the dual-stage carbon UX:
  * ``weight_kg`` present  -> user-verified weight (the precision audit value);
  * else ``box_area_px``   -> the blind pixel proxy, box_area / gamma
    (the same PIXEL_AREA_GAMMA calibration the /predict payload uses).

DESIGN GUARANTEES (CLAUDE.md):
  * Deterministic + 100% offline — the DMM never touches the network, needs
    no API key, and needs no Flask app context (thread-pool safe). Live
    region-scoped factors belong to Module 2's audit endpoint, not here.
  * The 7-class taxonomy stays in lockstep: DISPOSAL_PATHS, the factor
    matrix and EXPERT_KNOWLEDGE cover every material — tests enforce it.
  * Expected failures raise ``ApiError``; nothing 500s silently.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from app.services.carbon_service import (
    PIXEL_AREA_GAMMA,
    estimate_disposal_impact,
    get_disposal_factor,
)
from app.services.classification_service import DISPLAY_NAMES, MATERIAL_CLASSES
from app.utils.errors import ApiError

logger = logging.getLogger(__name__)

# Every item forks into exactly this many parallel end-of-life simulations.
_PARALLEL_PATHS = 3

# ---------------------------------------------------------------------------
# Taxonomy-branched path matrix: which 3 end-of-life methods each material
# forks into. Keys stay in lockstep with MATERIAL_CLASSES (tests enforce).
# ---------------------------------------------------------------------------
_DRY_RECYCLABLE_PATHS = ("recycling", "incineration", "landfill")
_ORGANIC_PATHS = ("composting", "anaerobic_digestion", "landfill")
_RESIDUAL_PATHS = ("material_recovery", "incineration", "landfill")

DISPOSAL_PATHS = {
    "plastic": _DRY_RECYCLABLE_PATHS,
    "glass": _DRY_RECYCLABLE_PATHS,
    "metal": _DRY_RECYCLABLE_PATHS,
    "cardboard": _DRY_RECYCLABLE_PATHS,
    "paper": _DRY_RECYCLABLE_PATHS,
    "biodegradable": _ORGANIC_PATHS,
    "general rubbish": _RESIDUAL_PATHS,
}

METHOD_DISPLAY_NAMES = {
    "recycling": "Recycling",
    "incineration": "Incineration (Energy-from-Waste)",
    "landfill": "Landfill",
    "composting": "Composting",
    "anaerobic_digestion": "Anaerobic Digestion",
    "material_recovery": "Material Recovery (MRF Sorting)",
}

# UX state indicator per rank — Rank 1 is the optimal green path,
# Rank 3 the worst-case baseline of the simulated set.
STATUS_TAGS = {1: "Optimal", 2: "Acceptable", 3: "Warning"}

# ---------------------------------------------------------------------------
# STRUCTURED EXPERT KNOWLEDGE BASE.
#
# Deep contextual analysis attached to every ranked choice: per
# (material, method), a professional rationale for WHY the route reduces
# environmental overhead (pros) and the explicit long-term cost of choosing
# it (cons). Pure data — deterministic, examiner-auditable, no LLM required.
# Tests enforce full coverage of every path in DISPOSAL_PATHS.
# ---------------------------------------------------------------------------
EXPERT_KNOWLEDGE = {
    "plastic": {
        "recycling": {
            "pros": "Closed-loop recycling displaces virgin polymer production, "
                    "avoiding the petroleum extraction, cracking and "
                    "polymerisation that dominate plastic's lifecycle footprint "
                    "— a net carbon credit per kilogram reclaimed.",
            "cons": "Polymer chains shorten with every cycle (downcycling), and "
                    "contaminated or mixed-resin streams are rejected at the "
                    "MRF and rerouted to residual disposal.",
        },
        "incineration": {
            "pros": "Energy-from-waste recovers part of plastic's high calorific "
                    "value (~40 MJ/kg, comparable to fuel oil), displacing some "
                    "grid generation while destroying the litter pathway.",
            "cons": "Burning fossil-derived polymers releases their locked "
                    "petroleum carbon straight to the atmosphere — the "
                    "highest-GHG route for plastic — and flue-gas residues "
                    "still require hazardous treatment.",
        },
        "landfill": {
            "pros": "Plastic is biologically inert, so a landfilled item "
                    "generates near-zero direct methane or CO2; its short-term "
                    "GHG burden is little more than collection logistics.",
            "cons": "The material persists for 400+ years, fragmenting into "
                    "microplastics that leach into soil and groundwater, and "
                    "every buried kilogram permanently forfeits the recycling "
                    "offset it could have earned.",
        },
    },
    "glass": {
        "recycling": {
            "pros": "Remelting cullet runs the furnace cooler than a virgin "
                    "batch (no soda-ash/limestone calcination), and glass "
                    "recycles infinitely with zero quality loss.",
            "cons": "Colour-mixed or contaminated cullet gets downcycled to "
                    "aggregate, and glass is heavy — long collection distances "
                    "erode the transport side of the credit.",
        },
        "incineration": {
            "pros": "Practically none — glass is non-combustible; at best it is "
                    "recovered from bottom ash as a low-grade aggregate.",
            "cons": "Inert glass absorbs furnace heat without yielding any "
                    "energy, lowering the plant's efficiency, and exits as slag "
                    "— the infinite remelt loop is broken for no gain.",
        },
        "landfill": {
            "pros": "Chemically stable and non-toxic in the ground — no "
                    "leachate chemistry and no gas generation.",
            "cons": "Occupies landfill volume essentially forever (glass takes "
                    "~1 million years to weather), and every buried tonne "
                    "forfeits the furnace-energy saving of remelting.",
        },
    },
    "metal": {
        "recycling": {
            "pros": "Remelting avoids ore mining and smelting — recycled "
                    "aluminium needs ~95% less energy than primary production "
                    "(steel ~70%) — the largest per-kg carbon credit of any "
                    "household material.",
            "cons": "Requires clean separation from food residue and "
                    "composites; repeatedly remelting mixed alloys can "
                    "downgrade the recovered material's quality.",
        },
        "incineration": {
            "pros": "Non-combustible metals pass through to bottom ash, from "
                    "which ferrous fractions are sometimes magnetically "
                    "recovered as a by-product.",
            "cons": "Metal contributes no energy to the burn; unrecovered "
                    "fractions are slagged and oxidised, and the enormous "
                    "smelting-avoidance credit goes unclaimed.",
        },
        "landfill": {
            "pros": "Structurally stable, and modern lined cells contain the "
                    "slow corrosion products.",
            "cons": "Burying refined metal permanently squanders the intense "
                    "embodied energy of smelting (up to ~9 kg CO2e per kg for "
                    "aluminium), and trace alloy elements can leach over time.",
        },
    },
    "cardboard": {
        "recycling": {
            "pros": "Repulping corrugated fibre displaces virgin kraft pulping "
                    "— saving trees, roughly a quarter of the production "
                    "energy, and the forestry-chain emissions; fibres survive "
                    "5–7 further cycles.",
            "cons": "Wet, greasy or wax-coated board (classic pizza-box "
                    "problem) contaminates the pulp stream and gets rejected "
                    "into residual waste.",
        },
        "incineration": {
            "pros": "Board is a biogenic fuel: combustion re-releases carbon "
                    "the tree absorbed, so the net fossil addition is small "
                    "while heat recovery displaces grid generation.",
            "cons": "The fibre value is destroyed in a single pass — burning "
                    "board forfeits 5+ future recycling cycles — and transport "
                    "plus flue-gas handling still carry a carbon cost.",
        },
        "landfill": {
            "pros": "Nothing beyond minimal handling cost and universal "
                    "availability.",
            "cons": "Buried cardboard decomposes anaerobically into methane "
                    "(~28x CO2's warming over a century) — the worst "
                    "end-of-life for fibre — while the raw material value is "
                    "lost entirely.",
        },
    },
    "paper": {
        "recycling": {
            "pros": "Recycled fibre displaces virgin pulping — the most "
                    "energy-intensive stage of papermaking — cutting water use "
                    "by roughly half and the chemical load alongside the "
                    "carbon credit.",
            "cons": "Fibres shorten with each loop (~5–7 cycles maximum), and "
                    "thermal receipts, tissues and laminated papers are "
                    "non-recyclable contaminants.",
        },
        "incineration": {
            "pros": "Paper's biogenic carbon makes energy-from-waste roughly "
                    "carbon-neutral on combustion, with genuine heat and power "
                    "recovery displacing fossil generation.",
            "cons": "A one-pass destruction of perfectly reusable fibre, and "
                    "ink or coating additives concentrate in the ash, which "
                    "needs managed disposal.",
        },
        "landfill": {
            "pros": "Nothing beyond minimal handling logistics.",
            "cons": "Paper is among the most methane-productive landfill "
                    "materials: anaerobic decomposition releases CH4 for "
                    "decades and gas-capture systems recover only part of it.",
        },
    },
    "biodegradable": {
        "composting": {
            "pros": "Aerobic composting returns nutrients and stable carbon to "
                    "soil, displacing synthetic fertiliser production and "
                    "improving water retention — a genuinely circular pathway "
                    "for food and garden waste.",
            "cons": "Poorly managed piles turn anaerobic and emit methane and "
                    "N2O, and industrial composting needs source-separated "
                    "feedstock free of plastic contamination.",
        },
        "anaerobic_digestion": {
            "pros": "Sealed digesters capture the methane as biogas for heat, "
                    "power or transport fuel — displacing fossil energy — while "
                    "the digestate substitutes mineral fertiliser; typically "
                    "the lowest-carbon organic route.",
            "cons": "Depends on dedicated infrastructure and consistent "
                    "feedstock; even a few percent of biogas leakage rapidly "
                    "erodes the climate benefit.",
        },
        "landfill": {
            "pros": "None — this is precisely the pathway organic waste should "
                    "avoid.",
            "cons": "Entombed organics decompose anaerobically into landfill "
                    "methane — the waste sector's single largest GHG source — "
                    "and even modern capture systems lose a large share of it "
                    "to the atmosphere.",
        },
    },
    "general rubbish": {
        "material_recovery": {
            "pros": "Mechanical sorting (MRF/MBT) pulls recyclable metals, "
                    "plastics and fibre back out of the residual stream before "
                    "disposal, reclaiming offsets that blind landfilling "
                    "forfeits.",
            "cons": "Sorting plants are energy-intensive and recovery rates on "
                    "contaminated mixed waste stay modest — source separation "
                    "upstream remains far superior.",
        },
        "incineration": {
            "pros": "Energy-from-waste shrinks residual volume by ~90% and "
                    "recovers heat and electricity, displacing landfill "
                    "methane and some grid generation.",
            "cons": "The stream's fossil fraction (plastics) burns straight to "
                    "CO2, and toxic fly ash must be stabilised in dedicated "
                    "hazardous cells.",
        },
        "landfill": {
            "pros": "The lowest immediate processing cost and universally "
                    "available.",
            "cons": "The organic fraction generates decades of methane, "
                    "leachate must be pumped and treated long after closure, "
                    "and the land is permanently committed — the worst-case "
                    "baseline for mixed waste.",
        },
    },
}


def _display_material(material: str) -> str:
    """Friendly material name (falls back to Title Case for unknowns)."""
    return DISPLAY_NAMES.get(material, material.title())


def _build_verdict(rank: int, material: str, method: str,
                   co2e_kg: float, best_co2e_kg: float,
                   worst_co2e_kg: float) -> str:
    """
    Rank-aware supportive copy for one simulated path.

    Composed at runtime from the SORTED outcome (never hand-tied to a rank)
    so a future factor recalibration can reshuffle the ranking without the
    tone drifting out of sync with the numbers.
    """
    mat = _display_material(material)
    meth = METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())
    saving_vs_worst = round(worst_co2e_kg - co2e_kg, 4)
    penalty_vs_best = round(co2e_kg - best_co2e_kg, 4)

    if rank == 1:
        offset = (" — it is carbon-NEGATIVE, offsetting more emissions than it "
                  "creates" if co2e_kg < 0 else "")
        return (f"Optimal green path! {meth} is the lowest-carbon fate for "
                f"{mat}{offset}. Choosing it avoids {saving_vs_worst} kg CO2e "
                f"versus the worst route here — outstanding work for the planet.")
    if rank == 2:
        return (f"Acceptable fallback. {meth} lands mid-table for {mat}: it "
                f"still avoids {saving_vs_worst} kg CO2e versus the worst "
                f"route, but the rank-1 path is the greener call whenever it "
                f"is available to you.")
    return (f"Warning: {meth} is the highest-carbon fate for {mat} in this "
            f"comparison, adding {penalty_vs_best} kg CO2e over the optimal "
            f"path. Please choose a higher-ranked route whenever local "
            f"facilities allow.")


def _resolve_weight(entry: dict) -> tuple:
    """
    (effective_weight_kg, weight_source) for one request item.

    A user-verified ``weight_kg`` (the Stage-B audit value) always wins; a
    ``box_area_px`` falls back to the blind pixel proxy, area / gamma — the
    exact calibration the /predict payload already uses. At least one of the
    two is required (the schema enforces this too; the service double-checks).
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


def simulate_disposal_paths(material: str, weight_kg: float, _pool=None) -> list:
    """
    Fork one item into its 3 taxonomy-branched end-of-life simulations,
    evaluate them IN PARALLEL, then rank ascending by net CO2e.

    Returns the ranked, fully annotated recommendation array::

        [
          {
            "method": "recycling",
            "method_display": "Recycling",
            "rank": 1,                       # 1 = optimal ... 3 = worst baseline
            "status_tag": "Optimal",         # Optimal | Acceptable | Warning
            "carbon_factor_kg_per_kg": -1.08,  # net factor (negative = credit)
            "carbon_impact_kg": -0.54,       # factor x weight (may be negative)
            "encouraging_verdict": "...",    # rank-aware supportive copy
            "environmental_pros": "...",     # knowledge base rationale
            "environmental_cons": "..."      # knowledge base long-term costs
          },
          ... rank 2, rank 3 ...
        ]

    Ties (which the current matrix never produces) fall back to method-name
    order so the ranking stays fully deterministic.
    """
    if material not in MATERIAL_CLASSES:
        raise ApiError(
            f"Unknown material '{material}'. Valid materials: "
            f"{', '.join(MATERIAL_CLASSES)}.",
            status_code=400,
        )
    if weight_kg <= 0:
        raise ApiError("weight_kg must be positive.", status_code=400)

    if _pool is None:
        with ThreadPoolExecutor(max_workers=_PARALLEL_PATHS) as pool:
            return simulate_disposal_paths(material, weight_kg, _pool=pool)

    methods = DISPOSAL_PATHS[material]
    # Parallel fan-out: each end-of-life path is priced concurrently by the
    # carbon engine (pure local arithmetic — thread-safe, no app context).
    co2e_values = list(_pool.map(
        lambda method: estimate_disposal_impact(material, method, weight_kg),
        methods,
    ))

    # SORTING ENGINE: ascending net CO2e — deepest offset / lowest burden wins.
    outcomes = sorted(zip(methods, co2e_values), key=lambda mc: (mc[1], mc[0]))
    best_co2e, worst_co2e = outcomes[0][1], outcomes[-1][1]

    ranked = []
    for rank, (method, co2e_kg) in enumerate(outcomes, start=1):
        knowledge = EXPERT_KNOWLEDGE[material][method]
        ranked.append({
            "method": method,
            "method_display": METHOD_DISPLAY_NAMES.get(
                method, method.replace("_", " ").title()),
            "rank": rank,
            "status_tag": STATUS_TAGS[rank],
            "carbon_factor_kg_per_kg": get_disposal_factor(material, method),
            "carbon_impact_kg": co2e_kg,
            "encouraging_verdict": _build_verdict(
                rank, material, method, co2e_kg, best_co2e, worst_co2e),
            "environmental_pros": knowledge["pros"],
            "environmental_cons": knowledge["cons"],
        })
    return ranked


def recommend_for_items(items: list) -> dict:
    """
    DMM aggregate entry for ``POST /api/recommend``.

    ``items`` is a list of ``{"material": str, "weight_kg": float?,
    "box_area_px": float?}`` dicts (type-validated by the pydantic schema at
    the route; each needs at least one of the two size fields). Returns::

        {
          "items": [
            { material, display_name, effective_weight_kg, weight_source,
              best_method, max_saving_kg, recommendations: [3 ranked paths] }
          ],
          "summary": { item_count, optimal_total_co2e_kg,
                       worst_total_co2e_kg, max_saving_kg },
          "provider": "local_knowledge_base"
        }

    The summary totals compare "user follows every rank-1 path" against
    "user follows every rank-3 path" — the headline saving the Step-7
    dashboard can celebrate. Totals may be NEGATIVE (net offsets).
    """
    results = []
    optimal_total = 0.0
    worst_total = 0.0

    # One pool for the whole request; every item forks its 3 paths onto it.
    with ThreadPoolExecutor(max_workers=_PARALLEL_PATHS) as pool:
        for entry in items:
            material = entry["material"]
            weight_kg, weight_source = _resolve_weight(entry)
            ranked = simulate_disposal_paths(material, weight_kg, _pool=pool)

            optimal_total += ranked[0]["carbon_impact_kg"]
            worst_total += ranked[-1]["carbon_impact_kg"]
            results.append({
                "material": material,
                "display_name": _display_material(material),
                "effective_weight_kg": round(weight_kg, 4),
                "weight_source": weight_source,
                "best_method": ranked[0]["method"],
                "max_saving_kg": round(
                    ranked[-1]["carbon_impact_kg"] - ranked[0]["carbon_impact_kg"], 4),
                "recommendations": ranked,
            })

    return {
        "items": results,
        "summary": {
            "item_count": len(results),
            "optimal_total_co2e_kg": round(optimal_total, 4),
            "worst_total_co2e_kg": round(worst_total, 4),
            "max_saving_kg": round(worst_total - optimal_total, 4),
        },
        "provider": "local_knowledge_base",
    }
