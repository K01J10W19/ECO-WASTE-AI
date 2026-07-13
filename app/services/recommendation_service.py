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
Rank 3 = worst-case baseline.

TEXT LAYER (v3.6 — "Hyper-Simple & Country-Aware"): the three literary fields
per path (``encouraging_verdict``, ``environmental_pros``,
``environmental_cons``) ship in plain, child-friendly language (1-2 punchy
sentences, <= 25 words, zero jargon):

  * DEFAULT / no LLM key — the local ``EXPERT_KNOWLEDGE`` grid + runtime
    verdicts below (deterministic; provider ``"local_knowledge_base"``).
  * LLM key configured — ONE batched call to any OpenAI-compatible
    chat-completions endpoint (free tiers: Groq / OpenRouter / Gemini compat /
    local Ollama; see config.py) rewrites the three fields per path,
    localized to the request's ``country`` (``"global average"`` when absent);
    provider ``"llm_enriched"``.
  * ANY LLM failure (auth, rate limit, timeout, malformed output, missing
    coverage) — seamless, atomic revert to the local grid; provider
    ``"local_fallback"``. Recommendations can never 502 on the LLM.

Weight resolution mirrors the dual-stage carbon UX (the SHARED
``carbon_service.resolve_effective_weight`` helper — the same substitution
``/api/calculate-impact`` applies):
  * ``weight_kg`` present  -> user-verified weight (the precision audit value);
  * else ``box_area_px``   -> the blind pixel proxy, box_area / gamma
    (the same PIXEL_AREA_GAMMA calibration the /predict payload uses).

DESIGN GUARANTEES (CLAUDE.md):
  * The NUMERIC core (simulation, ranking, factors) is deterministic + 100%
    offline and needs no Flask app context (thread-pool safe). The optional
    LLM text layer is the module's ONLY network touchpoint: it runs once per
    request in the request thread, may ONLY rewrite the three text fields —
    never ranks, methods or numbers — and always degrades to the local grid.
  * The 7-class taxonomy stays in lockstep: DISPOSAL_PATHS, the factor
    matrix and EXPERT_KNOWLEDGE cover every material — tests enforce it.
  * Expected failures raise ``ApiError``; nothing 500s silently.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from flask import current_app, has_app_context

from app.services.carbon_service import (
    estimate_disposal_impact,
    get_disposal_factor,
    resolve_effective_weight,
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
# LOCAL EXPERT KNOWLEDGE GRID (v3.6 hyper-simple register).
#
# The deterministic fallback copy behind the LLM layer — and the default text
# when no key is configured. Same constraints as the LLM fields: 1-2 punchy
# sentences, <= 25 words, plain language a child understands, grounded facts.
# Tests enforce full coverage of every path in DISPOSAL_PATHS AND the word
# budget, so fallback and LLM output always share one UX register.
# ---------------------------------------------------------------------------
EXPERT_KNOWLEDGE = {
    "plastic": {
        "recycling": {
            "pros": "Recycling plastic means factories pump less new oil from "
                    "the ground — a big saving in electricity and dirty air.",
            "cons": "Plastic gets weaker every time it is recycled, and dirty "
                    "or mixed plastic gets rejected and dumped instead.",
        },
        "incineration": {
            "pros": "Burning plastic makes electricity — one kilogram holds "
                    "about as much energy as fuel oil.",
            "cons": "Burning plastic is like burning oil: all its locked-up "
                    "pollution shoots straight into the sky we breathe.",
        },
        "landfill": {
            "pros": "Buried plastic just sits there — it does not rot, so it "
                    "makes almost no planet-warming gas.",
            "cons": "It stays buried for 400+ years, crumbling into toxic "
                    "microplastics that sneak into our water, our fish and "
                    "our food.",
        },
    },
    "glass": {
        "recycling": {
            "pros": "Old glass melts at a lower heat than glass made from "
                    "sand — less fuel burned, and it can be reborn forever.",
            "cons": "Mixed-colour or dirty glass often cannot become new "
                    "bottles, and heavy glass costs fuel to truck around.",
        },
        "incineration": {
            "pros": "Almost none — glass cannot burn. At best some bits get "
                    "scooped from the ashes for road filler.",
            "cons": "Glass soaks up the fire's heat without giving any energy "
                    "back, then ends up as waste slag anyway.",
        },
        "landfill": {
            "pros": "Glass is clean and safe in the ground — it does not rot, "
                    "leak or make gas.",
            "cons": "A buried bottle needs about a million years to "
                    "disappear, hogging space while new glass is made from "
                    "scratch.",
        },
    },
    "metal": {
        "recycling": {
            "pros": "Melting old cans uses up to 95% less electricity than "
                    "digging and smelting new metal — the biggest energy win "
                    "there is.",
            "cons": "Cans need to be fairly clean, and remelting mixed metals "
                    "again and again can lower their quality.",
        },
        "incineration": {
            "pros": "Metal does not burn, but magnets can rescue some of it "
                    "from the ashes afterwards.",
            "cons": "It adds no energy to the fire, and most of that precious "
                    "refined metal is wasted in the slag.",
        },
        "landfill": {
            "pros": "Metal sits fairly quietly in modern lined landfills "
                    "without causing much trouble.",
            "cons": "Burying a can throws away all the huge energy spent "
                    "making it, and slow rust can seep into the soil.",
        },
    },
    "cardboard": {
        "recycling": {
            "pros": "Recycled boxes mean fewer trees cut down and about a "
                    "quarter less factory energy — each box can go around "
                    "5-7 times.",
            "cons": "Greasy or wet boxes — the classic pizza box — spoil the "
                    "whole batch and get tossed out.",
        },
        "incineration": {
            "pros": "Burning cardboard turns the tree's stored energy into "
                    "electricity without adding much new pollution.",
            "cons": "One burn destroys fibres that could have been reused "
                    "five more times.",
        },
        "landfill": {
            "pros": "Nothing really — it is just the cheapest, laziest "
                    "option.",
            "cons": "Rotting buried cardboard burps out methane — a gas that "
                    "heats the planet about 28 times harder than car fumes.",
        },
    },
    "paper": {
        "recycling": {
            "pros": "New paper from old paper skips the most power-hungry "
                    "factory step and uses about half the water.",
            "cons": "Paper fibres wear out after 5-7 rounds, and receipts, "
                    "tissues and shiny paper cannot join in.",
        },
        "incineration": {
            "pros": "Burning paper for electricity is nearly "
                    "pollution-neutral, because the tree soaked up that "
                    "carbon while it grew.",
            "cons": "Perfectly reusable paper is gone in one flash, and the "
                    "inky ash still needs careful burying.",
        },
        "landfill": {
            "pros": "None worth naming — just cheap and easy.",
            "cons": "Buried paper is a methane machine: it rots for decades, "
                    "leaking planet-heating gas that capture pipes only "
                    "partly catch.",
        },
    },
    "biodegradable": {
        "composting": {
            "pros": "Food scraps become rich soil food, so farms need less "
                    "factory fertiliser and gardens hold water better.",
            "cons": "A badly-run pile turns smelly and leaks methane, and "
                    "plastic bits must be kept out.",
        },
        "anaerobic_digestion": {
            "pros": "Sealed tanks catch the rot-gas and burn it for "
                    "electricity, while the leftovers feed farm soil — the "
                    "greenest food-waste route.",
            "cons": "It needs special plants nearby, and even small gas leaks "
                    "quickly shrink the benefit.",
        },
        "landfill": {
            "pros": "None — this is exactly where food waste should never "
                    "go.",
            "cons": "Buried food rots without air and pumps out methane — "
                    "the single biggest climate problem in our rubbish.",
        },
    },
    "general rubbish": {
        "material_recovery": {
            "pros": "Sorting machines rescue metals, plastics and paper out "
                    "of mixed rubbish before dumping — a last-chance save.",
            "cons": "The machines use lots of power and only save a small "
                    "slice — sorting at home works far better.",
        },
        "incineration": {
            "pros": "Burning mixed rubbish shrinks it by 90% and makes "
                    "electricity instead of landfill gas.",
            "cons": "The plastic inside burns into sky pollution, and the "
                    "leftover toxic ash needs special burial.",
        },
        "landfill": {
            "pros": "The cheapest option, available everywhere.",
            "cons": "It leaks methane for decades, its dirty juice must be "
                    "pumped away for years, and the land is lost forever.",
        },
    },
}

# ---------------------------------------------------------------------------
# v3.6 LLM GENERATION PIPELINE ("Hyper-Simple & Country-Aware").
#
# One batched call per request to an OpenAI-compatible chat-completions
# endpoint (LLM_API_URL/LLM_API_KEY/LLM_MODEL in config — free tiers work:
# Groq, OpenRouter, Gemini's compat endpoint, or a fully local Ollama).
# The LLM may ONLY write the three literary fields; every number, rank and
# method id is computed locally and passed in read-only.
# ---------------------------------------------------------------------------
# Free-tier "thinking" models (e.g. gemini-flash-latest) can spend a long
# while reasoning before emitting the JSON — 30 s produced spurious timeouts
# in live testing, so the window is generous; any overrun still degrades
# cleanly to the local grid.
_LLM_TIMEOUT_S = 60
_LLM_MAX_TOKENS = 4096
_LLM_TEMPERATURE = 0.5
# Free tiers throw transient 503 "model overloaded" / 429 bursts that clear
# within seconds (observed live) — retry fast failures before falling back.
# A read TIMEOUT is never retried: that budget is already spent.
_LLM_MAX_ATTEMPTS = 3
_LLM_RETRY_BACKOFF_S = (2.0, 5.0)   # sleep before attempt 2, attempt 3

LLM_SYSTEM_PROMPT = """\
You are the friendly, plain-spoken voice of a family waste-sorting app.
You rewrite disposal advice so a 10-year-old instantly gets it.

You receive JSON: a "country" plus scanned waste "items". Each item has an
"index", a "material", its "weight_kg", and exactly 3 end-of-life "paths".
Each path gives: "method", "rank" (1 = best, 3 = worst), "status_tag",
"carbon_impact_kg" (negative = it REMOVES pollution), "saving_vs_worst_kg"
and "extra_vs_best_kg".

For EVERY path of EVERY item write exactly three fields:
1. "encouraging_verdict" — celebrate rank 1, gently nudge rank 2, sternly
   warn rank 3. MUST weave in the numbers provided (kg and/or rank).
2. "environmental_pros" — the immediate, everyday benefit of this choice.
3. "environmental_cons" — a stark but 100% truthful long-term consequence.

HARD RULES
- 1-2 punchy sentences per field. MAXIMUM 25 words per field.
- Words a child knows. NEVER use jargon like "carbon-negative", "offset",
  "displace", "biogenic", "anaerobic decomposition", "leachate", "CO2e",
  "emission factor". Prefer everyday images: "planet-warming gas",
  "saving electricity", "trash on our beaches", "burning coal".
- Use ONLY the numbers provided — never invent or change them.
- If "country" is a real country code, ground pros/cons in that country's
  everyday reality (its beaches and oceans if coastal, its crowded
  landfills, its power grid). If "country" is "global average", keep the
  text universal.
- Reply with STRICT JSON ONLY — no markdown, no commentary — shaped:
  {"items":[{"index":0,"paths":[{"method":"recycling",
  "encouraging_verdict":"...","environmental_pros":"...",
  "environmental_cons":"..."}]}]}
  Cover every item and every path exactly once, keeping the given
  "method" ids and "index" values unchanged.
"""


def _display_material(material: str) -> str:
    """Friendly material name (falls back to Title Case for unknowns)."""
    return DISPLAY_NAMES.get(material, material.title())


def _build_verdict(rank: int, material: str, method: str,
                   co2e_kg: float, best_co2e_kg: float,
                   worst_co2e_kg: float) -> str:
    """
    Local rank-aware verdict (v3.6 hyper-simple register, <= 25 words).

    Composed at runtime from the SORTED outcome (never hand-tied to a rank)
    so a factor recalibration can reshuffle ranks without the tone drifting
    out of sync with the numbers. This is the deterministic fallback copy;
    the LLM layer, when active, replaces it with localized text.
    """
    mat = _display_material(material).lower()
    meth = METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())
    saving_vs_worst = round(worst_co2e_kg - co2e_kg, 4)
    extra_vs_best = round(co2e_kg - best_co2e_kg, 4)

    if rank == 1:
        if co2e_kg < 0:
            return (f"Best choice! {meth} actually removes pollution for "
                    f"{mat} and dodges {saving_vs_worst} kg of planet-warming "
                    f"gas versus the worst route. Brilliant!")
        return (f"Best choice! {meth} is the cleanest fate for {mat}, "
                f"dodging {saving_vs_worst} kg of planet-warming gas. "
                f"Brilliant!")
    if rank == 2:
        return (f"Not bad — {meth} beats the worst option by "
                f"{saving_vs_worst} kg of pollution, but the number-1 choice "
                f"is kinder to the planet.")
    return (f"Careful! {meth} is the dirtiest route for {mat} — "
            f"{extra_vs_best} kg MORE planet-warming gas than the best "
            f"choice. Pick better if you can!")


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
            "environmental_pros": "...",     # knowledge grid benefit
            "environmental_cons": "..."      # knowledge grid long-term cost
          },
          ... rank 2, rank 3 ...
        ]

    Text fields carry the LOCAL knowledge-grid copy; the LLM layer in
    ``recommend_for_items`` may overwrite them (text only — numbers, ranks
    and method ids are immutable). Ties (which the current matrix never
    produces) fall back to method-name order so ranking stays deterministic.
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


def _llm_settings() -> tuple:
    """(api_key, model, api_url) from config; blanks outside an app context."""
    if not has_app_context():
        return "", "", ""
    cfg = current_app.config
    return (str(cfg.get("LLM_API_KEY", "") or ""),
            str(cfg.get("LLM_MODEL", "") or ""),
            str(cfg.get("LLM_API_URL", "") or ""))


def _llm_context(results: list, country: str) -> dict:
    """The read-only numeric context the LLM writes prose around."""
    return {
        "country": country or "global average",
        "items": [
            {
                "index": idx,
                "material": item["display_name"],
                "weight_kg": item["effective_weight_kg"],
                "paths": [
                    {
                        "method": path["method"],
                        "rank": path["rank"],
                        "status_tag": path["status_tag"],
                        "carbon_impact_kg": path["carbon_impact_kg"],
                        "saving_vs_worst_kg": round(
                            item["recommendations"][-1]["carbon_impact_kg"]
                            - path["carbon_impact_kg"], 4),
                        "extra_vs_best_kg": round(
                            path["carbon_impact_kg"]
                            - item["recommendations"][0]["carbon_impact_kg"], 4),
                    }
                    for path in item["recommendations"]
                ],
            }
            for idx, item in enumerate(results)
        ],
    }


def _extract_json(text: str) -> dict:
    """
    Parse the LLM reply leniently: tolerate stray prose or markdown fences by
    slicing from the first '{' to the last '}'. Anything unparseable raises
    (caught upstream -> local_fallback).
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in LLM reply")
    return json.loads(text[start:end + 1])


def _enrich_with_llm(results: list, country: str,
                     api_key: str, model: str, api_url: str) -> None:
    """
    Rewrite the three literary fields via the LLM, retrying transient blips.

    Fast failures (HTTP 429/5xx "model overloaded", network hiccups,
    malformed or partial output) are retried up to ``_LLM_MAX_ATTEMPTS``
    times with a short backoff — free tiers throw momentary 503s that clear
    in seconds. A read TIMEOUT is NOT retried (its 60 s budget is already
    spent). Raises on final failure — the caller serves the local grid.
    """
    import requests  # local import keeps module import light for tests

    last_error = None
    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(_LLM_RETRY_BACKOFF_S[attempt - 2])
            logger.info("LLM text layer retry %d/%d (previous attempt: %s)",
                        attempt, _LLM_MAX_ATTEMPTS, last_error)
        try:
            _generate_and_apply(results, country, api_key, model, api_url)
            return
        except requests.exceptions.Timeout:
            raise                    # window already burned — degrade now
        except Exception as exc:  # noqa: BLE001 - transient upstream noise
            last_error = exc
    raise last_error


def _generate_and_apply(results: list, country: str,
                        api_key: str, model: str, api_url: str) -> None:
    """
    ONE batched chat-completions call + strict validation + atomic apply.

    Raises on ANY problem (HTTP status, timeout, malformed JSON, missing
    item/path coverage, empty field). Replacements are STAGED and applied
    atomically, so a mid-validation failure never leaves the payload
    half-enriched.
    """
    import requests  # local import keeps module import light for tests

    body = {
        "model": model,
        "temperature": _LLM_TEMPERATURE,
        "max_tokens": _LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_llm_context(results, country))},
        ],
    }
    resp = requests.post(
        api_url,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_LLM_TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM endpoint returned HTTP {resp.status_code}")

    generated = _extract_json(resp.json()["choices"][0]["message"]["content"])
    by_index = {int(item["index"]): item for item in generated["items"]}

    staged = []
    for idx, item in enumerate(results):
        by_method = {p["method"]: p for p in by_index[idx]["paths"]}
        for path in item["recommendations"]:
            gen = by_method[path["method"]]   # KeyError -> fallback upstream
            for field in ("encouraging_verdict", "environmental_pros",
                          "environmental_cons"):
                value = str(gen[field]).strip()
                if not value:
                    raise ValueError(f"LLM returned an empty '{field}'")
                staged.append((path, field, value))
    for path, field, value in staged:        # atomic: only after FULL validation
        path[field] = value


def recommend_for_items(items: list, country: str = None) -> dict:
    """
    DMM aggregate entry for ``POST /api/recommend``.

    ``items`` is a list of ``{"material": str, "weight_kg": float?,
    "box_area_px": float?}`` dicts (type-validated by the pydantic schema at
    the route; each needs at least one of the two size fields). ``country``
    (optional ISO alpha-2, typically the frontend's IP-geolocated default)
    only flavours the v3.6 LLM text layer — the numbers stay identical.
    Returns::

        {
          "items": [
            { material, display_name, effective_weight_kg, weight_source,
              best_method, max_saving_kg, recommendations: [3 ranked paths] }
          ],
          "summary": { item_count, optimal_total_co2e_kg,
                       worst_total_co2e_kg, max_saving_kg },
          "country": "MY" | None,
          "provider": "llm_enriched" | "local_knowledge_base" | "local_fallback"
        }

    Provider semantics: ``llm_enriched`` = the batched LLM call rewrote the
    text fields; ``local_knowledge_base`` = no LLM key configured (the
    deterministic default); ``local_fallback`` = an LLM was configured but
    failed (rate limit, timeout, bad output) and the local grid took over
    seamlessly. The summary compares "user follows every rank-1 path"
    against "every rank-3 path" — totals may be NEGATIVE (net offsets).
    """
    results = []
    optimal_total = 0.0
    worst_total = 0.0

    # One pool for the whole request; every item forks its 3 paths onto it.
    with ThreadPoolExecutor(max_workers=_PARALLEL_PATHS) as pool:
        for entry in items:
            material = entry["material"]
            weight_kg, weight_source = resolve_effective_weight(entry)
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

    country_code = country.upper() if country else None

    # v3.6 text layer: LLM rewrite when configured, seamless local fallback.
    provider = "local_knowledge_base"
    api_key, model, api_url = _llm_settings()
    if api_key and model and api_url:
        try:
            _enrich_with_llm(results, country_code, api_key, model, api_url)
            provider = "llm_enriched"
        except Exception as exc:  # noqa: BLE001 - ANY LLM issue degrades gracefully
            logger.warning("LLM text layer failed (%s); serving the local "
                           "knowledge-base copy instead.", exc)
            provider = "local_fallback"

    return {
        "items": results,
        "summary": {
            "item_count": len(results),
            "optimal_total_co2e_kg": round(optimal_total, 4),
            "worst_total_co2e_kg": round(worst_total, 4),
            "max_saving_kg": round(worst_total - optimal_total, 4),
        },
        "country": country_code,
        "provider": provider,
    }
