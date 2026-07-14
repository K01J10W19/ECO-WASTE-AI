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
    BASE_GRID_INTENSITY,
    estimate_disposal_impact,
    get_disposal_factor,
    get_national_profile,
    get_scaled_disposal_factor,
    resolve_effective_weight,
    resolve_grid_intensity,
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

# ISO alpha-2 → full country name for the geopolitical LLM directive (the
# model grounds its advice in the named country's real infrastructure). Codes
# outside the CarbIQ selector fall back to the raw upper-cased code.
_COUNTRY_NAMES = {
    "MY": "Malaysia", "SG": "Singapore", "JP": "Japan", "CN": "China",
    "IN": "India", "US": "the United States", "GB": "the United Kingdom",
    "DE": "Germany", "FR": "France", "AU": "Australia", "NZ": "New Zealand",
}


def _country_display(code) -> str:
    """Full country name for the prompt; 'global average' when no country."""
    if not code:
        return "global average"
    code = str(code).strip().upper()
    return _COUNTRY_NAMES.get(code, code)

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
# LOCAL ACTION-PROTOCOL STEPS (companion to EXPERT_KNOWLEDGE).
#
# Exactly TWO ordered, child-simple do-this steps per path — the deterministic
# fallback for the ``action_steps`` field (and the default when no LLM key is
# set). Deliberately COUNTRY-NEUTRAL: the LLM layer localizes them to the
# request country's real bins/rules; the local grid stays universal. Keys stay
# in lockstep with DISPOSAL_PATHS (a test enforces 7x3 coverage + exactly 2
# steps). Same hyper-simple register as EXPERT_KNOWLEDGE.
# ---------------------------------------------------------------------------
EXPERT_ACTION_STEPS = {
    "plastic": {
        "recycling": ["Rinse the bottle and squash it flat to save space.",
                      "Drop it in the recycling bin — clean and dry."],
        "incineration": ["Empty and dry the plastic so it burns cleanly.",
                         "Put it in the general-waste bin for the energy plant."],
        "landfill": ["Only bin plastic too dirty or mixed to recycle.",
                     "Tie the bag well so nothing blows away outside."],
    },
    "glass": {
        "recycling": ["Rinse the jar and take off any metal lid.",
                      "Place it gently, unbroken, in the glass bank."],
        "incineration": ["Wrap broken glass in paper so no one gets cut.",
                         "Bin it — glass will not burn, but stays contained."],
        "landfill": ["Wrap sharp pieces safely before binning them.",
                     "Send only glass that cannot be recycled."],
    },
    "metal": {
        "recycling": ["Rinse the can and lightly crush it flat.",
                      "Drop it in the metals recycling bin."],
        "incineration": ["Scrape out any leftover food from the can.",
                         "Bin it — magnets pull the metal from the ash."],
        "landfill": ["Bin only metal that truly cannot be recycled.",
                     "Tuck sharp lids inside the can first."],
    },
    "cardboard": {
        "recycling": ["Flatten the box and peel off tape or plastic.",
                      "Stack it in the paper bin, kept dry."],
        "incineration": ["Keep greasy or wet card out of recycling.",
                         "Put it in general waste to burn for power."],
        "landfill": ["Bin only soiled cardboard that cannot be recycled.",
                     "Break it down so it packs flat."],
    },
    "paper": {
        "recycling": ["Keep the paper clean, dry and unfolded.",
                      "Add it to the paper recycling bin."],
        "incineration": ["Set aside greasy or shiny paper that cannot recycle.",
                         "Put it in the energy-from-waste bin."],
        "landfill": ["Landfill only soggy or food-stained paper.",
                     "Bag it so it stays contained."],
    },
    "biodegradable": {
        "composting": ["Collect food scraps in a small kitchen caddy.",
                       "Empty it into the compost or food-waste bin."],
        "anaerobic_digestion": ["Separate cooked food and peelings.",
                                "Put them in the food-waste bin for the biogas plant."],
        "landfill": ["Avoid this — buried food makes planet-warming gas.",
                     "If you must, bag the scraps tightly first."],
    },
    "general rubbish": {
        "material_recovery": ["Pull out any recyclables so sorting works better.",
                              "Bin the mixed leftovers for the sorting machines."],
        "incineration": ["Drain any liquids from the rubbish first.",
                         "Bag it up for the waste-to-energy plant."],
        "landfill": ["Use landfill only when nothing else fits.",
                     "Tie the bag so the dirty juice stays in."],
    },
}


# ---------------------------------------------------------------------------
# v3.6 LLM GENERATION PIPELINE ("Hyper-Simple & Country-Aware").
#
# One batched call per request to an OpenAI-compatible chat-completions
# endpoint (LLM_API_URL/LLM_API_KEY/LLM_MODEL in config — free tiers work:
# Groq, OpenRouter, Gemini's compat endpoint, or a fully local Ollama).
# The LLM may ONLY write the four text fields (verdict, pros, cons and the
# two-step action_steps); every number, rank and method id is computed
# locally and passed in read-only.
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
# Quality floor: some models over-compress into telegram fragments
# ("Great, rank 1!") — fields shorter than this many words are rejected,
# which triggers a retry and, if persistent, the local grid.
_LLM_MIN_FIELD_WORDS = 5
# -- 429 rate-limit sealing -------------------------------------------------
# The endpoint's Retry-After header is honoured when retrying, capped here:
# a header ABOVE the cap means the quota window is minutes away — retrying
# inside this request is doomed, so the layer degrades immediately instead.
_LLM_RETRY_AFTER_CAP_S = 8.0
# After a rate-limit failure the COOLDOWN BREAKER opens: every request inside
# the window serves the local grid instantly with ZERO outbound LLM flights
# (no retry stacking against an exhausted quota). Time-based, self-resetting.
_LLM_COOLDOWN_S = 45.0
_llm_cooldown_until = 0.0   # time.monotonic() deadline; module-level state


class _LlmHttpError(RuntimeError):
    """Non-200 from the LLM endpoint, carrying its rate-limit hints."""

    def __init__(self, status_code: int, retry_after: float = None):
        super().__init__(f"LLM endpoint returned HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


def _open_llm_cooldown(retry_after: float = None) -> None:
    """Open the breaker for max(server hint, default) seconds."""
    global _llm_cooldown_until
    window = max(float(retry_after or 0.0), _LLM_COOLDOWN_S)
    _llm_cooldown_until = time.monotonic() + window
    logger.warning("LLM rate limit hit — cooling the text layer down for "
                   "%.0f s (the local grid serves meanwhile).", window)


def _llm_cooling_down() -> bool:
    """True while the rate-limit breaker is open (skip the network)."""
    return time.monotonic() < _llm_cooldown_until

LLM_SYSTEM_PROMPT = """\
You are the friendly, plain-spoken voice of a family waste-sorting app.
You rewrite disposal advice so a 10-year-old instantly gets it.

You receive JSON: a "country_name" (the user's country) plus scanned waste
"items". Each item has an "index", a "material", its "weight_kg", and exactly
3 end-of-life "paths". Each path gives: "method", "method_display", "rank"
(1 = best, 3 = worst), "status_tag", "carbon_impact_kg" (negative = it REMOVES
pollution), "saving_vs_worst_kg" and "extra_vs_best_kg".

CRITICAL DIRECTIVE: the user is currently residing in the given
"country_name". You MUST analyse each disposal pathway strictly through the
lens of THAT country's actual recycling infrastructure, national sanitation
regulations, and native waste-separation habits. For example: if Germany,
build on the bottle-deposit return machines you take empties back to (Pfand);
if Malaysia, ground it in the community colour-sorted bins (e.g. brown for
glass); if Singapore, align with the single mixed bin where nearly everything
is burned for energy. Never supply generic Western tropes that contradict the
local reality. If "country_name" is "global average", keep the text universal.

For EVERY path of EVERY item write exactly FOUR fields:
1. "encouraging_verdict" — celebrate rank 1, gently nudge rank 2, sternly
   warn rank 3. MUST weave in the numbers provided (kg and/or rank).
2. "environmental_pros" — the immediate, everyday benefit of this choice.
3. "environmental_cons" — a stark but 100% truthful long-term consequence.
4. "action_steps" — an array of EXACTLY TWO short, ordered do-this steps that
   walk the user through this exact pathway using their country's REAL bins
   and rules. Step 1 = prepare it at home; Step 2 = where/how it goes. Each
   step is one plain imperative sentence, 4 to 20 words.

HARD RULES
- Text fields (1-3) are 1-2 COMPLETE sentences, 8 to 25 words. NEVER telegram
  fragments ("Great, rank 1!") and NEVER the word "None" alone — every
  field must say something real and specific.
- Field jobs: the verdict carries the feeling AND the numbers; the pros
  paint the everyday benefit in plain images (do NOT just repeat the kg
  figure); the cons describe one concrete long-term harm; the steps are
  practical and country-specific.
- Words a child knows. NEVER use jargon like "carbon-negative", "offset",
  "displace", "biogenic", "anaerobic decomposition", "leachate", "CO2e",
  "emission factor". Prefer everyday images: "planet-warming gas",
  "saving electricity", "trash on our beaches", "burning coal".
- Use ONLY the numbers provided — never invent or change them.
- Reply with STRICT JSON ONLY — no markdown, no commentary — shaped:
  {"items":[{"index":0,"paths":[{"method":"recycling",
  "encouraging_verdict":"...","environmental_pros":"...",
  "environmental_cons":"...","action_steps":["...","..."]}]}]}
  Cover every item and every path exactly once, keeping the given
  "method" ids and "index" values unchanged.

EXAMPLE of ONE well-written path (country_name "Malaysia") — match this
quality and length:
{"method":"recycling","encouraging_verdict":"Amazing pick — rank 1!
Recycling this saves 1.7 kg of planet-warming gas, like switching off the
lights for a whole week.","environmental_pros":"Old bottles become new
ones, so factories burn less oil and beaches stay clean.",
"environmental_cons":"Dirty or greasy plastic spoils the whole batch and
ends up dumped instead.","action_steps":["Rinse the bottle and squash it
flat at home.","Drop it in the orange recycling bin for plastics."]}
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


def _build_restricted_verdict(method: str, reason: str) -> str:
    """Deterministic policy copy for a nationally unavailable path (<= 25
    words, same hyper-simple register). Never LLM-rewritten — applicability
    is infrastructure fact, not narrative."""
    meth = METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())
    return f"{meth} is not an option where you live — {reason}."


def simulate_disposal_paths(material: str, weight_kg: float, _pool=None,
                            country: str = None,
                            grid_intensity: float = None) -> list:
    """
    Fork one item into its 3 taxonomy-branched end-of-life simulations,
    evaluate them IN PARALLEL, then rank ascending by net CO2e — EXCLUSIVELY
    among the paths the ``country``'s national infrastructure actually
    operates (v3.7 applicability matrix).

    ``grid_intensity`` (kgCO2e/kWh, resolved ONCE per request by the caller —
    never inside worker threads) drives the v3.8 Grid-Intensity Proxy Scaling
    Engine: every path's factor is re-derived from the GB-anchored baseline
    before ranking, so rank/status/savings all cascade from the regionalized
    numbers. None → the 0.207 anchor (baseline factors, ratio 1.0).

    Returns the fully annotated recommendation array: applicable paths first
    (rank 1..K ascending), then any nationally banned paths::

        [
          {
            "method": "recycling",
            "method_display": "Recycling",
            "rank": 1,                       # 1 = optimal (among APPLICABLE)
            "status_tag": "Optimal",         # Optimal | Acceptable | Warning
            "is_applicable": true,           # national infrastructure verdict
            "restriction_reason": null,      # set iff is_applicable is false
            "carbon_factor_kg_per_kg": -1.08,  # net factor (negative = credit)
            "carbon_impact_kg": -0.54,       # factor x weight (may be negative)
            "encouraging_verdict": "...",    # rank-aware supportive copy
            "environmental_pros": "...",     # knowledge grid benefit
            "environmental_cons": "..."      # knowledge grid long-term cost
          },
          ...,
          { "method": "landfill", "rank": null, "status_tag": null,
            "is_applicable": false, "restriction_reason": "Singapore ...", ... }
        ]

    Banned paths (e.g. landfill in zero-landfill Singapore) keep their priced
    CO2e for transparency but are the SUPREME BARRIER's casualties: rank and
    status_tag are None, their verdict is a fixed policy statement, and they
    never participate in ranking, verdict deltas, savings or summaries. If a
    profile were ever to ban EVERY path of a branch, the engine fails open
    (all paths applicable) rather than returning an unrankable item.

    Text fields carry the LOCAL knowledge-grid copy; the LLM layer in
    ``recommend_for_items`` may overwrite them for APPLICABLE paths only
    (text only — numbers, ranks and method ids are immutable). Ties (which
    the current matrix never produces) fall back to method-name order so
    ranking stays deterministic.
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
            return simulate_disposal_paths(material, weight_kg, _pool=pool,
                                           country=country,
                                           grid_intensity=grid_intensity)

    gi = BASE_GRID_INTENSITY if grid_intensity is None else float(grid_intensity)
    methods = DISPOSAL_PATHS[material]
    # Parallel fan-out: each end-of-life path is priced concurrently by the
    # carbon engine at the resolved grid intensity (pure local arithmetic —
    # thread-safe, no app context, no network inside the pool).
    co2e_values = list(_pool.map(
        lambda method: estimate_disposal_impact(material, method, weight_kg,
                                                grid_intensity=gi),
        methods,
    ))

    # APPLICABILITY PRE-FILTER (the supreme barrier): split the priced paths
    # by the country's national infrastructure profile BEFORE any ranking.
    profile = get_national_profile(country)
    banned = profile.get("banned_methods", frozenset())
    applicable = [(m, c) for m, c in zip(methods, co2e_values) if m not in banned]
    restricted = [(m, c) for m, c in zip(methods, co2e_values) if m in banned]
    if not applicable:   # a profile must never leave an item unrankable
        logger.warning("National profile for %r bans every %s path — "
                       "failing open (all paths applicable).",
                       country, material)
        applicable, restricted = list(zip(methods, co2e_values)), []

    # SORTING ENGINE: ascending net CO2e among APPLICABLE paths only —
    # deepest offset / lowest burden wins.
    outcomes = sorted(applicable, key=lambda mc: (mc[1], mc[0]))
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
            "is_applicable": True,
            "restriction_reason": None,
            # The regionalized runtime factor prices the path; the GB-anchored
            # baseline rides along so every scaling step stays auditable.
            "carbon_factor_kg_per_kg": round(
                get_scaled_disposal_factor(material, method, gi), 4),
            "base_factor_kg_per_kg": get_disposal_factor(material, method),
            "carbon_impact_kg": co2e_kg,
            "encouraging_verdict": _build_verdict(
                rank, material, method, co2e_kg, best_co2e, worst_co2e),
            "environmental_pros": knowledge["pros"],
            "environmental_cons": knowledge["cons"],
            # Two ordered do-this steps (LLM localizes; local grid is neutral).
            "action_steps": list(EXPERT_ACTION_STEPS[material][method]),
        })

    # Nationally banned paths ride LAST: numbers preserved for transparency,
    # rank stripped and status hard-set to "Banned" so no consumer can
    # mistake them for choices.
    for method, co2e_kg in sorted(restricted):
        knowledge = EXPERT_KNOWLEDGE[material][method]
        ranked.append({
            "method": method,
            "method_display": METHOD_DISPLAY_NAMES.get(
                method, method.replace("_", " ").title()),
            "rank": None,
            "status_tag": "Banned",
            "is_applicable": False,
            "restriction_reason": profile.get("reason", "not available"),
            "carbon_factor_kg_per_kg": round(
                get_scaled_disposal_factor(material, method, gi), 4),
            "base_factor_kg_per_kg": get_disposal_factor(material, method),
            "carbon_impact_kg": co2e_kg,
            "encouraging_verdict": _build_restricted_verdict(
                method, profile.get("reason", "not available")),
            "environmental_pros": knowledge["pros"],
            "environmental_cons": knowledge["cons"],
            # Carried for schema completeness; banned paths are never the
            # selected panel path, so these steps stay off-screen.
            "action_steps": list(EXPERT_ACTION_STEPS[material][method]),
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


def _applicable_paths(item: dict) -> list:
    """The nationally applicable (rank-carrying) subset, rank order."""
    return [p for p in item["recommendations"] if p["is_applicable"]]


def _llm_context(results: list, country: str) -> dict:
    """The read-only numeric context the LLM writes prose around.

    Nationally banned paths are EXCLUDED: their verdict is a fixed policy
    statement, and every comparative delta here is computed strictly inside
    the applicable pool (rank-1 best vs rank-K worst).
    """
    items = []
    for idx, item in enumerate(results):
        pool = _applicable_paths(item)
        best_kg = pool[0]["carbon_impact_kg"]
        worst_kg = pool[-1]["carbon_impact_kg"]
        items.append({
            "index": idx,
            "material": item["display_name"],
            "weight_kg": item["effective_weight_kg"],
            "paths": [
                {
                    "method": path["method"],
                    "method_display": path["method_display"],
                    "rank": path["rank"],
                    "status_tag": path["status_tag"],
                    "carbon_impact_kg": path["carbon_impact_kg"],
                    "saving_vs_worst_kg": round(
                        worst_kg - path["carbon_impact_kg"], 4),
                    "extra_vs_best_kg": round(
                        path["carbon_impact_kg"] - best_kg, 4),
                }
                for path in pool
            ],
        })
    # ``country`` stays the raw code (existing contract); ``country_name`` is
    # the full name the geopolitical directive grounds its advice in.
    return {
        "country": country or "global average",
        "country_name": _country_display(country),
        "items": items,
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
    Rewrite the four text fields via the LLM, retrying transient blips.

    Fast failures (HTTP 429/5xx "model overloaded", network hiccups,
    malformed or partial output) are retried up to ``_LLM_MAX_ATTEMPTS``
    times — free tiers throw momentary 503s that clear in seconds. Rate
    limiting is handled precisely: a ``Retry-After`` header replaces the
    default backoff (capped at ``_LLM_RETRY_AFTER_CAP_S``); a header ABOVE
    the cap means the quota window is far away, so the layer opens the
    cooldown breaker and degrades immediately instead of burning doomed
    retries. Exhausting all attempts on 429 also opens the breaker. A read
    TIMEOUT is NOT retried (its 60 s budget is already spent). Raises on
    final failure — the caller serves the local grid.
    """
    import requests  # local import keeps module import light for tests

    last_error = None
    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = _LLM_RETRY_BACKOFF_S[attempt - 2]
            if isinstance(last_error, _LlmHttpError) and last_error.retry_after:
                # The endpoint told us exactly when the window resets.
                delay = min(last_error.retry_after, _LLM_RETRY_AFTER_CAP_S)
            time.sleep(delay)
            logger.info("LLM text layer retry %d/%d (previous attempt: %s)",
                        attempt, _LLM_MAX_ATTEMPTS, last_error)
        try:
            _generate_and_apply(results, country, api_key, model, api_url)
            return
        except requests.exceptions.Timeout:
            raise                    # window already burned — degrade now
        except _LlmHttpError as exc:
            last_error = exc
            if (exc.status_code == 429 and exc.retry_after
                    and exc.retry_after > _LLM_RETRY_AFTER_CAP_S):
                # Quota resets minutes away — no retry can succeed inside
                # this request. Seal the vector and serve the grid NOW.
                _open_llm_cooldown(exc.retry_after)
                raise
        except Exception as exc:  # noqa: BLE001 - transient upstream noise
            last_error = exc

    if isinstance(last_error, _LlmHttpError) and last_error.status_code == 429:
        _open_llm_cooldown(last_error.retry_after)   # stop hammering the quota
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
        # Carry the endpoint's own rate-limit hint (Groq sends Retry-After
        # on 429) so the retry ladder can pace itself precisely.
        headers = getattr(resp, "headers", None) or {}
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
        raise _LlmHttpError(resp.status_code, retry_after)

    generated = _extract_json(resp.json()["choices"][0]["message"]["content"])
    by_index = {int(item["index"]): item for item in generated["items"]}

    staged = []
    for idx, item in enumerate(results):
        by_method = {p["method"]: p for p in by_index[idx]["paths"]}
        for path in item["recommendations"]:
            if not path["is_applicable"]:
                continue   # banned paths keep their deterministic policy copy
            gen = by_method[path["method"]]   # KeyError -> fallback upstream
            for field in ("encouraging_verdict", "environmental_pros",
                          "environmental_cons"):
                value = str(gen[field]).strip()
                if len(value.split()) < _LLM_MIN_FIELD_WORDS:
                    raise ValueError(
                        f"LLM '{field}' below the quality floor: {value!r}")
                staged.append((path, field, value))
            # action_steps: exactly TWO non-empty ordered strings (the model
            # can omit or malform them — any breach fails the whole enrichment
            # atomically, so the local neutral steps survive intact).
            steps = gen["action_steps"]       # KeyError -> fallback upstream
            if not isinstance(steps, list) or len(steps) != 2:
                raise ValueError(
                    f"LLM 'action_steps' must be exactly 2 steps: {steps!r}")
            steps = [str(s).strip() for s in steps]
            if not all(steps):
                raise ValueError("LLM 'action_steps' contains an empty step")
            staged.append((path, "action_steps", steps))
    for path, field, value in staged:        # atomic: only after FULL validation
        path[field] = value


def recommend_for_items(items: list, country: str = None) -> dict:
    """
    DMM aggregate entry for ``POST /api/recommend``.

    ``items`` is a list of ``{"material": str, "weight_kg": float?,
    "box_area_px": float?}`` dicts (type-validated by the pydantic schema at
    the route; each needs at least one of the two size fields). ``country``
    (optional ISO alpha-2, typically the frontend's IP-geolocated default)
    does three things: it selects the v3.8 GRID-INTENSITY scaling datum that
    re-derives every disposal factor from the GB-anchored baseline, it
    drives the v3.7 NATIONAL INFRASTRUCTURE applicability matrix (which
    end-of-life routes exist there at all), and it flavours the v3.6 LLM
    text layer. Returns::

        {
          "items": [
            { material, display_name, effective_weight_kg, weight_source,
              best_method, max_saving_kg, recommendations: [3 paths —
              applicable ranked first, banned flagged is_applicable=false] }
          ],
          "summary": { item_count, optimal_total_co2e_kg,
                       worst_total_co2e_kg, max_saving_kg },
          "grid": { country, intensity_kg_per_kwh,
                    base_intensity_kg_per_kwh, ratio, source },
          "country": "MY" | None,
          "provider": "llm_enriched" | "local_knowledge_base" | "local_fallback"
        }

    Provider semantics: ``llm_enriched`` = the batched LLM call rewrote the
    text fields; ``local_knowledge_base`` = no LLM key configured (the
    deterministic default); ``local_fallback`` = an LLM was configured but
    failed (rate limit, timeout, bad output) and the local grid took over
    seamlessly. ``best_method``, ``max_saving_kg`` and the summary compare
    "user follows every rank-1 path" against "every worst APPLICABLE path" —
    nationally banned paths never enter the subtraction, and totals may be
    NEGATIVE (net offsets).
    """
    country_code = str(country or "").strip().upper() or None

    # v3.8 grid engine: the country's live grid intensity is resolved ONCE
    # per request, in the request thread (cached Climatiq probe → local
    # average map → 0.207 anchor; never raises, never blocks the ranking).
    # Workers below receive only the resolved float — pure arithmetic.
    grid = resolve_grid_intensity(country_code)

    results = []
    optimal_total = 0.0
    worst_total = 0.0

    # One pool for the whole request; every item forks its 3 paths onto it.
    with ThreadPoolExecutor(max_workers=_PARALLEL_PATHS) as pool:
        for entry in items:
            material = entry["material"]
            weight_kg, weight_source = resolve_effective_weight(entry)
            ranked = simulate_disposal_paths(
                material, weight_kg, _pool=pool, country=country_code,
                grid_intensity=grid["intensity_kg_per_kwh"])

            # All comparative math draws STRICTLY from the applicable pool
            # (rank order: pool[0] = Optimal, pool[-1] = worst valid choice).
            pool_paths = [p for p in ranked if p["is_applicable"]]
            optimal_total += pool_paths[0]["carbon_impact_kg"]
            worst_total += pool_paths[-1]["carbon_impact_kg"]
            results.append({
                "material": material,
                "display_name": _display_material(material),
                "effective_weight_kg": round(weight_kg, 4),
                "weight_source": weight_source,
                "best_method": pool_paths[0]["method"],
                "max_saving_kg": round(
                    pool_paths[-1]["carbon_impact_kg"]
                    - pool_paths[0]["carbon_impact_kg"], 4),
                "recommendations": ranked,
            })

    # v3.6 text layer: LLM rewrite when configured, seamless local fallback.
    provider = "local_knowledge_base"
    api_key, model, api_url = _llm_settings()
    if api_key and model and api_url:
        if _llm_cooling_down():
            # Rate-limit breaker open: zero outbound flights — the previous
            # 429 told us the quota window; hammering it again helps nobody.
            logger.info("LLM text layer in rate-limit cooldown; serving the "
                        "local knowledge-base copy without a network call.")
            provider = "local_fallback"
        else:
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
        # v3.8 audit block: exactly which grid datum scaled this response.
        "grid": {
            "country": country_code,
            "intensity_kg_per_kwh": grid["intensity_kg_per_kwh"],
            "base_intensity_kg_per_kwh": BASE_GRID_INTENSITY,
            "ratio": grid["ratio"],
            "source": grid["source"],
        },
        "country": country_code,
        "provider": provider,
    }
