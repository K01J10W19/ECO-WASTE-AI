"""
Pydantic schemas for the recommendation API (POST /api/recommend, Step 6).

The request schema validates user input BEFORE it touches the Decision
Making Module; the response models double as living documentation of the
ranked-recommendation JSON contract. Note that carbon figures here may be
NEGATIVE — recycling/digestion paths carry avoided-burden credits.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.carbon_service import PIXEL_AREA_GAMMA

# Upper bound on an accepted box area. This is an INPUT-SANITY ceiling only —
# it is NOT the mass ceiling any more. The proxy weight is clamped to a
# plausible range inside carbon_service.proxy_weight_from_area, so a huge box
# can no longer translate into a huge mass (the v3.9 outlier fix).
_MAX_BOX_AREA_PX = 1000.0 * PIXEL_AREA_GAMMA


class RecommendationItemRequest(BaseModel):
    """One item to run through the 3-path disposal simulation."""

    material: str                         # one of the 7-class taxonomy
    # Stage-B audited weight — wins whenever present.
    weight_kg: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    # Stage-A blind proxy from /predict: the clamped geometric box area.
    box_area_px: Optional[float] = Field(default=None, gt=0.0, le=_MAX_BOX_AREA_PX)
    # Source-photo area (width x height) — normalises box_area_px into
    # resolution-invariant frame COVERAGE. Optional for back-compat; see
    # carbon_service.proxy_weight_from_area for why it matters.
    image_area_px: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _require_weight_or_area(self):
        if self.weight_kg is None and self.box_area_px is None:
            raise ValueError(
                "each item needs weight_kg (user-verified) or box_area_px "
                "(the blind pixel proxy)")
        return self


class RecommendRequest(BaseModel):
    """Body of POST /api/recommend."""

    items: List[RecommendationItemRequest] = Field(min_length=1, max_length=100)
    # ISO 3166-1 alpha-2 (e.g. the frontend's IP-geolocated default) — drives
    # the v3.7 national-infrastructure applicability matrix (which routes
    # exist there) and flavours the v3.6 LLM text layer; the per-path CO2e
    # arithmetic stays identical.
    country: Optional[str] = Field(default=None, pattern=r"^[A-Za-z]{2}$")

    @field_validator("country", mode="before")
    @classmethod
    def _blank_country_means_global(cls, value):
        """'' / whitespace → None: the text layer speaks in global averages."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class DisposalRecommendation(BaseModel):
    """One end-of-life path in the response (ranked iff nationally applicable).

    v3.7 applicability: a path banned by the request country's national
    infrastructure profile (e.g. landfill in zero-landfill Singapore) keeps
    its priced CO2e for transparency but carries ``is_applicable=false``,
    ``rank=null``, ``status_tag=null`` and a ``restriction_reason`` — and is
    excluded from every ranking, saving and summary computation.
    """

    method: str                           # e.g. "recycling", "composting"
    method_display: str                   # friendly UI label
    rank: Optional[int] = Field(default=None, ge=1, le=3)   # None = banned path
    status_tag: Optional[Literal["Optimal", "Acceptable", "Warning",
                                 "Banned"]] = None
    is_applicable: bool = True            # national-infrastructure verdict
    restriction_reason: Optional[str] = None   # set iff is_applicable is false
    # v3.8: the RUNTIME grid-scaled factor prices the path; the GB-anchored
    # baseline rides along so the scaling stays auditable per path.
    carbon_factor_kg_per_kg: float        # regionalized net factor; neg = credit
    base_factor_kg_per_kg: float          # GB-anchored baseline (0.207 kg/kWh)
    carbon_impact_kg: float               # scaled factor x weight; may be negative
    encouraging_verdict: str              # rank-aware copy | fixed policy copy
    environmental_pros: str               # knowledge-base rationale
    environmental_cons: str               # knowledge-base long-term costs
    # Exactly two ordered, country-localized do-this steps (LLM or local grid).
    action_steps: List[str] = Field(min_length=2, max_length=2)


class ActionMedia(BaseModel):
    """Option A item-level media block: the LLM writes a hyper-localized
    SEARCH QUERY (never a URL — hallucinated links are impossible) which the
    backend resolves to ONE live YouTube tutorial embed, plus an advanced
    expert tip. Keyless/offline → the verified universal fallback video."""

    video_search_query: str               # LLM-localized or local template
    expert_tip: str                       # authoritative material/country hack
    video_embed_url: str                  # https://www.youtube.com/embed/<id>
    video_title: str
    video_provider: Literal["youtube_live", "fallback"]


class RecommendedItem(BaseModel):
    """One simulated item with its full recommendation array (applicable
    paths ranked first, banned paths flagged at the tail)."""

    material: str
    display_name: str
    effective_weight_kg: float = Field(gt=0.0)
    weight_source: Literal["user_weight", "box_area_proxy"]
    best_method: str                      # the rank-1 APPLICABLE method id
    max_saving_kg: float = Field(ge=0.0)  # worst-vs-best among APPLICABLE paths
    recommendations: List[DisposalRecommendation] = Field(min_length=3, max_length=3)
    action_media: ActionMedia             # Option A tutorial video + expert tip


class RecommendSummary(BaseModel):
    """Aggregate across all items: optimal-vs-worst headline figures."""

    item_count: int = Field(ge=0)
    optimal_total_co2e_kg: float          # everyone follows rank 1 (may be < 0)
    worst_total_co2e_kg: float            # everyone follows rank 3
    max_saving_kg: float = Field(ge=0.0)


class GridScaling(BaseModel):
    """v3.8 audit block: the single grid datum that scaled this response."""

    country: Optional[str] = None         # None = global request (anchor used)
    intensity_kg_per_kwh: float = Field(gt=0.0)   # resolved local intensity
    base_intensity_kg_per_kwh: float = Field(gt=0.0)   # the GB anchor (0.207)
    ratio: float = Field(gt=0.0)          # intensity / anchor (1.0 = baseline)
    source: str                           # "climatiq" | "local_grid_map" | "baseline"


class RecommendResponse(BaseModel):
    """Full JSON body returned by POST /api/recommend."""

    items: List[RecommendedItem]
    summary: RecommendSummary
    grid: GridScaling                     # v3.8 grid-scaling audit block
    country: Optional[str] = None         # echoed ISO code (None = global)
    # "llm_enriched"        — v3.6 LLM text layer rewrote the literary fields
    # "local_knowledge_base" — no LLM key configured (deterministic default)
    # "local_fallback"      — LLM configured but failed; local grid took over
    provider: str
