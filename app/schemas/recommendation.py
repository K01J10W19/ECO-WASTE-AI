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

# Cap the blind pixel proxy at the same ceiling as user weights:
# area / gamma <= 1000 kg (mirrors WeightedItem's le=1000).
_MAX_BOX_AREA_PX = 1000.0 * PIXEL_AREA_GAMMA


class RecommendationItemRequest(BaseModel):
    """One item to run through the 3-path disposal simulation."""

    material: str                         # one of the 7-class taxonomy
    # Stage-B audited weight — wins whenever present.
    weight_kg: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    # Stage-A blind proxy from /predict: effective weight = area / gamma.
    box_area_px: Optional[float] = Field(default=None, gt=0.0, le=_MAX_BOX_AREA_PX)

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
    # ISO 3166-1 alpha-2 (e.g. the frontend's IP-geolocated default) — only
    # flavours the v3.6 LLM text layer; the carbon numbers stay identical.
    country: Optional[str] = Field(default=None, pattern=r"^[A-Za-z]{2}$")

    @field_validator("country", mode="before")
    @classmethod
    def _blank_country_means_global(cls, value):
        """'' / whitespace → None: the text layer speaks in global averages."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class DisposalRecommendation(BaseModel):
    """One ranked end-of-life path in the response."""

    method: str                           # e.g. "recycling", "composting"
    method_display: str                   # friendly UI label
    rank: int = Field(ge=1, le=3)         # 1 = optimal ... 3 = worst baseline
    status_tag: Literal["Optimal", "Acceptable", "Warning"]
    carbon_factor_kg_per_kg: float        # net factor; negative = credit
    carbon_impact_kg: float               # factor x weight; may be negative
    encouraging_verdict: str              # rank-aware supportive copy
    environmental_pros: str               # knowledge-base rationale
    environmental_cons: str               # knowledge-base long-term costs


class RecommendedItem(BaseModel):
    """One simulated item with its full ranked recommendation array."""

    material: str
    display_name: str
    effective_weight_kg: float = Field(gt=0.0)
    weight_source: Literal["user_weight", "box_area_proxy"]
    best_method: str                      # the rank-1 method identifier
    max_saving_kg: float = Field(ge=0.0)  # worst-path minus best-path CO2e
    recommendations: List[DisposalRecommendation] = Field(min_length=3, max_length=3)


class RecommendSummary(BaseModel):
    """Aggregate across all items: optimal-vs-worst headline figures."""

    item_count: int = Field(ge=0)
    optimal_total_co2e_kg: float          # everyone follows rank 1 (may be < 0)
    worst_total_co2e_kg: float            # everyone follows rank 3
    max_saving_kg: float = Field(ge=0.0)


class RecommendResponse(BaseModel):
    """Full JSON body returned by POST /api/recommend."""

    items: List[RecommendedItem]
    summary: RecommendSummary
    country: Optional[str] = None         # echoed ISO code (None = global)
    # "llm_enriched"        — v3.6 LLM text layer rewrote the literary fields
    # "local_knowledge_base" — no LLM key configured (deterministic default)
    # "local_fallback"      — LLM configured but failed; local grid took over
    provider: str
