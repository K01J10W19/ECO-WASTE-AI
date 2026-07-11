"""
Pydantic schemas for the carbon API (POST /api/calculate-impact, Step 5;
grid-sync + pixel-proxy upgrade in v3.5).

The request schema validates user input BEFORE it touches carbon_service;
the response models double as living documentation of the JSON contract.

v3.5 UX contract:
  * ``id`` — optional client row key (the /predict item id). Echoed back
    VERBATIM per response item so the split-screen frontend (image canvas ↔
    editable grid) can track edits bi-directionally without re-matching rows.
  * ``weight_kg`` optional — when absent, ``box_area_px / gamma`` becomes the
    blind pixel-proxy weight (at least one of the two is required).
  * ``country`` — optional ISO 3166-1 alpha-2, typically the frontend's
    IP-geolocated default. Blank/whitespace values coerce to None so the
    Climatiq client gracefully falls back to its global dataset.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.carbon_service import PIXEL_AREA_GAMMA

# Cap the blind pixel proxy at the same ceiling as user weights:
# area / gamma <= 1000 kg (mirrors weight_kg's le=1000).
_MAX_BOX_AREA_PX = 1000.0 * PIXEL_AREA_GAMMA


class WeightedItem(BaseModel):
    """One waste item to price in CO2e (audited weight or pixel proxy)."""

    id: Optional[int] = None                        # client grid/canvas row key
    material: str                                   # one of the 7-class taxonomy
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


class CalculateImpactRequest(BaseModel):
    """Body of POST /api/calculate-impact."""

    items: List[WeightedItem] = Field(min_length=1, max_length=100)
    # ISO 3166-1 alpha-2 (e.g. the frontend's IP-geolocated default).
    country: Optional[str] = Field(default=None, pattern=r"^[A-Za-z]{2}$")

    @field_validator("country", mode="before")
    @classmethod
    def _blank_country_means_global(cls, value):
        """'' / whitespace → None: omit region scoping (Climatiq global set)."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _ids_unique_when_provided(self):
        ids = [item.id for item in self.items if item.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique when provided "
                             "(they key the frontend grid rows)")
        return self


class ImpactItem(BaseModel):
    """One priced item in the response."""

    id: Optional[int] = None                        # echoed client row key
    material: str
    weight_kg: float = Field(gt=0.0)                # EFFECTIVE weight priced
    weight_source: Literal["user_weight", "box_area_proxy"]
    carbon_factor_kg_per_kg: float = Field(ge=0.0)
    co2e_kg: float = Field(ge=0.0)
    source: str                                     # "climatiq" | "local_dummy"


class CalculateImpactResponse(BaseModel):
    """Full JSON body returned by POST /api/calculate-impact."""

    items: List[ImpactItem]
    total_co2e_kg: float = Field(ge=0.0)
    country: Optional[str]
    provider: str                                   # "climatiq" | "local_dummy" | "mixed"
