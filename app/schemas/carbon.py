"""
Pydantic schemas for the carbon API (POST /api/calculate-impact, Step 5).

The request schema validates user input BEFORE it touches carbon_service;
the response models double as living documentation of the JSON contract.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class WeightedItem(BaseModel):
    """One user-weighted waste item to price in CO2e."""

    material: str                                   # one of the 7-class taxonomy
    weight_kg: float = Field(gt=0.0, le=1000.0)     # user-entered weight in kg


class CalculateImpactRequest(BaseModel):
    """Body of POST /api/calculate-impact."""

    items: List[WeightedItem] = Field(min_length=1, max_length=100)
    country: Optional[str] = Field(default=None, pattern=r"^[A-Za-z]{2}$")  # ISO 3166-1 alpha-2


class ImpactItem(BaseModel):
    """One priced item in the response."""

    material: str
    weight_kg: float
    carbon_factor_kg_per_kg: float = Field(ge=0.0)
    co2e_kg: float = Field(ge=0.0)
    source: str                                     # "climatiq" | "local_dummy"


class CalculateImpactResponse(BaseModel):
    """Full JSON body returned by POST /api/calculate-impact."""

    items: List[ImpactItem]
    total_co2e_kg: float = Field(ge=0.0)
    country: Optional[str]
    provider: str                                   # "climatiq" | "local_dummy" | "mixed"
