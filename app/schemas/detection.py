"""
Pydantic schemas for the detection API (Dual-Tower Hybrid pipeline).

These document and validate the shape returned by
``detection_service.analyze_waste_pipeline`` and the ``POST /api/predict``
endpoint. They double as living documentation of the JSON contract the
frontend consumes.
"""
from typing import List

from pydantic import BaseModel, Field


class MaterialScore(BaseModel):
    """One entry of the Stage-2 ViT softmax distribution for an item."""

    label: str                            # material class, e.g. "plastic"
    score: float = Field(ge=0.0, le=1.0)


class PhysicsInfo(BaseModel):
    """Method B evidence: classical-CV cues + the Plasticity Index psi."""

    laplacian_variance: float = Field(ge=0.0)   # micro-wrinkle texture variance
    edge_density: float = Field(ge=0.0, le=1.0)  # fraction of Canny edge pixels
    plasticity_index: float = Field(ge=0.0, le=1.0)  # psi: >=0.5 plastic-like
    tiebreak_applied: bool                # True when psi corrected the ViT ranking


class DetectionItem(BaseModel):
    """A single detected waste item (Stage-1 instance + Stage-2 material)."""

    id: int
    class_name: str                       # winning material, e.g. "plastic"
    display_name: str                     # friendly UI label, e.g. "Plastic"
    confidence: float = Field(ge=0.0, le=1.0)        # Stage-2 ViT softmax score
    box_confidence: float = Field(ge=0.0, le=1.0)    # Stage-1 localization score
    located_as: str                       # Stage-1 detector's own label (diagnostic only)
    bbox: List[int] = Field(min_length=4, max_length=4)  # [x1, y1, x2, y2] pixels
    box_area_px: float = Field(ge=0.0)    # (x2-x1)*(y2-y1) — volume/mass proxy
    material_scores: List[MaterialScore]  # ViT distribution (post tie-break), sorted desc
    physics: PhysicsInfo                  # Method B cues + tie-break audit flag
    carbon_factor_kg_per_kg: float = Field(ge=0.0)  # base material coefficient
    estimated_carbon_kg: float = Field(ge=0.0)      # base x (box_area_px / gamma)


class ImageInfo(BaseModel):
    """Pixel dimensions of the analysed image."""

    width: int
    height: int


class DetectionResult(BaseModel):
    """What ``detection_service.analyze_waste_pipeline`` returns."""

    items: List[DetectionItem]
    image: ImageInfo


class PredictImageInfo(ImageInfo):
    """Image info enriched by the endpoint with the saved file's name + URL."""

    filename: str
    url: str


class PredictResponse(BaseModel):
    """Full JSON body returned by ``POST /api/predict``."""

    items: List[DetectionItem]
    image: PredictImageInfo
