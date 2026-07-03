"""
Pydantic schemas for the detection API.

These document and validate the shape returned by ``detection_service`` and the
``POST /api/predict`` endpoint. They double as living documentation of the JSON
contract the frontend consumes.
"""
from typing import List

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """A single detected waste item."""

    id: int
    class_name: str                       # LOCKED internal name, e.g. "PLASTIC"
    display_name: str                     # friendly UI label, e.g. "Plastic"
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[int] = Field(min_length=4, max_length=4)  # [x1, y1, x2, y2] pixels


class ImageInfo(BaseModel):
    """Pixel dimensions of the analysed image."""

    width: int
    height: int


class DetectionResult(BaseModel):
    """What ``detection_service.run_detection`` returns."""

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
