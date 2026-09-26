from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class WasteClass(StrEnum):
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    UNKNOWN = "unknown"


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    UNKNOWN = "unknown"
    RESCAN = "rescan"


class SensorReading(BaseModel):
    metal_detected: bool | None = None
    magnetic: bool | None = None
    weight_grams: float | None = Field(default=None, ge=0)


class DecisionRequest(SensorReading):
    probabilities: dict[WasteClass, float]
    image_quality: float = Field(default=1.0, ge=0, le=1)
    multi_view_agreement: float = Field(default=1.0, ge=0, le=1)
    region: str = "Taiwan"

    @model_validator(mode="after")
    def validate_probabilities(self) -> "DecisionRequest":
        expected = set(WasteClass)
        if set(self.probabilities) != expected:
            raise ValueError(f"probabilities must contain exactly: {sorted(item.value for item in expected)}")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(self.probabilities.values()) - 1.0) > 0.02:
            raise ValueError("probabilities must sum to approximately 1")
        return self


class DecisionResult(BaseModel):
    final_class: WasteClass
    confidence: float
    status: DecisionStatus
    reason_codes: list[str]
    subtype: str | None = None


class PredictionResponse(BaseModel):
    result: DecisionResult
    probabilities: dict[WasteClass, float]
    image_quality: float
    advice: str
    region: str


class BoundingBox(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_extent(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")
        return self


class ObjectPrediction(BaseModel):
    object_id: int = Field(ge=1)
    box: BoundingBox
    detection_confidence: float = Field(ge=0, le=1)
    classification: DecisionResult
    probabilities: dict[WasteClass, float]
    advice: str


class MultiPredictionResponse(BaseModel):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_quality: float = Field(ge=0, le=1)
    objects: list[ObjectPrediction]
    region: str
