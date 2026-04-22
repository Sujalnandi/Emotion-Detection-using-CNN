from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    w: float = Field(..., ge=0.0)
    h: float = Field(..., ge=0.0)


class RankedPrediction(BaseModel):
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class FacePrediction(BaseModel):
    id: int = Field(..., ge=1)
    box: Optional[List[int]] = Field(default=None, min_length=4, max_length=4)
    bbox: FaceBox
    emotion: str
    display_label: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: Optional[str] = None
    low_confidence: Optional[bool] = None
    probabilities: Dict[str, float]
    top3: Optional[List[RankedPrediction]] = None


class PredictionResponse(BaseModel):
    emotion: str
    display_label: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: Optional[str] = None
    low_confidence: Optional[bool] = None
    probabilities: Dict[str, float]
    top3: Optional[List[RankedPrediction]] = None
    boxes: List[FaceBox]
    faces: List[FacePrediction]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
