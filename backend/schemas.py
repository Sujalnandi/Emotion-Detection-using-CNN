from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    w: float = Field(..., ge=0.0)
    h: float = Field(..., ge=0.0)


class FacePrediction(BaseModel):
    id: int = Field(..., ge=1)
    bbox: FaceBox
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    boxes: List[FaceBox]
    faces: List[FacePrediction]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
