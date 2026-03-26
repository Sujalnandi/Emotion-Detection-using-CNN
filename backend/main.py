from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference_service import infer_frame
from backend.model_loader import DETECTOR, MODEL, SMOOTHER
from backend.schemas import ErrorResponse, PredictionResponse
from backend.utils import decode_base64_image, decode_upload_bytes


class RealtimeRequest(BaseModel):
    frame: Optional[str] = None
    image_base64: Optional[str] = None
    image: Optional[str] = None


app = FastAPI(title="Facial Emotion Detection API", version="1.0.0")

logger = logging.getLogger("backend.api")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def _allowed_origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.get("/")
def root():
    return {"message": "Emotion Detection API Running"}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _validated_upload(upload: UploadFile) -> UploadFile:
    if upload.content_type and upload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use JPEG, PNG, or WEBP")
    return upload


def _check_upload_size(raw: bytes) -> None:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large. Max size is {MAX_UPLOAD_BYTES} bytes")


def _public_prediction(result: dict[str, Any]) -> PredictionResponse:
    if str(result.get("emotion", "")).strip().lower() == "no face":
        raise HTTPException(status_code=422, detail="No face detected")

    probabilities_raw = result.get("all_probabilities") or {}
    probabilities = {
        str(emotion).lower(): float(score)
        for emotion, score in dict(probabilities_raw).items()
    }
    confidence = float(result.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    return PredictionResponse(
        emotion=str(result.get("emotion", "unknown")).lower(),
        confidence=round(confidence, 4),
        probabilities=probabilities,
        boxes=result.get("boxes", []),
        faces=result.get("faces", []),
    )


@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
async def predict(image: UploadFile = File(None), file: UploadFile = File(None)) -> PredictionResponse:
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No image file was uploaded")

    upload = _validated_upload(upload)
    raw = await upload.read()
    _check_upload_size(raw)

    frame = decode_upload_bytes(raw)
    logger.info("Running image prediction")

    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=False,
    )
    return _public_prediction(result)


@app.post("/realtime", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
async def realtime(payload: RealtimeRequest) -> PredictionResponse:
    data = payload.image or payload.frame or payload.image_base64
    if not data:
        raise HTTPException(status_code=400, detail="Missing base64 frame payload")

    frame = decode_base64_image(data)
    logger.info("Running realtime prediction")
    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=True,
    )
    return _public_prediction(result)


# Backward-compatible aliases for existing frontend callers.
@app.post("/api/predict/image", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
async def predict_image_legacy(image: UploadFile = File(...)) -> PredictionResponse:
    image = _validated_upload(image)
    raw = await image.read()
    _check_upload_size(raw)
    frame = decode_upload_bytes(raw)

    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=False,
    )
    return _public_prediction(result)


@app.post("/api/predict/realtime", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
async def predict_realtime_legacy(frame: UploadFile = File(...)) -> PredictionResponse:
    frame = _validated_upload(frame)
    raw = await frame.read()
    _check_upload_size(raw)
    image = decode_upload_bytes(raw)

    result = await asyncio.to_thread(
        infer_frame,
        frame=image,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=True,
    )
    return _public_prediction(result)
