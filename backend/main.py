from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference_service import infer_frame
from backend.model_loader import DETECTOR, MODEL, SMOOTHER
from backend.utils import decode_base64_image, decode_upload_bytes


class RealtimeRequest(BaseModel):
    frame: Optional[str] = None
    image_base64: Optional[str] = None
    image: Optional[str] = None


app = FastAPI(title="Facial Emotion Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Emotion Detection API Running"}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _public_prediction(result: dict[str, Any]) -> dict[str, Any]:
    if str(result.get("emotion", "")).strip().lower() == "no face":
        return {"error": "No face detected"}

    probabilities = {
        str(emotion).lower(): float(score)
        for emotion, score in (result.get("all_probabilities") or {}).items()
    }
    confidence = float(result.get("confidence", 0.0))

    return {
        "emotion": str(result.get("emotion", "Unknown")),
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        # Keep compatibility for existing frontend widgets that still read these keys.
        "all_probabilities": probabilities,
        "boxes": result.get("boxes", []),
        "faces": result.get("faces", []),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(None), file: UploadFile = File(None)) -> dict:
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No image file was uploaded")

    frame = decode_upload_bytes(await upload.read())
    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=False,
    )
    return _public_prediction(result)


@app.post("/realtime")
async def realtime(payload: RealtimeRequest) -> dict:
    data = payload.image or payload.frame or payload.image_base64
    if not data:
        raise HTTPException(status_code=400, detail="Missing base64 frame payload")

    frame = decode_base64_image(data)
    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=True,
    )
    public = _public_prediction(result)
    if "error" in public:
        return public

    return {
        "emotion": public["emotion"],
        "confidence": public["confidence"],
        "probabilities": public["probabilities"],
        "all_probabilities": public["all_probabilities"],
        "boxes": public["boxes"],
        "faces": public["faces"],
    }


# Backward-compatible aliases for existing frontend callers.
@app.post("/api/predict/image")
async def predict_image_legacy(image: UploadFile = File(...)) -> dict:
    frame = decode_upload_bytes(await image.read())
    result = await asyncio.to_thread(
        infer_frame,
        frame=frame,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=False,
    )
    public = _public_prediction(result)
    if "error" in public:
        return public
    return {
        "emotion": public["emotion"],
        "confidence": float(public["confidence"]) / 100.0,
        "probabilities": {k.capitalize(): float(v) for k, v in public["probabilities"].items()},
        "boxes": public["boxes"],
        "faces_predictions": [
            {
                "id": face["id"],
                "bbox": face["bbox"],
                "emotion": face["emotion"],
                "confidence": face["confidence"],
                "probabilities": face["probabilities"],
                "probabilityMap": face["probabilityMap"],
            }
            for face in public["faces"]
        ],
    }


@app.post("/api/predict/realtime")
async def predict_realtime_legacy(frame: UploadFile = File(...)) -> dict:
    image = decode_upload_bytes(await frame.read())
    result = await asyncio.to_thread(
        infer_frame,
        frame=image,
        model=MODEL,
        detector=DETECTOR,
        smoother=SMOOTHER,
        use_smoothing=True,
    )
    public = _public_prediction(result)
    if "error" in public:
        return public
    return {
        "emotion": public["emotion"],
        "confidence": float(public["confidence"]) / 100.0,
        "probabilities": {k.capitalize(): float(v) for k, v in public["probabilities"].items()},
        "boxes": public["boxes"],
        "faces_predictions": [
            {
                "id": face["id"],
                "bbox": face["bbox"],
                "emotion": face["emotion"],
                "confidence": face["confidence"],
                "probabilities": face["probabilities"],
                "probabilityMap": face["probabilityMap"],
            }
            for face in public["faces"]
        ],
    }
