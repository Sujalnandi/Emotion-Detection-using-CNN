from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, PREDICTION_SMOOTHING_WINDOW
    from inference.inference_engine import (
        EMOTIONS,
        EmotionPredictor,
        init_face_detector,
        load_model_safe,
        predict_frame,
    )
except ModuleNotFoundError:
    from backend.config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, PREDICTION_SMOOTHING_WINDOW
    from backend.inference.inference_engine import (
        EMOTIONS,
        EmotionPredictor,
        init_face_detector,
        load_model_safe,
        predict_frame,
    )

app = FastAPI(title="Facial Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model_safe()
DETECTOR = init_face_detector()
SMOOTHER = EmotionPredictor(
    smoothing_window=PREDICTION_SMOOTHING_WINDOW,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    detection_interval=DETECTION_INTERVAL,
)


def _decode_upload(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return frame


def _response_payload(frame: np.ndarray, use_smoothing: bool) -> Dict:
    _, label, confidence, probs, boxes, faces_predictions, faces_results = predict_frame(
        frame,
        model=MODEL,
        detector=DETECTOR,
        predictor=SMOOTHER if use_smoothing else None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        detection_interval=DETECTION_INTERVAL,
        detector_size=(640, 480) if use_smoothing else None,
    )

    height, width = frame.shape[:2]
    norm_boxes = []
    for b in boxes:
        norm_boxes.append(
            {
                "x": b["x"] / max(width, 1),
                "y": b["y"] / max(height, 1),
                "w": b["w"] / max(width, 1),
                "h": b["h"] / max(height, 1),
            }
        )

    probabilities = {emotion.capitalize(): float(probs[i]) for i, emotion in enumerate(EMOTIONS)}

    faces_payload = []
    for face in faces_predictions:
        x, y, w, h = face["bbox"]
        prob_vector = [float(v) for v in face["probabilities"]]
        prob_map = {
            emotion.capitalize(): prob_vector[i]
            for i, emotion in enumerate(EMOTIONS)
            if i < len(prob_vector)
        }
        faces_payload.append(
            {
                "id": int(face["id"]),
                "bbox": {
                    "x": x / max(width, 1),
                    "y": y / max(height, 1),
                    "w": w / max(width, 1),
                    "h": h / max(height, 1),
                },
                "emotion": str(face["emotion"]),
                "confidence": float(face["confidence"]),
                "probabilities": prob_vector,
                "probabilityMap": prob_map,
            }
        )

    return {
        "emotion": label.capitalize(),
        "confidence": float(confidence),
        "probabilities": probabilities,
        "boxes": norm_boxes,
        "faces_predictions": faces_payload,
        "faces_results": faces_results,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Facial Emotion Detection API is running",
        "health": "/health",
        "docs": "/docs",
        "predict_image": "POST /api/predict/image",
        "predict_realtime": "POST /api/predict/realtime",
    }


@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404 logs when opening API root in a browser.
    return Response(status_code=204)


@app.post("/api/predict/image")
async def predict_image(image: UploadFile = File(...)):
    data = await image.read()
    frame = _decode_upload(data)
    return _response_payload(frame, use_smoothing=False)


@app.post("/api/predict/realtime")
async def predict_realtime(frame: UploadFile = File(...)):
    data = await frame.read()
    image = _decode_upload(data)
    return _response_payload(image, use_smoothing=True)
