from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from backend.config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL
from backend.inference.inference_engine import EMOTIONS, EmotionPredictor, predict_frame


def _normalize_boxes(boxes, width: int, height: int):
    out = []
    for b in boxes:
        out.append(
            {
                "x": b["x"] / max(width, 1),
                "y": b["y"] / max(height, 1),
                "w": b["w"] / max(width, 1),
                "h": b["h"] / max(height, 1),
            }
        )
    return out


def _faces_payload(faces_predictions, width: int, height: int):
    payload = []
    for face in faces_predictions:
        x, y, w, h = face["bbox"]
        prob_vector = [float(v) for v in face["probabilities"]]
        payload.append(
            {
                "id": int(face["id"]),
                "bbox": {
                    "x": x / max(width, 1),
                    "y": y / max(height, 1),
                    "w": w / max(width, 1),
                    "h": h / max(height, 1),
                },
                "emotion": str(face["emotion"]).lower(),
                "confidence": float(face["confidence"]),
                "probabilities": {
                    emotion: prob_vector[i]
                    for i, emotion in enumerate(EMOTIONS)
                    if i < len(prob_vector)
                },
            }
        )
    return payload


def infer_frame(
    frame: np.ndarray,
    model,
    detector,
    smoother: Optional[EmotionPredictor] = None,
    use_smoothing: bool = False,
) -> Dict:
    _, label, confidence, probs, boxes, faces_predictions, _ = predict_frame(
        frame,
        model=model,
        detector=detector,
        predictor=smoother if use_smoothing else None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        detection_interval=DETECTION_INTERVAL,
        detector_size=(640, 480) if use_smoothing else None,
    )

    frame_h, frame_w = frame.shape[:2]
    all_probabilities = {emotion: float(probs[i]) for i, emotion in enumerate(EMOTIONS)}

    return {
        "emotion": str(label).lower(),
        "confidence": float(confidence),
        "all_probabilities": all_probabilities,
        "boxes": _normalize_boxes(boxes, frame_w, frame_h),
        "faces": _faces_payload(faces_predictions, frame_w, frame_h),
    }
