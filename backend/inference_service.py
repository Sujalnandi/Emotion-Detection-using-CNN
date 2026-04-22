from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from backend.config import CONFIDENCE_THRESHOLD, DETECTION_CONFIDENCE_THRESHOLD
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
                "box": [int(x), int(y), int(w), int(h)],
                "bbox": {
                    "x": x / max(width, 1),
                    "y": y / max(height, 1),
                    "w": w / max(width, 1),
                    "h": h / max(height, 1),
                },
                "emotion": str(face["emotion"]).lower(),
                "display_label": str(face.get("display_label", face["emotion"])),
                "confidence": float(face["confidence"]),
                "confidence_level": str(face.get("confidence_level", "low")),
                "low_confidence": bool(face.get("low_confidence", float(face["confidence"]) < 0.4)),
                "probabilities": {
                    emotion: prob_vector[i]
                    for i, emotion in enumerate(EMOTIONS)
                    if i < len(prob_vector)
                },
                "top3": [
                    {
                        "emotion": str(item.get("emotion", "")).lower(),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                    for item in list(face.get("top3") or [])[:3]
                ],
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
        detection_interval=1,
        detection_confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detector_size=None,
        min_detection_width=640,
    )

    frame_h, frame_w = frame.shape[:2]
    all_probabilities = {emotion: float(probs[i]) for i, emotion in enumerate(EMOTIONS)}
    top_indices = list(np.argsort(probs)[::-1][: min(3, len(EMOTIONS))])
    top3 = [
        {"emotion": str(EMOTIONS[idx]).lower(), "confidence": float(probs[idx])}
        for idx in top_indices
    ]
    label_title = str(label).capitalize()

    return {
        "emotion": str(label).lower(),
        "display_label": f"{label_title} (low confidence {float(confidence) * 100.0:.0f}%)" if float(confidence) < 0.4 else label_title,
        "confidence": float(confidence),
        "confidence_level": "low" if float(confidence) < 0.4 else ("medium" if float(confidence) < 0.7 else "high"),
        "low_confidence": float(confidence) < 0.4,
        "all_probabilities": all_probabilities,
        "top3": top3,
        "boxes": _normalize_boxes(boxes, frame_w, frame_h),
        "faces": _faces_payload(faces_predictions, frame_w, frame_h),
    }
