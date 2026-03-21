from __future__ import annotations

from backend.inference.inference_engine import EmotionPredictor, init_face_detector, load_model_safe
from backend.config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, PREDICTION_SMOOTHING_WINDOW

# Load heavy resources once at startup and reuse across requests.
MODEL = load_model_safe()
DETECTOR = init_face_detector()
SMOOTHER = EmotionPredictor(
    smoothing_window=PREDICTION_SMOOTHING_WINDOW,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    detection_interval=DETECTION_INTERVAL,
)
