from __future__ import annotations

import logging

from backend.inference.inference_engine import EmotionPredictor, init_face_detector, load_model_safe
from backend.config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, PREDICTION_SMOOTHING_WINDOW

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("backend.model_loader")

# Load heavy resources once at startup and reuse across requests.
try:
    MODEL = load_model_safe()
    DETECTOR = init_face_detector()
    SMOOTHER = EmotionPredictor(
        smoothing_window=PREDICTION_SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        detection_interval=DETECTION_INTERVAL,
    )
    logger.info("Model resources loaded successfully")
except Exception as exc:
    logger.exception("Failed to initialize model resources at startup: %s", exc)
    raise RuntimeError(
        "Startup failed: unable to initialize model, detector, or smoothing pipeline. "
        "Verify model artifacts and detector dependencies."
    ) from exc
