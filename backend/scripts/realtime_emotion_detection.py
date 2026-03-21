import argparse
import os
import sys

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, PREDICTION_SMOOTHING_WINDOW
from inference.inference_engine import (
    EmotionPredictor,
    find_latest_model,
    init_face_detector,
    load_model_safe,
    predict_frame,
)


def main():
    parser = argparse.ArgumentParser(description="Real-time facial emotion detection via webcam.")
    parser.add_argument("--model", default=None, help="Path to trained model file.")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Webcam index (default: 0)."
    )
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold.")
    args = parser.parse_args()

    model_path = args.model if args.model else find_latest_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first using train_model.py")

    model = load_model_safe(model_path)
    face_detector = init_face_detector()
    smoother = EmotionPredictor(
        smoothing_window=PREDICTION_SMOOTHING_WINDOW,
        confidence_threshold=args.threshold,
        detection_interval=DETECTION_INTERVAL,
    )

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 480))

        annotated, _, _, _, _, _, _ = predict_frame(
            frame,
            model=model,
            detector=face_detector,
            predictor=smoother,
            confidence_threshold=args.threshold,
            detection_interval=DETECTION_INTERVAL,
            detector_size=(640, 480),
        )

        cv2.imshow("Real-Time Emotion Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
