import argparse
import os
import sys
from collections import deque

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


class TemporalSmoother:
    """
    Smooth predictions across frames using voting.
    
    This prevents jittery predictions and stabilizes emotion classification.
    Example: If last 5 frames predict [Happy, Happy, Sad, Happy, Happy],
    the smoothed result is Happy with confidence based on voting.
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, emotion: str, confidence: float):
        """Add prediction to buffer."""
        self.buffer.append((emotion, confidence))
    
    def get_smoothed_prediction(self):
        """
        Get most frequent emotion from buffer.
        Returns tuple: (emotion, smoothed_confidence)
        """
        if not self.buffer:
            return "neutral", 0.0
        
        # Count occurrences of each emotion
        emotion_votes = {}
        emotion_confidences = {}
        
        for emotion, confidence in self.buffer:
            emotion_votes[emotion] = emotion_votes.get(emotion, 0) + 1
            if emotion not in emotion_confidences:
                emotion_confidences[emotion] = []
            emotion_confidences[emotion].append(confidence)
        
        # Get most voted emotion
        best_emotion = max(emotion_votes.items(), key=lambda x: x[1])[0]
        
        # Average confidence for best emotion
        avg_confidence = sum(emotion_confidences[best_emotion]) / len(emotion_confidences[best_emotion])
        
        return best_emotion, avg_confidence


def main():
    parser = argparse.ArgumentParser(description="Real-time facial emotion detection via webcam.")
    parser.add_argument("--model", default=None, help="Path to trained model file.")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Webcam index (default: 0)."
    )
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold.")
    parser.add_argument(
        "--smoothing-window", type=int, default=PREDICTION_SMOOTHING_WINDOW, 
        help="Number of frames to use for temporal smoothing (default: 7)."
    )
    args = parser.parse_args()

    model_path = args.model if args.model else find_latest_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first using train_model.py")

    model = load_model_safe(model_path)
    face_detector = init_face_detector()
    
    # **IMPROVEMENT 1**: Enhanced EmotionPredictor with larger smoothing window
    smoother = EmotionPredictor(
        smoothing_window=args.smoothing_window,
        confidence_threshold=args.threshold,
        detection_interval=DETECTION_INTERVAL,
    )
    
    # **IMPROVEMENT 2**: Additional temporal smoothing buffer for extra stability
    temporal_smoother = TemporalSmoother(window_size=args.smoothing_window)

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print(f"Real-time Emotion Detection Started")
    print(f"  Model: {model_path}")
    print(f"  Confidence threshold: {args.threshold}")
    print(f"  Temporal smoothing window: {args.smoothing_window} frames")
    print(f"  Press 'q' to quit.\n")

    frame_count = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 480))

        annotated, emotion_label, confidence, probs, boxes, faces_predictions, faces_results = predict_frame(
            frame,
            model=model,
            detector=face_detector,
            predictor=smoother,
            confidence_threshold=args.threshold,
            detection_interval=DETECTION_INTERVAL,
            detector_size=(640, 480),
        )

        # **IMPROVEMENT 3**: Apply additional temporal smoothing to detected emotions
        if faces_predictions:
            for face in faces_predictions:
                emotion = face["emotion"].lower()
                face_confidence = face["confidence"]
                temporal_smoother.add_prediction(emotion, float(face_confidence))
        
        # Display frame information
        cv2.imshow("Real-Time Emotion Detection", annotated)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print stats every 30 frames (~1 second at 30 FPS)
            if temporal_smoother.buffer:
                smoothed_emotion, smoothed_conf = temporal_smoother.get_smoothed_prediction()
                print(f"Frame {frame_count}: Smoothed emotion = {smoothed_emotion} ({smoothed_conf:.2f})")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nStopped after {frame_count} frames.")


if __name__ == "__main__":
    main()
