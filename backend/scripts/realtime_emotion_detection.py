import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import (
    CONFIDENCE_THRESHOLD,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_INTERVAL,
    FACE_DETECTOR_BACKEND,
    PREDICTION_SMOOTHING_WINDOW,
)
from inference.inference_engine import (
    EmotionPredictor,
    find_latest_model,
    init_face_detector,
    load_model_safe,
    predict_frame,
)


def _parse_detector_size(value: str) -> Optional[Tuple[int, int]]:
    if value.lower() == "none":
        return None

    if "x" not in value:
        raise ValueError("detector-size must be WIDTHxHEIGHT, for example 640x480, or 'none'.")

    width_text, height_text = value.lower().split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width < 160 or height < 120:
        raise ValueError("detector-size is too small. Use at least 160x120.")
    return width, height


def main():
    parser = argparse.ArgumentParser(description="Real-time multi-face emotion detection via webcam.")
    parser.add_argument("--model", default=None, help="Path to trained model file.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Emotion confidence threshold for stable label updates (default: config value).",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DETECTION_CONFIDENCE_THRESHOLD,
        help="Face detector confidence threshold (default: 0.90).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=max(10, int(PREDICTION_SMOOTHING_WINDOW)),
        help="Temporal smoothing window (recommended 10-15).",
    )
    parser.add_argument(
        "--detector-backend",
        choices=["retinaface", "mtcnn", "haar", "auto"],
        default="auto",
        help="Face detector backend to use (default: auto -> config preference).",
    )
    parser.add_argument(
        "--detector-size",
        default="640x480",
        help="Detection resize resolution as WIDTHxHEIGHT, or 'none' for adaptive.",
    )
    parser.add_argument(
        "--display-size",
        default="640x480",
        help="Display frame size as WIDTHxHEIGHT, or 'none' to keep camera native size.",
    )
    args = parser.parse_args()

    smoothing_window = min(15, max(10, int(args.smoothing_window)))
    detector_size = _parse_detector_size(args.detector_size)
    display_size = _parse_detector_size(args.display_size)

    model_path = args.model if args.model else find_latest_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first using train_model.py")

    backend_preference = None if args.detector_backend == "auto" else args.detector_backend
    model = load_model_safe(model_path)
    face_detector = init_face_detector(backend_preference=backend_preference)

    smoother = EmotionPredictor(
        smoothing_window=smoothing_window,
        confidence_threshold=float(args.threshold),
        detection_interval=DETECTION_INTERVAL,
    )

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    selected_backend = face_detector.get("backend", FACE_DETECTOR_BACKEND)

    print("Real-time Multi-Face Emotion Detection Started")
    print(f"  Model: {model_path}")
    print(f"  Detector backend: {selected_backend}")
    print(f"  Emotion threshold: {args.threshold}")
    print(f"  Face detection threshold: {args.detection_threshold}")
    print(f"  Smoothing window: {smoothing_window} frames")
    print(f"  Detection interval: every {DETECTION_INTERVAL} frames")
    print("  Press 'q' to quit.\n")

    frame_count = 0
    fps = 0.0
    last_fps_update = time.time()
    fps_counter = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        if display_size is not None:
            frame = cv2.resize(frame, display_size)

        (
            annotated,
            _,
            _,
            _,
            _,
            faces_predictions,
            _,
        ) = predict_frame(
            frame,
            model=model,
            detector=face_detector,
            predictor=smoother,
            confidence_threshold=float(args.threshold),
            detection_interval=DETECTION_INTERVAL,
            detector_size=detector_size,
            detection_confidence_threshold=float(args.detection_threshold),
        )

        now = time.time()
        fps_counter += 1
        elapsed = now - last_fps_update
        if elapsed >= 0.5:
            fps = fps_counter / max(elapsed, 1e-6)
            fps_counter = 0
            last_fps_update = now

        stats_text = f"FPS: {fps:.1f} | Faces: {len(faces_predictions)} | Detector: {selected_backend}"
        cv2.putText(
            annotated,
            stats_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Real-Time Multi-Face Emotion Detection", annotated)

        frame_count += 1
        if frame_count % 60 == 0:
            print(
                f"Frame {frame_count}: faces={len(faces_predictions)}, fps={fps:.1f}, "
                f"backend={selected_backend}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nStopped after {frame_count} frames.")


if __name__ == "__main__":
    main()
