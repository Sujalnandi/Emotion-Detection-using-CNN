import argparse
import os
import sys

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from inference.inference_engine import init_face_detector, load_model_safe, predict_frame


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a facial image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        default=os.path.join(BACKEND_DIR, "artifacts", "best_model.h5"),
        help="Path to trained model file.",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold.")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model_safe(model_path)
    detector = init_face_detector()

    _, label, confidence, probs, _, faces_predictions, _ = predict_frame(
        image,
        model=model,
        detector=detector,
        predictor=None,
        confidence_threshold=args.threshold,
        detection_interval=1,
        detector_size=None,
    )

    if not faces_predictions:
        print("No face detected.")
        return

    print(f"Primary Emotion: {label.capitalize()}")
    print(f"Primary Confidence: {confidence * 100.0:.2f}%")
    print("\nDetected Faces:")
    for face in faces_predictions:
        x, y, w, h = face["bbox"]
        print(
            f"- Face {face['id']}: bbox=({x},{y},{w},{h}), emotion={face['emotion']}, "
            f"confidence={face['confidence'] * 100.0:.2f}%"
        )
        print(f"  Probabilities: {face['probabilities']}")

    print(f"\nPrimary Probability Vector: {probs.astype(float).tolist()}")


if __name__ == "__main__":
    main()
