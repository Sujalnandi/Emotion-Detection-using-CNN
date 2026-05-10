import argparse
from collections import defaultdict, deque
import os
import sys
import time
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import (
    CONFIDENCE_THRESHOLD,
    DETECTION_CONFIDENCE_THRESHOLD,
    FACE_DETECTOR_BACKEND,
    PREDICTION_SMOOTHING_WINDOW,
)
from inference.inference_engine import (
    EMOTIONS,
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


def _ensure_min_width(frame: np.ndarray, min_width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w >= min_width:
        return frame
    scale = float(min_width) / float(max(w, 1))
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _equalize_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _normalize_probs(probs: np.ndarray, classes: int) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if arr.size != classes:
        return np.full((classes,), 1.0 / float(classes), dtype=np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(arr))
    if total > 1e-12 and np.all(arr >= 0.0):
        normalized = arr / total
        if np.max(normalized) <= 1.0 + 1e-3:
            return normalized.astype(np.float32)

    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom <= 1e-12:
        return np.full((classes,), 1.0 / float(classes), dtype=np.float32)
    return (exps / denom).astype(np.float32)


def _confidence_color(confidence: float) -> Tuple[int, int, int]:
    if confidence > 0.6:
        return (0, 200, 0)  # Green
    if confidence >= 0.4:
        return (0, 220, 255)  # Yellow
    return (0, 0, 230)  # Red


def _format_label(emotion: str, confidence: float) -> str:
    readable = str(emotion).capitalize()
    if confidence < 0.4:
        return f"{readable} (low confidence)"
    return f"{readable} ({confidence * 100.0:.1f}%)"


def _top3_from_probs(probs: np.ndarray, class_names: List[str]) -> List[Tuple[str, float]]:
    idxs = np.argsort(probs)[::-1][: min(3, len(class_names))]
    return [(class_names[int(i)], float(probs[int(i)])) for i in idxs]


def _draw_face_overlay(frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]) -> None:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h

    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    box_top = max(0, y - th - baseline - 8)
    box_bottom = max(y, y - 2)
    box_right = min(frame.shape[1] - 1, x + tw + 8)

    cv2.rectangle(frame, (x, box_top), (box_right, box_bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x + 4, max(12, box_bottom - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class TemporalFaceSmoother:
    """Per-face temporal smoothing with fixed-size probability buffers."""

    def __init__(self, class_names: List[str], window_size: int = 12):
        self.class_names = list(class_names)
        self.window_size = max(10, min(15, int(window_size)))
        self._buffers: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._missed_frames: Dict[int, int] = defaultdict(int)
        self._max_missed = self.window_size * 2

    def update(self, face_id: int, probs: np.ndarray) -> np.ndarray:
        normalized = _normalize_probs(probs, classes=len(self.class_names))
        self._buffers[int(face_id)].append(normalized)
        self._missed_frames[int(face_id)] = 0

        stacked = np.stack(list(self._buffers[int(face_id)]), axis=0)
        mean_probs = np.mean(stacked, axis=0)
        return _normalize_probs(mean_probs, classes=len(self.class_names))

    def mark_missing(self, seen_face_ids: Set[int]) -> None:
        seen = {int(v) for v in seen_face_ids}
        for face_id in list(self._buffers.keys()):
            if face_id in seen:
                continue
            self._missed_frames[face_id] += 1
            if self._missed_frames[face_id] > self._max_missed:
                self._buffers.pop(face_id, None)
                self._missed_frames.pop(face_id, None)


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
        choices=["mediapipe", "mtcnn", "auto"],
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
    parser.add_argument(
        "--debug-predictions",
        action="store_true",
        help="Print per-face raw probabilities, selected class, confidence, and top-3.",
    )
    parser.add_argument(
        "--debug-boxes",
        action="store_true",
        help="Print per-face bounding box coordinates (x, y, w, h) for alignment debugging.",
    )
    parser.add_argument(
        "--debug-crop",
        action="store_true",
        help="Show model input face crops (224x224) and draw crop rectangles for verification.",
    )
    parser.add_argument(
        "--equalize-low-light",
        action="store_true",
        help="Apply histogram equalization on luminance channel to improve low-light stability.",
    )
    parser.add_argument(
        "--adaptive-skip",
        action="store_true",
        help="Adaptively skip inference frames to keep realtime FPS near target.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target FPS for adaptive skipping (default: 15).",
    )
    parser.add_argument(
        "--max-skip",
        type=int,
        default=3,
        help="Maximum inference stride for adaptive skip (1=max quality, 3=run every 3rd frame).",
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
        detection_interval=1,
    )
    temporal_smoother = TemporalFaceSmoother(class_names=EMOTIONS, window_size=smoothing_window)

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    selected_backend = face_detector.get("backend", FACE_DETECTOR_BACKEND)

    print("Real-time Multi-Face Emotion Detection Started")
    print(f"  Model: {model_path}")
    print(f"  Detector backend: {selected_backend}")
    print(f"  Emotion threshold: {args.threshold}")
    print(f"  Face detection threshold: {args.detection_threshold}")
    print(f"  Smoothing window: {smoothing_window} frames")
    print("  Detection interval: every frame")
    print("  Label policy: always top-1 emotion, low-confidence guidance below 0.4")
    print(f"  Debug face crop: {'enabled' if args.debug_crop else 'disabled'}")
    print(f"  Debug face boxes: {'enabled' if args.debug_boxes else 'disabled'}")
    if args.adaptive_skip:
        print(f"  Adaptive skip: enabled (target_fps={args.target_fps:.1f}, max_skip={max(1, int(args.max_skip))})")
    else:
        print("  Adaptive skip: disabled")
    print("  Press 'q' to quit.\n")

    frame_count = 0
    fps = 0.0
    last_fps_update = time.time()
    fps_counter = 0
    last_frame_time = time.time()
    fps_ema = 0.0

    inference_stride = 1
    max_skip = max(1, int(args.max_skip))
    cached_faces_predictions: List[Dict[str, object]] = []
    cached_overlays: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = _ensure_min_width(frame, min_width=640)

        if display_size is not None:
            frame = cv2.resize(frame, display_size)

        if args.equalize_low_light:
            frame = _equalize_low_light(frame)

        run_inference = (frame_count % max(1, inference_stride)) == 0
        if not args.adaptive_skip:
            run_inference = True

        if run_inference:
            (
                _,
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
                detection_interval=1,
                detector_size=detector_size,
                detection_confidence_threshold=float(args.detection_threshold),
            )
            cached_faces_predictions = faces_predictions
        else:
            faces_predictions = cached_faces_predictions

        annotated = frame.copy()

        if run_inference:
            seen_face_ids: Set[int] = set()
            live_overlays: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []

            for face in faces_predictions:
                if not isinstance(face, dict):
                    continue

                face_id_value = face.get("id", 0)
                if not isinstance(face_id_value, (int, float, str, np.integer)):
                    continue
                try:
                    face_id = int(face_id_value)
                except (TypeError, ValueError):
                    continue
                seen_face_ids.add(face_id)

                bbox_raw = face.get("bbox", (0, 0, 0, 0))
                if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                    continue
                try:
                    bbox = (
                        int(bbox_raw[0]),
                        int(bbox_raw[1]),
                        int(bbox_raw[2]),
                        int(bbox_raw[3]),
                    )
                except (TypeError, ValueError):
                    continue

                raw_probs = np.asarray(face.get("probabilities") or [], dtype=np.float32)
                smoothed_probs = temporal_smoother.update(face_id, raw_probs)

                emotion_index = int(np.argmax(smoothed_probs))
                emotion = EMOTIONS[emotion_index]
                confidence = float(smoothed_probs[emotion_index])

                label = _format_label(emotion, confidence)
                color = _confidence_color(confidence)
                _draw_face_overlay(annotated, bbox=bbox, label=label, color=color)
                live_overlays.append((bbox, label, color))

                if args.debug_boxes:
                    x, y, w, h = bbox
                    print(
                        f"Frame {frame_count + 1} | Face {face_id} | "
                        f"bbox=(x={x}, y={y}, w={w}, h={h})"
                    )

                if args.debug_predictions:
                    top3 = _top3_from_probs(smoothed_probs, EMOTIONS)
                    top3_text = ", ".join([f"{name}:{score * 100.0:.1f}%" for name, score in top3])
                    print(
                        f"Frame {frame_count + 1} | Face {face_id} | raw={raw_probs.tolist()} "
                        f"| selected={emotion} | confidence={confidence:.4f} | top3={top3_text}"
                    )

            temporal_smoother.mark_missing(seen_face_ids)
            cached_overlays = live_overlays
        else:
            for bbox, label, color in cached_overlays:
                _draw_face_overlay(annotated, bbox=bbox, label=label, color=color)

        if args.debug_crop:
            debug_crops: List[np.ndarray] = []
            frame_h, frame_w = frame.shape[:2]
            for face in faces_predictions:
                if not isinstance(face, dict):
                    continue

                bbox_raw = face.get("bbox", (0, 0, 0, 0))
                if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                    continue
                try:
                    bbox = (
                        int(bbox_raw[0]),
                        int(bbox_raw[1]),
                        int(bbox_raw[2]),
                        int(bbox_raw[3]),
                    )
                except (TypeError, ValueError):
                    continue

                x, y, w, h = [int(v) for v in bbox]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_w, x + max(0, w))
                y2 = min(frame_h, y + max(0, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                debug_crops.append(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            if debug_crops:
                preview = np.hstack(debug_crops[:4])
                cv2.imshow("Face Crop", preview)
            else:
                cv2.imshow("Face Crop", np.zeros((224, 224, 3), dtype=np.uint8))

        now = time.time()
        delta = max(now - last_frame_time, 1e-6)
        instant_fps = 1.0 / delta
        fps_ema = instant_fps if fps_ema <= 0.0 else (0.9 * fps_ema) + (0.1 * instant_fps)
        last_frame_time = now

        fps_counter += 1
        elapsed = now - last_fps_update
        if elapsed >= 0.5:
            fps = fps_counter / max(elapsed, 1e-6)
            fps_counter = 0
            last_fps_update = now

            if args.adaptive_skip:
                previous_stride = inference_stride
                if fps_ema < float(args.target_fps) and inference_stride < max_skip:
                    inference_stride += 1
                elif fps_ema > (float(args.target_fps) * 1.35) and inference_stride > 1:
                    inference_stride -= 1

                if previous_stride != inference_stride:
                    print(
                        f"Adaptive stride update: 1/{previous_stride} -> 1/{inference_stride} "
                        f"(ema_fps={fps_ema:.1f}, target={args.target_fps:.1f})"
                    )

        stats_text = (
            f"FPS: {fps:.1f} | Faces: {len(faces_predictions)} | Detector: {selected_backend} "
            f"| Infer: 1/{inference_stride}"
        )
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
    if args.debug_crop:
        cv2.destroyWindow("Face Crop")
    cv2.destroyAllWindows()
    print(f"\nStopped after {frame_count} frames.")


if __name__ == "__main__":
    main()
