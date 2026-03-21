import os
import importlib
from collections import deque
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

try:
    from config import BEST_MODEL_PATH, EMOTION_CLASSES, FACE_DETECTOR_BACKEND, HAAR_CASCADE_PATH
except ModuleNotFoundError:
    from backend.config import BEST_MODEL_PATH, EMOTION_CLASSES, FACE_DETECTOR_BACKEND, HAAR_CASCADE_PATH

MTCNN = None
try:
    _mtcnn_module = importlib.import_module("mtcnn")
    MTCNN = getattr(_mtcnn_module, "MTCNN", None)
except Exception:
    MTCNN = None

EMOTIONS = EMOTION_CLASSES
INPUT_SIZE = (48, 48)

EMOTION_COLORS = {
    "angry": (36, 28, 237),
    "disgust": (59, 180, 75),
    "fear": (0, 198, 255),
    "happy": (0, 223, 255),
    "neutral": (180, 180, 180),
    "sad": (255, 128, 0),
    "surprise": (220, 120, 255),
}


class EmotionPredictor:
    """Smooth emotion probabilities over recent frames to reduce jitter."""

    def __init__(
        self,
        smoothing_window: int = 2,
        confidence_threshold: float = 0.6,
        detection_interval: int = 2,
        window_size: Optional[int] = None,
    ):
        # Keep backward compatibility with previous `window_size` argument.
        if window_size is not None:
            smoothing_window = window_size

        self.smoothing_window = max(1, int(smoothing_window))
        self.confidence_threshold = float(confidence_threshold)
        self.detection_interval = max(1, int(detection_interval))

        self._history_by_face: Dict[int, deque[np.ndarray]] = {}
        self._frame_count = 0
        self._last_faces: List[Tuple[int, int, int, int]] = []
        self._stable_label_by_face: Dict[int, str] = {}
        self._stable_confidence_by_face: Dict[int, float] = {}
        self._label_history_by_face: Dict[int, deque[str]] = {}
        self._confidence_history_by_face: Dict[int, deque[float]] = {}

    def reset(self) -> None:
        self._history_by_face.clear()
        self._frame_count = 0
        self._last_faces = []
        self._stable_label_by_face.clear()
        self._stable_confidence_by_face.clear()
        self._label_history_by_face.clear()
        self._confidence_history_by_face.clear()

    def _vote_label(self, face_id: int, label: str, confidence: float) -> Tuple[str, float]:
        if face_id not in self._label_history_by_face:
            self._label_history_by_face[face_id] = deque(maxlen=self.smoothing_window)
            self._confidence_history_by_face[face_id] = deque(maxlen=self.smoothing_window)

        self._label_history_by_face[face_id].append(label)
        self._confidence_history_by_face[face_id].append(float(confidence))

        labels = list(self._label_history_by_face[face_id])
        confs = list(self._confidence_history_by_face[face_id])
        if not labels:
            return label, float(confidence)

        counts: Dict[str, int] = {}
        for item in labels:
            counts[item] = counts.get(item, 0) + 1

        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        winner_conf = [confs[i] for i, lbl in enumerate(labels) if lbl == winner]
        return winner, (sum(winner_conf) / max(1, len(winner_conf)))

    def should_detect(self) -> bool:
        run = (self._frame_count % self.detection_interval) == 0
        self._frame_count += 1
        return run

    def cache_faces(self, faces: List[Tuple[int, int, int, int]]) -> None:
        self._last_faces = faces

    def get_cached_faces(self) -> List[Tuple[int, int, int, int]]:
        return self._last_faces

    def smooth_for_face(self, face_id: int, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if face_id not in self._history_by_face:
            self._history_by_face[face_id] = deque(maxlen=self.smoothing_window)
        self._history_by_face[face_id].append(probs)
        stacked = np.stack(list(self._history_by_face[face_id]), axis=0)
        return np.mean(stacked, axis=0)

    def stable_label(self, face_id: int, probs: np.ndarray) -> Tuple[str, float, np.ndarray]:
        smoothed = self.smooth_for_face(face_id, probs)
        idx = int(np.argmax(smoothed))
        confidence = float(smoothed[idx])
        predicted = EMOTIONS[idx]

        voted_label, voted_confidence = self._vote_label(face_id, predicted, confidence)

        # Only update displayed emotion when confidence is high enough.
        if voted_confidence >= self.confidence_threshold:
            self._stable_label_by_face[face_id] = voted_label
            self._stable_confidence_by_face[face_id] = voted_confidence

        stable_label = self._stable_label_by_face.get(face_id, "No face")
        stable_conf = self._stable_confidence_by_face.get(face_id, 0.0)

        if stable_label == "No face":
            return "uncertain", confidence, smoothed

        return stable_label, stable_conf, smoothed


def find_latest_model(search_root: Optional[str] = None) -> str:
    """Find the most recent .h5 model, preferring best_model.h5 when available."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = search_root or base

    preferred = os.path.join(root, "artifacts", "best_model.h5")
    if os.path.exists(preferred):
        return preferred

    candidates = glob(os.path.join(root, "*.h5")) + glob(os.path.join(root, "artifacts", "*.h5"))
    if not candidates:
        raise FileNotFoundError("No .h5 model found. Train model first with train_model.py")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_model_safe(model_path: Optional[str] = None):
    """Load Keras model with safe fallback path resolution."""
    path = model_path or find_latest_model()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    # Compatibility path for some H5 files saved with extra Dense config keys
    # (e.g. quantization_config) that older/newer Keras versions may reject.
    try:
        return load_model(path, compile=False)
    except (TypeError, ValueError) as exc:
        if "quantization_config" not in str(exc):
            raise

        class DenseCompat(Dense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        return load_model(path, compile=False, custom_objects={"Dense": DenseCompat})


def init_face_detector(cascade_path: Optional[str] = None):
    """Initialize MTCNN detector with Haar fallback."""
    if FACE_DETECTOR_BACKEND.lower() == "mtcnn" and MTCNN is not None:
        return {"backend": "mtcnn", "detector": MTCNN()}

    path = cascade_path or HAAR_CASCADE_PATH
    # Try OpenCV default cascade name first, then fallback to configured absolute path.
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if haar.empty():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Haar cascade not found: {path}")
        haar = cv2.CascadeClassifier(path)
    if haar.empty():
        raise RuntimeError("Could not initialize Haar Cascade detector.")
    return {"backend": "haar", "detector": haar}


def _rotate_by_eyes(face_bgr: np.ndarray, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> np.ndarray:
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = face_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(face_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _extract_faces(detector_bundle, frame_bgr: np.ndarray, gray_small: np.ndarray):
    backend = detector_bundle.get("backend")
    detector = detector_bundle.get("detector")

    if backend == "mtcnn":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        faces = []
        for r in results:
            x, y, w, h = r.get("box", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            faces.append(
                {
                    "bbox": (int(max(0, x)), int(max(0, y)), int(w), int(h)),
                    "keypoints": r.get("keypoints", {}),
                }
            )
        return faces

    faces_small = detector.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )
    return [{"bbox": (int(x), int(y), int(w), int(h)), "keypoints": {}} for (x, y, w, h) in faces_small]


def preprocess_face(face_bgr: np.ndarray, input_size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """Exact model preprocessing pipeline for CNN-trained grayscale inputs."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, input_size)
    gray = cv2.equalizeHist(gray)
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    return gray


def preprocess_face_transfer(face_bgr: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Transfer-model preprocessing: equalized luminance as normalized RGB."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, input_size)
    gray = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)


def infer_model_profile(model) -> Tuple[Tuple[int, int], str]:
    """Infer target input size and mode from model signature."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) != 4:
        return INPUT_SIZE, "cnn"
    h, w, c = shape[1], shape[2], shape[3]
    h = int(h) if h is not None else INPUT_SIZE[0]
    w = int(w) if w is not None else INPUT_SIZE[1]
    mode = "transfer" if int(c or 1) == 3 else "cnn"
    return (h, w), mode


def _predict_face_probs(model, face_bgr: np.ndarray) -> np.ndarray:
    input_size, model_mode = infer_model_profile(model)
    if model_mode == "transfer":
        x = preprocess_face_transfer(face_bgr, input_size=input_size)
    else:
        x = preprocess_face(face_bgr, input_size=input_size)

    probs = model.predict(x, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 1 or probs.shape[0] != len(EMOTIONS):
        raise ValueError("Unexpected model output shape for emotion prediction.")
    return probs


def predict_frame(
    frame: np.ndarray,
    model=None,
    detector=None,
    predictor: Optional[EmotionPredictor] = None,
    confidence_threshold: float = 0.6,
    detection_interval: int = 2,
    detector_size: Optional[Tuple[int, int]] = (640, 480),
    max_detection_width: int = 800,
):
    """Predict emotions for faces in a BGR frame and draw annotations.

    Returns:
        annotated_frame, emotion_label, confidence, probability_vector, boxes, faces_predictions, faces_results
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame.")

    model = model or load_model_safe(BEST_MODEL_PATH)
    detector_bundle = detector or init_face_detector()

    annotated = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Detection image prep:
    # - use fixed detector_size for realtime if passed
    # - otherwise resize large images while preserving aspect ratio
    if detector_size is not None:
        detect_w, detect_h = detector_size
        resized = cv2.resize(frame, (detect_w, detect_h))
    else:
        if frame_w > max_detection_width:
            detect_w = max_detection_width
            detect_h = int(frame_h * (max_detection_width / float(frame_w)))
            resized = cv2.resize(frame, (detect_w, detect_h))
        else:
            detect_w, detect_h = frame_w, frame_h
            resized = frame

    gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.equalizeHist(gray_small)

    run_detection = True
    if predictor is not None:
        predictor.detection_interval = max(1, int(detection_interval))
        run_detection = predictor.should_detect()

    faces: List[Tuple[int, int, int, int]] = []
    keypoints_by_box: Dict[Tuple[int, int, int, int], Dict] = {}
    if run_detection or predictor is None:
        detections = _extract_faces(detector_bundle, resized, gray_small)

        sx = frame_w / float(detect_w)
        sy = frame_h / float(detect_h)
        for det in detections:
            x, y, w, h = det["bbox"]
            box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
            faces.append(box)

            scaled_keypoints = {}
            for name, point in det.get("keypoints", {}).items():
                px, py = point
                scaled_keypoints[name] = (int(px * sx), int(py * sy))
            keypoints_by_box[box] = scaled_keypoints

        if predictor is not None:
            predictor.cache_faces(faces)
    elif predictor is not None:
        faces = predictor.get_cached_faces()

    if len(faces) == 0:
        return annotated, "No face", 0.0, np.zeros(len(EMOTIONS), dtype=np.float32), [], [], []

    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    primary_probs: Optional[np.ndarray] = None
    primary_label = "No face"
    primary_confidence = 0.0
    boxes: List[Dict[str, int]] = []
    faces_predictions: List[Dict] = []

    for i, (x, y, w, h) in enumerate(faces_sorted):
        face_id = i + 1
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue

        face_keypoints = keypoints_by_box.get((x, y, w, h), {})
        if "left_eye" in face_keypoints and "right_eye" in face_keypoints:
            left_eye_global = face_keypoints["left_eye"]
            right_eye_global = face_keypoints["right_eye"]
            left_eye = (max(0, left_eye_global[0] - x), max(0, left_eye_global[1] - y))
            right_eye = (max(0, right_eye_global[0] - x), max(0, right_eye_global[1] - y))
            face_roi = _rotate_by_eyes(face_roi, left_eye=left_eye, right_eye=right_eye)

        probs = _predict_face_probs(model, face_roi)
        if predictor is not None:
            predictor.confidence_threshold = confidence_threshold
            display_label, conf, probs = predictor.stable_label(face_id, probs)
        else:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            display_label = EMOTIONS[idx] if conf >= confidence_threshold else "uncertain"

        color_key = display_label if display_label in EMOTION_COLORS else "neutral"
        color = EMOTION_COLORS.get(color_key, (120, 120, 120))
        text = f"Face {face_id} - {display_label.capitalize()} ({conf * 100:.0f}%)"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        faces_predictions.append(
            {
                "id": face_id,
                "bbox": (int(x), int(y), int(w), int(h)),
                "emotion": display_label.capitalize(),
                "confidence": float(conf),
                "probabilities": probs.astype(float).tolist(),
            }
        )

        if i == 0:
            primary_probs = probs
            primary_label = display_label
            primary_confidence = conf

    if primary_probs is None:
        primary_probs = np.zeros(len(EMOTIONS), dtype=np.float32)

    faces_results = [
        {
            "id": face["id"],
            "bbox": face["bbox"],
            "emotion": face["emotion"],
            "confidence": face["confidence"],
            "probabilities": face["probabilities"],
        }
        for face in faces_predictions
    ]

    return annotated, primary_label, primary_confidence, primary_probs, boxes, faces_predictions, faces_results
