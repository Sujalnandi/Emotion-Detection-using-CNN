import logging
import os
from collections import deque
from glob import glob
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

try:
    from config import (
        BEST_MODEL_PATH,
        DETECTION_CONFIDENCE_THRESHOLD,
        EMOTION_CLASSES,
    )
except ModuleNotFoundError:
    from backend.config import (
        BEST_MODEL_PATH,
        DETECTION_CONFIDENCE_THRESHOLD,
        EMOTION_CLASSES,
    )

MTCNN = None
try:
    from mtcnn import MTCNN
except ImportError:
    MTCNN = None

_GLOBAL_MTCNN_DETECTOR = None

logger = logging.getLogger("backend.inference_engine")

EMOTIONS = EMOTION_CLASSES
INPUT_SIZE = (48, 48)
FACE_CROP_MARGIN = 0.2

EMOTION_COLORS = {
    "angry": (36, 28, 237),
    "disgust": (59, 180, 75),
    "fear": (0, 198, 255),
    "happy": (0, 223, 255),
    "neutral": (180, 180, 180),
    "sad": (255, 128, 0),
    "surprise": (220, 120, 255),
    "uncertain": (140, 140, 140),
}

CONFIDENCE_GUIDANCE_THRESHOLD = 0.4
MEDIUM_CONFIDENCE_THRESHOLD = 0.7

CONFIDENCE_COLORS = {
    "high": (34, 197, 94),
    "medium": (0, 196, 255),
    "low": (45, 45, 220),
}

_PREPROCESS_RANGE_WARNED = False
_SOFTMAX_WARNED = False
_FLAT_PROBS_WARNED = False


def _confidence_level(confidence: float) -> str:
    if confidence < CONFIDENCE_GUIDANCE_THRESHOLD:
        return "low"
    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def _format_label(emotion: str, confidence: float) -> str:
    name = str(emotion).capitalize()
    if confidence < CONFIDENCE_GUIDANCE_THRESHOLD:
        return f"{name} (low confidence {confidence * 100.0:.0f}%)"
    return f"{name} ({confidence * 100.0:.0f}%)"


def _top_k_predictions(probabilities: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        return []

    top_k = min(int(k), int(probs.size))
    idxs = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "emotion": EMOTIONS[int(idx)],
            "confidence": float(probs[int(idx)]),
        }
        for idx in idxs
    ]


def _is_softmax_like(probabilities: np.ndarray) -> bool:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        return False
    if np.any(probs < 0.0):
        return False
    if np.any(probs > 1.0 + 1e-3):
        return False
    return abs(float(np.sum(probs)) - 1.0) <= 1e-2


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs.size != len(EMOTIONS):
        raise ValueError("Unexpected model output size for emotion prediction.")

    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if _is_softmax_like(probs):
        total = float(np.sum(probs))
        if total > 1e-12:
            return (probs / total).astype(np.float32)

    # Fall back to stable softmax for logits or malformed distributions.
    shifted = probs - np.max(probs)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom <= 1e-12:
        return np.full((len(EMOTIONS),), 1.0 / float(len(EMOTIONS)), dtype=np.float32)
    return (exps / denom).astype(np.float32)


def _model_has_softmax_output(model: Any) -> bool:
    if not getattr(model, "layers", None):
        return False
    activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", "")
    return activation == "softmax"


def _log_face_debug(face_id: int, raw_probs: np.ndarray, normalized_probs: np.ndarray) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    pred_idx = int(np.argmax(normalized_probs))
    pred_emotion = EMOTIONS[pred_idx]
    pred_conf = float(normalized_probs[pred_idx])
    logger.debug(
        "face=%s raw_probs=%s normalized_probs=%s selected_class=%s confidence=%.4f",
        face_id,
        np.array2string(np.asarray(raw_probs, dtype=np.float32), precision=4),
        np.array2string(np.asarray(normalized_probs, dtype=np.float32), precision=4),
        pred_emotion,
        pred_conf,
    )


class EmotionPredictor:
    """Face-aware temporal smoothing and ID tracking for stable realtime predictions."""

    def __init__(
        self,
        smoothing_window: int = 12,
        confidence_threshold: float = 0.6,
        detection_interval: int = 2,
        window_size: Optional[int] = None,
    ):
        if window_size is not None:
            smoothing_window = window_size

        self.smoothing_window = max(1, int(smoothing_window))
        self.confidence_threshold = float(confidence_threshold)
        self.detection_interval = max(1, int(detection_interval))

        self._ema_alpha = 0.4
        self._switch_margin = 0.12
        self._switch_vote_ratio = 0.6
        self._track_iou_threshold = 0.35
        self._max_missed_frames = max(3, self.smoothing_window)

        self._history_by_face: Dict[int, Deque[np.ndarray]] = {}
        self._ema_probs_by_face: Dict[int, np.ndarray] = {}
        self._stable_label_by_face: Dict[int, str] = {}
        self._stable_confidence_by_face: Dict[int, float] = {}
        self._label_history_by_face: Dict[int, Deque[str]] = {}
        self._confidence_history_by_face: Dict[int, Deque[float]] = {}

        self._frame_count = 0
        self._last_faces: List[Dict[str, Any]] = []
        self._track_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self._track_missed: Dict[int, int] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._history_by_face.clear()
        self._ema_probs_by_face.clear()
        self._stable_label_by_face.clear()
        self._stable_confidence_by_face.clear()
        self._label_history_by_face.clear()
        self._confidence_history_by_face.clear()
        self._frame_count = 0
        self._last_faces = []
        self._track_boxes.clear()
        self._track_missed.clear()
        self._next_track_id = 1

    def should_detect(self) -> bool:
        run = (self._frame_count % self.detection_interval) == 0
        self._frame_count += 1
        return run

    def cache_faces(self, faces: List[Dict[str, Any]]) -> None:
        self._last_faces = [dict(face) for face in faces]

    def get_cached_faces(self) -> List[Dict[str, Any]]:
        return [dict(face) for face in self._last_faces]

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        a_x2, a_y2 = ax + aw, ay + ah
        b_x2, b_y2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(1, aw * ah)
        area_b = max(1, bw * bh)
        union = area_a + area_b - inter_area
        return float(inter_area / max(union, 1))

    def _clear_face_state(self, face_id: int) -> None:
        self._history_by_face.pop(face_id, None)
        self._ema_probs_by_face.pop(face_id, None)
        self._stable_label_by_face.pop(face_id, None)
        self._stable_confidence_by_face.pop(face_id, None)
        self._label_history_by_face.pop(face_id, None)
        self._confidence_history_by_face.pop(face_id, None)
        self._track_boxes.pop(face_id, None)
        self._track_missed.pop(face_id, None)

    def _prune_missing_tracks(self) -> None:
        stale = [
            track_id
            for track_id, missed in self._track_missed.items()
            if missed > self._max_missed_frames
        ]
        for track_id in stale:
            self._clear_face_state(track_id)

    def assign_track_ids(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        if not boxes:
            for track_id in list(self._track_missed.keys()):
                self._track_missed[track_id] = self._track_missed.get(track_id, 0) + 1
            self._prune_missing_tracks()
            return []

        assignments = [-1] * len(boxes)
        track_ids = list(self._track_boxes.keys())

        if track_ids:
            candidates: List[Tuple[float, int, int]] = []
            for box_idx, box in enumerate(boxes):
                for track_id in track_ids:
                    iou = self._bbox_iou(box, self._track_boxes[track_id])
                    if iou >= self._track_iou_threshold:
                        candidates.append((iou, box_idx, track_id))

            candidates.sort(key=lambda item: item[0], reverse=True)
            used_boxes = set()
            used_tracks = set()
            for _, box_idx, track_id in candidates:
                if box_idx in used_boxes or track_id in used_tracks:
                    continue
                assignments[box_idx] = track_id
                used_boxes.add(box_idx)
                used_tracks.add(track_id)

            for track_id in track_ids:
                if track_id not in used_tracks:
                    self._track_missed[track_id] = self._track_missed.get(track_id, 0) + 1

        for box_idx, track_id in enumerate(assignments):
            if track_id == -1:
                track_id = self._next_track_id
                self._next_track_id += 1
                assignments[box_idx] = track_id

            self._track_boxes[track_id] = boxes[box_idx]
            self._track_missed[track_id] = 0

        self._prune_missing_tracks()
        return assignments

    def _append_history(self, face_id: int, label: str, confidence: float) -> None:
        if face_id not in self._label_history_by_face:
            self._label_history_by_face[face_id] = deque(maxlen=self.smoothing_window)
            self._confidence_history_by_face[face_id] = deque(maxlen=self.smoothing_window)

        self._label_history_by_face[face_id].append(label)
        self._confidence_history_by_face[face_id].append(float(confidence))

    def smooth_for_face(self, face_id: int, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if face_id not in self._history_by_face:
            self._history_by_face[face_id] = deque(maxlen=self.smoothing_window)
        self._history_by_face[face_id].append(probs)

        mean_probs = np.mean(np.stack(list(self._history_by_face[face_id]), axis=0), axis=0)
        prev_ema = self._ema_probs_by_face.get(face_id, mean_probs)
        ema_probs = (self._ema_alpha * mean_probs) + ((1.0 - self._ema_alpha) * prev_ema)
        self._ema_probs_by_face[face_id] = ema_probs.astype(np.float32)
        return self._ema_probs_by_face[face_id]

    def stable_label(self, face_id: int, probs: np.ndarray) -> Tuple[str, float, np.ndarray]:
        smoothed = self.smooth_for_face(face_id, probs)
        candidate_idx = int(np.argmax(smoothed))
        candidate_label = EMOTIONS[candidate_idx]
        candidate_confidence = float(smoothed[candidate_idx])

        self._append_history(face_id, candidate_label, candidate_confidence)

        labels = list(self._label_history_by_face.get(face_id, []))
        confidences = list(self._confidence_history_by_face.get(face_id, []))
        vote_counts: Dict[str, int] = {}
        for label in labels:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        voted_label = max(vote_counts.items(), key=lambda item: item[1])[0] if vote_counts else candidate_label
        voted_ratio = vote_counts.get(voted_label, 1) / max(1, len(labels))
        voted_confidences = [
            confidences[idx]
            for idx, label in enumerate(labels)
            if label == voted_label and idx < len(confidences)
        ]
        voted_confidence = (
            float(sum(voted_confidences) / max(1, len(voted_confidences)))
            if voted_confidences
            else candidate_confidence
        )

        current_label = self._stable_label_by_face.get(face_id)
        current_conf = self._stable_confidence_by_face.get(face_id, 0.0)

        if current_label is None:
            if voted_confidence >= self.confidence_threshold:
                self._stable_label_by_face[face_id] = voted_label
                self._stable_confidence_by_face[face_id] = voted_confidence
                return voted_label, voted_confidence, smoothed
            return "uncertain", voted_confidence, smoothed

        if voted_label != current_label:
            current_idx = EMOTIONS.index(current_label) if current_label in EMOTIONS else -1
            current_score = float(smoothed[current_idx]) if current_idx >= 0 else 0.0
            margin = voted_confidence - current_score

            should_switch = (
                voted_confidence >= self.confidence_threshold
                and voted_ratio >= self._switch_vote_ratio
                and margin >= self._switch_margin
            )
            if should_switch:
                self._stable_label_by_face[face_id] = voted_label
                self._stable_confidence_by_face[face_id] = voted_confidence
                return voted_label, voted_confidence, smoothed

            smoothed_conf = (0.7 * current_conf) + (0.3 * max(current_conf, current_score))
            self._stable_confidence_by_face[face_id] = float(smoothed_conf)
            return current_label, float(smoothed_conf), smoothed

        updated_conf = (0.6 * current_conf) + (0.4 * voted_confidence) if current_conf > 0 else voted_confidence
        self._stable_confidence_by_face[face_id] = float(updated_conf)
        return current_label, float(updated_conf), smoothed


def find_latest_model(search_root: Optional[str] = None) -> str:
    """Find the most recent saved model, preferring .keras checkpoints."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = search_root or base

    preferred = os.path.join(root, "artifacts", "best_model.keras")
    if os.path.exists(preferred):
        return preferred

    candidates = (
        glob(os.path.join(root, "*.keras"))
        + glob(os.path.join(root, "artifacts", "*.keras"))
        + glob(os.path.join(root, "*.h5"))
        + glob(os.path.join(root, "artifacts", "*.h5"))
    )
    if not candidates:
        raise FileNotFoundError("No saved model found (.keras or .h5). Train model first with train_model.py")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_model_safe(model_path: Optional[str] = None):
    """Load Keras model with safe fallback path resolution."""
    path = model_path or find_latest_model()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    logger.info("Loading model from %s", path)
    try:
        model = load_model(path, compile=False)
        logger.info("Model loaded successfully")
        return model
    except (TypeError, ValueError) as exc:
        if "quantization_config" not in str(exc):
            logger.exception("Model load failed")
            raise

        class DenseCompat(Dense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        logger.warning("Applying DenseCompat fallback while loading model")
        return load_model(path, compile=False, custom_objects={"Dense": DenseCompat})


def init_face_detector(cascade_path: Optional[str] = None, backend_preference: Optional[str] = None):
    """Initialize a global MTCNN detector and reuse it across requests/frames."""
    del cascade_path, backend_preference  # Kept for backward-compatible call signatures.

    global _GLOBAL_MTCNN_DETECTOR
    if _GLOBAL_MTCNN_DETECTOR is None:
        if MTCNN is None:
            raise RuntimeError(
                "MTCNN is not available. Install it with: pip install mtcnn"
            )
        logger.info("Initializing global MTCNN detector backend")
        _GLOBAL_MTCNN_DETECTOR = MTCNN()

    return {"backend": "mtcnn", "detector": _GLOBAL_MTCNN_DETECTOR}


def _rotate_by_eyes(face_bgr: np.ndarray, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> np.ndarray:
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = face_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(face_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _extract_faces(detector_bundle, frame_bgr: np.ndarray):
    detector = detector_bundle.get("detector")
    if detector is None:
        return []

    frame_h, frame_w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = detector.detect_faces(rgb)
    except Exception as exc:
        logger.warning("MTCNN detection failed on current frame: %s", exc)
        return []

    faces = []
    for item in results or []:
        if not isinstance(item, dict):
            continue

        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue

        x, y, w, h = [int(round(float(v))) for v in box[:4]]
        if w == 0 or h == 0:
            continue

        # MTCNN returns [x, y, w, h], and x/y may be negative at frame borders.
        if w < 0:
            x = x + w
            w = abs(w)
        if h < 0:
            y = y + h
            h = abs(h)

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_w, int(x + w))
        y2 = min(frame_h, int(y + h))

        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 1 or clipped_h <= 1:
            continue

        keypoints_raw = item.get("keypoints", {})
        keypoints = {}
        if isinstance(keypoints_raw, dict):
            for name in ("left_eye", "right_eye"):
                point = keypoints_raw.get(name)
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    keypoints[name] = (int(point[0]), int(point[1]))

        faces.append(
            {
                "bbox": (x1, y1, clipped_w, clipped_h),
                "keypoints": keypoints,
                "score": float(item.get("confidence", 0.0)),
            }
        )
    return faces


def _sanitize_bbox(
    box: Tuple[int, int, int, int], frame_w: int, frame_h: int, margin: float = FACE_CROP_MARGIN
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = box
    if w <= 1 or h <= 1:
        return None

    # Expand around the detector bbox so prediction sees context, not a tight crop.
    x = max(0, int(x))
    y = max(0, int(y))
    x1 = max(0, int(x - (w * float(margin))))
    y1 = max(0, int(y - (h * float(margin))))
    x2 = min(frame_w, int(x + (w * (1.0 + float(margin)))))
    y2 = min(frame_h, int(y + (h * (1.0 + float(margin)))))

    cw = x2 - x1
    ch = y2 - y1
    if cw <= 1 or ch <= 1:
        return None
    return x1, y1, cw, ch


def preprocess_face(face_bgr: np.ndarray, input_size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """Preprocess one face for grayscale CNN models."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, input_size)
    gray = cv2.equalizeHist(gray)
    gray = gray.astype("float32") / 255.0
    return np.expand_dims(gray, axis=-1)


def preprocess_face_transfer(face_bgr: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess one face for transfer models (BGR->RGB, resize, EfficientNet normalize)."""
    if face_bgr.ndim == 2:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2RGB)
    elif face_bgr.ndim == 3 and face_bgr.shape[-1] == 1:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    target_h, target_w = int(input_size[0]), int(input_size[1])
    if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
        interp = cv2.INTER_AREA if (rgb.shape[0] > target_h or rgb.shape[1] > target_w) else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=interp)

    rgb = rgb.astype("float32")
    return efficientnet_preprocess_input(rgb)


def _crop_face_roi(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None

    face = frame_bgr[y1:y2, x1:x2]
    if face is None or face.size == 0:
        return None
    return face


def infer_model_profile(model) -> Tuple[Tuple[int, int], str]:
    """Infer model input size and preprocessing mode from model signature."""
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


def _predict_faces_probs_batch(
    model, face_rois: List[np.ndarray], input_size: Tuple[int, int], model_mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    if not face_rois:
        empty = np.zeros((0, len(EMOTIONS)), dtype=np.float32)
        return empty, empty

    prepared = []
    for face_bgr in face_rois:
        if model_mode == "transfer":
            prepared.append(preprocess_face_transfer(face_bgr, input_size=input_size))
        else:
            prepared.append(preprocess_face(face_bgr, input_size=input_size))

    batch = np.stack(prepared, axis=0).astype(np.float32)
    if model_mode == "transfer":
        global _PREPROCESS_RANGE_WARNED
        bmin = float(np.min(batch))
        bmax = float(np.max(batch))
        looks_like_255_range = bmin >= -1e-3 and bmax <= 255.5
        looks_like_minus1_to1 = bmin >= -1.3 and bmax <= 1.3
        if not _PREPROCESS_RANGE_WARNED and not (looks_like_255_range or looks_like_minus1_to1):
            logger.warning(
                "Transfer preprocessing range looks unexpected (min=%.3f max=%.3f). "
                "Expected either [0,255] or [-1,1] depending on preprocessing path.",
                bmin,
                bmax,
            )
            _PREPROCESS_RANGE_WARNED = True

    raw_probs = model.predict(batch, verbose=0)
    if isinstance(raw_probs, list):
        raw_probs = raw_probs[0]

    raw_probs = np.asarray(raw_probs, dtype=np.float32)
    if raw_probs.ndim == 3 and raw_probs.shape[1] == 1:
        raw_probs = raw_probs[:, 0, :]
    if raw_probs.ndim != 2 or raw_probs.shape[1] != len(EMOTIONS):
        raise ValueError("Unexpected model output shape for emotion prediction.")

    global _SOFTMAX_WARNED
    if not _SOFTMAX_WARNED and not _model_has_softmax_output(model):
        logger.warning(
            "Model final activation is not softmax. Applying probability normalization on outputs."
        )
        _SOFTMAX_WARNED = True

    normalized = np.zeros_like(raw_probs, dtype=np.float32)
    for row_idx in range(raw_probs.shape[0]):
        normalized[row_idx] = _normalize_probabilities(raw_probs[row_idx])

    global _FLAT_PROBS_WARNED
    if not _FLAT_PROBS_WARNED:
        std_per_face = np.std(normalized, axis=1)
        if np.any(std_per_face < 0.02):
            logger.warning(
                "Flat probability distribution detected (std<0.02). "
                "This may indicate low signal quality or preprocessing mismatch."
            )
            _FLAT_PROBS_WARNED = True

    return raw_probs, normalized


def predict_frame(
    frame: np.ndarray,
    model: Any = None,
    detector: Optional[Dict[str, Any]] = None,
    predictor: Optional[EmotionPredictor] = None,
    confidence_threshold: float = 0.6,
    detection_interval: int = 1,
    detector_size: Optional[Tuple[int, int]] = (640, 480),
    min_detection_width: int = 640,
    max_detection_width: int = 800,
    detection_confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, str, float, np.ndarray, List[Dict[str, int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Predict emotions for all faces in a BGR frame and draw stable annotations."""
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame.")

    model = model or load_model_safe(BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else None)
    detector_bundle = detector or init_face_detector()

    annotated = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    if detector_size is not None:
        detect_w, detect_h = detector_size
        resized = cv2.resize(frame, (detect_w, detect_h))
    else:
        detect_scale = 1.0
        if frame_w < int(min_detection_width):
            detect_scale = float(min_detection_width) / float(max(frame_w, 1))
        elif frame_w > int(max_detection_width):
            detect_scale = float(max_detection_width) / float(max(frame_w, 1))

        if abs(detect_scale - 1.0) > 1e-6:
            detect_w = max(160, int(frame_w * detect_scale))
            detect_h = max(120, int(frame_h * detect_scale))
            resized = cv2.resize(frame, (detect_w, detect_h))
        else:
            detect_w, detect_h = frame_w, frame_h
            resized = frame

    run_detection = True
    if predictor is not None:
        predictor.detection_interval = max(1, int(detection_interval))
        run_detection = predictor.should_detect()

    tracked_faces: List[Dict[str, Any]] = []
    if run_detection or predictor is None:
        detections = _extract_faces(detector_bundle, resized)
        sx = frame_w / float(detect_w)
        sy = frame_h / float(detect_h)

        face_candidates: List[Dict[str, Any]] = []
        for det in detections:
            score = float(det.get("score", 1.0))
            if score < float(detection_confidence_threshold):
                continue

            x, y, w, h = det.get("bbox", (0, 0, 0, 0))
            scaled_box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
            fixed_box = _sanitize_bbox(scaled_box, frame_w=frame_w, frame_h=frame_h)
            if fixed_box is None:
                continue

            scaled_keypoints = {}
            for name, point in det.get("keypoints", {}).items():
                if not isinstance(point, (tuple, list)) or len(point) < 2:
                    continue
                px, py = point
                scaled_keypoints[name] = (int(px * sx), int(py * sy))

            face_candidates.append(
                {
                    "bbox": fixed_box,
                    "score": score,
                    "keypoints": scaled_keypoints,
                }
            )

        if predictor is not None:
            track_ids = predictor.assign_track_ids([face["bbox"] for face in face_candidates])
            for face, track_id in zip(face_candidates, track_ids):
                tracked_faces.append(
                    {
                        "id": int(track_id),
                        "bbox": face["bbox"],
                        "score": float(face.get("score", 1.0)),
                        "keypoints": face.get("keypoints", {}),
                    }
                )
            predictor.cache_faces(tracked_faces)
        else:
            for idx, face in enumerate(face_candidates, start=1):
                tracked_faces.append(
                    {
                        "id": int(idx),
                        "bbox": face["bbox"],
                        "score": float(face.get("score", 1.0)),
                        "keypoints": face.get("keypoints", {}),
                    }
                )
    elif predictor is not None:
        tracked_faces = predictor.get_cached_faces()

    if len(tracked_faces) == 0:
        return annotated, "No face", 0.0, np.zeros(len(EMOTIONS), dtype=np.float32), [], [], []

    faces_sorted = sorted(
        tracked_faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )

    model_input_size, model_mode = infer_model_profile(model)
    face_rois: List[np.ndarray] = []
    prepared_faces: List[Dict[str, Any]] = []

    for face in faces_sorted:
        x, y, w, h = face["bbox"]
        face_roi = _crop_face_roi(frame, face["bbox"])
        if face_roi is None:
            continue

        face_keypoints = face.get("keypoints", {})
        if "left_eye" in face_keypoints and "right_eye" in face_keypoints:
            left_eye_global = face_keypoints["left_eye"]
            right_eye_global = face_keypoints["right_eye"]
            left_eye = (max(0, left_eye_global[0] - x), max(0, left_eye_global[1] - y))
            right_eye = (max(0, right_eye_global[0] - x), max(0, right_eye_global[1] - y))
            face_roi = _rotate_by_eyes(face_roi, left_eye=left_eye, right_eye=right_eye)

        face_rois.append(face_roi)
        prepared_faces.append(face)

    if not face_rois:
        return annotated, "No face", 0.0, np.zeros(len(EMOTIONS), dtype=np.float32), [], [], []

    raw_batch_probs, batch_probs = _predict_faces_probs_batch(
        model=model,
        face_rois=face_rois,
        input_size=model_input_size,
        model_mode=model_mode,
    )

    primary_probs: Optional[np.ndarray] = None
    primary_label = "No face"
    primary_confidence = 0.0
    boxes: List[Dict[str, int]] = []
    faces_predictions: List[Dict[str, Any]] = []

    for idx, (face, raw_probs, probs) in enumerate(zip(prepared_faces, raw_batch_probs, batch_probs)):
        face_id = int(face["id"])
        x, y, w, h = face["bbox"]
        detection_score = float(face.get("score", 1.0))

        if predictor is not None:
            predictor.confidence_threshold = confidence_threshold
            used_probs = predictor.smooth_for_face(face_id, np.asarray(probs, dtype=np.float32))
        else:
            used_probs = np.asarray(probs, dtype=np.float32)

        used_probs = _normalize_probabilities(used_probs)
        pred_idx = int(np.argmax(used_probs))
        emotion = EMOTIONS[pred_idx]
        conf = float(used_probs[pred_idx])
        top3 = _top_k_predictions(used_probs, k=3)
        confidence_level = _confidence_level(conf)
        label_text = f"#{face_id} {_format_label(emotion, conf)}"
        color = CONFIDENCE_COLORS.get(confidence_level, (120, 120, 120))

        _log_face_debug(face_id=face_id, raw_probs=raw_probs, normalized_probs=used_probs)

        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)

        text_y = y - 8
        box_top = max(0, text_y - th - baseline - 4)
        box_bottom = max(th + baseline + 4, text_y + baseline)
        box_right = min(frame_w - 1, x + tw + 10)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(annotated, (x, box_top), (box_right, box_bottom), color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x + 5, max(th + 2, box_bottom - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        faces_predictions.append(
            {
                "id": face_id,
                "bbox": (int(x), int(y), int(w), int(h)),
                "emotion": str(emotion).lower(),
                "display_label": _format_label(emotion, conf),
                "confidence": float(conf),
                "confidence_percent": float(conf * 100.0),
                "confidence_level": confidence_level,
                "low_confidence": conf < CONFIDENCE_GUIDANCE_THRESHOLD,
                "detection_confidence": detection_score,
                "probabilities": used_probs.astype(float).tolist(),
                "probabilityMap": {
                    emotion: float(used_probs[i])
                    for i, emotion in enumerate(EMOTIONS)
                },
                "top3": top3,
            }
        )

        if idx == 0:
            primary_probs = used_probs
            primary_label = emotion
            primary_confidence = conf

    if primary_probs is None:
        primary_probs = np.zeros(len(EMOTIONS), dtype=np.float32)

    faces_results = [
        {
            "id": face["id"],
            "bbox": face["bbox"],
            "emotion": face["emotion"],
            "display_label": face.get("display_label"),
            "confidence": face["confidence"],
            "confidence_percent": face.get("confidence_percent", float(face["confidence"]) * 100.0),
            "confidence_level": face.get("confidence_level", _confidence_level(float(face["confidence"]))),
            "low_confidence": face.get("low_confidence", float(face["confidence"]) < CONFIDENCE_GUIDANCE_THRESHOLD),
            "detection_confidence": face.get("detection_confidence", 1.0),
            "probabilities": face["probabilities"],
            "top3": face.get("top3", _top_k_predictions(np.asarray(face["probabilities"], dtype=np.float32), k=3)),
        }
        for face in faces_predictions
    ]

    return annotated, primary_label, primary_confidence, primary_probs, boxes, faces_predictions, faces_results
