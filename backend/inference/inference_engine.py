import logging
import os
from collections import deque
from glob import glob
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

try:
    from backend.preprocessing.preprocess import preprocess_rgb_for_transfer
except ModuleNotFoundError:
    from preprocessing.preprocess import preprocess_rgb_for_transfer

try:
    from config import (
        BEST_MODEL_PATH,
        DETECTION_CONFIDENCE_THRESHOLD,
        EMOTION_CLASSES,
        FACE_DETECTOR_BACKEND,
        HAAR_CASCADE_PATH,
    )
except ModuleNotFoundError:
    from backend.config import (
        BEST_MODEL_PATH,
        DETECTION_CONFIDENCE_THRESHOLD,
        EMOTION_CLASSES,
        FACE_DETECTOR_BACKEND,
        HAAR_CASCADE_PATH,
    )

RETINAFACE = None
try:
    from retinaface import RetinaFace

    RETINAFACE = RetinaFace
except ImportError:
    RETINAFACE = None

MTCNN = None
try:
    from mtcnn import MTCNN
except ImportError:
    MTCNN = None

logger = logging.getLogger("backend.inference_engine")

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
    "uncertain": (140, 140, 140),
}


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


def _detector_priority(preference: str) -> List[str]:
    key = preference.strip().lower()
    if key in {"retinaface", "retina"}:
        return ["retinaface", "mtcnn", "haar"]
    if key in {"mtcnn"}:
        return ["mtcnn", "retinaface", "haar"]
    if key in {"haar", "haarcascade"}:
        return ["haar", "mtcnn", "retinaface"]
    return ["retinaface", "mtcnn", "haar"]


def init_face_detector(cascade_path: Optional[str] = None, backend_preference: Optional[str] = None):
    """Initialize detector with RetinaFace preference, then MTCNN, then Haar fallback."""
    preference = backend_preference or FACE_DETECTOR_BACKEND
    for backend in _detector_priority(str(preference)):
        if backend == "retinaface":
            if RETINAFACE is None:
                logger.warning("RetinaFace unavailable; trying next detector")
                continue
            logger.info("Using RetinaFace detector backend")
            return {"backend": "retinaface", "detector": RETINAFACE}

        if backend == "mtcnn":
            if MTCNN is None:
                logger.warning("MTCNN unavailable; trying next detector")
                continue
            try:
                logger.info("Using MTCNN detector backend")
                return {"backend": "mtcnn", "detector": MTCNN()}
            except Exception as exc:
                logger.warning("MTCNN initialization failed; trying next detector. Reason: %s", exc)
                continue

        if backend == "haar":
            path = cascade_path or HAAR_CASCADE_PATH
            if not os.path.exists(path):
                raise FileNotFoundError(f"Haar cascade not found: {path}")

            haar = cv2.CascadeClassifier(path)
            if haar.empty():
                raise RuntimeError("Could not initialize Haar Cascade detector.")
            logger.info("Using Haar Cascade face detector backend")
            return {"backend": "haar", "detector": haar}

    raise RuntimeError("No available face detector backend could be initialized")


def _rotate_by_eyes(face_bgr: np.ndarray, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> np.ndarray:
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = face_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(face_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _parse_retinaface_faces(results: Any) -> List[Dict[str, Any]]:
    if not results:
        return []

    if isinstance(results, dict):
        entries = list(results.values())
    elif isinstance(results, list):
        entries = results
    else:
        entries = []

    parsed = []
    for item in entries:
        if not isinstance(item, dict):
            continue

        has_facial_area = "facial_area" in item
        area = item.get("facial_area") or item.get("bbox") or [0, 0, 0, 0]
        if len(area) < 4:
            continue

        x1, y1, x2_or_w, y2_or_h = [int(v) for v in area[:4]]
        if has_facial_area:
            x, y, w, h = x1, y1, x2_or_w - x1, y2_or_h - y1
        else:
            x, y, w, h = x1, y1, x2_or_w, y2_or_h
            if w <= 0 or h <= 0:
                w = x2_or_w - x1
                h = y2_or_h - y1
        if w <= 0 or h <= 0:
            continue

        landmarks = item.get("landmarks", {})
        keypoints = {}
        if isinstance(landmarks, dict):
            for name in ("left_eye", "right_eye"):
                point = landmarks.get(name)
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    keypoints[name] = (int(point[0]), int(point[1]))

        score = float(item.get("score", item.get("confidence", 0.0)))
        parsed.append(
            {
                "bbox": (int(max(0, x)), int(max(0, y)), int(w), int(h)),
                "keypoints": keypoints,
                "score": score,
            }
        )
    return parsed


def _extract_faces(detector_bundle, frame_bgr: np.ndarray, gray_small: np.ndarray):
    backend = detector_bundle.get("backend")
    detector = detector_bundle.get("detector")

    if backend == "retinaface":
        try:
            results = detector.detect_faces(frame_bgr)
        except Exception as exc:
            logger.warning("RetinaFace detection failed on current frame: %s", exc)
            return []
        return _parse_retinaface_faces(results)

    if backend == "mtcnn":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        faces = []
        for item in results:
            x, y, w, h = item.get("box", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            faces.append(
                {
                    "bbox": (int(max(0, x)), int(max(0, y)), int(w), int(h)),
                    "keypoints": item.get("keypoints", {}),
                    "score": float(item.get("confidence", 0.0)),
                }
            )
        return faces

    faces_small = detector.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )
    return [
        {
            "bbox": (int(x), int(y), int(w), int(h)),
            "keypoints": {},
            "score": 1.0,
        }
        for (x, y, w, h) in faces_small
    ]


def _sanitize_bbox(
    box: Tuple[int, int, int, int], frame_w: int, frame_h: int, expand_ratio: float = 0.08
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = box
    if w <= 1 or h <= 1:
        return None

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    nw = int(w * (1.0 + expand_ratio))
    nh = int(h * (1.0 + expand_ratio))

    x1 = max(0, int(cx - (nw / 2.0)))
    y1 = max(0, int(cy - (nh / 2.0)))
    x2 = min(frame_w, int(cx + (nw / 2.0)))
    y2 = min(frame_h, int(cy + (nh / 2.0)))

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
    """Preprocess one face for transfer models (RGB + EfficientNetV2 preprocessing)."""
    if face_bgr.ndim == 2:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2RGB)
    elif face_bgr.ndim == 3 and face_bgr.shape[-1] == 1:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    return preprocess_rgb_for_transfer(rgb, target_size=input_size)


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
) -> np.ndarray:
    if not face_rois:
        return np.zeros((0, len(EMOTIONS)), dtype=np.float32)

    prepared = []
    for face_bgr in face_rois:
        if model_mode == "transfer":
            prepared.append(preprocess_face_transfer(face_bgr, input_size=input_size))
        else:
            prepared.append(preprocess_face(face_bgr, input_size=input_size))

    batch = np.stack(prepared, axis=0).astype(np.float32)
    probs = model.predict(batch, verbose=0)
    if isinstance(probs, list):
        probs = probs[0]

    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 3 and probs.shape[1] == 1:
        probs = probs[:, 0, :]
    if probs.ndim != 2 or probs.shape[1] != len(EMOTIONS):
        raise ValueError("Unexpected model output shape for emotion prediction.")
    return probs


def predict_frame(
    frame: np.ndarray,
    model: Any = None,
    detector: Optional[Dict[str, Any]] = None,
    predictor: Optional[EmotionPredictor] = None,
    confidence_threshold: float = 0.6,
    detection_interval: int = 2,
    detector_size: Optional[Tuple[int, int]] = (640, 480),
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

    tracked_faces: List[Dict[str, Any]] = []
    if run_detection or predictor is None:
        detections = _extract_faces(detector_bundle, resized, gray_small)
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
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
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

    batch_probs = _predict_faces_probs_batch(
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

    for idx, (face, probs) in enumerate(zip(prepared_faces, batch_probs)):
        face_id = int(face["id"])
        x, y, w, h = face["bbox"]
        detection_score = float(face.get("score", 1.0))

        if predictor is not None:
            predictor.confidence_threshold = confidence_threshold
            display_label, conf, smoothed_probs = predictor.stable_label(face_id, probs)
            used_probs = smoothed_probs
        else:
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            display_label = EMOTIONS[pred_idx] if conf >= confidence_threshold else "uncertain"
            used_probs = probs

        color_key = display_label if display_label in EMOTION_COLORS else "uncertain"
        color = EMOTION_COLORS.get(color_key, (120, 120, 120))

        label_text = f"#{face_id} {display_label.capitalize()} {conf * 100.0:.1f}%"
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
                "emotion": str(display_label).lower(),
                "confidence": float(conf),
                "confidence_percent": float(conf * 100.0),
                "detection_confidence": detection_score,
                "probabilities": used_probs.astype(float).tolist(),
                "probabilityMap": {
                    emotion: float(used_probs[i])
                    for i, emotion in enumerate(EMOTIONS)
                },
            }
        )

        if idx == 0:
            primary_probs = used_probs
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
            "confidence_percent": face.get("confidence_percent", float(face["confidence"]) * 100.0),
            "detection_confidence": face.get("detection_confidence", 1.0),
            "probabilities": face["probabilities"],
        }
        for face in faces_predictions
    ]

    return annotated, primary_label, primary_confidence, primary_probs, boxes, faces_predictions, faces_results
