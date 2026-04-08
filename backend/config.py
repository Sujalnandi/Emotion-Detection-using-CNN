import os
import cv2

# Fixed class order across training and inference to avoid label mismatch.
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NUM_CLASSES = len(EMOTION_CLASSES)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

def _has_train_test_dirs(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "test"))


def _normalize_dataset_dir(path: str) -> str:
    """Accept dataset root or direct train/test split path and normalize to dataset root."""
    if not path:
        return ""

    normalized = os.path.abspath(path)
    if _has_train_test_dirs(normalized):
        return normalized

    split_name = os.path.basename(normalized).lower()
    parent = os.path.dirname(normalized)
    if split_name in {"train", "test"} and _has_train_test_dirs(parent):
        return parent

    return ""


def _discover_kaggle_dataset_dir() -> str:
    kaggle_input_root = "/kaggle/input"
    if not os.path.isdir(kaggle_input_root):
        return ""

    try:
        level1 = sorted(os.listdir(kaggle_input_root))
    except OSError:
        return ""

    # Common Kaggle mounts are one or two levels below /kaggle/input.
    for name in level1:
        candidate = os.path.join(kaggle_input_root, name)
        if _has_train_test_dirs(candidate):
            return candidate

    for name in level1:
        parent = os.path.join(kaggle_input_root, name)
        if not os.path.isdir(parent):
            continue
        try:
            level2 = sorted(os.listdir(parent))
        except OSError:
            continue
        for child in level2:
            candidate = os.path.join(parent, child)
            if _has_train_test_dirs(candidate):
                return candidate

    return ""


# Dataset resolution priority:
# 1) FER_DATASET_DIR env var (dataset root, or direct train/test path)
# 2) Local backend path: ./dataset
# 3) Legacy project path: ../dataset
# 4) Kaggle auto-discovery under /kaggle/input
LOCAL_DATASET = os.path.join(BASE_DIR, "dataset")
LEGACY_PROJECT_DATASET = os.path.join(PROJECT_DIR, "dataset")
ENV_DATASET = os.environ.get("FER_DATASET_DIR", "").strip()

DATASET_DIR = ""
for candidate in [ENV_DATASET, LOCAL_DATASET, LEGACY_PROJECT_DATASET, _discover_kaggle_dataset_dir()]:
    resolved = _normalize_dataset_dir(candidate)
    if resolved:
        DATASET_DIR = resolved
        break

if not DATASET_DIR:
    DATASET_DIR = os.path.abspath(ENV_DATASET or LOCAL_DATASET)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMAGE_SIZE_CNN = (48, 48)
IMAGE_SIZE_RESNET = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
TRANSFER_LEARNING_RATE = 3e-4
FINE_TUNE_LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.2
PREDICTION_SMOOTHING_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.6
DETECTION_INTERVAL = 2
FACE_DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR_BACKEND", "haar")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
ROOT_BEST_MODEL_KERAS_PATH = os.path.join(PROJECT_DIR, "best_model.keras")
LEGACY_ROOT_BEST_MODEL_H5_PATH = os.path.join(PROJECT_DIR, "best_model.h5")
if os.path.exists(ROOT_BEST_MODEL_KERAS_PATH):
    BEST_MODEL_PATH = ROOT_BEST_MODEL_KERAS_PATH
else:
    BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.keras")
TRAINING_HISTORY_PLOT = os.path.join(ARTIFACTS_DIR, "training_history.png")
CNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "cnn_model.h5")
RESNET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "resnet50_model.h5")
EFFICIENTNET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "efficientnet_b3_best.keras")
LEGACY_EFFICIENTNET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "efficientnet_model.h5")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.txt")

HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
