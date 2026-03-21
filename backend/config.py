import os

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

# Dataset resolution priority:
# 1) FER_DATASET_DIR env var
# 2) Local backend path: ./dataset
# 3) Legacy project path: ../dataset
LOCAL_DATASET = os.path.join(BASE_DIR, "dataset")
LEGACY_PROJECT_DATASET = os.path.join(PROJECT_DIR, "dataset")
DATASET_DIR = os.environ.get("FER_DATASET_DIR", LOCAL_DATASET)
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = LEGACY_PROJECT_DATASET

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
ROOT_BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "best_model.h5")
BEST_MODEL_PATH = ROOT_BEST_MODEL_PATH if os.path.exists(ROOT_BEST_MODEL_PATH) else os.path.join(ARTIFACTS_DIR, "best_model.h5")
TRAINING_HISTORY_PLOT = os.path.join(ARTIFACTS_DIR, "training_history.png")
CNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "cnn_model.h5")
RESNET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "resnet50_model.h5")
EFFICIENTNET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "efficientnet_model.h5")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.txt")

HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(__import__("cv2").__file__),
    "data",
    "haarcascade_frontalface_default.xml",
)
