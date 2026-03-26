import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.utils.class_weight import compute_class_weight

try:
    from config import EMOTION_CLASSES
except ModuleNotFoundError:
    from backend.config import EMOTION_CLASSES


def _equalize_grayscale(gray_image):
    """Improve local contrast for in-the-wild lighting conditions."""
    gray_uint8 = gray_image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_uint8)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def preprocess_grayscale_image(x):
    """Apply grayscale contrast enhancement and normalize to [0, 1]."""
    if x.ndim == 3 and x.shape[-1] > 1:
        # flow_from_directory provides RGB arrays for color_mode="rgb".
        gray = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif x.ndim == 3 and x.shape[-1] == 1:
        gray = x[..., 0]
    else:
        gray = x

    gray = _equalize_grayscale(gray)
    gray = gray.astype("float32") / 255.0
    return np.expand_dims(gray, axis=-1)


def preprocess_rgb_for_transfer(x):
    """
    Proper EfficientNet preprocessing with batch dimension handling
    """

    rgb = x.astype("float32")

    # Convert grayscale to RGB if needed
    if rgb.ndim == 3 and rgb.shape[-1] == 1:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.ndim == 2:
        # Single channel image, convert to RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

    # Resize to EfficientNet input size
    rgb = cv2.resize(rgb, (224, 224))

    # Apply EfficientNet preprocessing [-1, 1]
    rgb = efficientnet_preprocess(rgb)

    return rgb


def build_train_datagen(validation_split=0.2):
    """Create training ImageDataGenerator with augmentation."""
    return ImageDataGenerator(
        preprocessing_function=preprocess_grayscale_image,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
        validation_split=validation_split,
    )


def build_transfer_train_datagen(validation_split=0.2):
    """Training datagen for transfer models with RGB tensors."""
    return ImageDataGenerator(
        preprocessing_function=preprocess_rgb_for_transfer,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.25,
        shear_range=0.12,
        brightness_range=(0.75, 1.3),
        fill_mode="nearest",
        validation_split=validation_split,
    )


def build_eval_datagen():
    """Create validation/test ImageDataGenerator without augmentation."""
    return ImageDataGenerator(preprocessing_function=preprocess_grayscale_image)


def build_transfer_eval_datagen():
    """Validation/test datagen for transfer models with RGB tensors."""
    return ImageDataGenerator(preprocessing_function=preprocess_rgb_for_transfer)


def create_flow_from_directory(
    datagen,
    directory,
    target_size,
    batch_size,
    subset=None,
    color_mode="grayscale",
    shuffle=True,
):
    """Shared helper to keep class ordering and generator params consistent."""
    return datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        classes=EMOTION_CLASSES,
        class_mode="categorical",
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        subset=subset,
    )


def preprocess_face(face_bgr, target_size=(48, 48), model_type="cnn"):
    """Preprocess a face crop for model inference."""
    model_type = model_type.lower()
    if model_type in {"resnet", "transfer", "efficientnet"}:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, target_size)
        face = preprocess_rgb_for_transfer(face_rgb)
    else:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size)
        gray = _equalize_grayscale(gray)
        face = gray.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)
    return face


def compute_generator_class_weights(train_generator):
    """Compute class weights from generator labels to reduce class imbalance."""
    labels = train_generator.classes
    class_ids = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=labels)
    return {int(cid): float(w) for cid, w in zip(class_ids, weights)}
