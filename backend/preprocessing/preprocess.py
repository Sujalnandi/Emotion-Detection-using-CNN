from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from config import EMOTION_CLASSES
except ModuleNotFoundError:
    from backend.config import EMOTION_CLASSES

EFFICIENTNET_INPUT_SIZE = (224, 224)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert 2D/1-channel image arrays to 3-channel RGB arrays."""
    x = np.asarray(image)

    if x.ndim == 2:
        return np.stack([x, x, x], axis=-1)

    if x.ndim == 3 and x.shape[-1] == 1:
        return np.repeat(x, 3, axis=-1)

    if x.ndim == 3 and x.shape[-1] > 3:
        return x[:, :, :3]

    return x


def preprocess_rgb_for_transfer(image: np.ndarray, target_size: Tuple[int, int] = EFFICIENTNET_INPUT_SIZE) -> np.ndarray:
    """Shared preprocessing for EfficientNet training, validation, and inference."""
    x = _to_rgb(image).astype("float32")

    if x.shape[0] != target_size[0] or x.shape[1] != target_size[1]:
        interp = cv2.INTER_AREA if (x.shape[0] > target_size[0] or x.shape[1] > target_size[1]) else cv2.INTER_LINEAR
        x = cv2.resize(x, (target_size[1], target_size[0]), interpolation=interp)

    return efficientnet_preprocess(x)


def build_transfer_train_datagen(validation_split: float = 0.2) -> ImageDataGenerator:
    """Transfer-learning datagen using shared EfficientNet preprocessing."""
    return ImageDataGenerator(
        preprocessing_function=preprocess_rgb_for_transfer,
        validation_split=validation_split,
    )


def build_transfer_eval_datagen() -> ImageDataGenerator:
    """Validation/test datagen using shared EfficientNet preprocessing."""
    return ImageDataGenerator(preprocessing_function=preprocess_rgb_for_transfer)


def create_flow_from_directory(
    datagen: ImageDataGenerator,
    directory: str,
    target_size: Tuple[int, int],
    batch_size: int,
    subset: str | None = None,
    color_mode: str = "rgb",
    shuffle: bool = True,
):
    """Shared directory-flow helper to keep class order fixed across splits."""
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


def preprocess_face(face_bgr: np.ndarray, target_size: Tuple[int, int] = EFFICIENTNET_INPUT_SIZE, model_type: str = "efficientnet"):
    """Preprocess a BGR face crop for inference."""
    model_key = str(model_type).lower()

    if model_key in {"resnet", "transfer", "efficientnet", "efficientnetb3"}:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = preprocess_rgb_for_transfer(face_rgb, target_size=target_size)
        return np.expand_dims(x, axis=0)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (target_size[1], target_size[0]))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return np.expand_dims(gray, axis=0)


def compute_generator_class_weights(train_generator, max_cap: float = 5.0):
    """Compute balanced class weights with optional max cap for stability."""
    labels = train_generator.classes
    class_ids = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=labels)
    return {int(cid): float(min(weight, max_cap)) for cid, weight in zip(class_ids, weights)}
