from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input


def build_optimizer(learning_rate: float, weight_decay: float = 1e-4):
    """Create AdamW when available, otherwise fall back to Adam."""
    if hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    experimental = getattr(tf.keras.optimizers, "experimental", None)
    if experimental and hasattr(experimental, "AdamW"):
        return experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def build_efficientnetv2_transfer(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    freeze_base: bool = True,
    dropout_rate: float = 0.5,
) -> Tuple[Model, Model]:
    """Build EfficientNetV2 transfer model with an FER classifier head."""
    inputs = Input(shape=input_shape, name="image")
    backbone = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=False,
    )
    backbone.trainable = not freeze_base

    x = backbone.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_head")(x)
    x = Dense(384, activation="relu", name="dense_384")(x)
    x = Dropout(dropout_rate, name="dropout_384")(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="emotion")(x)

    model = Model(inputs=inputs, outputs=outputs, name="efficientnetv2b0_emotion")
    model.compile(
        optimizer=build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model, backbone


def unfreeze_top_ratio(backbone: Model, ratio: float = 0.25) -> None:
    """Unfreeze only the top ratio of layers while keeping BN layers frozen."""
    ratio = max(0.0, min(1.0, float(ratio)))
    total = len(backbone.layers)
    split = int(total * (1.0 - ratio))

    for index, layer in enumerate(backbone.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = index >= split
