from __future__ import annotations

from typing import Literal, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Multiply,
    Reshape,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

BackboneName = Literal["efficientnetb0", "mobilenetv2"]
DEFAULT_IMAGENET_WEIGHTS = "imagenet"


@tf.keras.utils.register_keras_serializable(package="emotion")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def _squeeze_excite(features, ratio: int = 8):
    """
    Squeeze-and-Excitation attention mechanism to recalibrate channel-wise responses.
    
    Args:
        features: Input tensor
        ratio: Reduction ratio for bottleneck (default: 8). Lower values = more parameters,
               higher values = more compression. Typical range: 4-32.
    """
    channels = int(features.shape[-1]) if features.shape[-1] is not None else 256
    squeeze = GlobalAveragePooling2D(name="attn_gap")(features)
    squeeze = Reshape((1, 1, channels), name="attn_reshape")(squeeze)
    excitation = Dense(max(channels // ratio, 16), activation="relu", name="attn_fc1")(squeeze)
    excitation = Dense(channels, activation="sigmoid", name="attn_fc2")(excitation)
    return Multiply(name="attn_scale")([features, excitation])


def _build_backbone(name: BackboneName, inputs, weights: Optional[str] = DEFAULT_IMAGENET_WEIGHTS):
    if name == "mobilenetv2":
        try:
            return MobileNetV2(weights=weights, include_top=False, input_tensor=inputs)
        except Exception:
            return MobileNetV2(weights=None, include_top=False, input_tensor=inputs)

    try:
        return EfficientNetB0(weights=weights, include_top=False, input_tensor=inputs)
    except Exception:
        return EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)


def build_efficientnet_transfer(
    input_shape=(224, 224, 3),
    num_classes=7,
    learning_rate=3e-4,
    backbone: BackboneName = "efficientnetb0",
    pretrained_weights: Optional[str] = DEFAULT_IMAGENET_WEIGHTS,
    use_focal_loss: bool = True,
    freeze_base=True,
):
    """
    Build transfer model with attention head for facial emotion classification.
    
    Key improvements:
    - Proper base model freezing (trainable=False during stage 1)
    - Attention mechanism with squeeze-excitation blocks
    - Progressive dropout (0.35 → 0.45) to prevent overfitting
    - L2 regularization (1e-4) on dense layers
    - Batch normalization after each dense layer for stable training
    - Label smoothing via CategoricalCrossentropy or focal loss
    """
    inputs = Input(shape=input_shape)

    base_model = _build_backbone(backbone, inputs, weights=pretrained_weights)

    # **FIX 1**: Properly freeze base model for stage 1 training
    base_model.trainable = not freeze_base

    # Attention mechanism on base model output
    x = _squeeze_excite(base_model.output, ratio=8)
    x = GlobalAveragePooling2D(name="head_gap")(x)
    x = BatchNormalization(name="head_bn_1")(x)
    
    # **FIX 2**: Increased dropout for better regularization
    x = Dropout(0.5, name="head_dropout_1")(x)
    
    # Dense layer with regularization
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="head_dense_1")(x)
    x = BatchNormalization(name="head_bn_2")(x)
    
    # **FIX 3**: Further dropout to prevent overfitting
    x = Dropout(0.5, name="head_dropout_2")(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    # Use focal loss to handle hard samples, or CE with label smoothing
    loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.35) if use_focal_loss else CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_last_layers_for_finetune(base_model, unfreeze_from_ratio=0.7):
    """Unfreeze deeper blocks for second-stage fine-tuning."""
    total_layers = len(base_model.layers)
    start_idx = int(total_layers * float(unfreeze_from_ratio))
    for i, layer in enumerate(base_model.layers):
        # Keep BN layers frozen during fine-tuning for small FER datasets.
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = i >= start_idx


def compile_for_finetuning(model, learning_rate=1e-5, use_focal_loss: bool = True):
    """Re-compile model after layer unfreezing."""
    loss_fn = CategoricalFocalLoss(gamma=1.5, alpha=0.35) if use_focal_loss else CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model
