from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import (  # noqa: E402
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    EMOTION_CLASSES,
    EFFICIENTNET_MODEL_PATH,
    TEST_DIR,
    TRAINING_HISTORY_PLOT,
    TRAINING_LOG_PATH,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)
from preprocessing.preprocess import preprocess_rgb_for_transfer  # noqa: E402


INPUT_SIZE = (224, 224)
NUM_CLASSES = len(EMOTION_CLASSES)


@dataclass
class TrainConfig:
    batch_size: int = 32
    stage1_epochs: int = 18
    stage2_epochs: int = 42
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-5
    seed: int = 42
    class_weight_cap: float = 5.0
    unfreeze_last_n: int = 80


def ensure_paths() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")


def reset_log() -> None:
    if os.path.exists(TRAINING_LOG_PATH):
        os.remove(TRAINING_LOG_PATH)


def write_log(message: str) -> None:
    print(message)
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def configure_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        write_log(f"TensorFlow detected {len(gpus)} GPU(s). Memory growth enabled.")
        write_log("Mixed precision enabled: mixed_float16")
        return True

    tf.keras.mixed_precision.set_global_policy("float32")
    write_log("No GPU detected. Training will run on CPU.")
    write_log("Mixed precision disabled (float32 policy).")
    return False


def _focal_loss() -> tf.keras.losses.Loss:
    # Focal loss improves minority-class recall for hard FER classes (e.g., fear, angry).
    return tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)


def _effective_batch_size(batch_size: int, gpu_available: bool) -> int:
    if gpu_available:
        return max(8, int(batch_size))
    # Keep CPU training stable and avoid OOM.
    return min(max(8, int(batch_size)), 16)


def build_datagens(validation_split: float) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_rgb_for_transfer,
        validation_split=validation_split,
    )

    eval_gen = ImageDataGenerator(
        preprocessing_function=preprocess_rgb_for_transfer,
        validation_split=validation_split,
    )
    return train_gen, eval_gen


def create_generators(config: TrainConfig):
    train_datagen, eval_datagen = build_datagens(validation_split=VALIDATION_SPLIT)

    train_gen = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=INPUT_SIZE,
        classes=EMOTION_CLASSES,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=config.batch_size,
        subset="training",
        shuffle=True,
        seed=config.seed,
    )

    val_gen = eval_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=INPUT_SIZE,
        classes=EMOTION_CLASSES,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=config.batch_size,
        subset="validation",
        shuffle=False,
        seed=config.seed,
    )

    test_gen = ImageDataGenerator(preprocessing_function=preprocess_rgb_for_transfer).flow_from_directory(
        directory=TEST_DIR,
        target_size=INPUT_SIZE,
        classes=EMOTION_CLASSES,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=config.batch_size,
        shuffle=False,
    )

    if train_gen.class_indices != {name: idx for idx, name in enumerate(EMOTION_CLASSES)}:
        raise RuntimeError(
            "Class index mismatch detected. Ensure TRAIN_DIR and TEST_DIR folders follow EMOTION_CLASSES order."
        )

    return train_gen, val_gen, test_gen


def compute_balanced_class_weights(train_gen, cap: float) -> Dict[int, float]:
    y = train_gen.classes
    class_ids = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=y)
    capped = {int(k): float(min(v, cap)) for k, v in zip(class_ids, weights)}

    full_weights: Dict[int, float] = {}
    for class_id in range(NUM_CLASSES):
        full_weights[class_id] = float(capped.get(class_id, 1.0))

    return full_weights


def build_model(learning_rate: float, freeze_base: bool = True) -> Tuple[Model, Model]:
    inputs = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name="image")

    # Online augmentation for robust generalization on FER classes.
    aug = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.2),
            RandomZoom(0.2),
            RandomContrast(0.2),
        ],
        name="augmentation",
    )

    x_in = aug(inputs)
    backbone = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=x_in)
    backbone.trainable = not freeze_base

    x = backbone.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_256")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.5, name="drop_256")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.4, name="drop_128")(x)

    # Keep classifier output in float32 for numeric stability under mixed precision.
    outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="emotion")(x)

    model = Model(inputs=inputs, outputs=outputs, name="efficientnetb3_emotion")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_focal_loss(),
        metrics=["accuracy"],
    )
    return model, backbone


def unfreeze_last_layers(backbone: Model, last_n: int = 50) -> None:
    split = max(0, len(backbone.layers) - int(last_n))
    for index, layer in enumerate(backbone.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = index >= split


def get_callbacks(model_path: str):
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ]


def concat_histories(history1, history2) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        out[key] = history1.history.get(key, []) + history2.history.get(key, [])
    return out


def save_history_plot(history: Dict[str, list]) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PLOT, dpi=220)
    plt.close()


def evaluate_and_save_reports(model: Model, test_gen) -> float:
    eval_out = model.evaluate(test_gen, verbose=1)
    if isinstance(eval_out, (list, tuple)):
        test_loss = float(eval_out[0])
        test_acc = float(eval_out[1]) if len(eval_out) > 1 else 0.0
    else:
        test_loss = float(eval_out)
        test_acc = 0.0
    write_log(f"Final test loss: {test_loss:.4f}")
    write_log(f"Final test accuracy: {test_acc:.4f}")

    test_gen.reset()
    probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(NUM_CLASSES)
    names = [name.capitalize() for name in EMOTION_CLASSES]
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=220)
    plt.close()

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[name.capitalize() for name in EMOTION_CLASSES],
        digits=4,
    )
    if not isinstance(report_text, str):
        report_text = str(report_text)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    write_log("\nClassification report:\n" + report_text)
    return float(test_acc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EfficientNetB3 transfer-learning pipeline for facial emotion detection.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=18)
    parser.add_argument("--stage2-epochs", type=int, default=42)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--class-weight-cap", type=float, default=5.0)
    parser.add_argument("--unfreeze-last-n", type=int, default=80, help="Number of final backbone layers to unfreeze (50-100 recommended).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        batch_size=max(1, int(args.batch_size)),
        stage1_epochs=max(1, int(args.stage1_epochs)),
        stage2_epochs=max(1, int(args.stage2_epochs)),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        class_weight_cap=max(1.0, float(args.class_weight_cap)),
        unfreeze_last_n=min(100, max(50, int(args.unfreeze_last_n))),
    )

    ensure_paths()
    reset_log()
    has_gpu = configure_gpu()
    config.batch_size = _effective_batch_size(config.batch_size, has_gpu)

    write_log("=== EfficientNetB3 Transfer FER Pipeline ===")
    write_log(f"Train dir: {TRAIN_DIR}")
    write_log(f"Test dir: {TEST_DIR}")
    write_log(f"Input size: {INPUT_SIZE}")
    write_log(f"Batch size: {config.batch_size}")
    write_log(f"Stage 1 epochs: {config.stage1_epochs}")
    write_log(f"Stage 2 epochs: {config.stage2_epochs}")
    write_log(f"Fine-tune layers: last {config.unfreeze_last_n}")
    write_log(f"Class weight cap: {config.class_weight_cap}")
    write_log("Loss: CategoricalFocalCrossentropy")
    write_log("Backbone: EfficientNetB3 (ImageNet pretrained)")

    train_gen, val_gen, test_gen = create_generators(config)
    class_weights = compute_balanced_class_weights(train_gen, cap=config.class_weight_cap)
    write_log(f"Class weights (capped): {class_weights}")

    model, backbone = build_model(learning_rate=config.stage1_lr, freeze_base=True)
    model.summary(print_fn=write_log)
    write_log("Stage 1: backbone frozen")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.stage1_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(EFFICIENTNET_MODEL_PATH),
        verbose=1,
    )

    write_log(f"Stage 2: fine-tuning last {config.unfreeze_last_n} layers")
    unfreeze_last_layers(backbone, last_n=config.unfreeze_last_n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.stage2_lr),
        loss=_focal_loss(),
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=config.stage1_epochs,
        epochs=config.stage1_epochs + config.stage2_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(EFFICIENTNET_MODEL_PATH),
        verbose=1,
    )

    history = concat_histories(history1, history2)
    best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    write_log(f"Best validation accuracy: {best_val_acc:.4f}")

    if os.path.exists(EFFICIENTNET_MODEL_PATH):
        shutil.copy2(EFFICIENTNET_MODEL_PATH, BEST_MODEL_PATH)
        write_log(f"Best model copied to: {BEST_MODEL_PATH}")

    test_acc = evaluate_and_save_reports(model, test_gen)
    save_history_plot(history)

    val_test_gap = max(0.0, float(best_val_acc) - float(test_acc))
    write_log(f"Validation-test gap: {val_test_gap:.4f}")
    write_log(f"Training history plot: {TRAINING_HISTORY_PLOT}")
    write_log(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    write_log(f"Classification report: {CLASSIFICATION_REPORT_PATH}")
    write_log(f"Training log: {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()