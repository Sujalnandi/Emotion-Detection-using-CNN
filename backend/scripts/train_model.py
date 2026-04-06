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
    EFFICIENTNET_MODEL_PATH,
    EMOTION_CLASSES,
    TEST_DIR,
    TRAIN_DIR,
    TRAINING_HISTORY_PLOT,
    TRAINING_LOG_PATH,
    VALIDATION_SPLIT,
)
from preprocessing.preprocess import preprocess_rgb_for_transfer  # noqa: E402

INPUT_SIZE = (224, 224)
NUM_CLASSES = len(EMOTION_CLASSES)


@dataclass
class TrainConfig:
    batch_size: int = 32
    stage1_epochs: int = 12
    stage2_epochs: int = 30
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


def configure_runtime() -> bool:
    xla_enabled = os.environ.get("FER_ENABLE_XLA", "0").strip() == "1"
    tf.config.optimizer.set_jit(xla_enabled)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        write_log(f"Detected {len(gpus)} GPU(s). Memory growth enabled.")
        write_log("Mixed precision enabled: mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        write_log("No GPU detected. Mixed precision disabled (float32 policy).")

    write_log(f"XLA JIT enabled: {xla_enabled}")
    return bool(gpus)


def focal_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)


def effective_batch_size(batch_size: int, gpu_available: bool) -> int:
    if gpu_available:
        return max(8, int(batch_size))
    return min(max(8, int(batch_size)), 16)


def create_generators(config: TrainConfig):
    base_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_rgb_for_transfer,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = base_datagen.flow_from_directory(
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

    val_gen = base_datagen.flow_from_directory(
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

    expected = {name: idx for idx, name in enumerate(EMOTION_CLASSES)}
    if train_gen.class_indices != expected:
        raise RuntimeError(
            "Class index mismatch. Ensure dataset folder names and EMOTION_CLASSES ordering are identical."
        )

    return train_gen, val_gen, test_gen


def compute_capped_class_weights(train_gen, cap: float) -> Dict[int, float]:
    labels = train_gen.classes
    class_ids = np.unique(labels)
    raw_weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=labels)

    capped: Dict[int, float] = {}
    for class_id, weight in zip(class_ids, raw_weights):
        capped[int(class_id)] = float(min(float(weight), float(cap)))

    full_weights: Dict[int, float] = {}
    for class_id in range(NUM_CLASSES):
        full_weights[class_id] = float(capped.get(class_id, 1.0))
    return full_weights


def build_model(learning_rate: float, freeze_backbone: bool = True) -> Tuple[Model, Model]:
    inputs = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            RandomContrast(0.1),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    backbone = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    backbone.trainable = not freeze_backbone

    x = backbone(x, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_head")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.5, name="dropout_256")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.4, name="dropout_128")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="emotion")(x)

    model = Model(inputs=inputs, outputs=outputs, name="efficientnetb3_emotion")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(),
        metrics=["accuracy"],
    )
    return model, backbone


def unfreeze_last_layers(backbone: Model, last_n: int) -> None:
    split_index = max(0, len(backbone.layers) - int(last_n))
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= split_index


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


def save_history_plot(history: Dict[str, list], stage1_epochs: int) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.axvline(max(0, stage1_epochs - 1), linestyle="--", color="gray", label="fine-tune start")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.axvline(max(0, stage1_epochs - 1), linestyle="--", color="gray", label="fine-tune start")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PLOT, dpi=220)
    plt.close()


def load_best_model_for_eval(model_path: str, learning_rate: float) -> Model:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(),
        metrics=["accuracy"],
    )
    return model


def evaluate_and_save_reports(model: Model, test_gen) -> float:
    metrics = model.evaluate(test_gen, verbose=1)
    if isinstance(metrics, (tuple, list)):
        test_loss = float(metrics[0])
        test_acc = float(metrics[1]) if len(metrics) > 1 else 0.0
    else:
        test_loss = float(metrics)
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
    tick_ids = np.arange(NUM_CLASSES)
    names = [name.capitalize() for name in EMOTION_CLASSES]
    plt.xticks(tick_ids, names, rotation=45, ha="right")
    plt.yticks(tick_ids, names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=220)
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        target_names=[name.capitalize() for name in EMOTION_CLASSES],
        digits=4,
    )
    report_text = str(report)
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    write_log("\nClassification report:\n" + report_text)
    return test_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train facial emotion detector with EfficientNetB3 transfer learning.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=12, help="Backbone frozen stage. Recommended 10-15.")
    parser.add_argument("--stage2-epochs", type=int, default=30, help="Fine-tuning stage after unfreezing layers.")
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--class-weight-cap", type=float, default=5.0)
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=80,
        help="Number of final EfficientNet layers to unfreeze. Recommended 50-100.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        batch_size=max(1, int(args.batch_size)),
        stage1_epochs=min(15, max(10, int(args.stage1_epochs))),
        stage2_epochs=max(1, int(args.stage2_epochs)),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        class_weight_cap=max(1.0, float(args.class_weight_cap)),
        unfreeze_last_n=min(100, max(50, int(args.unfreeze_last_n))),
    )

    ensure_paths()
    reset_log()
    gpu_available = configure_runtime()
    config.batch_size = effective_batch_size(config.batch_size, gpu_available)

    write_log("=== EfficientNetB3 Transfer Learning Pipeline ===")
    write_log(f"Train directory: {TRAIN_DIR}")
    write_log(f"Validation split: {VALIDATION_SPLIT}")
    write_log(f"Test directory: {TEST_DIR}")
    write_log(f"Input size: {INPUT_SIZE}")
    write_log(f"Batch size: {config.batch_size}")
    write_log(f"Stage 1 epochs (frozen): {config.stage1_epochs}")
    write_log(f"Stage 2 epochs (fine-tune): {config.stage2_epochs}")
    write_log(f"Stage 1 learning rate: {config.stage1_lr}")
    write_log(f"Stage 2 learning rate: {config.stage2_lr}")
    write_log(f"Unfreeze last layers: {config.unfreeze_last_n}")
    write_log(f"Class-weight cap: {config.class_weight_cap}")
    write_log("Loss: CategoricalFocalCrossentropy(gamma=2.0)")
    write_log("Backbone: EfficientNetB3(include_top=False, weights='imagenet')")

    train_gen, val_gen, test_gen = create_generators(config)
    class_weights = compute_capped_class_weights(train_gen, cap=config.class_weight_cap)
    write_log(f"Class weights (capped): {class_weights}")

    model, backbone = build_model(learning_rate=config.stage1_lr, freeze_backbone=True)
    model.summary(print_fn=write_log)

    callbacks = get_callbacks(EFFICIENTNET_MODEL_PATH)

    write_log("\nStage 1 training started (backbone frozen)")
    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    write_log(f"\nStage 2 training started (unfreezing last {config.unfreeze_last_n} layers)")
    unfreeze_last_layers(backbone, last_n=config.unfreeze_last_n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.stage2_lr),
        loss=focal_loss(),
        metrics=["accuracy"],
    )

    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=config.stage1_epochs,
        epochs=config.stage1_epochs + config.stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history = concat_histories(history_stage1, history_stage2)
    best_val_acc = max(history.get("val_accuracy", [0.0]))
    write_log(f"Best validation accuracy: {best_val_acc:.4f}")

    eval_model = model
    if os.path.exists(EFFICIENTNET_MODEL_PATH):
        eval_model = load_best_model_for_eval(EFFICIENTNET_MODEL_PATH, learning_rate=config.stage2_lr)
        write_log(f"Loaded best checkpoint for evaluation: {EFFICIENTNET_MODEL_PATH}")

        if os.path.abspath(EFFICIENTNET_MODEL_PATH) != os.path.abspath(BEST_MODEL_PATH):
            shutil.copy2(EFFICIENTNET_MODEL_PATH, BEST_MODEL_PATH)
            write_log(f"Best model copied to: {BEST_MODEL_PATH}")

    test_acc = evaluate_and_save_reports(eval_model, test_gen)
    save_history_plot(history, stage1_epochs=config.stage1_epochs)

    val_test_gap = max(0.0, float(best_val_acc) - float(test_acc))
    write_log(f"Validation-test gap: {val_test_gap:.4f}")
    write_log(f"Training history: {TRAINING_HISTORY_PLOT}")
    write_log(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    write_log(f"Classification report: {CLASSIFICATION_REPORT_PATH}")
    write_log(f"Training log: {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
