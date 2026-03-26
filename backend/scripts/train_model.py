from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import (  # noqa: E402
    ARTIFACTS_DIR,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    EFFICIENTNET_MODEL_PATH,
    EMOTION_CLASSES,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_SIZE_RESNET,
    TEST_DIR,
    TRAINING_HISTORY_PLOT,
    TRAINING_LOG_PATH,
    TRAIN_DIR,
    TRANSFER_LEARNING_RATE,
    VALIDATION_SPLIT,
)
from models.efficientnet_transfer import (  # noqa: E402
    build_efficientnet_transfer,
    compile_for_finetuning,
    unfreeze_last_layers_for_finetune,
)
from preprocessing.preprocess import (  # noqa: E402
    build_transfer_eval_datagen,
    build_transfer_train_datagen,
    compute_generator_class_weights,
    create_flow_from_directory,
)


class CosineWarmRestart(Callback):
    """Cosine warm-restart scheduler for smoother fine-tuning convergence."""

    def __init__(self, base_lr: float, min_lr: float, cycle_length: int = 8, cycle_mult: float = 1.4):
        super().__init__()
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.cycle_length = max(1, int(cycle_length))
        self.cycle_mult = float(cycle_mult)
        self.epoch_in_cycle = 0

    def on_epoch_begin(self, epoch, logs=None):
        frac = self.epoch_in_cycle / float(max(self.cycle_length, 1))
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * frac))
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "learning_rate"):
            lr_obj = optimizer.learning_rate
            try:
                # Preferred path for TF variable learning rates.
                lr_obj.assign(lr)
            except Exception:
                # Fallback for immutable or schedule-backed LR objects.
                optimizer.learning_rate = lr

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_in_cycle += 1
        if self.epoch_in_cycle >= self.cycle_length:
            self.epoch_in_cycle = 0
            self.cycle_length = max(2, int(self.cycle_length * self.cycle_mult))


def ensure_directories():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def write_log(message: str):
    print(message)
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def validate_dataset_structure():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")


def _extract_labels_from_dataset(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[np.ndarray] = []
    for _, y in dataset:
        labels.append(np.asarray(y))
    if not labels:
        return np.array([], dtype=np.int32)
    return np.concatenate(labels, axis=0).astype(np.int32)


def _preprocess_batch(images: tf.Tensor, labels: tf.Tensor, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    images = tf.cast(images, tf.float32)
    images = efficientnet_preprocess(images)
    labels = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
    return images, labels


def _build_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="fer_augmenter",
    )


def _compute_class_weights_from_labels(labels: np.ndarray) -> Dict[int, float]:
    class_ids = np.unique(labels)
    if class_ids.size == 0:
        return {}
    weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=labels)
    return {int(cid): float(w) for cid, w in zip(class_ids, weights)}


def get_transfer_tfdata_datasets(batch_size: int, validation_split: float, seed: int = 42):
    num_classes = len(EMOTION_CLASSES)
    augmenter = _build_augmenter()

    train_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=EMOTION_CLASSES,
        color_mode="rgb",
        image_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=EMOTION_CLASSES,
        color_mode="rgb",
        image_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        shuffle=False,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    test_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=EMOTION_CLASSES,
        color_mode="rgb",
        image_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        shuffle=False,
    )

    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = (
        train_raw
        .with_options(options)
        .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: _preprocess_batch(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw
        .map(lambda x, y: _preprocess_batch(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_raw
        .map(lambda x, y: _preprocess_batch(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_labels = _extract_labels_from_dataset(train_raw.unbatch().batch(4096))
    class_weights = _compute_class_weights_from_labels(train_labels)
    return train_ds, val_ds, test_ds, class_weights


def get_transfer_generators(batch_size: int):
    train_datagen = build_transfer_train_datagen(validation_split=VALIDATION_SPLIT)
    eval_datagen = build_transfer_eval_datagen()

    train_gen = create_flow_from_directory(
        datagen=train_datagen,
        directory=TRAIN_DIR,
        target_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        subset="training",
        color_mode="rgb",
        shuffle=True,
    )

    val_gen = create_flow_from_directory(
        datagen=train_datagen,
        directory=TRAIN_DIR,
        target_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        subset="validation",
        color_mode="rgb",
        shuffle=False,
    )

    test_gen = create_flow_from_directory(
        datagen=eval_datagen,
        directory=TEST_DIR,
        target_size=IMAGE_SIZE_RESNET,
        batch_size=batch_size,
        subset=None,
        color_mode="rgb",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_stage1_callbacks(model_save_path: str):
    return [
        ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]


def build_stage2_callbacks(model_save_path: str, fine_tune_lr: float):
    return [
        ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=5e-7, verbose=1),
        CosineWarmRestart(base_lr=fine_tune_lr, min_lr=max(fine_tune_lr * 0.05, 5e-7), cycle_length=8),
    ]


def unfreeze_last_n_layers(base_model: tf.keras.Model, last_n: int = 25) -> None:
    """Unfreeze only the last N layers for stable stage-2 fine-tuning."""
    last_n = max(1, int(last_n))
    split_idx = max(0, len(base_model.layers) - last_n)
    for idx, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = idx >= split_idx


def evaluate_on_test_and_log(model, test_data, model_name: str):
    eval_out = model.evaluate(test_data, verbose=1)
    if isinstance(eval_out, (list, tuple)):
        test_loss = float(eval_out[0])
        test_acc = float(eval_out[1]) if len(eval_out) > 1 else 0.0
    else:
        test_loss = float(eval_out)
        test_acc = 0.0

    write_log(f"{model_name} test_loss: {test_loss:.4f}")
    write_log(f"{model_name} test_accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def save_training_plot(history_stage1, history_stage2):
    acc = history_stage1.history.get("accuracy", []) + history_stage2.history.get("accuracy", [])
    val_acc = history_stage1.history.get("val_accuracy", []) + history_stage2.history.get("val_accuracy", [])
    loss = history_stage1.history.get("loss", []) + history_stage2.history.get("loss", [])
    val_loss = history_stage1.history.get("val_loss", []) + history_stage2.history.get("val_loss", [])

    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(x=len(history_stage1.history.get("accuracy", [])) - 1, linestyle="--", color="gray", linewidth=1)
    plt.title("Transfer Learning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(x=len(history_stage1.history.get("loss", [])) - 1, linestyle="--", color="gray", linewidth=1)
    plt.title("Transfer Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PLOT, dpi=220)
    plt.close()


def _dataset_true_labels(test_data) -> np.ndarray:
    if hasattr(test_data, "classes"):
        return np.asarray(test_data.classes)

    labels: list[np.ndarray] = []
    for _, y_batch in test_data:
        y_np = y_batch.numpy()
        if y_np.ndim > 1:
            y_np = np.argmax(y_np, axis=1)
        labels.append(y_np.astype(np.int32))
    if not labels:
        return np.array([], dtype=np.int32)
    return np.concatenate(labels, axis=0)


def save_confusion_matrix_and_report(model, test_data):
    if hasattr(test_data, "reset"):
        test_data.reset()
    y_true = _dataset_true_labels(test_data)
    y_prob = model.predict(test_data, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[e.capitalize() for e in EMOTION_CLASSES],
        yticklabels=[e.capitalize() for e in EMOTION_CLASSES],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=220)
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        target_names=[e.capitalize() for e in EMOTION_CLASSES],
        digits=4,
    )
    report_text = str(report)
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    write_log("\nClassification Report:\n" + report_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved transfer model for facial emotion recognition.")
    parser.add_argument("--model", choices=["efficientnetb0", "mobilenetv2"], default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs across both stages (recommended: 80-100).")
    parser.add_argument("--stage1-epochs", type=int, default=30, help="Frozen-backbone warmup epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pipeline", choices=["tfdata", "generator"], default="tfdata")
    parser.add_argument("--workers", type=int, default=4, help="Used only for generator pipeline")
    parser.add_argument("--unfreeze-layers", type=int, default=25, help="Number of backbone layers to unfreeze in stage 2")
    parser.add_argument("--freeze-ratio", type=float, default=0.70, help="Fraction of backbone to keep frozen during fine-tuning.")
    parser.add_argument("--learning-rate", type=float, default=TRANSFER_LEARNING_RATE)
    parser.add_argument("--fine-tune-lr", type=float, default=FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--disable-focal-loss", action="store_true", help="Use CE loss instead of focal loss.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Training device preference. 'gpu' fails if no GPU is available.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision on GPU for faster training (if supported).",
    )
    return parser.parse_args()


def configure_runtime_device(device: str, mixed_precision: bool) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    if device == "gpu" and not gpus:
        raise RuntimeError(
            "--device gpu was requested, but no GPU is visible to TensorFlow. "
            "On native Windows with TensorFlow >= 2.11, use WSL2 CUDA or TensorFlow-DirectML."
        )

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        tf.keras.mixed_precision.set_global_policy("float32")
        return f"Device: CPU only ({len(cpus)} CPU(s) visible, {len(gpus)} GPU(s) hidden)"

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            return f"Device: GPU ({len(gpus)} GPU(s) visible), mixed precision: ON"

        tf.keras.mixed_precision.set_global_policy("float32")
        return f"Device: GPU ({len(gpus)} GPU(s) visible), mixed precision: OFF"

    tf.keras.mixed_precision.set_global_policy("float32")
    return (
        "Device: CPU fallback (no TensorFlow GPU detected). "
        "On native Windows with TensorFlow >= 2.11, GPU is not supported without WSL2/DirectML."
    )


def log_gpu_info() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    write_log(f"TensorFlow GPUs: {gpus}")
    if gpus:
        for idx, _ in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpus[idx])
                write_log(f"GPU[{idx}] details: {details}")
            except Exception:
                write_log(f"GPU[{idx}] details unavailable")


def main():
    args = parse_args()
    ensure_directories()

    if os.path.exists(TRAINING_LOG_PATH):
        os.remove(TRAINING_LOG_PATH)

    validate_dataset_structure()
    device_summary = configure_runtime_device(device=args.device, mixed_precision=args.mixed_precision)

    total_epochs = max(80, int(args.epochs))
    stage1_epochs = max(12, min(int(args.stage1_epochs), total_epochs - 10))
    stage2_epochs = max(20, total_epochs - stage1_epochs)

    write_log("=== Improved FER Training Pipeline ===")
    write_log(device_summary)
    log_gpu_info()
    write_log(f"Backbone: {args.model}")
    write_log(f"Pipeline: {args.pipeline}")
    write_log(f"Stage-2 unfreeze layers: {args.unfreeze_layers}")
    write_log(f"Train directory: {TRAIN_DIR}")
    write_log(f"Test directory: {TEST_DIR}")
    write_log(f"Input resolution: {IMAGE_SIZE_RESNET[0]}x{IMAGE_SIZE_RESNET[1]} RGB")
    write_log(f"Epoch plan -> stage1: {stage1_epochs}, stage2: {stage2_epochs}, total: {stage1_epochs + stage2_epochs}")

    if args.pipeline == "tfdata":
        train_data, val_data, test_data, class_weights = get_transfer_tfdata_datasets(
            batch_size=args.batch_size,
            validation_split=VALIDATION_SPLIT,
            seed=42,
        )
        fit_kwargs = {}
    else:
        train_gen, val_gen, test_gen = get_transfer_generators(batch_size=args.batch_size)
        train_data, val_data, test_data = train_gen, val_gen, test_gen
        class_weights = compute_generator_class_weights(train_gen)
        fit_kwargs = {
            "workers": int(args.workers),
            "use_multiprocessing": True,
            "max_queue_size": 32,
        }

    class_weights = class_weights if class_weights else {}
    write_log(f"Computed class weights: {class_weights}")

    model, base_model = build_efficientnet_transfer(
        input_shape=(IMAGE_SIZE_RESNET[0], IMAGE_SIZE_RESNET[1], 3),
        num_classes=len(EMOTION_CLASSES),
        learning_rate=float(args.learning_rate),
        backbone=args.model,
        use_focal_loss=not args.disable_focal_loss,
        freeze_base=True,
    )

    write_log("\nStage 1: training attention head with frozen backbone...")
    history_stage1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=stage1_epochs,
        callbacks=build_stage1_callbacks(EFFICIENTNET_MODEL_PATH),
        class_weight=class_weights,
        verbose=1,
        **fit_kwargs,
    )

    write_log("\nStage 2: fine-tuning top backbone layers with low learning rate...")
    unfreeze_last_n_layers(base_model, last_n=int(args.unfreeze_layers))
    compile_for_finetuning(
        model,
        learning_rate=float(args.fine_tune_lr),
        use_focal_loss=not args.disable_focal_loss,
    )

    history_stage2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=stage1_epochs,
        callbacks=build_stage2_callbacks(EFFICIENTNET_MODEL_PATH, fine_tune_lr=float(args.fine_tune_lr)),
        class_weight=class_weights,
        verbose=1,
        **fit_kwargs,
    )

    val_acc_all = history_stage1.history.get("val_accuracy", []) + history_stage2.history.get("val_accuracy", [])
    best_val_acc = float(np.max(val_acc_all)) if val_acc_all else 0.0
    write_log(f"Best validation accuracy: {best_val_acc:.4f}")

    shutil.copy2(EFFICIENTNET_MODEL_PATH, BEST_MODEL_PATH)
    write_log(f"Saved best model to: {BEST_MODEL_PATH}")

    _, test_acc = evaluate_on_test_and_log(model, test_data, f"{args.model} transfer")
    write_log(f"Final test accuracy: {test_acc:.4f}")

    save_training_plot(history_stage1, history_stage2)
    write_log(f"Saved training plot to: {TRAINING_HISTORY_PLOT}")

    save_confusion_matrix_and_report(model, test_data)
    write_log(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    write_log(f"Saved classification report to: {CLASSIFICATION_REPORT_PATH}")
    write_log(f"Saved training log to: {TRAINING_LOG_PATH}")

    target = 0.75
    if best_val_acc >= target:
        write_log(f"Target reached: val_accuracy {best_val_acc:.4f} >= {target:.2f}")
    else:
        write_log(
            "Target not reached in this run. Re-run with --model efficientnetb0 and tune --freeze-ratio/--fine-tune-lr."
        )


if __name__ == "__main__":
    main()
