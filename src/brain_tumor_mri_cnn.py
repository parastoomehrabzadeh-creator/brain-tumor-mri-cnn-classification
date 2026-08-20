"""
Brain Tumor MRI Classification with a Custom CNN

This script trains and evaluates a convolutional neural network for
four-class brain MRI image classification:

- glioma
- meningioma
- notumor
- pituitary

Expected dataset structure:

data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

The dataset is not included in this repository because of file-size and
data-sharing restrictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_reproducibility(seed: int = 20) -> None:
    """Set random seeds for more reproducible experiments."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    """Create a dataframe with image paths and class labels."""
    rows: List[Dict[str, str]] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No valid image files were found in {data_dir}")

    return df


def verify_images(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Verify that image files can be opened.

    Invalid or corrupted images are removed from the returned dataframe.
    """
    valid_rows = []
    invalid_files: List[str] = []

    for _, row in df.iterrows():
        image_path = row["image_path"]

        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_rows.append(row)
        except Exception:
            invalid_files.append(image_path)

    clean_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    return clean_df, invalid_files


def create_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: Tuple[int, int],
    batch_size: int,
) -> Tuple[tf.keras.preprocessing.image.DataFrameIterator, ...]:
    """Create Keras image data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="label",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    valid_gen = eval_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col="image_path",
        y_col="label",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col="label",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, valid_gen, test_gen


def build_custom_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build a compact custom CNN model."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def compute_multiclass_specificity(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute one-vs-rest TP, TN, FP, FN, sensitivity, and specificity."""
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = tp / np.maximum(tp + fn, 1)
    specificity = tn / np.maximum(tn + fp, 1)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation curves."""
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history_dict["loss"], label="Training loss")
    axes[0, 0].plot(epochs, history_dict["val_loss"], label="Validation loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history_dict["accuracy"], label="Training accuracy")
    axes[0, 1].plot(epochs, history_dict["val_accuracy"], label="Validation accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history_dict["precision"], label="Training precision")
    axes[1, 0].plot(epochs, history_dict["val_precision"], label="Validation precision")
    axes[1, 0].set_title("Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history_dict["recall"], label="Training recall")
    axes[1, 1].plot(epochs, history_dict["val_recall"], label="Validation recall")
    axes[1, 1].set_title("Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    fig.suptitle("Model Training Metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    test_gen: tf.keras.preprocessing.image.DataFrameIterator,
    output_dir: Path,
) -> Dict[str, object]:
    """Evaluate the model and save reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    cm = confusion_matrix(y_true, y_pred)
    one_vs_rest = compute_multiclass_specificity(cm)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "class_order": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "per_class_sensitivity": {
            cls: float(value)
            for cls, value in zip(class_names, one_vs_rest["sensitivity"])
        },
        "per_class_specificity": {
            cls: float(value)
            for cls, value in zip(class_names, one_vs_rest["specificity"])
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "classification_report.csv")
    save_confusion_matrix_plot(cm, class_names, output_dir / "confusion_matrix.png")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a custom CNN for brain tumor MRI classification."
    )

    parser.add_argument("--train-dir", type=Path, default=Path("data/Training"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/Testing"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_dataframe(args.train_dir)
    test_df = build_dataframe(args.test_dir)

    train_df, invalid_train = verify_images(train_df)
    test_df, invalid_test = verify_images(test_df)

    if invalid_train or invalid_test:
        invalid_report = {
            "invalid_train_files": invalid_train,
            "invalid_test_files": invalid_test,
        }
        (args.output_dir / "invalid_files.json").write_text(
            json.dumps(invalid_report, indent=2),
            encoding="utf-8",
        )

    train_df_final, valid_df = train_test_split(
        train_df,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    image_size = (args.image_size, args.image_size)

    train_gen, valid_gen, test_gen = create_generators(
        train_df=train_df_final,
        valid_df=valid_df,
        test_df=test_df,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    num_classes = len(train_gen.class_indices)
    model = build_custom_cnn(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes,
    )

    model.summary()

    checkpoint_path = args.model_dir / "best_brain_tumor_cnn.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        TensorBoard(log_dir=str(args.log_dir)),
    ]

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.model_dir / "final_brain_tumor_cnn.keras")
    save_training_curves(history, args.output_dir / "training_curves.png")

    metrics = evaluate_model(model, test_gen, args.output_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
