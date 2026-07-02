"""Visualization utilities for training curves, confusion matrices, and ROC curves."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                 'train_acc', 'val_acc', each mapping to a list of per-epoch values.
        save_path: File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix from sklearn.
        save_path: File path to save the figure.
        class_names: Labels for the axes. Defaults to ['Real', 'Screen'].
    """
    if class_names is None:
        class_names = ["Real", "Screen"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> None:
    """Plot ROC curve with AUC annotation.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        save_path: File path to save the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_samples(
    images: List[np.ndarray],
    labels: List[int],
    predictions: List[float],
    save_path: Path,
    num_samples: int = 8,
) -> None:
    """Plot a grid of sample images with their true labels and predicted probabilities.

    Args:
        images: List of RGB images as numpy arrays.
        labels: Ground truth labels (0=Real, 1=Screen).
        predictions: Predicted probabilities.
        save_path: File path to save the figure.
        num_samples: Number of samples to display (max).
    """
    n = min(num_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    label_map = {0: "Real", 1: "Screen"}

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        true_label = label_map[labels[i]]
        pred_prob = predictions[i]
        pred_label = label_map[int(pred_prob >= 0.5)]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.3f})", color=color)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
