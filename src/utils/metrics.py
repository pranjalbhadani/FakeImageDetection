"""Metrics computation for binary classification evaluation."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all binary classification metrics from true labels and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        threshold: Decision threshold for converting probabilities to class predictions.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and roc_auc.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC requires both classes to be present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix for binary classification.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        2x2 confusion matrix as numpy array.
        [[TN, FP],
         [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred)
