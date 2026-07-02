"""Utility helpers for reproducibility, device selection, and class weighting."""

import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file until it finds the directory containing 'dataset/'.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "dataset").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no 'dataset/' folder found).")


def compute_class_weights(labels: Union[List[int], np.ndarray]) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight = num_negative / num_positive
    For our dataset: real=0 (negative), screen=1 (positive).
    With 52 real and 103 screen images, pos_weight ≈ 0.505.
    This downweights the majority class to balance the loss.
    """
    labels = np.array(labels)
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)

    if num_positive == 0:
        raise ValueError("No positive samples found in labels.")

    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32)
    return pos_weight
