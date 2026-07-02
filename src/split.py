"""Stratified train/validation/test splitting for the screen photo dataset."""

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    paths: List[str],
    labels: List[int],
    random_state: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Split image paths and labels into stratified train/val/test sets.

    Split ratio: ~70% train, ~15% validation, ~15% test.
    Stratification ensures each split preserves the class distribution.

    Args:
        paths: List of image file paths.
        labels: Corresponding binary labels.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels,
                  test_paths, test_labels).
    """
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=random_state,
    )

    # 0.1765 of 85% ≈ 15% of total
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.1765,
        stratify=train_val_labels,
        random_state=random_state,
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    )