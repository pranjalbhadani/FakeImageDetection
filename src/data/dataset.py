"""Dataset class and data loading utilities for the screen photo detector."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ScreenPhotoDataset(Dataset):
    """PyTorch Dataset for binary screen photo detection.

    Each sample is an image loaded as RGB, optionally transformed via
    an Albumentations pipeline, and paired with a binary label.

    Labels:
        0 = Real photo
        1 = Photo of a screen
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[object] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = self.labels[idx]
        return image, label

    def __repr__(self) -> str:
        n_real = sum(1 for l in self.labels if l == 0)
        n_screen = sum(1 for l in self.labels if l == 1)
        return (
            f"ScreenPhotoDataset(total={len(self)}, "
            f"real={n_real}, screen={n_screen}, "
            f"transform={'yes' if self.transform else 'no'})"
        )


def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
    """Scan the dataset directory and return image paths with labels.

    Expected structure:
        dataset_path/
            real/   -> label 0
            screen/ -> label 1

    Args:
        dataset_path: Path to the root dataset directory.

    Returns:
        Tuple of (image_paths, labels).
    """
    dataset_path = Path(dataset_path)
    image_paths: List[str] = []
    labels: List[int] = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img in sorted((dataset_path / "real").glob("*")):
        if img.suffix.lower() in valid_extensions:
            image_paths.append(str(img))
            labels.append(0)

    for img in sorted((dataset_path / "screen").glob("*")):
        if img.suffix.lower() in valid_extensions:
            image_paths.append(str(img))
            labels.append(1)

    return image_paths, labels