"""Image augmentation pipelines for training and validation.

Training transforms are designed specifically for screen-photo detection:
- Preserves moiré patterns and screen textures (no heavy blur, no vertical flips)
- Adds realistic variations (brightness, contrast, noise, perspective)
"""

from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE: int = 224


def get_train_transform() -> A.Compose:
    """Return the training augmentation pipeline.

    Augmentations are chosen to simulate real-world variance without
    destroying screen-specific artifacts like moiré, pixel grids, or glare.
    """
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.4,
            ),
            A.GaussNoise(
                std_range=(0.02, 0.08),
                p=0.3,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),
            A.ImageCompression(
                quality_range=(70, 100),
                p=0.4,
            ),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.Perspective(
                scale=(0.02, 0.05),
                p=0.3,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_val_transform() -> A.Compose:
    """Return the validation/inference augmentation pipeline.

    Only deterministic transforms: resize and normalize.
    """
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


# Module-level instances for backward compatibility
train_transform: A.Compose = get_train_transform()
val_transform: A.Compose = get_val_transform()