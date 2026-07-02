"""MobileNetV3-Small based binary classifier for screen photo detection.

Uses transfer learning: a pretrained MobileNetV3-Small backbone with a
custom single-output classification head for binary prediction.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ScreenDetectorMobileNet(nn.Module):
    """Binary classifier built on MobileNetV3-Small.

    Architecture:
        - MobileNetV3-Small backbone (pretrained on ImageNet)
        - Custom classifier: AdaptiveAvgPool2d → Flatten → Dropout → Linear(576, 1)
        - Output: raw logit (apply sigmoid for probability)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)

        # Extract feature layers (everything before the classifier)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small last conv outputs 576 channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(576, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Logits tensor of shape (B, 1).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Stage 1 (head-only training)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_blocks_to_unfreeze: int = 4) -> None:
        """Unfreeze the last N inverted residual blocks for Stage 2 fine-tuning.

        MobileNetV3-Small has 13 feature blocks (indices 0-12).
        By default, unfreezes the last 4 blocks (indices 9-12).
        """
        total_blocks = len(self.features)
        freeze_until = max(0, total_blocks - num_blocks_to_unfreeze)

        for i, block in enumerate(self.features):
            for param in block.parameters():
                param.requires_grad = (i >= freeze_until)

    def get_num_params(self) -> dict:
        """Return count of total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
