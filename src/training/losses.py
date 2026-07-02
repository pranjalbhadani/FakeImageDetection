"""Loss functions for the screen photo detector.

Uses BCEWithLogitsLoss with pos_weight to handle class imbalance.
The model outputs raw logits; BCEWithLogitsLoss applies sigmoid internally,
which is numerically more stable than BCELoss + manual sigmoid.
"""

import torch
import torch.nn as nn


def get_criterion(
    pos_weight: torch.Tensor,
    device: torch.device,
) -> nn.BCEWithLogitsLoss:
    """Create a weighted binary cross-entropy loss.

    Args:
        pos_weight: Weight for the positive class (screen).
                    Computed as num_negative / num_positive.
        device: Target device for the loss tensor.

    Returns:
        Configured BCEWithLogitsLoss instance.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
