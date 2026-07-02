"""Inference script for screen photo detection.

Usage:
    python src/predict.py image.jpg

Outputs a single probability between 0 and 1:
    0.0 = Real photo
    1.0 = Photo of a screen
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as 'python src/predict.py'
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from PIL import Image

from src.data.transformer import get_val_transform
from src.models.mobilenet import ScreenDetectorMobileNet
from src.utils.helpers import get_project_root


def predict(image_path: str) -> float:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image.

    Returns:
        Probability that the image is a photo of a screen (0.0 to 1.0).
    """
    root = get_project_root()
    weights_path = root / "weights" / "best_model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Train first.")

    device = torch.device("cpu")

    model = ScreenDetectorMobileNet(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    transform = get_val_transform()
    tensor = transform(image=image)["image"].unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    return probability


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    prob = predict(image_path)
    print(f"{prob:.2f}")
