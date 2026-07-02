"""Handcrafted computer vision features for screen photo detection.

These features capture physical artifacts of photographed screens:
- Moiré patterns appear in the frequency domain (FFT)
- Screen pixels create uniform edge patterns (edge density)
- Screen glare produces near-saturated regions
- Display sub-pixel grids create distinctive textures (LBP)
"""

from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift


def extract_fft_high_freq_energy(gray: np.ndarray) -> float:
    """Ratio of high-frequency energy in the Fourier spectrum.

    Screen photos exhibit periodic patterns (moiré) that concentrate
    energy at specific high frequencies.
    """
    f_transform = fft2(gray.astype(np.float32))
    f_shifted = fftshift(f_transform)
    magnitude = np.abs(f_shifted)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Define low-frequency region as central 10% of the spectrum
    radius = int(min(rows, cols) * 0.1)
    mask = np.zeros_like(magnitude, dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    mask[(y - crow) ** 2 + (x - ccol) ** 2 <= radius ** 2] = True

    total_energy = np.sum(magnitude ** 2) + 1e-10
    low_freq_energy = np.sum(magnitude[mask] ** 2)
    high_freq_ratio = 1.0 - (low_freq_energy / total_energy)

    return float(high_freq_ratio)


def extract_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — measures image sharpness.

    Screen photos often have slightly different sharpness characteristics
    due to lens-screen interaction and refocusing.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def extract_edge_density(gray: np.ndarray) -> float:
    """Ratio of edge pixels detected by Canny.

    Screen photos may have more uniform edge density from pixel grids.
    """
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size)


def extract_lbp_histogram(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute simplified Local Binary Pattern histogram.

    Screens have distinctive micro-textures from sub-pixel rendering
    that LBP can capture.
    """
    # Simplified uniform LBP implementation
    rows, cols = gray.shape
    lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

    # 8-connected neighborhood
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:rows - 1 + dy, 1 + dx:cols - 1 + dx]
        center = gray[1:rows - 1, 1:cols - 1]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist.astype(np.float32)


def extract_glare_ratio(image_rgb: np.ndarray) -> float:
    """Percentage of near-saturated (glare) pixels.

    Screen photos frequently have specular reflections that appear
    as bright spots near pixel value 255.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
    threshold = 240
    glare_pixels = np.sum(gray >= threshold)
    return float(glare_pixels / gray.size)


def extract_color_uniformity(image_rgb: np.ndarray) -> float:
    """Standard deviation of color channels.

    Screen photos tend to have more uniform color regions compared
    to natural photos.
    """
    std_per_channel = np.std(image_rgb, axis=(0, 1))
    return float(np.mean(std_per_channel))


def extract_features(image_path: str) -> Dict[str, float]:
    """Extract all handcrafted features from a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary mapping feature names to their scalar values,
        plus the LBP histogram as a separate entry.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize for consistent feature extraction
    resized = cv2.resize(image_rgb, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    lbp_hist = extract_lbp_histogram(gray)

    features: Dict[str, float] = {
        "fft_high_freq_energy": extract_fft_high_freq_energy(gray),
        "laplacian_variance": extract_laplacian_variance(gray),
        "edge_density": extract_edge_density(gray),
        "glare_ratio": extract_glare_ratio(resized),
        "color_uniformity": extract_color_uniformity(resized),
    }

    # Add LBP histogram bins as individual features
    for i, val in enumerate(lbp_hist):
        features[f"lbp_{i}"] = float(val)

    return features


def extract_batch_features(image_paths: List[str]) -> pd.DataFrame:
    """Extract features from a batch of images.

    Args:
        image_paths: List of image file paths.

    Returns:
        DataFrame where each row is one image and columns are feature names.
    """
    all_features = []
    for path in image_paths:
        try:
            feats = extract_features(path)
            feats["image_path"] = path
            all_features.append(feats)
        except ValueError as e:
            print(f"Warning: Skipping {path} — {e}")

    return pd.DataFrame(all_features)
