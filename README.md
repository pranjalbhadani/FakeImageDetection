# Screen Photo Detector

A lightweight binary image classifier that determines whether an input image is a **real photograph** or a **photo of a screen** (phone, laptop, monitor, tablet, printed image).

## Key Features

- **MobileNetV3-Small** backbone — only 928K parameters, designed for mobile deployment
- **Two-stage transfer learning** — frozen backbone → fine-tuned end-to-end
- **>95% target accuracy** on unseen images
- **ONNX + TorchScript export** for production deployment
- **Handcrafted CV features** (FFT, LBP, edge density) for interpretability

## Quick Start

### Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Prediction

```bash
python src/predict.py image.jpg
```

Output: a single probability (0.0 = real, 1.0 = screen photo)

```
0.94
```

### Training

```bash
python -m src.training.train
```

This runs two-stage training:
1. **Stage 1**: Trains classifier head with frozen backbone (~15 epochs)
2. **Stage 2**: Fine-tunes last 4 backbone blocks (~30 epochs)

Best model is saved to `weights/best_model.pth`.

### Evaluation

```bash
python -m src.training.evaluate
```

Runs evaluation on the held-out test set and generates plots.

### Cross-Validation

The evaluation module also supports 5-fold stratified cross-validation:

```python
from src.training.evaluate import run_cross_validation
results = run_cross_validation()
```

## Project Structure

```
screen-photo-detector/
├── dataset/
│   ├── real/              # 52 real photographs
│   └── screen/            # 103 screen photos
├── outputs/
│   ├── figures/           # Training curves, confusion matrix, ROC
│   ├── logs/              # Training logs (JSON)
│   └── predictions/       # Test set predictions
├── src/
│   ├── data/
│   │   ├── dataset.py     # PyTorch Dataset + data loader
│   │   └── transformer.py # Albumentations augmentation pipelines
│   ├── models/
│   │   ├── mobilenet.py   # MobileNetV3-Small classifier
│   │   └── handcrafted_features.py  # FFT, LBP, edge, glare features
│   ├── training/
│   │   ├── train.py       # Two-stage training orchestrator
│   │   ├── evaluate.py    # Evaluation + cross-validation
│   │   └── losses.py      # Weighted BCE loss
│   ├── utils/
│   │   ├── helpers.py     # Seed, device, class weights
│   │   ├── metrics.py     # Accuracy, F1, ROC-AUC, confusion matrix
│   │   └── visualization.py  # Training curves, plots
│   ├── predict.py         # CLI inference script
│   └── split.py           # Stratified train/val/test splitting
├── weights/               # Saved model weights
├── requirements.txt
├── README.md
└── report.md
```

## Architecture

### Model
- **Backbone**: MobileNetV3-Small (pretrained on ImageNet)
- **Head**: AdaptiveAvgPool2d → Dropout(0.2) → Linear(576, 1)
- **Output**: Raw logit → Sigmoid → Probability
- **Total parameters**: ~928K
- **Input size**: 224×224 RGB

### Training Strategy
- **Loss**: BCEWithLogitsLoss with pos_weight for class imbalance
- **Stage 1**: Freeze backbone, train head (LR=1e-3, 15 epochs)
- **Stage 2**: Unfreeze last 4 blocks, fine-tune (LR=1e-4, 30 epochs)
- **Early stopping**: Patience=7 on validation loss
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Mixed precision**: Automatic when GPU is available

### Data Augmentation
Augmentations preserve screen-specific artifacts (moiré, pixel grids):
- Resize, Horizontal Flip, Brightness/Contrast, Color Jitter
- Gaussian Noise, Light Blur, Affine transforms, Perspective

### Handcrafted Features
Additional interpretable features in `handcrafted_features.py`:
- FFT high-frequency energy (moiré detection)
- Laplacian variance (sharpness)
- Edge density (pixel grid patterns)
- LBP texture histogram (sub-pixel rendering)
- Glare ratio (screen reflections)
- Color uniformity (uniform backlight)

## Dataset

| Class | Count | Label |
|-------|-------|-------|
| Real  | 52    | 0     |
| Screen| 103   | 1     |
| **Total** | **155** | — |

- Resolution: 3072×3072
- Split: 70% train / 15% val / 15% test (stratified)

## Requirements

- Python 3.10+
- PyTorch 2.3+
- See `requirements.txt` for full list

## License

This project is for educational purposes.
