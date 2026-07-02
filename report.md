# Screen Photo Detector — Technical Report

## 1. Problem Statement

Build a binary classifier to distinguish between:
- **Class 0 (Real)**: Genuine photographs taken directly with a camera
- **Class 1 (Screen)**: Photos of screens (phone, laptop, monitor, tablet, printed image)

### Constraints
- Model size must be small enough for mobile deployment
- Inference latency must be extremely low
- Must achieve >95% accuracy on unseen data
- Must generalize to diverse screen types and lighting conditions

## 2. Dataset Analysis

### Overview
| Property | Value |
|----------|-------|
| Total images | 155 |
| Real photos | 52 (33.5%) |
| Screen photos | 103 (66.5%) |
| Resolution | 3072×3072 |
| Class ratio | 1:1.98 |

### Key Observations
1. **Class imbalance**: Screen photos outnumber real photos ~2:1. Addressed via `pos_weight` in the loss function.
2. **High resolution**: Images are 3072×3072 — we resize to 224×224 for MobileNetV3, which preserves global structure while being efficient.
3. **Small dataset**: Only 155 images total. Transfer learning is essential — training from scratch would overfit immediately.

### Split Strategy
- **70/15/15** stratified split (train/validation/test)
- Stratification preserves class ratios across all splits
- Random state fixed for reproducibility

## 3. Methodology

### 3.1 Model Architecture

**MobileNetV3-Small** was selected for the following reasons:

| Criterion | MobileNetV3-Small | ResNet-18 | EfficientNet-B0 |
|-----------|-------------------|-----------|-----------------|
| Parameters | 928K | 11.7M | 5.3M |
| Model size | ~3.7 MB | ~47 MB | ~21 MB |
| Mobile-optimized | ✅ Yes | ❌ No | ⚠️ Partially |
| ImageNet Top-1 | 67.7% | 69.8% | 77.1% |
| Inference speed | Very fast | Moderate | Moderate |

MobileNetV3-Small offers the best tradeoff between accuracy and efficiency for this binary task. It was specifically designed for mobile deployment with hardware-aware neural architecture search.

**Custom head**:
```
Features(576 channels) → AdaptiveAvgPool2d(1) → Flatten → Dropout(0.2) → Linear(576, 1)
```

### 3.2 Training Strategy

#### Two-Stage Transfer Learning

**Stage 1 — Head Training** (backbone frozen):
- Only 577 trainable parameters
- Learning rate: 1e-3
- Fast convergence to establish good decision boundary
- Prevents catastrophic forgetting of ImageNet features

**Stage 2 — Fine-Tuning** (last 4 blocks unfrozen):
- ~737K trainable parameters
- Learning rate: 1e-4 (10x lower to preserve backbone features)
- Allows model to adapt frequency-domain features to screen detection

#### Regularization
- **Early stopping**: Patience=7 on validation loss
- **Learning rate scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Weight decay**: 1e-4
- **Dropout**: 0.2 in classifier head
- **Class weighting**: pos_weight = num_real / num_screen ≈ 0.505

### 3.3 Data Augmentation

Augmentations are specifically designed to simulate real-world variance **without destroying screen-specific artifacts**:

| Augmentation | Probability | Rationale |
|-------------|-------------|-----------|
| Resize(224) | 1.0 | MobileNet input size |
| HorizontalFlip | 0.5 | Invariant to left-right orientation |
| BrightnessContrast | 0.5 | Lighting variation |
| HueSaturationValue | 0.4 | Color temperature variation |
| GaussNoise | 0.3 | Camera sensor noise |
| GaussianBlur(3-5) | 0.2 | Slight defocus |
| ImageCompression(70-100) | 0.4 | JPEG artifacts |
| Affine | 0.5 | Slight rotation/scale/translate |
| Perspective | 0.3 | Viewing angle changes |

**Deliberately excluded**: Vertical flips, 90° rotations, heavy blur, aggressive compression — these destroy moiré patterns and screen textures.

### 3.4 Handcrafted Features

Additional interpretable features complement the CNN:

| Feature | Detection Target |
|---------|-----------------|
| FFT high-frequency energy | Moiré patterns from screen pixel grids |
| Laplacian variance | Sharpness differences (lens-screen interaction) |
| Edge density | Uniform edge patterns from display pixels |
| LBP texture histogram | Sub-pixel rendering artifacts |
| Glare ratio | Specular reflections on screen glass |
| Color uniformity | Uniform backlight illumination |

These features can be used for model interpretability, ensemble methods, or as a lightweight fallback classifier.

### 3.5 Loss Function

**BCEWithLogitsLoss** with class weighting:
- Combines sigmoid + BCE in a single numerically stable operation
- `pos_weight ≈ 0.505` downweights the majority class (screen)
- Ensures the model doesn't bias toward always predicting "screen"

## 4. Results

### Training Summary

| Stage | Epochs Run | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |
|-------|-----------|-----------------|----------------|---------------|--------------|
| Stage 1 (Head only) | 15 | 0.3203 | 84.1% | 0.2822 | 83.3% |
| Stage 2 (Fine-tune) | 30 | 0.0489 | 97.2% | 0.0495 | 95.8% |

### Final Validation Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.83%** |
| **Precision** | **100.0%** |
| **Recall** | **93.75%** |
| **F1 Score** | **96.77%** |
| **ROC-AUC** | **100.0%** |
| Loss | 0.0495 |

### Test Set Evaluation (24 images: 8 real, 16 screen)

| Metric | Value |
|--------|-------|
| **Accuracy** | **100.0%** |
| **Precision** | **100.0%** |
| **Recall** | **100.0%** |
| **F1 Score** | **100.0%** |
| **ROC-AUC** | **100.0%** |

Confusion Matrix:
```
              Predicted
              Real  Screen
Actual Real  [  8     0  ]
       Screen[  0    16  ]
```

Zero misclassifications on the held-out test set.

### Key Observations
1. **Stage 1** converged steadily from 41% to 83% validation accuracy in 15 epochs
2. **Stage 2** pushed accuracy to 95.8% validation / 100% test, with loss dropping from 0.28 to 0.05
3. Early stopping triggered at epoch 30 (Stage 2) — patience=7 after best at epoch 23
4. ReduceLROnPlateau reduced LR from 1e-4 to 5e-5 at epoch 27
5. Perfect precision (no false positives) indicates the model is conservative about labeling real photos as screen

### Model Size

| Format | Size |
|--------|------|
| PyTorch (.pth) | 3.8 MB |
| TorchScript (.pt) | 4.1 MB |
| ONNX (.onnx) | ~3.7 MB |

### Output Artifacts
- Training curves: `outputs/figures/training_curves.png`
- Confusion matrix: `outputs/figures/confusion_matrix.png`
- ROC curve: `outputs/figures/roc_curve.png`
- Training log: `outputs/logs/training_log.json`
- Test predictions: `outputs/predictions/test_predictions.json`

## 5. Model Export

The trained model is exported in three formats:

| Format | File | Use Case |
|--------|------|----------|
| PyTorch | `weights/best_model.pth` | Python inference |
| TorchScript | `weights/model_scripted.pt` | C++ / mobile deployment |
| ONNX | `weights/model.onnx` | Cross-platform inference |

## 6. Reproducibility

- Fixed random seed (42) across Python, NumPy, PyTorch, CUDA
- Deterministic cuDNN operations
- Stratified splitting ensures consistent class ratios
- All hyperparameters logged to `outputs/logs/training_log.json`

## 7. Limitations & Future Work

### Current Limitations
- Small dataset (155 images) — model may not generalize to all screen types
- No ensemble with handcrafted features (features are extracted but not fused)
- CPU-only training may be slow for larger datasets

### Potential Improvements
1. **Feature fusion**: Concatenate CNN embeddings with handcrafted features
2. **Test-time augmentation (TTA)**: Average predictions over augmented copies
3. **Knowledge distillation**: Compress a larger teacher model
4. **Larger dataset**: Collect more diverse screen photos
5. **Quantization**: INT8 quantization for even faster mobile inference
