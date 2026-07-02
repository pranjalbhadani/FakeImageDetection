"""Model evaluation and 5-fold stratified cross-validation.

Provides:
- Single model evaluation on a test set with all metrics
- 5-fold stratified cross-validation with aggregated results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ScreenPhotoDataset, load_dataset
from src.data.transformer import get_train_transform, get_val_transform
from src.models.mobilenet import ScreenDetectorMobileNet
from src.split import create_splits
from src.training.losses import get_criterion
from src.training.train import train_one_epoch, validate
from src.utils.helpers import compute_class_weights, get_device, get_project_root, set_seed
from src.utils.metrics import compute_confusion_matrix, compute_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_prediction_samples,
    plot_roc_curve,
)

from sklearn.model_selection import StratifiedKFold


def evaluate_model(
    model_path: str,
    test_paths: List[str],
    test_labels: List[int],
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Evaluate a trained model on a test set.

    Args:
        model_path: Path to the saved model weights (.pth).
        test_paths: List of test image paths.
        test_labels: Ground truth labels for test images.
        device: Compute device (auto-detected if None).

    Returns:
        Dictionary with all evaluation metrics, predictions, and raw arrays.
    """
    if device is None:
        device = get_device()

    root = get_project_root()

    model = ScreenDetectorMobileNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_dataset = ScreenPhotoDataset(test_paths, test_labels, transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_true, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred)

    # Save predictions
    predictions_dir = root / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions_data = []
    for path, label, prob in zip(test_paths, all_labels, all_probs):
        predictions_data.append({
            "image_path": str(path),
            "true_label": label,
            "predicted_prob": round(prob, 6),
            "predicted_label": int(prob >= 0.5),
        })

    with open(predictions_dir / "test_predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)

    # Save plots
    figures_dir = root / "outputs" / "figures"
    plot_confusion_matrix(cm, figures_dir / "test_confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, figures_dir / "test_roc_curve.png")

    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def run_cross_validation(
    dataset_path: Optional[str] = None,
    n_folds: int = 5,
    config: Optional[Dict] = None,
) -> Dict[str, object]:
    """Run stratified k-fold cross-validation.

    Trains a fresh model on each fold and aggregates metrics across folds
    to estimate true model performance.

    Args:
        dataset_path: Path to dataset directory (auto-detected if None).
        n_folds: Number of CV folds.
        config: Training hyperparameters override.

    Returns:
        Dictionary with per-fold and aggregated metrics.
    """
    cfg = {
        "seed": 42,
        "batch_size": 16,
        "stage1_epochs": 10,
        "stage1_lr": 1e-3,
        "stage2_epochs": 20,
        "stage2_lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "num_workers": 0,
        "unfreeze_blocks": 4,
        "dropout": 0.2,
    }
    if config:
        cfg.update(config)

    set_seed(cfg["seed"])
    device = get_device()
    root = get_project_root()

    if dataset_path is None:
        dataset_path = str(root / "dataset")

    paths, labels = load_dataset(dataset_path)
    paths = np.array(paths)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])

    fold_metrics: List[Dict[str, float]] = []

    print(f"\n{'='*60}")
    print(f"Running {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        train_paths_fold = paths[train_idx].tolist()
        train_labels_fold = labels[train_idx].tolist()
        val_paths_fold = paths[val_idx].tolist()
        val_labels_fold = labels[val_idx].tolist()

        train_ds = ScreenPhotoDataset(train_paths_fold, train_labels_fold, transform=get_train_transform())
        val_ds = ScreenPhotoDataset(val_paths_fold, val_labels_fold, transform=get_val_transform())

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

        model = ScreenDetectorMobileNet(pretrained=True, dropout=cfg["dropout"]).to(device)
        pos_weight = compute_class_weights(train_labels_fold)
        criterion = get_criterion(pos_weight, device)

        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        # Stage 1
        model.freeze_backbone()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["stage1_lr"], weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, cfg["stage1_epochs"] + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_result = validate(model, val_loader, criterion, device)
            scheduler.step(val_result["loss"])

            if val_result["loss"] < best_loss:
                best_loss = val_result["loss"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # Stage 2
        if best_state:
            model.load_state_dict(best_state)
        model.unfreeze_backbone(cfg["unfreeze_blocks"])
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["stage2_lr"], weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        patience_counter = 0

        for epoch in range(1, cfg["stage2_epochs"] + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_result = validate(model, val_loader, criterion, device)
            scheduler.step(val_result["loss"])

            if val_result["loss"] < best_loss:
                best_loss = val_result["loss"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # Evaluate fold
        if best_state:
            model.load_state_dict(best_state)
        final_metrics = validate(model, val_loader, criterion, device)

        fold_result = {
            k: v for k, v in final_metrics.items()
            if k not in ("y_true", "y_prob")
        }
        fold_metrics.append(fold_result)

        print(f"  Fold {fold} — Acc: {fold_result['accuracy']:.4f}, "
              f"F1: {fold_result['f1']:.4f}, AUC: {fold_result['roc_auc']:.4f}")

    # Aggregate results
    aggregated = {}
    metric_names = fold_metrics[0].keys()
    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        aggregated[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": values,
        }

    print(f"\n{'='*60}")
    print("Cross-Validation Results (mean ± std)")
    print(f"{'='*60}")
    for name, stats in aggregated.items():
        print(f"  {name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save CV results
    logs_dir = root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "cv_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


if __name__ == "__main__":
    root = get_project_root()
    model_path = root / "weights" / "best_model.pth"

    if model_path.exists():
        # Load test split info
        paths, labels = load_dataset(root / "dataset")
        _, _, _, _, test_paths, test_labels = create_splits(paths, labels)
        evaluate_model(str(model_path), test_paths, test_labels)
    else:
        print("No trained model found. Run training first.")
        print("Running cross-validation instead...")
        run_cross_validation()
