"""Training orchestrator for two-stage transfer learning.

Stage 1: Freeze backbone, train classifier head only (fast convergence).
Stage 2: Unfreeze last N backbone blocks, fine-tune end-to-end (slow, careful).

Features:
- Early stopping on validation loss
- ReduceLROnPlateau learning rate scheduler
- Mixed precision training (AMP) when GPU is available
- Saves best model by validation loss
- Logs and plots training history
- Exports to TorchScript and ONNX after training
"""

import json
import time
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
from src.utils.helpers import compute_class_weights, get_device, get_project_root, set_seed
from src.utils.metrics import compute_confusion_matrix, compute_metrics
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_curves


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns:
        Dictionary with 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Run validation and compute all metrics.

    Returns:
        Dictionary with loss, accuracy, precision, recall, f1, roc_auc,
        and raw arrays y_true / y_prob.
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_t = labels.float().to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels_t)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = running_loss / len(y_true)
    metrics["y_true"] = y_true
    metrics["y_prob"] = y_prob

    return metrics


def train_model(config: Optional[Dict] = None) -> Dict[str, object]:
    """Full two-stage training orchestrator.

    Args:
        config: Optional dictionary to override default hyperparameters.

    Returns:
        Dictionary with final metrics and training history.
    """
    # Default configuration
    cfg = {
        "seed": 42,
        "batch_size": 16,
        "stage1_epochs": 15,
        "stage1_lr": 1e-3,
        "stage2_epochs": 30,
        "stage2_lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 7,
        "num_workers": 0,
        "unfreeze_blocks": 4,
        "dropout": 0.2,
    }
    if config:
        cfg.update(config)

    set_seed(cfg["seed"])
    device = get_device()
    root = get_project_root()

    print(f"Device: {device}")
    print(f"Project root: {root}")

    # Load and split data
    paths, labels = load_dataset(root / "dataset")
    print(f"Dataset: {len(paths)} images ({sum(1 for l in labels if l == 0)} real, "
          f"{sum(1 for l in labels if l == 1)} screen)")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_splits(
        paths, labels, random_state=cfg["seed"]
    )
    print(f"Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Build datasets and loaders
    train_dataset = ScreenPhotoDataset(train_paths, train_labels, transform=get_train_transform())
    val_dataset = ScreenPhotoDataset(val_paths, val_labels, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    # Model, loss, optimizer
    model = ScreenDetectorMobileNet(pretrained=True, dropout=cfg["dropout"]).to(device)
    pos_weight = compute_class_weights(train_labels)
    criterion = get_criterion(pos_weight, device)

    # Prepare output directories
    figures_dir = root / "outputs" / "figures"
    logs_dir = root / "outputs" / "logs"
    weights_dir = root / "weights"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = weights_dir / "best_model.pth"

    # ==================== STAGE 1: Train Head Only ====================
    print("\n" + "=" * 60)
    print("STAGE 1: Training classifier head (backbone frozen)")
    print("=" * 60)

    model.freeze_backbone()
    params = model.get_num_params()
    print(f"Parameters — total: {params['total']:,}, trainable: {params['trainable']:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:2d}/{cfg['stage1_epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ==================== STAGE 2: Fine-Tune Backbone ====================
    print("\n" + "=" * 60)
    print(f"STAGE 2: Fine-tuning last {cfg['unfreeze_blocks']} backbone blocks")
    print("=" * 60)

    # Load best Stage 1 model
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.unfreeze_backbone(num_blocks_to_unfreeze=cfg["unfreeze_blocks"])
    params = model.get_num_params()
    print(f"Parameters — total: {params['total']:,}, trainable: {params['trainable']:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage2_lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    patience_counter = 0

    for epoch in range(1, cfg["stage2_epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        current_lr = optimizer.param_groups[0]["lr"]
        total_epoch = len(history["train_loss"])
        print(
            f"  Epoch {epoch:2d}/{cfg['stage2_epochs']} (total: {total_epoch}) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ==================== Final Evaluation ====================
    print("\n" + "=" * 60)
    print("Final evaluation on validation set")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    final_val = validate(model, val_loader, criterion, device)

    print(f"  Val Loss:      {final_val['loss']:.4f}")
    print(f"  Val Accuracy:  {final_val['accuracy']:.4f}")
    print(f"  Val Precision: {final_val['precision']:.4f}")
    print(f"  Val Recall:    {final_val['recall']:.4f}")
    print(f"  Val F1:        {final_val['f1']:.4f}")
    print(f"  Val ROC-AUC:   {final_val['roc_auc']:.4f}")

    # ==================== Save Plots & Logs ====================
    plot_training_curves(history, figures_dir / "training_curves.png")

    y_pred = (final_val["y_prob"] >= 0.5).astype(int)
    cm = compute_confusion_matrix(final_val["y_true"], y_pred)
    plot_confusion_matrix(cm, figures_dir / "confusion_matrix.png")
    plot_roc_curve(final_val["y_true"], final_val["y_prob"], figures_dir / "roc_curve.png")

    # Save training log
    log_data = {
        "config": cfg,
        "history": {k: v for k, v in history.items()},
        "final_val_metrics": {
            k: v for k, v in final_val.items()
            if k not in ("y_true", "y_prob")
        },
        "best_val_loss": best_val_loss,
    }
    with open(logs_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # ==================== Export Models ====================
    print("\nExporting models...")

    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 3, 224, 224)

    # TorchScript
    try:
        scripted = torch.jit.trace(model, dummy_input)
        scripted.save(str(weights_dir / "model_scripted.pt"))
        print(f"  TorchScript saved: {weights_dir / 'model_scripted.pt'}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")

    # ONNX
    try:
        import os
        os.environ["PYTHONIOENCODING"] = "utf-8"
        torch.onnx.export(
            model,
            dummy_input,
            str(weights_dir / "model.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
        )
        print(f"  ONNX saved: {weights_dir / 'model.onnx'}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    print("\nTraining complete!")
    print(f"Best model: {best_model_path}")

    return {
        "history": history,
        "final_metrics": {
            k: v for k, v in final_val.items()
            if k not in ("y_true", "y_prob")
        },
        "best_val_loss": best_val_loss,
        "test_paths": test_paths,
        "test_labels": test_labels,
    }


if __name__ == "__main__":
    results = train_model()
