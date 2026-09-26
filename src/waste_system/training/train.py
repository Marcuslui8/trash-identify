from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from waste_system.training.dataset import (
    CLASS_NAMES,
    WasteCsvDataset,
    balanced_class_weights,
    build_loader,
    build_transform,
)
from waste_system.training.metrics import macro_f1
from waste_system.training.model import build_model, freeze_features, unfreeze_tail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EfficientNet-B0 waste classifier")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/realwaste_split.csv"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    losses = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses += float(loss.item()) * images.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    accuracy = sum(a == b for a, b in zip(targets, predictions, strict=True)) / len(targets)
    return losses / len(targets), accuracy, macro_f1(targets, predictions, len(CLASS_NAMES))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    project_root = args.project_root.resolve()
    annotations = (project_root / args.annotations).resolve() if not args.annotations.is_absolute() else args.annotations
    output = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output
    reports = (project_root / args.reports).resolve() if not args.reports.is_absolute() else args.reports
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    train_dataset = WasteCsvDataset(annotations, project_root, "train", build_transform(True))
    validation_dataset = WasteCsvDataset(annotations, project_root, "validation", build_transform(False))
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = build_loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = build_model(pretrained=True).to(device)
    freeze_features(model)
    criterion = nn.CrossEntropyLoss(weight=balanced_class_weights(train_dataset).to(device))
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    stale_epochs = 0

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)}; validation images: {len(validation_dataset)}")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            unfreeze_tail(model, blocks=2)
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.fine_tune_learning_rate,
                weight_decay=args.weight_decay,
            )
            print("Unfroze the last two EfficientNet feature blocks")

        train_loss, train_accuracy, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        with torch.inference_mode():
            validation_loss, validation_accuracy, validation_f1 = run_epoch(
                model, validation_loader, criterion, device, None, scaler
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.4f}, f1={train_f1:.4f}; "
            f"validation loss={validation_loss:.4f}, f1={validation_f1:.4f}"
        )

        checkpoint = {
            "architecture": "efficientnet_b0",
            "classes": CLASS_NAMES,
            "epoch": epoch,
            "validation_macro_f1": validation_f1,
            "model_state": model.state_dict(),
        }
        torch.save(checkpoint, output / "efficientnet_b0_last.pth")
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            stale_epochs = 0
            torch.save(checkpoint, output / "efficientnet_b0_best.pth")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    history_path = reports / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Best validation macro F1: {best_f1:.4f}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
