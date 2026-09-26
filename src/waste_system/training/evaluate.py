from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from waste_system.training.dataset import CLASS_NAMES, WasteCsvDataset, build_loader, build_transform
from waste_system.training.model import load_checkpoint
from waste_system.training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained waste classifier")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/efficientnet_b0_best.pth"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/realwaste_split.csv"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checkpoint = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    annotations = (root / args.annotations).resolve() if not args.annotations.is_absolute() else args.annotations
    reports = (root / args.reports).resolve() if not args.reports.is_absolute() else args.reports
    reports.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = WasteCsvDataset(annotations, root, "test", build_transform(False))
    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = load_checkpoint(checkpoint, device)
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    paths: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            scores, predicted = probabilities.max(dim=1)
            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(scores.cpu().tolist())
            paths.extend(batch_paths)
    elapsed = time.perf_counter() - started

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "images": len(dataset),
        "device": str(device),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "average_inference_ms_per_image": elapsed / len(dataset) * 1000,
    }
    (reports / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Waste classification confusion matrix")
    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(reports / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    with (reports / "misclassified.csv").open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "actual", "predicted", "confidence"],
        )
        writer.writeheader()
        for path, target, prediction, confidence in zip(
            paths, targets, predictions, confidences, strict=True
        ):
            if target != prediction:
                writer.writerow(
                    {
                        "image_path": path,
                        "actual": CLASS_NAMES[target],
                        "predicted": CLASS_NAMES[prediction],
                        "confidence": f"{confidence:.6f}",
                    }
                )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
