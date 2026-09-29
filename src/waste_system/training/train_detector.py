from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the one-class YOLO waste detector")
    parser.add_argument("--data", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is the stable Windows setting",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional initial learning rate (lr0), useful for checkpoint fine-tuning",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Ultralytics optimizer name; use AdamW for controlled fine-tuning",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-learning-rate", type=float, default=0.1)
    parser.add_argument("--project", type=Path, default=Path("runs/detection"))
    parser.add_argument("--name", default="waste_detector")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume an interrupted Ultralytics run from its last.pt checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: python -m pip install -e '.[detection]'"
        ) from exc
    if args.resume_from is not None:
        checkpoint = args.resume_from.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = YOLO(str(checkpoint)).train(resume=True)
        print(result)
        return
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    model = YOLO(args.base_model)
    train_options = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_learning_rate,
        seed=42,
    )
    if args.learning_rate is not None:
        train_options["lr0"] = args.learning_rate
    result = model.train(**train_options)
    print(result)
    print(
        "After validation, copy runs/detection/waste_detector/weights/best.pt "
        "to models/waste_detector.pt"
    )


if __name__ == "__main__":
    main()
