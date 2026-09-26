from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split RealWaste into train, validation and test sets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/annotations/realwaste.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/realwaste_split.csv"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_ratio = 1.0 - args.train_ratio - args.validation_ratio

    if args.train_ratio <= 0:
        raise ValueError("train ratio must be greater than zero")

    if args.validation_ratio <= 0:
        raise ValueError("validation ratio must be greater than zero")

    if test_ratio <= 0:
        raise ValueError("test ratio must be greater than zero")

    with args.input.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError("Input CSV is empty")

    required_columns = {
        "image_path",
        "source_category",
        "target_class",
        "item_id",
    }

    missing = required_columns - set(fieldnames)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # 按照目標類別及原始子類別分層。
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = (
            row["target_class"],
            row["source_category"],
        )
        strata[key].append(row)

    random_generator = random.Random(args.seed)

    for key, group in sorted(strata.items()):
        random_generator.shuffle(group)

        count = len(group)
        train_count = int(count * args.train_ratio)
        validation_count = int(
            count * args.validation_ratio
        )

        for index, row in enumerate(group):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "test"

    # 保持輸出順序穩定。
    rows.sort(
        key=lambda row: (
            row["split"],
            row["target_class"],
            row["source_category"],
            row["image_path"],
        )
    )

    if "split" not in fieldnames:
        fieldnames.append("split")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(
        row["split"] for row in rows
    )

    target_split_counts = Counter(
        (row["split"], row["target_class"])
        for row in rows
    )

    print(f"Input images: {len(rows)}")
    print(f"Output: {args.output}")
    print()

    print("Split counts:")

    for split in ("train", "validation", "test"):
        print(
            f"- {split}: "
            f"{split_counts[split]}"
        )

    print()
    print("Class counts:")

    for split in ("train", "validation", "test"):
        print(f"[{split}]")

        for target_class in (
            "glass",
            "metal",
            "paper",
            "plastic",
            "unknown",
        ):
            count = target_split_counts[
                (split, target_class)
            ]

            print(
                f"- {target_class}: {count}"
            )


if __name__ == "__main__":
    main()