from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DATASET_ID = "rkh7nz2rcp"
DATASET_VERSION = 1
DATASET_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}"
API_BASE = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
FOLDER_NAMES = {"Negative", "Critical_negative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import unique clean-scene images as empty-label YOLO training negatives"
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/detection"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/mendeley_illegal_dumping")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/annotations/mendeley_negatives_import.csv"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def download_bytes(url: str, *, timeout: int, retries: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trash-identify/0.7.1 academic dataset importer",
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def load_json(
    url: str, cache: Path, *, timeout: int, retries: int
) -> list[dict[str, Any]]:
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(download_bytes(url, timeout=timeout, retries=retries))
    value = json.loads(cache.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list from {url}")
    return value


def fetch_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    folders = load_json(
        f"{API_BASE}/folders/{DATASET_VERSION}",
        args.cache_root / "folders.json",
        timeout=args.timeout,
        retries=args.retries,
    )
    selected_folders = {
        folder["id"]: folder["name"]
        for folder in folders
        if folder.get("name") in FOLDER_NAMES
    }
    if set(selected_folders.values()) != FOLDER_NAMES:
        raise ValueError(f"Missing expected folders: {sorted(FOLDER_NAMES)}")

    files: list[dict[str, Any]] = []
    for folder_id, folder_name in selected_folders.items():
        query = urllib.parse.urlencode(
            {
                "folder_id": folder_id,
                "version": DATASET_VERSION,
                "$start": 0,
                "$limit": 1000,
            }
        )
        rows = load_json(
            f"{API_BASE}/files?{query}",
            args.cache_root / f"{folder_name}_files.json",
            timeout=args.timeout,
            retries=args.retries,
        )
        for row in rows:
            row = dict(row)
            row["source_group"] = folder_name
            files.append(row)

    unique: dict[str, dict[str, Any]] = {}
    for row in files:
        details = row.get("content_details") or {}
        source_hash = details.get("sha256_hash")
        download_url = details.get("download_url")
        if (
            row.get("status") == "COMPLETED"
            and isinstance(source_hash, str)
            and isinstance(download_url, str)
            and details.get("content_type", "").startswith("image/")
        ):
            unique.setdefault(source_hash, row)

    candidates = [unique[key] for key in sorted(unique)]
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]
    return candidates, len(files) - len(unique)


def save_negative(
    row: dict[str, Any], args: argparse.Namespace
) -> dict[str, str | int]:
    details = row["content_details"]
    source_hash = details["sha256_hash"]
    stem = f"mendeley_neg_{source_hash[:20]}"
    image_path = args.output_root / "images" / "train" / f"{stem}.jpg"
    label_path = args.output_root / "labels" / "train" / f"{stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path.is_file() and label_path.is_file():
        with Image.open(image_path) as existing:
            width, height = existing.size
        stored_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    else:
        content = download_bytes(
            details["download_url"], timeout=args.timeout, retries=args.retries
        )
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != source_hash:
            raise ValueError(
                f"SHA-256 mismatch for {row['filename']}: {actual_hash} != {source_hash}"
            )
        with Image.open(io.BytesIO(content)) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            width, height = rgb.size
            temporary = image_path.with_suffix(".part.jpg")
            rgb.save(temporary, format="JPEG", quality=90, optimize=True)
            temporary.replace(image_path)
        label_path.write_text("", encoding="utf-8")
        stored_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    return {
        "image": image_path.as_posix(),
        "label": label_path.as_posix(),
        "split": "train",
        "source_group": row["source_group"],
        "source_filename": row["filename"],
        "source_sha256": source_hash,
        "stored_sha256": stored_hash,
        "width": width,
        "height": height,
        "boxes": 0,
        "license": "CC BY-NC 4.0",
        "source_url": DATASET_URL,
    }


def write_manifest(rows: list[dict[str, str | int]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["image"])))


def main() -> None:
    args = parse_args()
    candidates, duplicates = fetch_candidates(args)
    print(
        f"Found {len(candidates)} unique clean scenes; ignored {duplicates} duplicate files.",
        flush=True,
    )
    rows: list[dict[str, str | int]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(save_negative, row, args): row for row in candidates}
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(f"{row.get('filename')}: {exc}")
            if index % 25 == 0 or index == len(candidates):
                print(
                    f"Processed {index}/{len(candidates)}; successful={len(rows)}, "
                    f"failed={len(failures)}",
                    flush=True,
                )

    write_manifest(rows, args.manifest)
    if failures:
        failure_path = args.cache_root / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"{len(failures)} downloads failed; see {failure_path}. Re-run to retry."
        )
    print(f"Imported {len(rows)} empty-label YOLO training negatives.")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
