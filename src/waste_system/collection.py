from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from waste_system.inference.preprocessing import assess_quality


CATEGORIES = {"positive", "negative", "hard"}
TARGETS = {"positive": 350, "negative": 100, "hard": 50}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_LONG_SIDE = 1600
_SAFE_SESSION = re.compile(r"^[a-zA-Z0-9_-]{1,48}$")
_SAVE_LOCK = threading.Lock()


class CollectionError(ValueError):
    pass


def validate_session(value: str) -> str:
    session = value.strip()
    if not _SAFE_SESSION.fullmatch(session):
        raise CollectionError("Session 只能使用英文字母、數字、底線或連字號，長度 1～48")
    return session


def _resize(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_LONG_SIDE:
        return image
    scale = MAX_LONG_SIDE / longest
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _difference_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _existing_records(records_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not records_root.is_dir():
        return records
    for path in records_root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _hash_distance(first: str, second: str) -> int:
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError:
        return 64


def save_capture(
    content: bytes,
    *,
    category: str,
    session: str,
    collection_root: Path,
    notes: str = "",
    captured_at: str | None = None,
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise CollectionError(f"不支援的類別：{category}")
    session = validate_session(session)
    if not content:
        raise CollectionError("照片內容是空的")
    if len(content) > MAX_UPLOAD_BYTES:
        raise CollectionError("照片超過 12 MB")

    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise CollectionError("無法讀取照片")
    height, width = image.shape[:2]
    if min(height, width) < 320:
        raise CollectionError("照片太小，短邊至少需要 320 px")

    image = _resize(image)
    quality = assess_quality(
        image,
        blur_threshold=60.0,
        minimum_brightness=35.0,
        maximum_brightness=225.0,
    )
    # Ordinary positive/negative samples should remain readable. Deliberately
    # difficult frames are accepted under the hard category and retain issues.
    if category != "hard" and quality.score < 0.35:
        raise CollectionError("照片太模糊或曝光嚴重，請重拍；若刻意蒐集困難案例請選『困難案例』")

    success, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise CollectionError("照片重新編碼失敗")
    sanitized = jpeg.tobytes()
    digest = hashlib.sha256(sanitized).hexdigest()
    difference_hash = _difference_hash(image)
    records_root = collection_root / "records"

    with _SAVE_LOCK:
        records = _existing_records(records_root)
        if any(record.get("sha256") == digest for record in records):
            raise CollectionError("這張照片與已保存照片完全相同，請移動手機或更換場景")
        for record in records:
            if record.get("session") != session or record.get("category") != category:
                continue
            existing_hash = str(record.get("difference_hash", ""))
            if existing_hash and _hash_distance(existing_hash, difference_hash) <= 2:
                raise CollectionError("畫面與同一場景的既有照片太相似，請改變距離、角度或物品配置")

        now = datetime.now(timezone.utc)
        capture_id = f"local_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        image_dir = collection_root / "images" / session / category
        image_dir.mkdir(parents=True, exist_ok=True)
        records_root.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{capture_id}.jpg"
        record_path = records_root / f"{capture_id}.json"
        image_path.write_bytes(sanitized)
        saved_height, saved_width = image.shape[:2]
        record = {
            "capture_id": capture_id,
            "category": category,
            "session": session,
            "image_path": image_path.relative_to(collection_root).as_posix(),
            "width": saved_width,
            "height": saved_height,
            "sha256": digest,
            "difference_hash": difference_hash,
            "quality_score": quality.score,
            "brightness": round(quality.brightness, 3),
            "blur_variance": round(quality.blur_variance, 3),
            "quality_issues": list(quality.issues),
            "notes": notes.strip()[:200],
            "captured_at_client": (captured_at or "")[:64],
            "saved_at_utc": now.isoformat(),
            "annotation_status": "not_required" if category == "negative" else "pending",
        }
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def collection_status(collection_root: Path) -> dict[str, Any]:
    counts = {category: 0 for category in CATEGORIES}
    sessions: dict[str, dict[str, int]] = {}
    records_root = collection_root / "records"
    if records_root.is_dir():
        for path in records_root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            category = record.get("category")
            session = record.get("session")
            if category not in CATEGORIES or not isinstance(session, str):
                continue
            counts[category] += 1
            session_counts = sessions.setdefault(session, {item: 0 for item in CATEGORIES})
            session_counts[category] += 1
    total = sum(counts.values())
    return {
        "total": total,
        "target_total": sum(TARGETS.values()),
        "counts": counts,
        "targets": TARGETS,
        "remaining": {key: max(0, TARGETS[key] - counts[key]) for key in CATEGORIES},
        "sessions": sessions,
    }
