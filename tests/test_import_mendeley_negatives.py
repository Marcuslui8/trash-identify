from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.import_mendeley_negatives import fetch_candidates


def _file(filename: str, source_hash: str, *, status: str = "COMPLETED") -> dict:
    return {
        "filename": filename,
        "status": status,
        "content_details": {
            "sha256_hash": source_hash,
            "download_url": f"https://example.invalid/{filename}",
            "content_type": "image/jpeg",
        },
    }


def test_fetch_candidates_uses_only_expected_folders_and_deduplicates(tmp_path) -> None:
    folders = [
        {"id": "negative-id", "name": "Negative"},
        {"id": "critical-id", "name": "Critical_negative"},
        {"id": "positive-id", "name": "Positive"},
    ]
    (tmp_path / "folders.json").write_text(json.dumps(folders), encoding="utf-8")
    (tmp_path / "Negative_files.json").write_text(
        json.dumps([_file("clean-a.jpg", "a" * 64), _file("copy.jpg", "a" * 64)]),
        encoding="utf-8",
    )
    (tmp_path / "Critical_negative_files.json").write_text(
        json.dumps([_file("hard-b.jpg", "b" * 64)]), encoding="utf-8"
    )
    args = SimpleNamespace(cache_root=tmp_path, timeout=1, retries=1, limit=None)

    candidates, duplicates = fetch_candidates(args)

    assert [row["source_group"] for row in candidates] == [
        "Negative",
        "Critical_negative",
    ]
    assert duplicates == 1


def test_fetch_candidates_limit_is_deterministic(tmp_path) -> None:
    folders = [
        {"id": "negative-id", "name": "Negative"},
        {"id": "critical-id", "name": "Critical_negative"},
    ]
    (tmp_path / "folders.json").write_text(json.dumps(folders), encoding="utf-8")
    (tmp_path / "Negative_files.json").write_text(
        json.dumps([_file("z.jpg", "f" * 64)]), encoding="utf-8"
    )
    (tmp_path / "Critical_negative_files.json").write_text(
        json.dumps([_file("a.jpg", "0" * 64)]), encoding="utf-8"
    )
    args = SimpleNamespace(cache_root=tmp_path, timeout=1, retries=1, limit=1)

    candidates, _ = fetch_candidates(args)

    assert len(candidates) == 1
    assert candidates[0]["filename"] == "a.jpg"
