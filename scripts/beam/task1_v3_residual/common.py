"""Shared deterministic helpers for the V3 residual policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recall(gold: set[str], top5: Iterable[str]) -> float:
    return len(gold & {str(doc_id) for doc_id in top5}) / len(gold) if gold else 0.0


def load_fold_map(path: Path) -> dict[str, int]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict) and isinstance(value.get("folds"), list):
        result: dict[str, int] = {}
        for record in value["folds"]:
            for query_id in record.get("validation_ids", []):
                result[str(query_id)] = int(record["fold"])
        return result
    if isinstance(value, list):
        return {str(row["query_id"]): int(row["fold"]) for row in value}
    raise ValueError(f"unsupported folds schema: {path}")


def sort_query_ids(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (int(value) if str(value).isdigit() else str(value)))
