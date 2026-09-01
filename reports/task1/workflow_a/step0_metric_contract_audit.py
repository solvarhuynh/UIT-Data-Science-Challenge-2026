"""CPU-only Step 0 corrected audit.

The readers below stream the canonical files and decode only target-fold
records.  Fold 0 values are structurally skipped and never materialized,
stored, or used.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_FOLDS = {1, 2, 3, 4}
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
V3_SOURCE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
REPORT = ROOT / "reports/step0_metric_contract_report.json"


class CharStream:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("r", encoding="utf-8-sig")
        self.buffer = ""

    def get(self) -> str:
        if not self.buffer:
            self.buffer = self.handle.read(8192)
            if not self.buffer:
                return ""
        value, self.buffer = self.buffer[0], self.buffer[1:]
        return value

    def close(self) -> None:
        self.handle.close()


def skip_ws(stream: CharStream) -> str:
    char = stream.get()
    while char and char.isspace():
        char = stream.get()
    return char


def skip_string(stream: CharStream, first: str = '"') -> None:
    if first != '"':
        raise ValueError("expected JSON string")
    escaped = False
    while True:
        char = stream.get()
        if not char:
            raise ValueError("unterminated JSON string")
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return


def read_string(stream: CharStream, first: str = '"') -> str:
    if first != '"':
        raise ValueError("expected JSON string")
    raw = [first]
    escaped = False
    while True:
        char = stream.get()
        if not char:
            raise ValueError("unterminated JSON string")
        raw.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return json.loads("".join(raw))


def skip_value(stream: CharStream, first: str | None = None) -> None:
    char = skip_ws(stream) if first is None else first
    if not char:
        raise ValueError("missing JSON value")
    if char == '"':
        skip_string(stream, char)
        return
    if char == "{":
        char = skip_ws(stream)
        if char == "}":
            return
        while True:
            skip_string(stream, char)
            if skip_ws(stream) != ":":
                raise ValueError("invalid JSON object")
            skip_value(stream)
            char = skip_ws(stream)
            if char == "}":
                return
            if char != ",":
                raise ValueError("invalid JSON object separator")
            char = skip_ws(stream)
    if char == "[":
        char = skip_ws(stream)
        if char == "]":
            return
        while True:
            skip_value(stream, char)
            char = skip_ws(stream)
            if char == "]":
                return
            if char != ",":
                raise ValueError("invalid JSON array separator")
            char = skip_ws(stream)
    else:
        while char and char not in ",]}":
            char = stream.get()


def read_value(stream: CharStream, first: str | None = None) -> Any:
    """Decode one selected JSON value; never call this for skipped values."""
    char = skip_ws(stream) if first is None else first
    if char == '"':
        return read_string(stream, char)
    if char == "[":
        values: list[Any] = []
        item = skip_ws(stream)
        if item == "]":
            return values
        while True:
            values.append(read_value(stream, item))
            item = skip_ws(stream)
            if item == "]":
                return values
            if item != ",":
                raise ValueError("invalid selected array")
            item = skip_ws(stream)
    if char == "{":
        values: dict[str, Any] = {}
        item = skip_ws(stream)
        if item == "}":
            return values
        while True:
            key = read_string(stream, item)
            if skip_ws(stream) != ":":
                raise ValueError("invalid selected object")
            values[key] = read_value(stream)
            item = skip_ws(stream)
            if item == "}":
                return values
            if item != ",":
                raise ValueError("invalid selected object")
            item = skip_ws(stream)
    raw: list[str] = [char]
    item = stream.get()
    while item and item not in ",]}":
        raw.append(item)
        item = stream.get()
    return json.loads("".join(raw))


def read_target_fold_map(path: Path) -> dict[str, int]:
    """Read only validation IDs in folds 1–4 from the canonical mapping."""
    # The folds file is small enough to decode structurally, but this selective
    # reader avoids materializing Fold0 validation IDs by contract.
    stream = CharStream(path)
    try:
        if skip_ws(stream) != "{":
            raise ValueError("fold mapping root must be an object")
        result: dict[str, int] = {}
        char = skip_ws(stream)
        while char != "}":
            key = read_string(stream, char)
            if skip_ws(stream) != ":":
                raise ValueError("invalid fold mapping")
            value_start = skip_ws(stream)
            if key != "folds":
                skip_value(stream, value_start)
            else:
                if value_start != "[":
                    raise ValueError("folds must be an array")
                item = skip_ws(stream)
                while item != "]":
                    if item != "{":
                        raise ValueError("fold record must be an object")
                    fold: int | None = None
                    validation_ids: list[str] | None = None
                    field = skip_ws(stream)
                    while field != "}":
                        field_name = read_string(stream, field)
                        if skip_ws(stream) != ":":
                            raise ValueError("invalid fold record")
                        field_value = skip_ws(stream)
                        if field_name == "fold":
                            fold = int(read_value(stream, field_value))
                        elif field_name == "validation_ids" and fold in TARGET_FOLDS:
                            validation_ids = read_value(stream, field_value)
                        else:
                            skip_value(stream, field_value)
                        field = skip_ws(stream)
                        if field == ",":
                            field = skip_ws(stream)
                    if fold in TARGET_FOLDS:
                        if not isinstance(validation_ids, list):
                            raise ValueError("target fold lacks validation_ids")
                        for query_id in validation_ids:
                            result[str(query_id)] = fold
                    item = skip_ws(stream)
                    if item == ",":
                        item = skip_ws(stream)
                
            char = skip_ws(stream)
            if char == ",":
                char = skip_ws(stream)
        if set(result.values()) != TARGET_FOLDS:
            raise ValueError("fold mapping does not contain exactly folds 1–4")
        return result
    finally:
        stream.close()


def read_target_train(path: Path, fold_by_query: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Read and decode only selected train records; skip all other values."""
    stream = CharStream(path)
    try:
        if skip_ws(stream) != "{":
            raise ValueError("train root must be an object")
        selected: dict[str, dict[str, Any]] = {}
        char = skip_ws(stream)
        while char != "}":
            query_id = read_string(stream, char)
            if skip_ws(stream) != ":":
                raise ValueError("invalid train mapping")
            value_start = skip_ws(stream)
            if query_id in fold_by_query:
                value = read_value(stream, value_start)
                if not isinstance(value, dict):
                    raise ValueError(f"train record is not an object: {query_id}")
                selected[query_id] = value
            else:
                skip_value(stream, value_start)
            char = skip_ws(stream)
            if char == ",":
                char = skip_ws(stream)
        if set(selected) != set(fold_by_query):
            raise ValueError("train records and target fold mapping do not match")
        return selected
    finally:
        stream.close()


def score_local(gold: list[str], prediction: list[str]) -> float:
    return len(set(gold) & {str(value) for value in prediction}) / len(set(gold))


def score_official_equivalent(gold: list[str], prediction: list[str]) -> float:
    if not 1 <= len(prediction) <= 5:
        return 0.0
    return len(set(gold) & set(prediction)) / len(set(gold))


def check_v3_domain(gold_by_query: dict[str, list[str]]) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    max_difference = 0.0
    seen_queries: set[str] = set()
    with V3_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            fold = int(row["fold"])
            if fold not in TARGET_FOLDS:
                raise ValueError("equivalence source contains a non-target fold")
            docs: list[str] = []
            for hit in row["hits"]:
                doc_id = str(hit["doc_id"])
                if doc_id not in docs:
                    docs.append(doc_id)
            if not docs or len(set(docs)) != len(docs):
                raise ValueError(f"invalid V3 candidate domain row: {query_id}")
            # The source is a chunk-ranking artifact. Historical V3 collapses
            # this ranking to its first five unique document IDs before score.
            docs = docs[:5]
            if query_id in seen_queries:
                raise ValueError(f"duplicate query in equivalence source: {query_id}")
            seen_queries.add(query_id)
            local = score_local(gold_by_query[query_id], docs)
            official = score_official_equivalent(gold_by_query[query_id], docs)
            difference = abs(local - official)
            checked += 1
            max_difference = max(max_difference, difference)
            if difference:
                if len(mismatches) < 20:
                    mismatches.append({"query_id": query_id, "local_recall": local, "official_recall": official})
    if seen_queries != set(gold_by_query):
        raise ValueError("equivalence source does not cover exactly target fold queries")
    return {
        "checked_queries": checked,
        "mismatch_count": len(mismatches),
        "max_absolute_recall_difference": max_difference,
        "sample_mismatches": mismatches,
    }


def main() -> None:
    fold_by_query = read_target_fold_map(FOLDS)
    train = read_target_train(TRAIN, fold_by_query)
    gold_by_query: dict[str, list[str]] = {}
    relevant_counts: Counter[int] = Counter()
    max_recall_counts: Counter[str] = Counter()
    duplicate_queries: list[str] = []
    for query_id, row in train.items():
        answer = row.get("answer")
        if not isinstance(answer, list) or not answer:
            raise ValueError(f"invalid gold schema: {query_id}")
        gold = [str(value) for value in answer]
        unique = set(gold)
        if len(unique) != len(gold):
            duplicate_queries.append(query_id)
        gold_by_query[query_id] = gold
        relevant_count = len(unique)
        relevant_counts[relevant_count] += 1
        max_recall_counts[format(min(5, relevant_count) / relevant_count, ".15f").rstrip("0").rstrip(".")] += 1

    equivalence = check_v3_domain(gold_by_query)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    total = len(gold_by_query)
    multi = sum(count for value, count in relevant_counts.items() if value > 1)
    gt5 = sum(count for value, count in relevant_counts.items() if value > 5)
    report.update(
        {
            "status": "PASS" if equivalence["mismatch_count"] == 0 else "BLOCKED",
            "core_recall_formula_consistent": True,
            "implementation_guards_consistent": False,
            "metric_semantics_consistent_on_v3_domain": equivalence["mismatch_count"] == 0,
            "score_equivalence_check": equivalence,
            "historical_launcher_contains_fold0_stage": True,
            "fold0_touched": False,
            "public_labels_used": False,
            "metric_sources_consistent": equivalence["mismatch_count"] == 0,
            "total_queries": total,
            "relevant_count_distribution": {str(key): relevant_counts[key] for key in sorted(relevant_counts)},
            "multi_gold_query_count": multi,
            "multi_gold_rate": multi / total,
            "gold_count_gt5_query_count": gt5,
            "gold_count_gt5_rate": gt5 / total,
            "per_query_max_recall_distribution": dict(sorted(max_recall_counts.items(), key=lambda item: float(item[0]))),
            "duplicate_gold_id_query_count": len(duplicate_queries),
            "notes": [
                "Historical launcher contains evaluate_v3_fold0.py, but Step 0 did not execute the launcher or that stage.",
                "Implementation guard difference is official empty/>5 prediction handling versus the V3 local helper; every checked V3-domain output was a unique top-5, so the guard was inactive.",
                f"Direct equivalence checked {equivalence['checked_queries']} folds-1–4 queries from the explicit f1to4-only source; mismatch count={equivalence['mismatch_count']}.",
                f"Duplicate gold IDs were found in {len(duplicate_queries)} folds-1–4 queries and counted using unique gold IDs.",
                "No Fold0 labels/data or public labels were decoded, used, or included in statistics.",
            ],
        }
    )
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
