"""Check complete Task1 candidate and P12 coverage before P13 training.

The checker is intentionally CPU-only and model-free.  It reports exact
question coverage, duplicate candidate rows, per-fold negative coverage, and
returns a non-zero exit code unless the strict five-fold prerequisites are
complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_train_ids(path: Path) -> set[str]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("--train must be a LegalIR object mapping")
    return {
        str(query_id)
        for query_id, item in payload.items()
        if isinstance(item, dict)
        and str(item.get("question", "")).strip()
        and isinstance(item.get("answer"), list)
        and item["answer"]
    }


def _load_fold_map(path: Path) -> dict[str, int]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "legal-ir-strict-cv-v2"
    ):
        raise ValueError("--folds must be a legal-ir-strict-cv-v2 artifact")
    result: dict[str, int] = {}
    for row in payload.get("folds", []):
        if not isinstance(row, dict):
            continue
        fold = row.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError("strict fold number must be an integer")
        for query_id in row.get("validation_ids", []):
            query_id = str(query_id)
            if query_id in result:
                raise ValueError(f"duplicate strict-CV query ID: {query_id}")
            result[query_id] = fold
    if set(result.values()) != set(range(5)):
        raise ValueError("P13 requires exactly folds 0, 1, 2, 3, and 4")
    return result


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return
    payload = _load_json(path)
    rows = payload if isinstance(payload, list) else payload.get("predictions", [])
    if not isinstance(rows, list):
        raise ValueError(f"unsupported ranking artifact: {path}")
    yield from (row for row in rows if isinstance(row, dict))


def _candidate_ids(path: Path) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    occurrences: Counter[str] = Counter()
    for row in _iter_records(path):
        query_id = str(row.get("question_id", row.get("id", ""))).strip()
        if query_id:
            ids.add(query_id)
            occurrences[query_id] += 1
    return ids, dict(occurrences)


def _negative_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing P12 fold artifact: {path}")
    return list(_iter_records(path))


def inspect_coverage(
    *,
    train: Path,
    folds: Path,
    candidates: Path,
    negatives_dir: Path | None = None,
    expected_query_count: int = 7000,
) -> dict[str, Any]:
    """Return a JSON-safe prerequisite report without loading any model."""

    train_ids = _load_train_ids(train)
    fold_map = _load_fold_map(folds)
    required_ids = train_ids.intersection(fold_map)
    candidate_ids, candidate_occurrences = _candidate_ids(candidates)
    duplicate_candidate_ids = sorted(
        query_id for query_id, count in candidate_occurrences.items() if count != 1
    )
    missing_candidates = sorted(required_ids.difference(candidate_ids))
    unexpected_candidates = sorted(candidate_ids.difference(required_ids))

    if negatives_dir is None:
        complete = not any(
            (len(required_ids) != expected_query_count, missing_candidates,
             unexpected_candidates, duplicate_candidate_ids)
        )
        return {
            "schema_version": "task1-p13-prerequisites-v1",
            "mode": "candidates-only",
            "coverage_status": "complete" if complete else "incomplete",
            "expected_query_count": expected_query_count,
            "total_queries": len(required_ids),
            "candidate_query_ids": len(candidate_ids),
            "missing_candidate_query_ids": missing_candidates,
            "unexpected_candidate_query_ids": unexpected_candidates,
            "duplicate_candidate_query_ids": duplicate_candidate_ids,
            "next_step": "P12 mining may start." if complete else "STOP: complete candidates first.",
        }

    fold_reports: dict[str, Any] = {}
    all_negative_ids: set[str] = set()
    all_positive_ids: set[str] = set()
    all_hard_semi_ids: set[str] = set()
    fold_leaks: list[str] = []
    missing_fold_negatives: dict[str, list[str]] = {}
    for fold in range(5):
        rows = _negative_rows(negatives_dir / f"fold_{fold}.jsonl")
        ids = {str(row.get("query_id", "")).strip() for row in rows}
        ids.discard("")
        all_negative_ids.update(ids)
        all_positive_ids.update(
            str(row.get("query_id", "")).strip()
            for row in rows
            if str(row.get("positive_doc", "")).strip()
        )
        all_hard_semi_ids.update(
            str(row.get("query_id", "")).strip()
            for row in rows
            if str(row.get("negative_type", ""))
            in {
                "hard_false_positive",
                "semantic_confuser",
                "semi_hard",
                "lexical_confuser",
                "same_law",
            }
        )
        for row in rows:
            query_id = str(row.get("query_id", "")).strip()
            if query_id and fold_map.get(query_id) == fold:
                fold_leaks.append(query_id)
        expected_training_ids = {
            query_id for query_id in required_ids if fold_map[query_id] != fold
        }
        missing_fold_negatives[str(fold)] = sorted(
            expected_training_ids.difference(ids)
        )
        fold_reports[str(fold)] = {
            "record_count": len(rows),
            "query_count": len(ids),
            "training_query_count": len(expected_training_ids),
            "queries_missing_negatives": len(missing_fold_negatives[str(fold)]),
            "leakage_query_ids": sorted(
                {query_id for query_id in ids if fold_map.get(query_id) == fold}
            ),
        }

    missing_negatives = sorted(required_ids.difference(all_negative_ids))
    missing_positives = sorted(required_ids.difference(all_positive_ids))
    missing_hard_semi = sorted(required_ids.difference(all_hard_semi_ids))
    complete = not any(
        (
            len(required_ids) != expected_query_count,
            len(candidate_ids) != len(required_ids),
            missing_candidates,
            unexpected_candidates,
            duplicate_candidate_ids,
            missing_negatives,
            missing_positives,
            missing_hard_semi,
            fold_leaks,
            any(missing_fold_negatives.values()),
        )
    )
    return {
        "schema_version": "task1-p13-prerequisites-v1",
        "coverage_status": "complete" if complete else "incomplete",
        "expected_query_count": expected_query_count,
        "total_queries": len(required_ids),
        "required_query_ids": len(required_ids),
        "candidate_query_ids": len(candidate_ids),
        "missing_candidate_query_ids": missing_candidates,
        "unexpected_candidate_query_ids": unexpected_candidates,
        "duplicate_candidate_query_ids": duplicate_candidate_ids,
        "queries_with_positives": len(all_positive_ids),
        "queries_with_hard_or_semi_hard_negatives": len(all_hard_semi_ids),
        "queries_missing_candidates": missing_candidates,
        "queries_missing_negatives": missing_negatives,
        "queries_missing_positive_records": missing_positives,
        "queries_missing_hard_or_semi_hard_negatives": missing_hard_semi,
        "fold_leakage_query_ids": sorted(set(fold_leaks)),
        "folds": fold_reports,
        "missing_fold_negatives": missing_fold_negatives,
        "next_step": (
            "P13 may start after recording this report."
            if complete
            else "STOP: complete candidate/P12 coverage before P13 training."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--negatives-dir", type=Path)
    parser.add_argument("--candidates-only", action="store_true",
                        help="Check strict candidate coverage without requiring P12 negatives.")
    parser.add_argument("--expected-query-count", type=int, default=7000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidates_only and args.negatives_dir is None:
        print("P13 prerequisite error: --negatives-dir is required unless --candidates-only is used", file=sys.stderr)
        return 2
    try:
        report = inspect_coverage(
            train=args.train,
            folds=args.folds,
            candidates=args.candidates,
            negatives_dir=None if args.candidates_only else args.negatives_dir,
            expected_query_count=args.expected_query_count,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"P13 prerequisite error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
        print(args.output)
    print(serialized, end="")
    return 0 if report["coverage_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
