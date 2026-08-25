"""Build strict-fold raw-chunk groups for the gated Task1 fine-tuning V3 design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from udsc2026.training.task1_chunk_training_v3 import RankBands, build_training_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _folds(path: Path) -> dict[int, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        int(item["fold"]): {str(value) for value in item["validation_ids"]}
        for item in payload["folds"]
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-end", type=int, default=20)
    parser.add_argument("--medium-end", type=int, default=100)
    parser.add_argument("--positives-per-query", type=int, default=3)
    parser.add_argument("--negatives-per-band", type=int, default=3)
    parser.add_argument("--no-same-law", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    forbidden = [Path("models/reranker").resolve(), Path("data/processed_v3").resolve()]
    output = args.output_dir.resolve()
    if any(output == path or path in output.parents for path in forbidden):
        raise ValueError("V3 output must not be inside canonical model/corpus paths")
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    folds = _folds(args.folds)
    if len(folds) != 5:
        raise ValueError("expected five strict folds")
    all_ids = set(map(str, train))
    if set().union(*folds.values()) != all_ids:
        raise ValueError("fold validation union does not match training queries")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = {}
    temporary_paths = {}
    counts: dict[int, Counter[str]] = {fold: Counter() for fold in folds}
    try:
        for fold in folds:
            fold_dir = args.output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            destination = fold_dir / "train_groups.jsonl"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            streams[fold] = temporary.open("w", encoding="utf-8", newline="\n")
            temporary_paths[fold] = (temporary, destination)
        with args.raw_candidates.open(encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row.get("query_id", row.get("question_id", "")))
                if query_id not in train:
                    raise ValueError(f"unknown raw candidate query {query_id!r}")
                question_row = train[query_id]
                for fold, validation_ids in folds.items():
                    if query_id in validation_ids:
                        counts[fold]["excluded_validation_queries"] += 1
                        continue
                    record = build_training_record(
                        query_id=query_id,
                        question=str(question_row["question"]),
                        gold_documents=[str(value) for value in question_row["answer"]],
                        raw_hits=row.get("hits", []),
                        bands=RankBands(args.hard_end, args.medium_end),
                        positives_per_query=args.positives_per_query,
                        negatives_per_band=args.negatives_per_band,
                        include_same_law=not args.no_same_law,
                    )
                    if record is None:
                        counts[fold]["skipped_no_retrieved_positive"] += 1
                        continue
                    streams[fold].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[fold]["training_queries"] += 1
    finally:
        for stream in streams.values():
            stream.close()
    for temporary, destination in temporary_paths.values():
        temporary.replace(destination)
    manifest = {
        "schema_version": "task1-chunk-training-v3-v1",
        "status": "complete_data_build_only",
        "representation": (
            "positive and negative retrieved raw chunks from inference pool"
        ),
        "strict_cv": "each fold output excludes that fold's validation IDs",
        "train": str(args.train),
        "train_sha256": _sha256(args.train),
        "folds": str(args.folds),
        "folds_sha256": _sha256(args.folds),
        "raw_candidates": str(args.raw_candidates),
        "raw_candidates_sha256": _sha256(args.raw_candidates),
        "rank_bands": {
            "hard": [1, args.hard_end],
            "medium": [args.hard_end + 1, args.medium_end],
            "easy": [args.medium_end + 1, None],
        },
        "sampling": {
            "positives_per_query": args.positives_per_query,
            "negatives_per_band": args.negatives_per_band,
            "same_law": not args.no_same_law,
        },
        "fold_counts": {str(fold): dict(value) for fold, value in counts.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task1 V3 data builder error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["fold_counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
