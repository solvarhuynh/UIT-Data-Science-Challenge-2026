"""Mine distinct-document positives and negatives from base-scored chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from udsc2026.evaluation.reporting import write_run_manifest  # noqa: E402
from udsc2026.evaluation.retrieval_metrics import (  # noqa: E402
    normalize_legal_ir_matching_question,
)
from udsc2026.training.chunk_training import build_training_record  # noqa: E402
from udsc2026.training.provenance import (  # noqa: E402
    question_mapping_bytes,
    read_json,
    sha256,
    validate_folds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--reranked-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    train = read_json(args.train)
    questions = {qid: record["question"] for qid, record in train.items()}
    folds = validate_folds(args.folds, questions)
    fold_by_id = {qid: fold for fold, ids in folds.items() for qid in ids}
    groups_by_id = {
        qid: normalize_legal_ir_matching_question(question)
        for qid, question in questions.items()
    }
    group_gold: dict[str, set[str]] = defaultdict(set)
    for qid, record in train.items():
        group_gold[groups_by_id[qid]].update(record["answer"])
    source = read_json(args.reranked_manifest)
    question_hash = hashlib.sha256(question_mapping_bytes(questions)).hexdigest()
    if (
        source.get("schema_version") != "reranker-run-v2"
        or source.get("complete") is not True
        or source.get("model") != "AITeamVN/Vietnamese_Reranker"
        or source.get("questions_sha256") != question_hash
        or source.get("completed_count") != len(questions)
        or source.get("output_sha256") != sha256(args.reranked)
    ):
        raise ValueError("reranked source fails model, question or checksum validation")
    depth = source["rerank_k"]
    if not 0 < depth <= source["candidate_k"]:
        raise ValueError("invalid base-scored candidate depth")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "groups.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("remove the previous generated groups before rebuilding")
    seen = set()
    counts = {fold: Counter() for fold in folds}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=args.output_dir, delete=False
        ) as temporary, args.reranked.open(encoding="utf-8-sig") as rows:
            temporary_path = Path(temporary.name)
            for line in rows:
                row = json.loads(line)
                qid = row["question_id"]
                if qid in seen or qid not in questions:
                    raise ValueError(f"unknown or duplicate candidate ID {qid}")
                seen.add(qid)
                fold = fold_by_id[qid]
                group = build_training_record(
                    qid,
                    questions[qid].strip(),
                    train[qid]["answer"],
                    row["hits"][:depth],
                    excluded_documents=group_gold[groups_by_id[qid]],
                )
                if group is None:
                    counts[fold]["skipped_without_both_labels"] += 1
                    continue
                counts[fold]["usable_queries"] += 1
                for band, hits in group["negative_chunks"].items():
                    counts[fold][f"negatives_{band}"] += len(hits)
                temporary.write(json.dumps(group, ensure_ascii=False) + "\n")
        if seen != questions.keys():
            raise ValueError(f"candidate file misses {len(questions.keys() - seen)} IDs")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": "retrieval-training-groups-v2",
        "train_sha256": sha256(args.train),
        "questions_sha256": question_hash,
        "folds_sha256": sha256(args.folds),
        "source_manifest_sha256": sha256(args.reranked_manifest),
        "source_sha256": source["output_sha256"],
        "base_model_revision": source["model_revision"],
        "rerank_k": depth,
        "max_length": source["max_length"],
        "groups_sha256": sha256(destination),
        "validation_fold_counts": counts,
    }
    write_run_manifest(manifest, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
