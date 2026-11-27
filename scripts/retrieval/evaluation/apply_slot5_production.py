"""Apply the leakage-safe slot-five guard to an existing production rerank.

The rule is selected with grouped cross-validation over the complete LegalIR
training set, then document priors are fitted on that training set.  The
production reranked file is never modified; all outputs are written to a new
variant directory with a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from udsc2026.evaluation.models import PredictionSample
from udsc2026.evaluation.retrieval_metrics import LegalIRReference, load_warmup

from run_slot5_probe import (
    FINAL_DOCUMENTS,
    METHOD,
    SCORE_FIELD,
    _fit_prior_stats,
    _load_bundles,
    _load_folds,
    _parameter_grid,
    _rescue_documents,
    _score_documents,
    QueryBundle,
    RuleParams,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty mapping")
    result: dict[str, str] = {}
    for question_id, row in payload.items():
        if not isinstance(question_id, str) or not isinstance(row, dict):
            raise ValueError("invalid question mapping")
        question = row.get("question")
        if set(row) != {"question"} or not isinstance(question, str) or not question.strip():
            raise ValueError("questions must contain only non-empty question objects")
        result[question_id] = question.strip()
    return result


def choose_params(
    bundles: dict[str, QueryBundle],
    references: dict[str, LegalIRReference],
    fold_by_id: dict[str, int],
) -> tuple[RuleParams, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for candidate in _parameter_grid():
        recalls: list[float] = []
        precisions: list[float] = []
        swaps = 0
        for fold in range(5):
            validation_ids = [qid for qid, value in fold_by_id.items() if value == fold]
            train_ids = [qid for qid, value in fold_by_id.items() if value != fold]
            stats = _fit_prior_stats(bundles, references, train_ids)
            recall, precision, fold_swaps, _ = _score_documents(
                validation_ids, bundles, references, stats, candidate
            )
            recalls.append(recall)
            precisions.append(precision)
            swaps += fold_swaps
        rows.append(
            {
                "params": candidate.as_dict(),
                "recall": sum(recalls) / len(recalls),
                "precision": sum(precisions) / len(precisions),
                "swaps": swaps,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["recall"]),
            -float(row["precision"]),
            int(row["swaps"]),
            json.dumps(row["params"], sort_keys=True),
        )
    )
    selected = rows[0]["params"]
    assert isinstance(selected, dict)
    return (
        RuleParams(
            prior_weight=float(selected["prior_weight"]),
            raw_score_ratio=float(selected["raw_score_ratio"]),
            minimum_prior=float(selected["minimum_prior"]),
            maximum_candidate_rank=int(selected["maximum_candidate_rank"]),
        ),
        rows[:10],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--train-reranked-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--reranked-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    questions = read_questions(args.questions)
    source_manifest = json.loads(args.reranked_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("complete") is not True:
        raise ValueError("production reranked manifest is incomplete")
    if source_manifest.get("output_sha256") != sha256(args.reranked):
        raise ValueError("production reranked hash does not match manifest")
    if source_manifest.get("questions_sha256") != sha256(args.questions):
        raise ValueError("production question mapping does not match reranked manifest")

    references = {
        sample.id: LegalIRReference(id=sample.id, gold_documents=list(sample.gold_documents))
        for sample in load_warmup(args.gold)
    }
    fold_by_id = _load_folds(args.folds)
    if set(references) != set(fold_by_id):
        raise ValueError("gold and grouped folds do not have identical IDs")
    train_bundles: dict[str, QueryBundle] = {}
    for fold in range(5):
        path = args.train_reranked_root / f"fold_{fold}" / "reranked.jsonl"
        expected = {qid for qid, value in fold_by_id.items() if value == fold}
        train_bundles.update(_load_bundles(path, expected))
    params, top_candidates = choose_params(train_bundles, references, fold_by_id)
    train_stats = _fit_prior_stats(train_bundles, references, references)

    production_bundles = _load_bundles(args.reranked, set(questions))
    submissions: list[dict[str, object]] = []
    contexts_path = args.output_root / "contexts.jsonl"
    audit_path = args.output_root / "audit.jsonl"
    swaps = 0
    with args.reranked.open(encoding="utf-8-sig") as source, contexts_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as contexts, audit_path.open("w", encoding="utf-8", newline="\n") as audit:
        for line in source:
            if not line.strip():
                continue
            sample = PredictionSample.model_validate_json(line, strict=True)
            selected, info = _rescue_documents(production_bundles[sample.question_id], train_stats, params)
            if info["swapped"]:
                swaps += 1
            selected_set = set(selected)
            hits = [
                hit.model_copy(update={"rank": rank})
                for rank, hit in enumerate(
                    (hit for hit in sample.hits if hit.doc_id in selected_set), start=1
                )
            ]
            contexts.write(sample.model_copy(update={"hits": hits}).model_dump_json(exclude_none=True) + "\n")
            submissions.append({"id": sample.question_id, "documents": list(selected)})
            audit.write(json.dumps({"id": sample.question_id, **info}, ensure_ascii=False) + "\n")
    if len(submissions) != len(questions) or {row["id"] for row in submissions} != set(questions):
        raise ValueError("production coverage mismatch")
    submission_path = args.output_root / "submission_candidates.json"
    submission_path.write_text(json.dumps(submissions, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "legalir-slot5-production-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "score_field": SCORE_FIELD,
        "final_documents": FINAL_DOCUMENTS,
        "selection": "five-fold grouped CV over all LegalIR train IDs; priors fitted on all train IDs",
        "selected_params": params.as_dict(),
        "top_cv_candidates": top_candidates,
        "question_count": len(submissions),
        "swap_count": swaps,
        "gold_sha256": sha256(args.gold),
        "folds_sha256": sha256(args.folds),
        "questions_sha256": sha256(args.questions),
        "source_reranked_sha256": sha256(args.reranked),
        "source_reranked_manifest_sha256": sha256(args.reranked_manifest),
        "contexts_sha256": sha256(contexts_path),
        "submission_sha256": sha256(submission_path),
        "complete": True,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context_manifest = {
        "schema_version": "legalir-production-context-v1",
        "created_at_utc": manifest["created_at_utc"],
        "questions_sha256": sha256(args.questions),
        "source_reranked_sha256": sha256(args.reranked),
        "source_reranked_manifest_sha256": sha256(args.reranked_manifest),
        "model": source_manifest.get("model"),
        "candidate_k": source_manifest.get("candidate_k"),
        "rerank_k": source_manifest.get("rerank_k"),
        "method": METHOD,
        "score_field": SCORE_FIELD,
        "max_documents": FINAL_DOCUMENTS,
        "evidence_limit": 3,
        "rrf_k": 60,
        "dense_rrf_k": 30,
        "dense_signal_weight": 0.30,
        "output_sha256": sha256(contexts_path),
        "question_count": len(submissions),
        "complete": True,
    }
    (args.output_root / "contexts.manifest.json").write_text(
        json.dumps(context_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(args.output_root), "params": params.as_dict(), "swap_count": swaps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
