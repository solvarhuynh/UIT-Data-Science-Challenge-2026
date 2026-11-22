"""Collapse reranked chunks into the production document list and QA contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.evaluation.retrieval_candidates import (  # noqa: E402
    DEFAULT_DENSE_RRF_K,
    DEFAULT_DENSE_SIGNAL_WEIGHT,
    aggregate_to_legal_ir_prediction,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty ID mapping")
    questions: dict[str, str] = {}
    for question_id, record in payload.items():
        if (
            not isinstance(question_id, str)
            or not isinstance(record, dict)
            or set(record) != {"question"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
        ):
            raise ValueError("questions must contain only non-empty question objects")
        questions[question_id] = record["question"].strip()
    return questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--reranked-manifest", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--method", default="rrf_cap2_dense_signal")
    parser.add_argument("--score-field", default="rerank_score")
    parser.add_argument("--max-documents", type=int, default=5)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-rrf-k", type=int, default=DEFAULT_DENSE_RRF_K)
    parser.add_argument(
        "--dense-signal-weight",
        type=float,
        default=DEFAULT_DENSE_SIGNAL_WEIGHT,
    )
    return parser


def run(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    if args.max_documents <= 0 or args.evidence_limit <= 0:
        raise ValueError("document and evidence limits must be positive")
    if args.submission.exists() or args.contexts.exists():
        raise FileExistsError("production aggregation outputs already exist")
    source_manifest = json.loads(
        args.reranked_manifest.read_text(encoding="utf-8")
    )
    if not isinstance(source_manifest, dict) or source_manifest.get("complete") is not True:
        raise ValueError("reranked manifest is missing or incomplete")
    if source_manifest.get("output_sha256") != _sha256(args.reranked):
        raise ValueError("reranked output does not match its manifest")
    if source_manifest.get("questions_sha256") != _sha256(args.questions):
        raise ValueError("reranked output uses a different question mapping")

    submissions: list[dict[str, object]] = []
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    args.contexts.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with args.reranked.open("r", encoding="utf-8") as source, args.contexts.open(
        "w", encoding="utf-8"
    ) as context_file:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            sample = PredictionSample.model_validate_json(line, strict=True)
            if sample.question_id in seen or sample.question_id not in questions:
                raise ValueError(f"unexpected or duplicate question ID at line {line_number}")
            seen.add(sample.question_id)
            prediction = aggregate_to_legal_ir_prediction(
                sample,
                method=args.method,
                max_documents=args.max_documents,
                evidence_limit=args.evidence_limit,
                rrf_k=args.rrf_k,
                dense_rrf_k=args.dense_rrf_k,
                dense_signal_weight=args.dense_signal_weight,
                score_field=args.score_field,
            )
            selected = set(prediction.documents)
            context_hits = [
                hit.model_copy(update={"rank": rank})
                for rank, hit in enumerate(
                    (hit for hit in sample.hits if hit.doc_id in selected),
                    start=1,
                )
            ]
            context_sample = sample.model_copy(update={"hits": context_hits})
            context_file.write(
                context_sample.model_dump_json(exclude_none=True) + "\n"
            )
            submissions.append(
                {"id": prediction.id, "documents": list(prediction.documents)}
            )
    if seen != set(questions):
        missing = sorted(set(questions) - seen)
        raise ValueError(f"aggregation coverage mismatch; missing IDs: {missing[:5]}")
    args.submission.write_text(
        json.dumps(submissions, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    context_manifest = {
        "schema_version": "legalir-production-context-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_sha256": _sha256(args.questions),
        "source_reranked_sha256": _sha256(args.reranked),
        "source_reranked_manifest_sha256": _sha256(args.reranked_manifest),
        "model": source_manifest.get("model"),
        "candidate_k": source_manifest.get("candidate_k"),
        "rerank_k": source_manifest.get("rerank_k"),
        "method": args.method,
        "score_field": args.score_field,
        "max_documents": args.max_documents,
        "evidence_limit": args.evidence_limit,
        "rrf_k": args.rrf_k,
        "dense_rrf_k": args.dense_rrf_k,
        "dense_signal_weight": args.dense_signal_weight,
        "output_sha256": _sha256(args.contexts),
        "question_count": len(seen),
        "complete": True,
    }
    args.context_manifest.write_text(
        json.dumps(context_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"question_count": len(seen), "submission": str(args.submission)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, TypeError, ValueError) as exc:
        print(f"production aggregation error: {exc}", file=sys.stderr)
        raise SystemExit(2)
