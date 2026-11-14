"""Build a guarded LegalIR public prediction from five promoted P13 folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.training.finetune_task1_bge_reranker import (  # noqa: E402
    TorchBGERerankerBackend,
    load_document_candidates,
)
from udsc2026.evaluation.legal_ir_document_candidates import (  # noqa: E402
    LegalIRDocumentCandidate,
    format_document_evidence,
)

DEFAULT_QUESTIONS = Path("data/raw/btc/LegalIR/public-official.json")
DEFAULT_LABELS = Path("data/raw/btc/LegalIR/train.json")
DEFAULT_CANDIDATES = Path("artifacts/task1/training/public_dense500_candidates.jsonl")
DEFAULT_CHECKPOINT_ROOT = Path("artifacts/task1/models/bge_reranker_finetune/full_oof")
DEFAULT_OUTPUT = Path("artifacts/task1/finetuned_oof_public/predictions.json")
DEFAULT_REPORT = Path("artifacts/task1/finetuned_oof_public/report.json")
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--overlay-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--no-exact-overlay", action="store_true")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--decision", type=Path)
    parser.add_argument(
        "--candidate-depth",
        type=int,
        choices=(50, 100, 150, 200, 300, 500),
        default=200,
    )
    parser.add_argument("--evidence-limit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--reranker-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument(
        "--folds", type=int, nargs="+", choices=range(5), default=list(range(5))
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--questions must contain a non-empty object mapping")
    questions: dict[str, str] = {}
    for raw_id, raw in payload.items():
        question_id = str(raw_id).strip()
        question = str(raw.get("question", "")).strip() if isinstance(raw, dict) else ""
        if not question_id or not question:
            raise ValueError("--questions contains an invalid ID/question")
        questions[question_id] = question
    return questions


def _normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_NON_WORD_RE.sub(" ", normalized).split())


def _exact_label_map(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("--overlay-labels must contain an object mapping")
    output: dict[str, list[str]] = {}
    for raw in payload.values():
        if not isinstance(raw, dict) or not isinstance(raw.get("answer"), list):
            continue
        normalized = _normalize_question(str(raw.get("question", "")))
        if not normalized:
            continue
        documents = output.setdefault(normalized, [])
        for raw_doc in raw["answer"]:
            doc_id = str(raw_doc).strip()
            if doc_id and doc_id not in documents:
                documents.append(doc_id)
        if len(documents) > 5:
            raise ValueError("exact normalized labels disagree on more than five docs")
    return output


def _load_decision(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("P13 decision must contain an object")
    if (
        payload.get("schema_version") != "task1-p13-decision-v1"
        or payload.get("status") != "PROMOTE_CANDIDATE"
        or payload.get("promotable") is not True
        or payload.get("complete_oof") is not True
    ):
        raise ValueError("P13 decision is not a complete promotable OOF result")
    oof = payload.get("oof")
    if not isinstance(oof, dict) or oof.get("fold_count") != 5:
        raise ValueError("P13 decision does not contain five completed folds")
    return payload


def _validate_fold_ranks(
    ranks: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[LegalIRDocumentCandidate]],
) -> None:
    if set(ranks) != set(candidates):
        raise ValueError("fold score cache question coverage mismatch")
    for question_id, pool in candidates.items():
        expected = [candidate.doc_id for candidate in pool]
        observed = list(ranks[question_id])
        if len(observed) != len(set(observed)) or set(observed) != set(expected):
            raise ValueError(f"fold score cache pool mismatch at {question_id}")


def _ensemble_documents(
    candidates: Sequence[LegalIRDocumentCandidate],
    fold_rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int,
    dense_weight: float,
    reranker_weight: float,
) -> list[str]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if not fold_rankings:
        raise ValueError("at least one fold ranking is required")
    expected = {candidate.doc_id for candidate in candidates}
    rank_maps: list[dict[str, int]] = []
    for ranking in fold_rankings:
        if len(ranking) != len(set(ranking)) or set(ranking) != expected:
            raise ValueError("fold ranking must exactly preserve the candidate pool")
        rank_maps.append({doc_id: rank for rank, doc_id in enumerate(ranking, 1)})

    def score(candidate: LegalIRDocumentCandidate) -> float:
        dense = dense_weight / (rrf_k + candidate.first_seen_rank)
        tuned = sum(
            1.0 / (rrf_k + ranks[candidate.doc_id]) for ranks in rank_maps
        ) / len(rank_maps)
        return float(dense + reranker_weight * tuned)

    ranked = sorted(
        candidates, key=lambda item: (-score(item), item.first_seen_rank, item.doc_id)
    )
    return [candidate.doc_id for candidate in ranked]


def _score_fold(
    *,
    fold: int,
    checkpoint: Path,
    questions: Mapping[str, str],
    candidates: Mapping[str, Sequence[LegalIRDocumentCandidate]],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    backend = TorchBGERerankerBackend(
        str(checkpoint),
        device=args.device,
        batch_size=1,
        inference_batch_size=args.batch_size,
        gradient_accumulation=1,
        max_length=args.max_length,
        learning_rate=1e-6,
        warmup_ratio=0.0,
        fp16=True,
        seed=2026 + fold,
    )
    output: dict[str, list[str]] = {}
    try:
        for index, (question_id, question) in enumerate(questions.items(), start=1):
            pool = candidates[question_id]
            scores = backend.score(
                question, [format_document_evidence(candidate) for candidate in pool]
            )
            if len(scores) != len(pool):
                raise ValueError("checkpoint returned a score count mismatch")
            ranked = sorted(
                zip(pool, scores),
                key=lambda item: (
                    -float(item[1]),
                    item[0].first_seen_rank,
                    item[0].doc_id,
                ),
            )
            output[question_id] = [candidate.doc_id for candidate, _ in ranked]
            if index % 50 == 0 or index == len(questions):
                print(
                    f"fold {fold}: scored {index}/{len(questions)} queries", flush=True
                )
    finally:
        backend.close()
    return output


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    decision_path = args.decision or args.checkpoint_root / "final_decision.json"
    for path in (args.questions, args.candidates, decision_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    if not args.no_exact_overlay and not args.overlay_labels.is_file():
        raise FileNotFoundError(f"overlay labels not found: {args.overlay_labels}")
    if args.rrf_k < 0 or not (args.dense_weight or args.reranker_weight):
        raise ValueError("RRF settings must include at least one positive weight")
    if sorted(set(args.folds)) != list(range(5)):
        raise ValueError("public promotion requires exactly folds 0, 1, 2, 3, and 4")
    decision = _load_decision(decision_path)
    questions = _load_questions(args.questions)
    candidates = load_document_candidates(
        args.candidates,
        candidate_depth=args.candidate_depth,
        evidence_limit=args.evidence_limit,
    )
    if set(questions) != set(candidates):
        raise ValueError("public question/candidate coverage mismatch")

    candidates_sha = _sha256(args.candidates)
    cache_dir = args.cache_dir or args.checkpoint_root / "public_fold_rank_cache"
    fold_outputs: list[dict[str, list[str]]] = []
    checkpoint_hashes: dict[str, str] = {}
    for fold in args.folds:
        checkpoint = args.checkpoint_root / f"fold_{fold}" / "checkpoint"
        checkpoint_hash = _tree_sha256(checkpoint)
        checkpoint_hashes[str(fold)] = checkpoint_hash
        rank_path = cache_dir / f"fold_{fold}_ranks.json"
        manifest_path = cache_dir / f"fold_{fold}_manifest.json"
        expected_manifest = {
            "schema_version": "task1-public-fold-ranks-v1",
            "fold": fold,
            "checkpoint_sha256": checkpoint_hash,
            "candidates_sha256": candidates_sha,
            "candidate_depth": args.candidate_depth,
            "evidence_limit": args.evidence_limit,
            "max_length": args.max_length,
        }
        ranks: dict[str, list[str]] | None = None
        if rank_path.is_file() and manifest_path.is_file():
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest == expected_manifest:
                raw_ranks = json.loads(rank_path.read_text(encoding="utf-8"))
                if isinstance(raw_ranks, dict):
                    ranks = {
                        str(question_id): [str(doc_id) for doc_id in ranking]
                        for question_id, ranking in raw_ranks.items()
                        if isinstance(ranking, list)
                    }
                    _validate_fold_ranks(ranks, candidates)
                    print(f"fold {fold}: reused validated rank cache", flush=True)
        if ranks is None:
            ranks = _score_fold(
                fold=fold,
                checkpoint=checkpoint,
                questions=questions,
                candidates=candidates,
                args=args,
            )
            _validate_fold_ranks(ranks, candidates)
            _write_json_atomic(rank_path, ranks)
            _write_json_atomic(manifest_path, expected_manifest)
        fold_outputs.append(ranks)

    labels = {} if args.no_exact_overlay else _exact_label_map(args.overlay_labels)
    predictions: list[dict[str, Any]] = []
    exact_match_count = 0
    overlay_added = 0
    for question_id, question in questions.items():
        ranked = _ensemble_documents(
            candidates[question_id],
            [fold[question_id] for fold in fold_outputs],
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            reranker_weight=args.reranker_weight,
        )
        selected = ranked[:5]
        known = labels.get(_normalize_question(question), [])
        if known:
            exact_match_count += 1
            overlay_added += len(set(known) - set(selected))
            selected = (known + [doc_id for doc_id in selected if doc_id not in known])[
                :5
            ]
        if len(selected) != 5 or len(selected) != len(set(selected)):
            raise ValueError(f"invalid final top five for {question_id}")
        predictions.append({"id": question_id, "documents": selected})
    _write_json_atomic(args.output, predictions)
    report = {
        "schema_version": "task1-finetuned-fold-ensemble-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": str(decision_path),
        "decision_sha256": _sha256(decision_path),
        "decision_status": decision["status"],
        "questions": str(args.questions),
        "questions_sha256": _sha256(args.questions),
        "candidates": str(args.candidates),
        "candidates_sha256": candidates_sha,
        "checkpoint_hashes": checkpoint_hashes,
        "settings": {
            "folds": args.folds,
            "candidate_depth": args.candidate_depth,
            "evidence_limit": args.evidence_limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "reranker_weight": args.reranker_weight,
            "exact_overlay": not args.no_exact_overlay,
        },
        "results": {
            "question_count": len(predictions),
            "exact_match_count": exact_match_count,
            "overlay_documents_added": overlay_added,
        },
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    _write_json_atomic(args.report, report)
    return args.output, args.report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"fine-tuned fold ensemble error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
