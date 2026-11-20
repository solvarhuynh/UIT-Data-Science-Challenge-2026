"""Build LegalQA answers from RRF-ranked, bounded parent contexts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.contracts.retrieval import RetrievalHit  # noqa: E402
from udsc2026.retrieval.parent_context import (  # noqa: E402
    JsonlParentStore,
    ParentContextExpander,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parents-dir", type=Path, default=Path("data/processed/parents")
    )
    parser.add_argument("--dense-weight", type=float, default=0.2)
    parser.add_argument("--reranker-weight", type=float, default=0.8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-chunks", type=int, default=3)
    parser.add_argument("--parent-max-tokens", type=int, default=384)
    parser.add_argument("--parent-total-tokens", type=int, default=1152)
    parser.add_argument(
        "--question-mapping",
        type=Path,
        help="Question-only mapping that limits output to those IDs, such as one held-out fold.",
    )
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def _validated_hits(row: dict[str, Any]) -> list[RetrievalHit]:
    raw_hits = row.get("hits")
    if not isinstance(raw_hits, list) or not raw_hits:
        raise ValueError(f"question {row.get('question_id')!r} has no hits")
    hits = [RetrievalHit.model_validate(hit) for hit in raw_hits]
    chunk_ids = [hit.chunk_id for hit in hits]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"question {row.get('question_id')!r} repeats chunk IDs")
    return hits


def _load_mapping(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{label} must be a non-empty JSON object: {path}")
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{label} has an invalid question ID")
        if not isinstance(record, dict):
            raise ValueError(f"{label} record {question_id!r} must be an object")
    return payload


def _load_question_order(path: Path) -> list[str]:
    mapping = _load_mapping(path, label="question-only mapping")
    for question_id, record in mapping.items():
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no question text")
    return list(mapping)


def main() -> int:
    args = build_parser().parse_args()
    weights = (args.dense_weight, args.reranker_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fusion weights must sum to 1")
    if args.rrf_k < 0 or args.top_chunks <= 0:
        raise ValueError("rrf-k must be non-negative and top-chunks must be positive")
    if args.parent_max_tokens <= 0 or args.parent_total_tokens <= 0:
        raise ValueError("parent token limits must be positive")
    if args.parent_max_tokens > args.parent_total_tokens:
        raise ValueError("parent-max-tokens must not exceed parent-total-tokens")

    dense_rows = _load_jsonl(args.dense)
    reranked_rows = _load_jsonl(args.reranked)
    if len(dense_rows) != len(reranked_rows):
        raise ValueError("dense and reranked files have different row counts")
    question_order = (
        _load_question_order(args.question_mapping)
        if args.question_mapping is not None
        else None
    )
    selected_question_ids = set(question_order) if question_order is not None else None

    expander = ParentContextExpander(
        JsonlParentStore(args.parents_dir, max_cached_documents=256),
        max_parent_tokens=args.parent_max_tokens,
        max_total_tokens=args.parent_total_tokens,
    )
    predictions: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for dense_row, reranked_row in zip(dense_rows, reranked_rows):
        question_id = dense_row.get("question_id")
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or question_id in seen_questions
        ):
            raise ValueError("question IDs must be non-empty and unique")
        if reranked_row.get("question_id") != question_id:
            raise ValueError(f"question order mismatch at {question_id!r}")
        seen_questions.add(question_id)
        if (
            selected_question_ids is not None
            and question_id not in selected_question_ids
        ):
            continue

        dense_hits = _validated_hits(dense_row)
        reranked_hits = _validated_hits(reranked_row)
        dense_rank = {hit.chunk_id: rank for rank, hit in enumerate(dense_hits, 1)}
        reranker_rank = {
            hit.chunk_id: rank for rank, hit in enumerate(reranked_hits, 1)
        }
        if set(dense_rank) != set(reranker_rank):
            raise ValueError(f"reranker candidate pool mismatch at {question_id!r}")
        hit_by_id = {hit.chunk_id: hit for hit in reranked_hits}
        ranked_chunk_ids = sorted(
            hit_by_id,
            key=lambda chunk_id: (
                -(
                    args.dense_weight / (args.rrf_k + dense_rank[chunk_id])
                    + args.reranker_weight
                    / (args.rrf_k + reranker_rank[chunk_id])
                ),
                dense_rank[chunk_id],
                reranker_rank[chunk_id],
                chunk_id,
            ),
        )
        selected = [
            hit_by_id[chunk_id] for chunk_id in ranked_chunk_ids[: args.top_chunks]
        ]
        answer = "\n".join(
            hit.text.strip() for hit in expander.expand(selected)
        ).strip()
        if not answer:
            raise ValueError(f"question {question_id!r} produced an empty answer")
        predictions.append({"id": question_id, "answer": answer})

    if question_order is not None:
        predicted_ids = {prediction["id"] for prediction in predictions}
        missing_ids = [
            question_id
            for question_id in question_order
            if question_id not in predicted_ids
        ]
        if missing_ids:
            raise ValueError(
                f"retrieval predictions are missing selected IDs: {missing_ids[:5]}"
            )
        order = {question_id: index for index, question_id in enumerate(question_order)}
        predictions.sort(key=lambda prediction: order[prediction["id"]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"questions={len(predictions)} empty_answers=0")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
