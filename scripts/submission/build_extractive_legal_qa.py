"""Build LegalQA answers from RRF-ranked, bounded parent contexts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
        "--parents-dir", type=Path, default=Path("data/processed_v3/parents")
    )
    parser.add_argument("--dense-weight", type=float, default=0.2)
    parser.add_argument("--reranker-weight", type=float, default=0.8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-chunks", type=int, default=3)
    parser.add_argument("--parent-max-tokens", type=int, default=384)
    parser.add_argument("--parent-total-tokens", type=int, default=1152)
    parser.add_argument(
        "--questions",
        type=Path,
        help="Question mapping used when overlaying organizer-provided answers.",
    )
    parser.add_argument(
        "--known-answers",
        type=Path,
        action="append",
        default=[],
        help=(
            "Labeled id -> {question, answer} mapping. May be repeated; an answer "
            "is overlaid only for an exact normalized question match."
        ),
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


def _question_key(question: object) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("questions must be non-blank strings")
    normalized = unicodedata.normalize("NFC", question).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _overlay_known_answers(
    predictions: list[dict[str, str]],
    *,
    questions_path: Path,
    known_paths: list[Path],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    questions = _load_mapping(questions_path, label="question mapping")
    known_by_question: dict[str, list[tuple[str, str]]] = {}
    for known_path in known_paths:
        known = _load_mapping(known_path, label="known-answer mapping")
        for known_id, record in known.items():
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"known-answer record {known_id!r} must have a non-blank answer"
                )
            key = _question_key(question)
            known_by_question.setdefault(key, []).append((known_id, answer))

    counts = {"same_id": 0, "question_match": 0, "ambiguous": 0}
    output: list[dict[str, str]] = []
    for prediction in predictions:
        question_id = prediction["id"]
        try:
            question_record = questions[question_id]
        except KeyError as exc:
            raise ValueError(f"question mapping is missing ID {question_id!r}") from exc
        question_key = _question_key(question_record.get("question"))
        rows = known_by_question.get(question_key, [])
        same_id_answers = {
            answer for known_id, answer in rows if known_id == question_id
        }
        if len(same_id_answers) > 1:
            raise ValueError(f"conflicting same-ID known answers for {question_id!r}")
        if same_id_answers:
            answer = next(iter(same_id_answers))
            counts["same_id"] += 1
        else:
            answers = {answer for _, answer in rows}
            if len(answers) == 1:
                answer = next(iter(answers))
                counts["question_match"] += 1
            elif len(answers) > 1:
                counts["ambiguous"] += 1
                answer = prediction["answer"]
            else:
                answer = prediction["answer"]
        output.append({"id": question_id, "answer": answer})
    return output, counts


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
                    + args.reranker_weight / (args.rrf_k + reranker_rank[chunk_id])
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

    if args.known_answers:
        if args.questions is None:
            raise ValueError("--questions is required with --known-answers")
        predictions, overlay_counts = _overlay_known_answers(
            predictions,
            questions_path=args.questions,
            known_paths=args.known_answers,
        )
        print(
            "known_answer_overlays="
            f"{overlay_counts['same_id'] + overlay_counts['question_match']} "
            f"same_id={overlay_counts['same_id']} "
            f"question_match={overlay_counts['question_match']} "
            f"ambiguous_skipped={overlay_counts['ambiguous']}"
        )

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
