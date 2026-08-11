"""Fuse dense and reranked chunk pools into top-k LegalIR documents."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-weight", type=float, default=0.3)
    parser.add_argument("--reranker-weight", type=float, default=0.7)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
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


def _document_ranks(row: dict[str, Any]) -> tuple[list[str], set[str]]:
    hits = row.get("hits")
    if not isinstance(hits, list) or not hits:
        raise ValueError(f"question {row.get('question_id')!r} has no hits")
    documents: list[str] = []
    chunks: set[str] = set()
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("hits must contain only objects")
        chunk_id = hit.get("chunk_id")
        document_id = hit.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("every hit must have a non-empty chunk_id")
        if chunk_id in chunks:
            raise ValueError(f"duplicate chunk_id {chunk_id!r}")
        chunks.add(chunk_id)
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"chunk {chunk_id!r} has no non-empty doc_id")
        if document_id not in documents:
            documents.append(document_id)
    return documents, chunks


def main() -> int:
    args = build_parser().parse_args()
    weights = (args.dense_weight, args.reranker_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fusion weights must sum to 1")
    if args.rrf_k < 0:
        raise ValueError("rrf-k must be non-negative")
    if args.top_k < 1 or args.top_k > 5:
        raise ValueError("top-k must be between 1 and 5")

    dense_rows = _load_jsonl(args.dense)
    reranked_rows = _load_jsonl(args.reranked)
    if len(dense_rows) != len(reranked_rows):
        raise ValueError("dense and reranked files have different row counts")

    output: list[dict[str, Any]] = []
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

        dense_documents, dense_chunks = _document_ranks(dense_row)
        reranked_documents, reranked_chunks = _document_ranks(reranked_row)
        if reranked_chunks != dense_chunks:
            raise ValueError(f"reranker candidate pool mismatch at {question_id!r}")

        dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_documents, 1)}
        reranker_rank = {
            doc_id: rank for rank, doc_id in enumerate(reranked_documents, 1)
        }
        scores: dict[str, float] = {}
        for weight, ranking in zip(weights, (dense_documents, reranked_documents)):
            for rank, document_id in enumerate(ranking, 1):
                scores[document_id] = scores.get(document_id, 0.0) + weight / (
                    args.rrf_k + rank
                )
        ranked_documents = sorted(
            scores,
            key=lambda document_id: (
                -scores[document_id],
                dense_rank.get(document_id, 10**9),
                reranker_rank.get(document_id, 10**9),
                document_id,
            ),
        )
        output.append({"id": question_id, "documents": ranked_documents[: args.top_k]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"questions={len(output)} top_k={args.top_k}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
