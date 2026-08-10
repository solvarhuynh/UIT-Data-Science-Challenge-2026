"""Adapt organizer question mappings to the evaluation benchmark contract.

The resulting JSONL is an internal, label-free transport artifact.  Its dummy
reference fields exist only because the shared dense/reranker CLIs validate the
benchmark schema; they must never be interpreted as evaluation labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("public question input must be a non-empty JSON object")

    rows: list[str] = []
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question IDs must be non-empty strings")
        if not isinstance(record, dict):
            raise ValueError(f"question {question_id!r} must be an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no non-empty text")
        row = {
            "question_id": question_id,
            "question": question,
            "answer": "LABEL_NOT_AVAILABLE",
            "gold_chunk_ids": ["LABEL_NOT_AVAILABLE"],
            "gold_citations": [],
            "difficulty": "easy",
            "question_type": "public_official_unlabeled",
            "metadata": {"unlabeled_public_question": True},
        }
        rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(rows), encoding="utf-8", newline="\n")
    print(f"adapted_questions={len(rows)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
