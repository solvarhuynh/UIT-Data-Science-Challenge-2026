"""Convert aligned extractive answers into source-only generation rankings.

This lets P15 reuse the already paid-for P14 retrieval/cross-encoder output.
No references are read, and no new parent-scoring GPU pass is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--extractive", type=Path)
    source.add_argument(
        "--submission",
        type=Path,
        help="Recover the bounded evidence prefix from a prior submission zip.",
    )
    parser.add_argument("--evidence-words", type=int, default=352)
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Also save the aligned evidence rows used to build rankings.",
    )
    return parser


def build_rankings(
    questions: dict[str, Any],
    predictions: list[dict[str, Any]],
    question_ids: list[str],
) -> list[dict[str, Any]]:
    mapped: dict[str, str] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError("extractive rows must contain exactly id and answer")
        question_id = str(row["id"])
        answer = row["answer"]
        if question_id in mapped or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"invalid extractive prediction {question_id!r}")
        mapped[question_id] = answer.strip()
    if missing := set(question_ids) - set(questions):
        raise ValueError(f"questions are missing IDs: {sorted(missing)[:5]}")
    if missing := set(question_ids) - set(mapped):
        raise ValueError(
            f"extractive predictions are missing IDs: {sorted(missing)[:5]}"
        )
    return [
        {
            "question_id": question_id,
            "parents": [
                {
                    "rank": 1,
                    "candidate_rank": 1,
                    "doc_id": "p15-extractive",
                    "parent_id": f"p15-extractive-{question_id}",
                    "law_name": "",
                    "article": "",
                    "anchor_text": "",
                    "parent_text": mapped[question_id],
                }
            ],
        }
        for question_id in question_ids
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_submission_prefixes(path: Path, evidence_words: int) -> list[dict[str, str]]:
    if evidence_words < 1:
        raise ValueError("evidence-words must be positive")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if names != ["submission.json"]:
            raise ValueError("submission zip must contain exactly submission.json")
        payload = json.loads(archive.read("submission.json").decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("submission.json must be an object")
    rows: list[dict[str, str]] = []
    for question_id, value in payload.items():
        answer = value.get("answer") if isinstance(value, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"submission answer is invalid for {question_id!r}")
        prefix = " ".join(answer.split()[:evidence_words])
        rows.append({"id": str(question_id), "answer": prefix})
    return rows


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    predictions = (
        _load_submission_prefixes(args.submission, args.evidence_words)
        if args.submission is not None
        else json.loads(args.extractive.read_text(encoding="utf-8-sig"))
    )
    if not isinstance(questions, dict) or not isinstance(predictions, list):
        raise ValueError("invalid questions or extractive payload")
    question_ids = (
        [str(value) for value in json.loads(args.question_ids.read_text())]
        if args.question_ids is not None
        else list(questions)
    )
    rows = build_rankings(questions, predictions, question_ids)
    _write(args.output, rows)
    aligned_predictions = [
        {"id": row["question_id"], "answer": row["parents"][0]["parent_text"]}
        for row in rows
    ]
    if args.predictions_output is not None:
        _write_json(args.predictions_output, aligned_predictions)
    manifest = args.output.with_name(f"{args.output.stem}.manifest.json")
    _write_json(
        manifest,
        {
            "schema_version": "task2-p15-extractive-rankings-v1",
            "question_count": len(rows),
            "output": str(args.output),
            "output_sha256": _sha256(args.output),
            "questions": str(args.questions),
            "extractive": str(args.extractive) if args.extractive else None,
            "submission": str(args.submission) if args.submission else None,
            "evidence_words": (
                args.evidence_words if args.submission is not None else None
            ),
            "question_ids": str(args.question_ids) if args.question_ids else None,
            "predictions_output": (
                str(args.predictions_output) if args.predictions_output else None
            ),
            "label_free": True,
        },
    )
    return args.output, manifest


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task 2 P15 ranking conversion error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
