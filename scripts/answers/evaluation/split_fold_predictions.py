"""Split complete predictions into the existing held-out answer folds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--filename", default="predictions.json")
    parser.add_argument("--fold-count", type=int, default=5)
    return parser


def _load_question_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"question mapping must be a non-empty object: {path}")
    ids: list[str] = []
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"invalid question ID in {path}")
        if not isinstance(record, dict):
            raise ValueError(f"question {question_id!r} in {path} must be an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} in {path} has no text")
        ids.append(question_id)
    return ids


def _load_predictions(path: Path) -> dict[str, dict[str, str]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"predictions must be a non-empty array: {path}")
    predictions: dict[str, dict[str, str]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError(f"prediction {index} must contain exactly id and answer")
        question_id = row["id"]
        answer = row["answer"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"prediction {index} has an invalid ID")
        if question_id in predictions:
            raise ValueError(f"duplicate prediction ID: {question_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"prediction {question_id!r} has an empty answer")
        predictions[question_id] = {"id": question_id, "answer": answer}
    return predictions


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run(args: argparse.Namespace) -> list[Path]:
    if args.fold_count < 2:
        raise ValueError("fold-count must be at least two")
    if Path(args.filename).name != args.filename:
        raise ValueError("filename must be a single path segment")
    if Path(args.filename).suffix.casefold() != ".json":
        raise ValueError("filename must use .json")
    reserved_filenames = {
        "questions.json",
        "references.json",
        "folds.json",
        "manifest.json",
    }
    if args.filename.casefold() in reserved_filenames:
        raise ValueError("filename is reserved for fold input metadata")
    predictions = _load_predictions(args.predictions)
    assignments: list[tuple[Path, list[str]]] = []
    seen: set[str] = set()
    for fold_index in range(args.fold_count):
        fold_dir = args.folds_dir / f"fold_{fold_index}"
        question_ids = _load_question_ids(fold_dir / "questions.json")
        overlap = seen.intersection(question_ids)
        if overlap:
            raise ValueError(f"question IDs occur in multiple folds: {sorted(overlap)[:5]}")
        seen.update(question_ids)
        assignments.append((fold_dir / args.filename, question_ids))

    if seen != set(predictions):
        raise ValueError(
            "fold/prediction coverage mismatch: "
            f"missing={sorted(seen - set(predictions))[:5]}, "
            f"extra={sorted(set(predictions) - seen)[:5]}"
        )

    input_path = args.predictions.resolve()
    collisions = [
        path for path, _ in assignments if path.resolve() == input_path
    ]
    if collisions:
        raise ValueError(f"fold output would overwrite its input: {collisions[0]}")

    written: list[Path] = []
    for path, question_ids in assignments:
        _write_json(path, [predictions[question_id] for question_id in question_ids])
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        for path in run(args):
            print(path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalQA fold split error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
