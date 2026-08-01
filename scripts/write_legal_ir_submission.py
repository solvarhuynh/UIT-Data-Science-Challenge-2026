"""Validate and write an official DSC2026 LegalIR JSON/ZIP submission."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    LegalIRPredictionSet,
    load_legal_ir_question_ids,
)
from udsc2026.evaluation.legal_ir_submission import (  # noqa: E402
    LegalIRSubmissionItem,
    load_legal_ir_submission,
    write_legal_ir_submission,
)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard numeric constants in manifest files."""

    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys in manifest files."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    """Load a strict UTF-8 JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"manifest must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _validate_id(value: object, *, label: str) -> str:
    """Validate an opaque ID without coercion or normalization."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must contain only strings")
    if not value:
        raise ValueError(f"{label} must not contain empty IDs")
    if value != value.strip():
        raise ValueError(f"{label} IDs must not contain surrounding whitespace")
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        raise ValueError(f"{label} IDs must not contain control characters")
    return value


def _validate_unique_ids(values: list[object], *, label: str) -> list[str]:
    """Validate a non-empty, duplicate-free JSON array of strings."""

    if not values:
        raise ValueError(f"{label} must not be empty")
    validated = [_validate_id(value, label=label) for value in values]
    if len(validated) != len(set(validated)):
        raise ValueError(f"{label} must contain unique IDs")
    return validated


def _load_question_ids(path: Path) -> list[str]:
    """Delegate phase-question validation to the shared LegalIR contract."""

    return load_legal_ir_question_ids(path)


def _load_corpus_ids(path: Path) -> list[str]:
    """Load the deliberately simple corpus manifest contract."""

    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("corpus manifest root must be a JSON array of string IDs")
    return _validate_unique_ids(payload, label="corpus manifest")


def _load_candidate_items(path: Path) -> list[LegalIRSubmissionItem]:
    """Load internal prediction JSON or an already official JSON/ZIP artifact."""

    suffix = path.suffix.casefold()
    if suffix == ".zip":
        return load_legal_ir_submission(path)
    if suffix != ".json":
        raise ValueError("candidate input must use .json or .zip")

    payload = _load_json(path)
    if isinstance(payload, dict):
        return load_legal_ir_submission(path)
    if not isinstance(payload, list):
        raise ValueError(
            "candidate must be an internal JSON array of {id, documents} objects "
            "or an official question-id object"
        )
    try:
        predictions = LegalIRPredictionSet.model_validate(payload, strict=True).root
    except ValidationError as exc:
        raise ValueError(
            "internal candidate predictions must be a JSON array of exact "
            f"{{id, documents}} objects: {exc}"
        ) from exc
    return [
        LegalIRSubmissionItem(
            id=prediction.id,
            documents=tuple(prediction.documents),
        )
        for prediction in predictions
    ]


def _paths_collide(left: Path, right: Path) -> bool:
    """Detect resolved, symlink, and hard-link path aliases."""

    try:
        if left.resolve(strict=True) == right.resolve(strict=False):
            return True
    except OSError:
        pass
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    return False


def build_parser() -> argparse.ArgumentParser:
    """Build the LegalIR submission-writer interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate exact question coverage and write deterministic "
            "submission.json or submission.zip."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Internal prediction .json array of {id, documents}, or an existing "
            "official object-shaped .json/.zip artifact."
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help=(
            "Expected IDs as a JSON string array, organizer mapping "
            "id -> {question, answer?}, or an array of {id, question?, "
            "answer?} objects. Exact coverage is always enforced."
        ),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Optional strict JSON array containing every valid document ID.",
    )
    parser.add_argument(
        "--append-missing-corpus",
        action="store_true",
        help=(
            "Append every unranked corpus ID in manifest order while preserving "
            "the model ranking prefix. Requires --corpus-manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.getenv(
                "LEGAL_IR_SUBMISSION_PATH",
                "artifacts/task1/submission.zip",
            )
        ),
        help=(
            "Destination .json or .zip (default: LEGAL_IR_SUBMISSION_PATH or "
            "artifacts/task1/submission.zip)."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalIRSubmissionItem]:
    """Validate, write, and independently reload a submission artifact."""

    protected = [args.input, args.questions]
    if args.corpus_manifest is not None:
        protected.append(args.corpus_manifest)
    for source in protected:
        if _paths_collide(source, args.output):
            raise ValueError(
                "submission output must not overwrite an input file: "
                f"{args.output} conflicts with {source}"
            )
    if args.append_missing_corpus and args.corpus_manifest is None:
        raise ValueError("--append-missing-corpus requires --corpus-manifest")

    question_ids = _load_question_ids(args.questions)
    corpus_ids = (
        _load_corpus_ids(args.corpus_manifest)
        if args.corpus_manifest is not None
        else None
    )
    items = _load_candidate_items(args.input)
    write_legal_ir_submission(
        items,
        args.output,
        expected_question_ids=question_ids,
        allowed_document_ids=(None if args.append_missing_corpus else corpus_ids),
        complete_with_document_ids=(corpus_ids if args.append_missing_corpus else None),
    )
    return load_legal_ir_submission(
        args.output,
        expected_question_ids=question_ids,
        allowed_document_ids=corpus_ids,
        require_complete_ranking=args.append_missing_corpus,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR submission write error: {exc}", file=sys.stderr)
        return 2
    print(
        f"valid questions={len(items)} "
        f"min_documents={min(len(item.documents) for item in items)} "
        f"max_documents={max(len(item.documents) for item in items)}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
