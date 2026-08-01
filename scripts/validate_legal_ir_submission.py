"""Validate the official object-shaped DSC2026 LegalIR JSON/ZIP artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import load_legal_ir_question_ids  # noqa: E402
from udsc2026.evaluation.legal_ir_submission import (  # noqa: E402
    LegalIRSubmissionItem,
    load_legal_ir_submission,
)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard numeric constants in manifests."""

    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys in manifests."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    """Load a strict UTF-8 manifest file."""

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


def _validate_ids(payload: object, *, label: str) -> list[str]:
    """Validate a strict, non-empty JSON string-array manifest."""

    if not isinstance(payload, list):
        raise ValueError(f"{label} root must be a JSON array of string IDs")
    if not payload:
        raise ValueError(f"{label} must not be empty")
    validated: list[str] = []
    for value in payload:
        if not isinstance(value, str):
            raise TypeError(f"{label} must contain only strings")
        if not value:
            raise ValueError(f"{label} must not contain empty IDs")
        if value != value.strip():
            raise ValueError(f"{label} IDs must not contain surrounding whitespace")
        if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
            raise ValueError(f"{label} IDs must not contain control characters")
        validated.append(value)
    if len(validated) != len(set(validated)):
        raise ValueError(f"{label} must contain unique IDs")
    return validated


def _load_question_ids(path: Path) -> list[str]:
    """Delegate phase-question validation to the shared LegalIR contract."""

    return load_legal_ir_question_ids(path)


def _load_corpus_ids(path: Path) -> list[str]:
    """Load the strict corpus manifest contract."""

    return _validate_ids(_load_json(path), label="corpus manifest")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone validation interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the exact {question_id: {answer: [document_ids]}} LegalIR "
            "submission structure. Optional manifests enable question coverage, "
            "corpus membership, and full-ranking checks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Official object-shaped submission.json or submission.zip to validate.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help=(
            "Optional expected IDs as a JSON string array, organizer mapping, "
            "or an array of question objects; enables exact coverage validation."
        ),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Optional strict JSON array containing every valid document ID.",
    )
    parser.add_argument(
        "--require-complete-ranking",
        action="store_true",
        help=(
            "Require every question to rank the full corpus. This requires "
            "--corpus-manifest."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalIRSubmissionItem]:
    """Load optional manifests and validate the complete artifact."""

    if args.require_complete_ranking and args.corpus_manifest is None:
        raise ValueError("--require-complete-ranking requires --corpus-manifest")
    question_ids = (
        _load_question_ids(args.questions) if args.questions is not None else None
    )
    corpus_ids = (
        _load_corpus_ids(args.corpus_manifest)
        if args.corpus_manifest is not None
        else None
    )
    return load_legal_ir_submission(
        args.input,
        expected_question_ids=question_ids,
        allowed_document_ids=corpus_ids,
        require_complete_ranking=args.require_complete_ranking,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero for a valid artifact and two for every validation failure."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR submission validation error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "max_documents_per_question": max(len(item.documents) for item in items),
        "min_documents_per_question": min(len(item.documents) for item in items),
        "question_count": len(items),
        "status": "valid",
    }
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
