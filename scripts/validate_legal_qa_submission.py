"""Strictly validate an official DSC2026 Task 2 LegalQA submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_qa import (  # noqa: E402
    load_legal_qa_question_ids,
)
from udsc2026.evaluation.legal_qa_submission import (  # noqa: E402
    EmptyAnswerPolicy,
    LegalQASubmissionItem,
    load_legal_qa_submission,
)

DEFAULT_QUESTIONS = Path("data/task2/warmup.json")


def _load_question_ids(path: Path) -> list[str]:
    """Delegate question-source validation to the shared Task 2 contract."""

    return load_legal_qa_question_ids(path)


def _answer_policy(value: str) -> EmptyAnswerPolicy:
    """Convert the stable CLI representation to the core enum."""

    return EmptyAnswerPolicy(value)


def _summary(
    items: list[LegalQASubmissionItem],
    input_path: Path,
) -> dict[str, object]:
    """Return a compact machine-readable validation summary."""

    try:
        checksum = hashlib.sha256(input_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise OSError(f"cannot checksum LegalQA submission: {exc}") from exc
    return {
        "artifact_sha256": checksum,
        "empty_answer_count": sum(not item.answer.strip() for item in items),
        "input": str(input_path),
        "member": "submission.json",
        "question_count": len(items),
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Task 2 ZIP validator interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate an official Task 2 submission ZIP whose sole "
            "submission.json member is an object keyed by question ID, plus "
            "exact coverage, raw answer preservation, and empty-answer policy."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Official submission.zip; submission.json must use "
            "{question_id: {answer: string}}."
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"Expected question source (default: {DEFAULT_QUESTIONS}).",
    )
    parser.add_argument(
        "--empty-answer-policy",
        choices=("allow", "warn", "error"),
        default="error",
        help="How to handle empty answers (default: error).",
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalQASubmissionItem]:
    """Load question coverage and strictly validate one official ZIP."""

    if args.input.suffix.casefold() != ".zip":
        raise ValueError("official LegalQA submission input must use .zip")
    return load_legal_qa_submission(
        args.input,
        expected_question_ids=_load_question_ids(args.questions),
        empty_answer_policy=_answer_policy(args.empty_answer_policy),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero for a valid ZIP and two for every validation failure."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
        summary = _summary(items, args.input)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        print(f"LegalQA submission validation error: {exc}", file=sys.stderr)
        return 2
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
