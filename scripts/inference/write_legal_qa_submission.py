"""Validate and atomically package an official Task 2 LegalQA submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_qa import (  # noqa: E402
    load_legal_qa_predictions,
    load_legal_qa_question_ids,
)
from udsc2026.evaluation.legal_qa_submission import (  # noqa: E402
    EmptyAnswerPolicy,
    LegalQASubmissionItem,
    load_legal_qa_submission,
    write_legal_qa_submission_zip,
)

DEFAULT_QUESTIONS = Path("data/task2/warmup.json")
DEFAULT_OUTPUT = Path(
    os.getenv("LEGAL_QA_SUBMISSION_PATH", "artifacts/task2/submission.zip")
)


def _load_question_ids(path: Path) -> list[str]:
    """Delegate question-source validation to the shared Task 2 contract."""

    return load_legal_qa_question_ids(path)


def _paths_collide(left: Path, right: Path) -> bool:
    """Detect lexical, resolved, symlink, and hard-link path collisions."""

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


def _answer_policy(value: str) -> EmptyAnswerPolicy:
    """Convert a stable CLI string to the core empty-answer policy."""

    return EmptyAnswerPolicy(value)


def _summary(
    items: list[LegalQASubmissionItem],
    output: Path,
) -> dict[str, object]:
    """Return a compact machine-readable packaging summary."""

    try:
        checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    except OSError as exc:
        raise OSError(f"cannot checksum LegalQA submission: {exc}") from exc
    return {
        "artifact_sha256": checksum,
        "empty_answer_count": sum(not item.answer.strip() for item in items),
        "member": "submission.json",
        "output": str(output),
        "question_count": len(items),
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the official LegalQA submission writer interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate an internal Task 2 prediction JSON array and write a "
            "deterministic official submission.zip. Its sole submission.json "
            "member is an object keyed by question ID."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Internal prediction .json: a root array whose items contain "
            "exactly string fields id and answer."
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
        help="How to handle empty answers before packaging (default: error).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination ZIP (default: LEGAL_QA_SUBMISSION_PATH or "
            "artifacts/task2/submission.zip)."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalQASubmissionItem]:
    """Validate, package, and independently reload the resulting ZIP."""

    if args.input.suffix.casefold() != ".json":
        raise ValueError("internal LegalQA prediction input must use .json")
    if args.output.suffix.casefold() != ".zip":
        raise ValueError("official LegalQA submission output must use .zip")
    for input_path in (args.input, args.questions):
        if _paths_collide(input_path, args.output):
            raise ValueError(
                "LegalQA submission output must not overwrite an input file: "
                f"{args.output} conflicts with {input_path}"
            )
    expected_ids = _load_question_ids(args.questions)
    policy = _answer_policy(args.empty_answer_policy)
    predictions = load_legal_qa_predictions(args.input)
    items = [
        LegalQASubmissionItem(id=prediction.id, answer=prediction.answer)
        for prediction in predictions
    ]
    write_legal_qa_submission_zip(
        items,
        args.output,
        expected_question_ids=expected_ids,
        empty_answer_policy=policy,
    )
    return load_legal_qa_submission(
        args.output,
        expected_question_ids=expected_ids,
        empty_answer_policy=policy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
        summary = _summary(items, args.output)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        print(f"LegalQA submission write error: {exc}", file=sys.stderr)
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
