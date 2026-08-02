"""Create a label-leaking warm-up artifact for local pipeline smoke tests only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import load_warmup  # noqa: E402
from udsc2026.evaluation.legal_ir_submission import (  # noqa: E402
    LegalIRSubmissionItem,
    load_legal_ir_submission,
    write_legal_ir_submission_zip,
)

DEFAULT_OUTPUT = Path("artifacts/task1/oracle_DO_NOT_SUBMIT.zip")
WARNING = (
    "WARNING: THIS ARTIFACT DIRECTLY LEAKS WARM-UP ANSWER LABELS. "
    "IT IS ONLY FOR LOCAL PIPELINE TESTING. NEVER UPLOAD OR SUBMIT IT."
)


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
    """Build the guarded oracle-smoke command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic answer-label oracle solely to smoke-test "
            "local JSON/ZIP/evaluation plumbing. Never upload this artifact."
        )
    )
    parser.add_argument(
        "--warmup",
        type=Path,
        default=Path("data/task1/warmup.json"),
        help=("Task 1 Warm-up root-mapping JSON (default: data/task1/warmup.json)."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output ZIP whose filename must contain DO_NOT_SUBMIT "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--acknowledge-label-leakage",
        action="store_true",
        help=(
            "Required acknowledgement that this uses answer labels and must "
            "never be uploaded as a competition submission."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalIRSubmissionItem]:
    """Create and revalidate the guarded, deterministic oracle ZIP."""

    if not args.acknowledge_label_leakage:
        raise ValueError(
            "refusing to expose answer labels without --acknowledge-label-leakage"
        )
    if args.output.suffix.casefold() != ".zip":
        raise ValueError("oracle smoke output must use .zip")
    if "DO_NOT_SUBMIT" not in args.output.name:
        raise ValueError("oracle smoke output filename must contain DO_NOT_SUBMIT")
    if _paths_collide(args.warmup, args.output):
        raise ValueError("oracle smoke output must not overwrite warm-up input")

    samples = load_warmup(args.warmup)
    labeled_document_ids = sorted(
        {document_id for sample in samples for document_id in sample.gold_documents}
    )
    items = [
        LegalIRSubmissionItem(
            id=sample.id,
            documents=tuple(sample.gold_documents),
        )
        for sample in samples
    ]

    question_ids = [sample.id for sample in samples]
    write_legal_ir_submission_zip(
        items,
        args.output,
        expected_question_ids=question_ids,
        allowed_document_ids=labeled_document_ids,
    )
    return load_legal_ir_submission(
        args.output,
        expected_question_ids=question_ids,
        allowed_document_ids=labeled_document_ids,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on guarded success and two for every refusal/failure."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR oracle smoke error: {exc}", file=sys.stderr)
        return 2
    print("=" * 79, file=sys.stderr)
    print(WARNING, file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    print(
        f"local-only oracle questions={len(items)} "
        f"min_answer_documents={min(len(item.documents) for item in items)} "
        f"max_answer_documents={max(len(item.documents) for item in items)}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
