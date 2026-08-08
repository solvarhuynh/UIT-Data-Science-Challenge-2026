"""Create a guarded answer-leaking Task 2 ZIP for local smoke tests only."""

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

from udsc2026.evaluation.legal_qa import load_legal_qa_warmup  # noqa: E402
from udsc2026.evaluation.legal_qa_submission import (  # noqa: E402
    EmptyAnswerPolicy,
    LegalQASubmissionItem,
    load_legal_qa_submission,
    write_legal_qa_submission_zip,
)

DEFAULT_WARMUP = Path("data/task2/warmup.json")
DEFAULT_OUTPUT = Path("artifacts/task2/oracle_DO_NOT_SUBMIT.zip")
WARNING = (
    "WARNING: THIS TASK 2 ARTIFACT DIRECTLY COPIES REFERENCE ANSWERS. "
    "IT LEAKS LABELS, IS ONLY FOR LOCAL PIPELINE TESTING, AND MUST NEVER BE "
    "UPLOADED OR SUBMITTED."
)


def _paths_collide(left: Path, right: Path) -> bool:
    """Detect lexical, resolved, symlink, and hard-link path aliases."""

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


def _summary(
    items: list[LegalQASubmissionItem],
    output: Path,
) -> dict[str, object]:
    """Return a machine-readable local-only artifact summary."""

    try:
        checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    except OSError as exc:
        raise OSError(f"cannot checksum LegalQA oracle smoke ZIP: {exc}") from exc
    return {
        "artifact_sha256": checksum,
        "label_leakage": True,
        "output": str(output),
        "question_count": len(items),
        "status": "local_only_do_not_submit",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the guarded Task 2 oracle-smoke interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Copy warm-up references into an official object-shaped ZIP "
            "solely to test local packaging. This artifact must never be "
            "uploaded."
        )
    )
    parser.add_argument(
        "--warmup",
        type=Path,
        default=DEFAULT_WARMUP,
        help=f"Canonical Task 2 warm-up JSON (default: {DEFAULT_WARMUP}).",
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
            "Required acknowledgement that references are copied and the ZIP "
            "must never be uploaded or treated as a model baseline."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[LegalQASubmissionItem]:
    """Create and revalidate the explicitly acknowledged oracle artifact."""

    if not args.acknowledge_label_leakage:
        raise ValueError(
            "refusing to copy reference answers without --acknowledge-label-leakage"
        )
    if args.output.suffix.casefold() != ".zip":
        raise ValueError("LegalQA oracle smoke output must use .zip")
    if "DO_NOT_SUBMIT" not in args.output.name:
        raise ValueError(
            "LegalQA oracle smoke output filename must contain DO_NOT_SUBMIT"
        )
    if _paths_collide(args.warmup, args.output):
        raise ValueError("LegalQA oracle output must not overwrite warm-up input")

    samples = load_legal_qa_warmup(args.warmup)
    payload = {sample.id: {"answer": sample.answer} for sample in samples}
    expected_ids = [sample.id for sample in samples]
    write_legal_qa_submission_zip(
        payload,
        args.output,
        expected_question_ids=expected_ids,
        empty_answer_policy=EmptyAnswerPolicy.ERROR,
    )
    items = load_legal_qa_submission(
        args.output,
        expected_question_ids=expected_ids,
        empty_answer_policy=EmptyAnswerPolicy.ERROR,
    )
    item_by_id = {item.id: item.answer for item in items}
    for sample in samples:
        if item_by_id.get(sample.id) != sample.answer:
            raise ValueError("oracle ZIP round-trip changed an ID or reference answer")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on guarded success and two for any refusal or failure."""

    args = build_parser().parse_args(argv)
    try:
        items = run(args)
        summary = _summary(items, args.output)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        print(f"LegalQA oracle smoke error: {exc}", file=sys.stderr)
        return 2
    print("=" * 79, file=sys.stderr)
    print(WARNING, file=sys.stderr)
    print("=" * 79, file=sys.stderr)
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
