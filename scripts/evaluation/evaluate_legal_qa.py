"""Evaluate Task 2 LegalQA predictions with the local diagnostic profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_qa import (  # noqa: E402
    LegalQAEvaluationReport,
    LegalQAPrediction,
    evaluate_legal_qa,
    load_legal_qa_predictions,
    load_legal_qa_warmup,
)
from udsc2026.evaluation.legal_qa_submission import (  # noqa: E402
    EmptyAnswerPolicy,
    load_legal_qa_submission,
)

DEFAULT_REFERENCES = Path("data/task2/warmup.json")
DEFAULT_OUTPUT = Path("artifacts/task2/evaluation/diagnostic_report.json")


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


def _write_report_atomic(report: LegalQAEvaluationReport, path: Path) -> None:
    """Write one canonical report through a same-directory atomic replace."""

    if path.suffix.casefold() != ".json":
        raise ValueError("LegalQA evaluation report output must use .json")
    if path.is_symlink():
        raise ValueError("LegalQA evaluation output must not be a symbolic link")
    if path.is_dir():
        raise ValueError("LegalQA evaluation output must not be a directory")
    if path.exists() and not path.is_file():
        raise ValueError("LegalQA evaluation output must be a regular file")
    encoded = (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError("LegalQA evaluation output became a symbolic link")
        if path.exists() and not path.is_file():
            raise ValueError("LegalQA evaluation output is no longer a regular file")
        os.replace(temporary_name, path)
        replaced = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not replaced:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _summary(
    report: LegalQAEvaluationReport,
    output: Path,
) -> dict[str, object]:
    """Return a compact machine-readable diagnostic summary."""

    profile = report.profile
    if hasattr(profile, "model_dump"):
        profile_value: object = profile.model_dump(mode="json")
    else:
        profile_value = profile
    return {
        "empty_prediction_count": report.aggregate.empty_prediction_count,
        "evaluation_scope": report.evaluation_scope,
        "meteor": report.aggregate.meteor,
        "official_scorer_parity": report.official_scorer_parity,
        "output": str(output),
        "profile": profile_value,
        "question_count": report.aggregate.sample_count,
        "rouge_l": report.aggregate.rouge_l,
        "status": "valid",
    }


def _load_predictions(
    path: Path,
    *,
    expected_question_ids: list[str],
) -> list[LegalQAPrediction]:
    """Load the declared internal JSON or official ZIP representation."""

    suffix = path.suffix.casefold()
    if suffix == ".json":
        return load_legal_qa_predictions(path)
    if suffix == ".zip":
        submission_items = load_legal_qa_submission(
            path,
            expected_question_ids=expected_question_ids,
            empty_answer_policy=EmptyAnswerPolicy.ALLOW,
        )
        return [
            LegalQAPrediction(id=item.id, answer=item.answer)
            for item in submission_items
        ]
    raise ValueError("LegalQA predictions must use internal .json or official .zip")


def build_parser() -> argparse.ArgumentParser:
    """Build the local diagnostic LegalQA evaluator interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an internal Task 2 prediction JSON array or an official "
            "object-shaped submission ZIP with deterministic diagnostic "
            "METEOR and ROUGE-L. This does not claim official scorer parity."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES,
        help=f"Reference warm-up mapping (default: {DEFAULT_REFERENCES}).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Internal .json array of exact {id, answer} records, or official "
            ".zip with object-shaped submission.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Atomic diagnostic JSON report (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def run(args: argparse.Namespace) -> LegalQAEvaluationReport:
    """Load aligned inputs, evaluate them, and persist the report."""

    for input_path in (args.references, args.predictions):
        if _paths_collide(input_path, args.output):
            raise ValueError(
                "LegalQA evaluation output must not overwrite an input file: "
                f"{args.output} conflicts with {input_path}"
            )
    references = load_legal_qa_warmup(args.references)
    predictions = _load_predictions(
        args.predictions,
        expected_question_ids=[sample.id for sample in references],
    )
    report = evaluate_legal_qa(references, predictions)
    _write_report_atomic(report, args.output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        summary = _summary(report, args.output)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        print(f"LegalQA diagnostic evaluation error: {exc}", file=sys.stderr)
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
