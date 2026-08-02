"""Evaluate DSC2026 LegalIR document rankings with auditable metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn, Sequence

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    LegalIREvaluationReport,
    LegalIRPrediction,
    LegalIRPredictionSet,
    LegalIRReference,
    LegalIRReferenceSet,
    evaluate_legal_ir,
    load_warmup,
)
from udsc2026.evaluation.legal_ir_submission import (  # noqa: E402
    load_legal_ir_submission,
)

OFFICIAL_MODE = "official-set"


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard constants accepted by Python's JSON decoder."""

    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode an object without silently accepting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> object:
    """Load strict UTF-8 JSON from a regular file."""

    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON file must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _load_official_references(path: Path) -> list[LegalIRReference]:
    """Load official one-or-more-gold references from supported strict shapes."""

    payload = _load_strict_json(path)
    if isinstance(payload, dict):
        samples = load_warmup(path)
        return [
            LegalIRReference(
                id=sample.id,
                gold_documents=list(sample.gold_documents),
            )
            for sample in samples
        ]

    try:
        return list(LegalIRReferenceSet.model_validate(payload, strict=True).root)
    except ValidationError as exc:
        raise ValueError(
            "official references must be a JSON array of exact "
            "{id, gold_documents} objects, or a warm-up root mapping: "
            f"{exc}"
        ) from exc


def _load_predictions(path: Path) -> list[LegalIRPrediction]:
    """Load an internal prediction array or an official object-shaped artifact."""

    suffix = path.suffix.casefold()
    if suffix == ".zip":
        items = load_legal_ir_submission(path)
        return [
            LegalIRPrediction(id=item.id, documents=list(item.documents))
            for item in items
        ]
    if suffix != ".json":
        raise ValueError("prediction input must use .json or .zip")

    payload = _load_strict_json(path)
    if isinstance(payload, dict):
        items = load_legal_ir_submission(path)
        return [
            LegalIRPrediction(id=item.id, documents=list(item.documents))
            for item in items
        ]
    if isinstance(payload, list):
        try:
            return list(LegalIRPredictionSet.model_validate(payload, strict=True).root)
        except ValidationError as exc:
            raise ValueError(
                "internal predictions must be a JSON array of exact "
                f"{{id, documents}} objects: {exc}"
            ) from exc
    raise ValueError(
        "predictions must be an internal JSON array of {id, documents} objects "
        "or an official question-id object"
    )


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


def _validate_output_path(path: Path) -> None:
    """Require a safe JSON report destination."""

    if path.suffix.casefold() != ".json":
        raise ValueError("evaluation report output must use .json")
    if path.is_symlink():
        raise ValueError("evaluation report output must not be a symbolic link")
    if path.is_dir():
        raise ValueError("evaluation report output must not be a directory")
    if path.exists() and not path.is_file():
        raise ValueError("evaluation report output must be a regular file")


def _write_report(report: LegalIREvaluationReport, path: Path) -> None:
    """Write a canonical report through a same-directory atomic replacement."""

    _validate_output_path(path)
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
            raise ValueError("evaluation report output became a symbolic link")
        if path.exists() and not path.is_file():
            raise ValueError("evaluation report output is no longer a regular file")
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


def build_parser() -> argparse.ArgumentParser:
    """Build the LegalIR evaluator command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LegalIR document IDs with official macro Recall (primary) "
            "and macro Precision (secondary). Multiple gold documents are valid; "
            "ranking order does not directly affect either set metric."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Official reference JSON or warm-up root-mapping JSON.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Internal .json array of {id, documents}, or official object-shaped "
            "submission.json/submission.zip."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(OFFICIAL_MODE,),
        default=OFFICIAL_MODE,
        help="Metric semantics (default and only current mode: official-set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task1/evaluation/report.json"),
        help="Atomic JSON report path.",
    )
    return parser


def run(args: argparse.Namespace) -> LegalIREvaluationReport:
    """Evaluate parsed CLI inputs and persist the deterministic report."""

    for input_path in (args.references, args.predictions):
        if _paths_collide(input_path, args.output):
            raise ValueError(
                "evaluation report must not overwrite an input file: "
                f"{args.output} conflicts with {input_path}"
            )

    if args.mode != OFFICIAL_MODE:
        raise ValueError(f"unsupported LegalIR evaluation mode: {args.mode}")
    references = _load_official_references(args.references)
    predictions = _load_predictions(args.predictions)
    report = evaluate_legal_ir(references, predictions)
    _write_report(report, args.output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR evaluation error: {exc}", file=sys.stderr)
        return 2
    print(
        f"mode={report.evaluation_mode} "
        f"questions={report.aggregate.sample_count} "
        f"Recall(primary)={report.aggregate.recall:.12f} "
        f"Precision(secondary)={report.aggregate.precision:.12f}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
