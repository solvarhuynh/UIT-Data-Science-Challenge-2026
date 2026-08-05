"""Audit DSC2026 Task 2 LegalQA warm-up data without scoring a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_qa import (  # noqa: E402
    LegalQAWarmupSample,
    load_legal_qa_warmup,
)
from udsc2026.evaluation.legal_qa_metrics import (  # noqa: E402
    tokenize_legal_qa_text,
)

DEFAULT_INPUT = Path("data/task2/warmup.json")
DEFAULT_OUTPUT = Path("artifacts/task2/warmup_audit.json")


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


def _length_summary(values: list[int]) -> dict[str, int | float]:
    """Summarize non-empty integer lengths deterministically."""

    if not values:
        raise ValueError("length summary requires at least one value")
    ordered = sorted(values)
    percentile_index = int(0.95 * (len(ordered) - 1))
    return {
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p95_nearest_rank": ordered[percentile_index],
    }


def _duplicate_summary(values: list[str]) -> dict[str, int]:
    """Count duplicate groups and all records participating in those groups."""

    frequencies = Counter(values)
    duplicate_counts = [count for count in frequencies.values() if count > 1]
    return {
        "group_count": len(duplicate_counts),
        "record_count": sum(duplicate_counts),
    }


def _normalized_for_duplicate_check(value: str) -> str:
    """Return a comparison-only NFC/case/whitespace representation."""

    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _character_anomaly(
    samples: list[LegalQAWarmupSample],
    values: list[str],
    *,
    character: str,
) -> dict[str, object]:
    """Report exact record and occurrence counts for one Unicode character."""

    ids = sorted(
        sample.id for sample, value in zip(samples, values) if character in value
    )
    return {
        "ids": ids,
        "occurrence_count": sum(value.count(character) for value in values),
        "record_count": len(ids),
    }


def _text_audit(
    samples: list[LegalQAWarmupSample],
    *,
    field: str,
) -> dict[str, object]:
    """Build text-integrity and diagnostic-length statistics for one field."""

    values = [getattr(sample, field) for sample in samples]
    non_nfc_ids = sorted(
        sample.id
        for sample, value in zip(samples, values)
        if not unicodedata.is_normalized("NFC", value)
    )
    surrounding_whitespace_ids = sorted(
        sample.id for sample, value in zip(samples, values) if value != value.strip()
    )
    collapsed_whitespace_ids = sorted(
        sample.id
        for sample, value in zip(samples, values)
        if value != " ".join(value.split())
    )
    multiline_ids = sorted(
        sample.id for sample, value in zip(samples, values) if "\n" in value
    )
    carriage_return_ids = sorted(
        sample.id for sample, value in zip(samples, values) if "\r" in value
    )
    empty_ids = sorted(
        sample.id for sample, value in zip(samples, values) if not value.strip()
    )
    format_character_ids = sorted(
        sample.id
        for sample, value in zip(samples, values)
        if any(unicodedata.category(character) == "Cf" for character in value)
    )
    blank_line_ids = sorted(
        sample.id
        for sample, value in zip(samples, values)
        if any(not line.strip() for line in value.splitlines())
    )
    trailing_line_whitespace_ids = sorted(
        sample.id
        for sample, value in zip(samples, values)
        if any(line != line.rstrip(" \t\u00a0") for line in value.splitlines())
    )
    return {
        "blank_line": {
            "ids": blank_line_ids,
            "occurrence_count": sum(
                sum(not line.strip() for line in value.splitlines()) for value in values
            ),
            "record_count": len(blank_line_ids),
        },
        "character_length": _length_summary([len(value) for value in values]),
        "collapsed_whitespace_change_count": len(collapsed_whitespace_ids),
        "collapsed_whitespace_change_ids": collapsed_whitespace_ids,
        "contains_carriage_return_count": len(carriage_return_ids),
        "contains_carriage_return_ids": carriage_return_ids,
        "duplicate_normalized": _duplicate_summary(
            [_normalized_for_duplicate_check(value) for value in values]
        ),
        "duplicate_raw": _duplicate_summary(values),
        "empty_or_blank_count": len(empty_ids),
        "empty_or_blank_ids": empty_ids,
        "format_character_cf": {
            "ids": format_character_ids,
            "occurrence_count": sum(
                sum(unicodedata.category(character) == "Cf" for character in value)
                for value in values
            ),
            "record_count": len(format_character_ids),
        },
        "embedded_bom_feff": _character_anomaly(
            samples,
            values,
            character="\ufeff",
        ),
        "multiline_count": len(multiline_ids),
        "multiline_ids": multiline_ids,
        "non_breaking_space": _character_anomaly(
            samples,
            values,
            character="\u00a0",
        ),
        "non_nfc_count": len(non_nfc_ids),
        "non_nfc_ids": non_nfc_ids,
        "soft_hyphen": _character_anomaly(
            samples,
            values,
            character="\u00ad",
        ),
        "surrounding_whitespace_count": len(surrounding_whitespace_ids),
        "surrounding_whitespace_ids": surrounding_whitespace_ids,
        "token_length_diagnostic": _length_summary(
            [len(tokenize_legal_qa_text(value)) for value in values]
        ),
        "trailing_line_whitespace": {
            "ids": trailing_line_whitespace_ids,
            "occurrence_count": sum(
                sum(line != line.rstrip(" \t\u00a0") for line in value.splitlines())
                for value in values
            ),
            "record_count": len(trailing_line_whitespace_ids),
        },
    }


def build_audit(
    samples: list[LegalQAWarmupSample],
    *,
    dataset_sha256: str,
    source_size_bytes: int,
) -> dict[str, object]:
    """Build the deterministic Task 2 warm-up audit payload."""

    if not samples:
        raise ValueError("LegalQA warm-up audit requires at least one question")
    ids = [sample.id for sample in samples]
    return {
        "answer": _text_audit(samples, field="answer"),
        "dataset_sha256": dataset_sha256,
        "id": {
            "all_numeric_looking_count": sum(value.isdigit() for value in ids),
            "duplicate": _duplicate_summary(ids),
            "max_length": max(map(len, ids)),
            "min_length": min(map(len, ids)),
        },
        "question": _text_audit(samples, field="question"),
        "question_count": len(samples),
        "reference_answer_count": len(samples),
        "schema_version": "legal-qa-warmup-audit-v1",
        "source_size_bytes": source_size_bytes,
    }


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    """Write canonical UTF-8 JSON through a same-directory atomic replace."""

    if path.suffix.casefold() != ".json":
        raise ValueError("LegalQA warm-up audit output must use .json")
    if path.is_symlink():
        raise ValueError("LegalQA warm-up audit output must not be a symbolic link")
    if path.is_dir():
        raise ValueError("LegalQA warm-up audit output must not be a directory")
    if path.exists() and not path.is_file():
        raise ValueError("LegalQA warm-up audit output must be a regular file")
    encoded = (
        json.dumps(
            payload,
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
            raise ValueError("LegalQA warm-up audit output became a symbolic link")
        if path.exists() and not path.is_file():
            raise ValueError("LegalQA warm-up audit output is no longer a regular file")
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


def _summary(report: dict[str, object], output: Path) -> dict[str, object]:
    """Return a compact machine-readable success summary."""

    question = report["question"]
    answer = report["answer"]
    if not isinstance(question, dict) or not isinstance(answer, dict):
        raise TypeError("audit report text sections must be objects")
    return {
        "answer_multiline_count": answer["multiline_count"],
        "answer_non_nfc_count": answer["non_nfc_count"],
        "dataset_sha256": report["dataset_sha256"],
        "output": str(output),
        "question_count": report["question_count"],
        "question_non_nfc_count": question["non_nfc_count"],
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the LegalQA warm-up audit interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate Task 2 warmup.json and report Unicode, "
            "whitespace, duplicate, length, and checksum diagnostics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Canonical Task 2 warm-up JSON (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Atomic audit JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Validate, audit, and atomically persist one warm-up dataset."""

    if _paths_collide(args.input, args.output):
        raise ValueError("LegalQA audit output must not overwrite its input")
    samples = load_legal_qa_warmup(args.input)
    try:
        source = args.input.read_bytes()
    except OSError as exc:
        raise OSError(f"cannot fingerprint LegalQA warm-up input: {exc}") from exc
    report = build_audit(
        samples,
        dataset_sha256=hashlib.sha256(source).hexdigest(),
        source_size_bytes=len(source),
    )
    _write_json_atomic(report, args.output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        summary = _summary(report, args.output)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        print(f"LegalQA warm-up audit error: {exc}", file=sys.stderr)
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
