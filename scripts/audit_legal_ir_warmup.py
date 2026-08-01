"""Audit the supplied DSC2026 LegalIR warm-up data without scoring a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    WarmupSample,
    load_warmup,
    normalize_legal_ir_query,
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


def _duplicate_stats(values: list[str]) -> tuple[int, int]:
    """Return duplicate group count and total records in those groups."""

    frequencies = Counter(values)
    duplicates = [count for count in frequencies.values() if count > 1]
    return len(duplicates), sum(duplicates)


def build_audit(samples: list[WarmupSample], *, raw_sha256: str) -> dict[str, object]:
    """Build deterministic schema, label, and Unicode diagnostics."""

    if not samples:
        raise ValueError("warm-up audit requires at least one question")
    answer_counts = Counter(len(sample.gold_documents) for sample in samples)
    labeled_document_ids = {
        document_id for sample in samples for document_id in sample.gold_documents
    }
    whitespace_ids = sorted(
        sample.id
        for sample in samples
        if sample.raw_question != " ".join(sample.raw_question.split())
    )
    non_nfc_ids = sorted(
        sample.id
        for sample in samples
        if not unicodedata.is_normalized("NFC", sample.raw_question)
    )
    normalized_change_ids = sorted(
        sample.id
        for sample in samples
        if sample.raw_question != normalize_legal_ir_query(sample.raw_question)
    )
    multi_gold_ids = sorted(
        sample.id for sample in samples if len(sample.gold_documents) > 1
    )
    raw_duplicate_groups, raw_duplicate_records = _duplicate_stats(
        [sample.raw_question for sample in samples]
    )
    normalized_duplicate_groups, normalized_duplicate_records = _duplicate_stats(
        [sample.question for sample in samples]
    )
    return {
        "answer_count_distribution": {
            str(count): frequency for count, frequency in sorted(answer_counts.items())
        },
        "dataset_sha256": raw_sha256,
        "duplicate_normalized_question_group_count": normalized_duplicate_groups,
        "duplicate_normalized_question_record_count": normalized_duplicate_records,
        "duplicate_raw_question_group_count": raw_duplicate_groups,
        "duplicate_raw_question_record_count": raw_duplicate_records,
        "max_answer_count": max(answer_counts),
        "multi_gold_question_count": len(multi_gold_ids),
        "multi_gold_question_ids": multi_gold_ids,
        "non_nfc_question_count": len(non_nfc_ids),
        "non_nfc_question_ids": non_nfc_ids,
        "normalized_question_change_count": len(normalized_change_ids),
        "normalized_question_change_ids": normalized_change_ids,
        "question_count": len(samples),
        "schema_version": "legal-ir-warmup-audit-v1",
        "single_gold_question_count": len(samples) - len(multi_gold_ids),
        "total_answer_label_count": sum(answer_counts.elements()),
        "unique_labeled_document_count": len(labeled_document_ids),
        "whitespace_normalization_question_count": len(whitespace_ids),
        "whitespace_normalization_question_ids": whitespace_ids,
    }


def _write_json(payload: dict[str, object], path: Path) -> None:
    """Atomically write a canonical UTF-8 JSON audit report."""

    if path.suffix.casefold() != ".json":
        raise ValueError("warm-up audit output must use .json")
    if path.is_symlink():
        raise ValueError("warm-up audit output must not be a symbolic link")
    if path.is_dir():
        raise ValueError("warm-up audit output must not be a directory")
    if path.exists() and not path.is_file():
        raise ValueError("warm-up audit output must be a regular file")
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
            raise ValueError("warm-up audit output became a symbolic link")
        if path.exists() and not path.is_file():
            raise ValueError("warm-up audit output is no longer a regular file")
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
    """Build the warm-up audit command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate warmup.json and report counts, multi-gold records, "
            "whitespace, and Unicode normalization issues."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/task1/warmup.json"),
        help=("Task 1 Warm-up root-mapping JSON (default: data/task1/warmup.json)."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task1/warmup_audit.json"),
        help="Atomic JSON audit report path.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Validate, audit, and persist the selected warm-up dataset."""

    if _paths_collide(args.input, args.output):
        raise ValueError("warm-up audit output must not overwrite the input file")
    samples = load_warmup(args.input)
    try:
        raw_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    except OSError as exc:
        raise OSError(f"cannot fingerprint warm-up input: {exc}") from exc
    report = build_audit(samples, raw_sha256=raw_sha256)
    _write_json(report, args.output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Return zero on success and two for validation or I/O failures."""

    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR warm-up audit error: {exc}", file=sys.stderr)
        return 2
    print(
        f"questions={report['question_count']} "
        f"answer_labels={report['total_answer_label_count']} "
        f"unique_labeled_documents={report['unique_labeled_document_count']} "
        f"multi_gold={report['multi_gold_question_count']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
