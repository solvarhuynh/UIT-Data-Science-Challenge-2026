"""Recover searchable fallback chunks for cleaned documents missed by parsing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.ingestion.chunking.chunker import (  # noqa: E402
    chunk_unstructured_document,
)
from udsc2026.ingestion.cleaners.models import CleanDocument  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents-dir", type=Path, default=Path("data/processed/documents")
    )
    parser.add_argument(
        "--chunks-dir", type=Path, default=Path("data/processed/chunks")
    )
    parser.add_argument(
        "--parents-dir", type=Path, default=Path("data/processed/parents")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/metadata/empty_chunk_repair.json"),
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=192)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write repairs. Without this flag the command is a read-only audit.",
    )
    return parser


def _has_records(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as stream:
        return any(
            line.strip() and not line.lstrip().startswith("#") for line in stream
        )


def _jsonl(records: Sequence[Any]) -> str:
    return "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must be non-negative and less than chunk-size")
    document_paths = sorted(args.documents_dir.glob("*.json"))
    if not document_paths:
        raise FileNotFoundError(f"no cleaned documents found in {args.documents_dir}")

    repaired_document_ids: list[str] = []
    unrecoverable_document_ids: list[str] = []
    invalid_documents: list[dict[str, str]] = []
    recovered_chunks = 0
    recovered_parents = 0
    for document_path in document_paths:
        chunk_path = args.chunks_dir / f"{document_path.stem}.jsonl"
        if _has_records(chunk_path):
            continue
        try:
            document = CleanDocument.model_validate_json(
                document_path.read_text(encoding="utf-8")
            )
            result = chunk_unstructured_document(
                document,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except (OSError, TypeError, ValueError) as exc:
            invalid_documents.append({"path": str(document_path), "error": str(exc)})
            continue
        if not result.chunks or not result.parents:
            unrecoverable_document_ids.append(document.doc_id)
            continue
        if args.apply:
            _write_atomic(chunk_path, _jsonl(result.chunks))
            _write_atomic(
                args.parents_dir / f"{document.doc_id}.jsonl",
                _jsonl(result.parents),
            )
        repaired_document_ids.append(document.doc_id)
        recovered_chunks += len(result.chunks)
        recovered_parents += len(result.parents)

    report = {
        "schema_version": 1,
        "mode": "apply" if args.apply else "audit",
        "document_count": len(document_paths),
        "recoverable_document_count": len(repaired_document_ids),
        "recovered_chunk_count": recovered_chunks,
        "recovered_parent_count": recovered_parents,
        "unrecoverable_document_ids": unrecoverable_document_ids,
        "invalid_documents": invalid_documents,
        "repaired_document_ids": repaired_document_ids,
    }
    if args.apply:
        _write_atomic(
            args.report,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"empty-chunk repair error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "mode": report["mode"],
        "document_count": report["document_count"],
        "recoverable_document_count": report["recoverable_document_count"],
        "recovered_chunk_count": report["recovered_chunk_count"],
        "recovered_parent_count": report["recovered_parent_count"],
        "unrecoverable_document_count": len(report["unrecoverable_document_ids"]),
        "invalid_document_count": len(report["invalid_documents"]),
        "report": str(args.report) if args.apply else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not report["invalid_documents"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
