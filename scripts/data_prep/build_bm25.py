"""Build a validated BM25 artifact from ingestion-produced LegalChunk files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.contracts import LegalChunk  # noqa: E402
from udsc2026.retrieval.sparse import BM25Retriever  # noqa: E402


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _discover_inputs(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("chunk input file must use .json or .jsonl")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"chunk input does not exist: {source}")
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise ValueError(f"no .json or .jsonl chunk files found in {source}")
    return files


def _load_json(path: Path) -> list[object]:
    with path.open("r", encoding="utf-8-sig") as stream:
        payload: object = json.load(
            stream,
            parse_constant=_reject_json_constant,
        )
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and set(payload) == {"chunks"}:
        chunks = payload["chunks"]
        if isinstance(chunks, list):
            return chunks
    raise ValueError(f"{path}: JSON root must be an array or a single 'chunks' object")


def _load_jsonl(path: Path) -> list[object]:
    records: list[object] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(
                    json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: JSONL file contains no records")
    return records


def load_chunks(source: Path, output: Path) -> list[LegalChunk]:
    """Load every input deterministically and reject output/input collisions."""

    input_files = _discover_inputs(source)
    output_resolved = output.resolve()
    if source.is_dir():
        try:
            output_resolved.relative_to(source.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("BM25 output must be outside the chunk input directory")
    if any(path.resolve() == output_resolved for path in input_files):
        raise ValueError("BM25 output must not overwrite a chunk input file")

    chunks: list[LegalChunk] = []
    for path in input_files:
        raw_records = (
            _load_jsonl(path) if path.suffix.lower() == ".jsonl" else _load_json(path)
        )
        for record_index, raw_record in enumerate(raw_records):
            try:
                chunks.append(LegalChunk.model_validate(raw_record, strict=True))
            except ValidationError as exc:
                record_label = f"{path}: record {record_index + 1}"
                raise ValueError(
                    f"{record_label} is not a valid LegalChunk: {exc}"
                ) from exc
    return chunks


def build_index(source: Path, output: Path) -> int:
    """Build and atomically save one BM25 artifact; return indexed chunk count."""

    chunks = load_chunks(source, output)
    retriever = BM25Retriever(str(output))
    retriever.build_index(chunks)
    retriever.save()
    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed_v3/chunks"),
        help="LegalChunk .json/.jsonl file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.getenv(
                "BM25_INDEX_PATH",
                "data/vector_store/bm25/index.json",
            )
        ),
        help="Destination index JSON (default: BM25_INDEX_PATH or project path).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        chunk_count = build_index(args.chunks, args.output)
    except (OSError, TypeError, ValueError) as exc:
        print(f"BM25 build error: {exc}", file=sys.stderr)
        return 2
    print(f"Indexed {chunk_count} chunks into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
