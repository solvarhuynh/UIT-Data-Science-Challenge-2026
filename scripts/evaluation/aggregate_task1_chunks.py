"""Aggregate a raw scored Task1 chunk JSONL into reusable document rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from udsc2026.evaluation.legal_ir_chunk_aggregation import aggregate_chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="rank_cap10_k5")
    parser.add_argument("--rank-source", choices=("bge", "dense"), default="bge")
    parser.add_argument("--top-k", type=int, default=200)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5")
    output_dir = args.output_dir
    ranking_path = output_dir / "document_rankings.jsonl"
    manifest_path = output_dir / "manifest.json"
    if args.input.resolve() in {ranking_path.resolve(), manifest_path.resolve()}:
        raise ValueError("output must not overwrite the raw input")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = ranking_path.with_name(f".{ranking_path.name}.{os.getpid()}.tmp")
    count = 0
    seen: set[str] = set()
    with (
        args.input.open(encoding="utf-8-sig") as source,
        temporary.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(
                row.get("query_id", row.get("question_id", row.get("id", "")))
            ).strip()
            if not query_id or query_id in seen:
                raise ValueError(f"invalid/duplicate query ID at line {line_number}")
            seen.add(query_id)
            ranking = aggregate_chunks(
                row.get("hits", ()), method=args.method, rank_source=args.rank_source
            )[: args.top_k]
            payload = {
                "query_id": query_id,
                "documents": [item.doc_id for item in ranking],
                "document_aggregates": [item.as_dict() for item in ranking],
            }
            destination.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(ranking_path)
    manifest = {
        "schema_version": "task1-raw-chunk-document-aggregation-v1",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "method": args.method,
        "rank_source": args.rank_source,
        "top_k": args.top_k,
        "query_count": count,
        "output": str(ranking_path),
        "output_sha256": _sha256(ranking_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"chunk aggregation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
