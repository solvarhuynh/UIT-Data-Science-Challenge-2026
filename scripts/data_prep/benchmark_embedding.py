"""Benchmark resilient document embedding over LegalChunk JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("benchmark_embedding")


def iter_chunks(chunks_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield valid chunk IDs and texts without retaining the corpus in RAM."""
    from udsc2026.contracts import LegalChunk

    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                try:
                    chunk = LegalChunk.model_validate(json.loads(raw))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    LOGGER.warning("Skipping %s:%d: %s", path, line_number, exc)
                    continue
                yield chunk.chunk_id, chunk.text


def iter_batches(
    items: Iterator[tuple[str, str]], batch_size: int
) -> Iterator[list[tuple[str, str]]]:
    """Yield bounded batches suitable for resilient embedding."""
    batch: list[tuple[str, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed_v3/chunks", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--error-report",
        type=Path,
        default=Path("data/processed_v3/metadata/embedding_errors.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from udsc2026.infrastructure.config import load_config
    from udsc2026.infrastructure.embedding.client import EmbeddingClient

    config = load_config("development")
    embedding_config: dict[str, Any] = config.get("embedding", {})
    batch_size = args.batch_size or int(embedding_config.get("batch_size", 32))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    client = EmbeddingClient(
        model_path=str(embedding_config.get("model_path", "./models/dek21-v2")),
        device=str(embedding_config.get("device", "cpu")),
        batch_size=batch_size,
        max_length=int(embedding_config.get("max_length", 256)),
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
        output_dimension=embedding_config.get("output_dimension"),
        window_long_texts=bool(embedding_config.get("window_long_texts", False)),
        window_overlap_tokens=int(embedding_config.get("window_overlap_tokens", 32)),
    )
    started = time.perf_counter()
    chunk_count = 0
    encoded_count = 0
    batch_count = 0
    errors: list[dict[str, str]] = []
    for batch in iter_batches(iter_chunks(args.chunks_dir), batch_size):
        encoded, batch_errors = client.embed_documents_resilient(
            batch, batch_size=batch_size
        )
        chunk_count += len(batch)
        encoded_count += len(encoded)
        batch_count += 1
        errors.extend(batch_errors)
    if chunk_count == 0:
        raise SystemExit(f"No valid chunks found in {args.chunks_dir}")
    elapsed = time.perf_counter() - started
    args.error_report.parent.mkdir(parents=True, exist_ok=True)
    args.error_report.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Chunks: {chunk_count}")
    print(f"Encoded: {encoded_count}")
    print(f"Failed: {len(errors)}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Average seconds/batch: {elapsed / batch_count:.4f}")
    print(
        f"Chunks/second: {encoded_count / elapsed:.2f}"
        if elapsed
        else "Chunks/second: inf"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
