"""Benchmark resilient document embedding over LegalChunk JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("benchmark_embedding")


def read_chunks(chunks_dir: Path) -> list[dict[str, str]]:
    """Read valid chunk IDs and texts, skipping blank/comment lines."""
    from udsc2026.contracts import LegalChunk

    result: list[dict[str, str]] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            try:
                chunk = LegalChunk.model_validate(json.loads(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                LOGGER.warning("Skipping %s:%d: %s", path, line_number, exc)
                continue
            result.append({"chunk_id": chunk.chunk_id, "text": chunk.text})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed/chunks", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--error-report",
        type=Path,
        default=Path("data/processed/metadata/embedding_errors.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from udsc2026.infrastructure.config import load_config
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient

    config = load_config("development")
    embedding_config: dict[str, Any] = config.get("embedding", {})
    batch_size = args.batch_size or int(embedding_config.get("batch_size", 32))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    chunks = read_chunks(args.chunks_dir)
    if not chunks:
        raise SystemExit(f"No valid chunks found in {args.chunks_dir}")
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
    encoded, errors = client.embed_documents_resilient(
        [(item["chunk_id"], item["text"]) for item in chunks], batch_size=batch_size
    )
    elapsed = time.perf_counter() - started
    args.error_report.parent.mkdir(parents=True, exist_ok=True)
    args.error_report.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Chunks: {len(chunks)}")
    print(f"Encoded: {len(encoded)}")
    print(f"Failed: {len(errors)}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Average seconds/batch: {elapsed / batches:.4f}")
    print(
        f"Chunks/second: {len(encoded) / elapsed:.2f}"
        if elapsed
        else "Chunks/second: inf"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
