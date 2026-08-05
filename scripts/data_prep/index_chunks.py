"""Index TV4 JSONL chunks into dense and sparse retrieval backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow running this file directly from a source checkout without installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("index_chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    parser.add_argument("--config-env", default="development")
    parser.add_argument("--vector-db-type", choices=("qdrant", "faiss"))
    parser.add_argument("--bm25-index-path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even when a matching manifest already exists.",
    )
    return parser.parse_args()


def calculate_corpus_hash(chunks: list[Any]) -> str:
    """Return a stable hash for the searchable chunk identity and text."""
    payload = [
        {"chunk_id": chunk.chunk_id, "text": chunk.text}
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def calculate_model_hash(model_path: str, embedding_config: dict[str, Any]) -> str:
    """Hash model identity/config without reading potentially large weights."""
    identity = {
        "model_path": model_path,
        "device": embedding_config.get("device", "cpu"),
        "max_length": embedding_config.get("max_length", 256),
        "normalize_embeddings": embedding_config.get("normalize_embeddings", True),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str | None:
    """Return the current commit when the repository metadata is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest_paths(
    vector_config: dict[str, Any], vector_db_type: str, bm25_path: str
) -> tuple[Path, Path]:
    """Resolve vector and BM25 manifest paths beside their artifacts."""
    if vector_db_type == "faiss":
        vector_root = Path(vector_config["faiss_index_path"])
    else:
        vector_root = Path("data/vector_store/qdrant") / vector_config["collection_name"]
    return vector_root / "manifest.json", Path(bm25_path).with_name("manifest.json")


def is_current_manifest(path: Path, expected: dict[str, Any]) -> bool:
    """Check whether a manifest matches the current corpus and index settings."""
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    keys = ("corpus_hash", "model_hash", "vector_db_type", "collection_name")
    return all(actual.get(key) == expected.get(key) for key in keys)


def write_manifests(
    paths: tuple[Path, Path], manifest: dict[str, Any]
) -> None:
    """Persist identical version metadata for vector and sparse artifacts."""
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


def read_chunks(chunks_dir: str, error_path: Path) -> tuple[list[Any], int]:
    from udsc2026.contracts import LegalChunk

    chunks: list[LegalChunk] = []
    errors: list[dict[str, Any]] = []
    files = sorted(Path(chunks_dir).glob("*.jsonl"))
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, raw_line in enumerate(jsonl_file, start=1):
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                try:
                    chunks.append(LegalChunk.model_validate(json.loads(raw_line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    errors.append(
                        {"file": str(file_path), "line": line_number, "error": str(exc)}
                    )
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chunks, len(errors)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.perf_counter()
    from udsc2026.infrastructure.config import load_config

    config = load_config(args.config_env)
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
    from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

    embedding_config = config.get("embedding", {})
    vector_config = dict(config.get("vector_db", {}))
    if args.vector_db_type:
        vector_config["type"] = args.vector_db_type
    config["vector_db"] = vector_config
    batch_size = int(embedding_config.get("batch_size", 32))

    error_path = Path("data/processed/metadata/index_errors.json")
    chunks, error_count = read_chunks(args.chunks_dir, error_path)
    if not chunks:
        LOGGER.error("No valid chunks found in %s", args.chunks_dir)
        return 1

    model_path = str(embedding_config.get("model_path", "./models/bkai-bi-encoder"))
    vector_db_type = str(vector_config.get("type", "qdrant"))
    bm25_path = args.bm25_index_path or config.get("sparse", {}).get(
        "bm25_index_path", "data/vector_store/bm25/index.pkl"
    )
    manifest_file_paths = manifest_paths(vector_config, vector_db_type, bm25_path)
    manifest = {
        "timestamp": None,
        "chunk_count": len(chunks),
        "corpus_hash": calculate_corpus_hash(chunks),
        "model_path": model_path,
        "model_hash": calculate_model_hash(model_path, embedding_config),
        "vector_db_type": vector_db_type,
        "collection_name": vector_config["collection_name"],
        "bm25_index_path": str(bm25_path),
        "git_commit": git_commit(),
    }
    if (
        not args.force
        and is_current_manifest(manifest_file_paths[0], manifest)
        and is_current_manifest(manifest_file_paths[1], manifest)
    ):
        print("Index đã up to date, bỏ qua")
        return 0

    embedder = EmbeddingClient(
        model_path=model_path,
        device=embedding_config.get("device", "cpu"),
        batch_size=batch_size,
        max_length=int(embedding_config.get("max_length", 256)),
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
    )
    vector_db = get_vector_db_adapter(config)
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch_texts = [chunk.text for chunk in chunks[start : start + batch_size]]
        embeddings.extend(embedder.embed_documents(batch_texts, batch_size=batch_size))
        LOGGER.info("Embedded %d/%d chunks", len(embeddings), len(chunks))

    vector_db.create_collection(
        vector_config["collection_name"],
        len(embeddings[0]),
        vector_config.get("distance", "cosine"),
    )
    for start in range(0, len(chunks), batch_size):
        vector_db.upsert(
            chunks[start : start + batch_size], embeddings[start : start + batch_size]
        )
        LOGGER.info(
            "Upserted %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks)
        )

    bm25 = BM25Retriever(bm25_path)
    bm25.build_index(chunks)
    bm25.save()
    manifest["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifests(manifest_file_paths, manifest)
    elapsed = time.perf_counter() - started
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Skipped errors: {error_count}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
