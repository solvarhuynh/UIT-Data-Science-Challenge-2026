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
        "--batch-size",
        type=int,
        help="Override embedding and Qdrant upsert batch size from configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even when a matching manifest already exists.",
    )
    return parser.parse_args()


class RunningCorpusHash:
    """Order-independent incremental hash so we never hold the whole corpus in RAM.

    NOTE: this changes the manifest's corpus_hash algorithm from
    "sha256(sorted json list)" to "XOR of per-chunk sha256 digests". It is
    still stable/deterministic and still detects any change to a chunk's
    (chunk_id, text), it just no longer requires buffering every chunk to
    sort them first. Old manifests will simply be treated as stale once and
    rebuilt (safe, one-time cost).
    """

    def __init__(self) -> None:
        self._acc = bytearray(32)
        self.count = 0

    def update(self, chunk_id: str, text: str) -> None:
        payload = json.dumps(
            {"chunk_id": chunk_id, "text": text},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        for i in range(32):
            self._acc[i] ^= digest[i]
        self.count += 1

    def hexdigest(self) -> str:
        return bytes(self._acc).hex()


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


def iter_chunks(chunks_dir: str, errors: list[dict[str, Any]]):
    """Yield one LegalChunk at a time (generator) instead of loading them all.

    This is the core streaming/lazy-loading fix: at any instant only the
    current line + current chunk object are alive, not the full corpus.
    ``errors`` is appended to in place so callers can still persist them.
    """
    from udsc2026.contracts import LegalChunk

    files = sorted(Path(chunks_dir).glob("*.jsonl"))
    if not files:
        LOGGER.warning("No .jsonl files found in %s", chunks_dir)
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, raw_line in enumerate(jsonl_file, start=1):
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                try:
                    yield LegalChunk.model_validate(json.loads(raw_line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    errors.append(
                        {"file": str(file_path), "line": line_number, "error": str(exc)}
                    )


def iter_batches(iterable, batch_size: int):
    """Chunk a generator into lists of at most ``batch_size`` items."""
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_chunks(chunks_dir: str) -> int:
    """Cheap pre-pass just counting non-blank/non-comment JSONL lines for tqdm totals."""
    total = 0
    for file_path in sorted(Path(chunks_dir).glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as jsonl_file:
            for raw_line in jsonl_file:
                if raw_line.strip() and not raw_line.lstrip().startswith("#"):
                    total += 1
    return total


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback keeps script runnable without tqdm
        def tqdm(iterable=None, total=None, desc=None, unit=None):  # type: ignore
            return iterable if iterable is not None else range(0)

    started = time.perf_counter()
    from udsc2026.infrastructure.config import load_config

    config = load_config(args.config_env)
    from udsc2026.infrastructure.embedding.client import EmbeddingClient
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
    from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

    embedding_config = config.get("embedding", {})
    vector_config = dict(config.get("vector_db", {}))
    if args.vector_db_type:
        vector_config["type"] = args.vector_db_type
    config["vector_db"] = vector_config
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(embedding_config.get("batch_size", 32))
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    LOGGER.info("Embedding and upsert batch size: %d", batch_size)

    error_path = Path("data/processed/metadata/index_errors.json")
    errors: list[dict[str, Any]] = []

    LOGGER.info("Counting chunks in %s (cheap pre-pass, does not load text)...", args.chunks_dir)
    total_chunks = count_chunks(args.chunks_dir)
    if total_chunks == 0:
        LOGGER.error("No valid chunks found in %s", args.chunks_dir)
        return 1
    LOGGER.info("Found %d chunks to index", total_chunks)

    model_path = str(embedding_config.get("model_path", "./models/hcmute-embedding-v2"))
    vector_db_type = str(vector_config.get("type", "qdrant"))
    bm25_path = args.bm25_index_path or config.get("sparse", {}).get(
        "bm25_index_path", "data/vector_store/bm25/index.pkl"
    )
    manifest_file_paths = manifest_paths(vector_config, vector_db_type, bm25_path)

    if not args.force:
        LOGGER.info("Checking cache (cheap text-only pre-pass, no embedding)...")
        precheck_hash = RunningCorpusHash()
        precheck_errors: list[dict[str, Any]] = []
        for chunk in iter_chunks(args.chunks_dir, precheck_errors):
            precheck_hash.update(chunk.chunk_id, chunk.text)
        precheck_manifest = {
            "corpus_hash": precheck_hash.hexdigest(),
            "model_hash": calculate_model_hash(model_path, embedding_config),
            "vector_db_type": vector_db_type,
            "collection_name": vector_config["collection_name"],
        }
        if is_current_manifest(manifest_file_paths[0], precheck_manifest) and is_current_manifest(
            manifest_file_paths[1], precheck_manifest
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

    # --- Pass 1: stream chunks -> embed batch -> upsert batch -> discard batch. ---
    # At no point do we hold more than `batch_size` chunks + their embeddings
    # in RAM simultaneously; this is the fix for the OOM/"unexpected EOF" crash.
    running_hash = RunningCorpusHash()
    collection_created = False
    processed = 0
    progress_log_interval = max(batch_size * 100, 10_000)
    progress = tqdm(total=total_chunks, desc="Embedding + upserting", unit="chunk")
    for batch in iter_batches(iter_chunks(args.chunks_dir, errors), batch_size):
        batch_texts = [chunk.text for chunk in batch]
        batch_embeddings = embedder.embed_documents(batch_texts, batch_size=batch_size)

        if not collection_created:
            vector_db.create_collection(
                vector_config["collection_name"],
                len(batch_embeddings[0]),
                vector_config.get("distance", "cosine"),
            )
            collection_created = True

        vector_db.upsert(batch, batch_embeddings)
        for chunk in batch:
            running_hash.update(chunk.chunk_id, chunk.text)

        processed += len(batch)
        progress.update(len(batch))
        if processed % progress_log_interval < len(batch) or processed == total_chunks:
            LOGGER.info("Embedded+upserted %d/%d chunks", processed, total_chunks)
    progress.close()

    if processed == 0:
        LOGGER.error("No valid chunks found in %s", args.chunks_dir)
        return 1

    manifest = {
        "timestamp": None,
        "chunk_count": processed,
        "corpus_hash": running_hash.hexdigest(),
        "model_path": model_path,
        "model_hash": calculate_model_hash(model_path, embedding_config),
        "vector_db_type": vector_db_type,
        "collection_name": vector_config["collection_name"],
        "bm25_index_path": str(bm25_path),
        "git_commit": git_commit(),
    }

    # --- Pass 2: BM25 build. ---
    # rank_bm25's BM25Okapi computes IDF over the whole corpus, so unlike the
    # vector step this genuinely needs every tokenized chunk resident in RAM
    # at once -- there is no streaming algorithm for classic BM25 here. We
    # still stream chunks off disk (not from the pass-1 list, which we never
    # kept) so vector-store memory has already been freed by the time this
    # runs, and we only pay the "hold everything" cost once instead of twice.
    LOGGER.info("Building BM25 index (this pass must hold all chunks in RAM)...")
    bm25_chunks: list[Any] = []
    for chunk in tqdm(
        iter_chunks(args.chunks_dir, errors), total=total_chunks, desc="Loading for BM25", unit="chunk"
    ):
        bm25_chunks.append(chunk)
    bm25 = BM25Retriever(bm25_path)
    bm25.build_index(bm25_chunks)
    bm25.save()
    del bm25_chunks

    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifests(manifest_file_paths, manifest)
    elapsed = time.perf_counter() - started
    print(f"Indexed chunks: {processed}")
    print(f"Skipped errors: {len(errors)}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
