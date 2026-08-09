"""Build reproducible dense and optional BM25 indexes from TV4 JSONL chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("index_chunks")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed_v3/chunks")
    parser.add_argument("--config-env", default="development")
    parser.add_argument("--vector-db-type", choices=("qdrant", "faiss"))
    parser.add_argument("--bm25-index-path")
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Build dense only; recommended for the 992k-chunk BTC corpus.",
    )
    parser.add_argument(
        "--max-chunks",
        type=_positive_int,
        help="Bound input size for a smoke run.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Override the configured embedding and vector upsert batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the source/model manifest matches.",
    )
    return parser.parse_args(argv)


class RunningCorpusHash:
    """Order-independent streaming fingerprint for ``(chunk_id, text)`` pairs."""

    def __init__(self) -> None:
        self._accumulator = bytearray(32)
        self.count = 0

    def update(self, chunk_id: str, text: str) -> None:
        payload = json.dumps(
            {"chunk_id": chunk_id, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        for index, value in enumerate(digest):
            self._accumulator[index] ^= value
        self.count += 1

    def hexdigest(self) -> str:
        return bytes(self._accumulator).hex()


def calculate_corpus_hash(chunks: Iterable[Any]) -> str:
    """Compatibility helper used by unit tests and small audit utilities."""

    fingerprint = RunningCorpusHash()
    for chunk in chunks:
        fingerprint.update(chunk.chunk_id, chunk.text)
    return fingerprint.hexdigest()


def calculate_model_hash(model_path: str, config: dict[str, Any]) -> str:
    identity = {
        "model_path": model_path,
        "model_id": config.get("model_id"),
        "max_length": config.get("max_length", 256),
        "output_dimension": config.get("output_dimension"),
        "normalize_embeddings": config.get("normalize_embeddings", True),
        "window_long_texts": config.get("window_long_texts", False),
        "window_overlap_tokens": config.get("window_overlap_tokens", 32),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(chunks_dir: str | Path, max_chunks: int | None) -> str:
    """Cheap cache key over the selected input file inventory."""

    root = Path(chunks_dir)
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.jsonl")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    digest.update(str(max_chunks).encode())
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def iter_chunks(
    chunks_dir: str | Path,
    errors: list[dict[str, Any]],
    *,
    max_chunks: int | None = None,
) -> Iterator[Any]:
    """Yield strict LegalChunk records without retaining the corpus in RAM."""

    from udsc2026.contracts import LegalChunk

    yielded = 0
    files = sorted(Path(chunks_dir).glob("*.jsonl"))
    if not files:
        LOGGER.warning("No .jsonl files found in %s", chunks_dir)
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                try:
                    chunk = LegalChunk.model_validate(json.loads(raw_line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "error": str(exc),
                        }
                    )
                    continue
                yield chunk
                yielded += 1
                if max_chunks is not None and yielded >= max_chunks:
                    return


def iter_batches(iterable: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_chunks(chunks_dir: str | Path, max_chunks: int | None = None) -> int:
    count = 0
    for file_path in sorted(Path(chunks_dir).glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                if raw_line.strip() and not raw_line.lstrip().startswith("#"):
                    count += 1
                    if max_chunks is not None and count >= max_chunks:
                        return count
    return count


def is_current_manifest(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    keys = (
        "source_fingerprint",
        "corpus_hash",
        "model_hash",
        "vector_db_type",
        "collection_name",
        "max_chunks",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _vector_manifest_path(config: dict[str, Any], vector_type: str) -> Path:
    collection = str(config["collection_name"])
    if vector_type == "faiss":
        return Path(config["faiss_index_path"]) / collection / "manifest.json"
    return Path("data/vector_store/qdrant") / collection / "manifest.json"


def _load_download_manifest() -> dict[str, Any] | None:
    path = Path("models/download_manifest.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover

        class _Progress:
            def update(self, _: int) -> None:
                return None

            def close(self) -> None:
                return None

        def tqdm(iterable=None, **_: Any):  # type: ignore[no-untyped-def]
            return iterable if iterable is not None else _Progress()

    from udsc2026.infrastructure.config import load_config
    from udsc2026.infrastructure.embedding.client import EmbeddingClient
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter

    config = load_config(args.config_env)
    embedding_config = dict(config.get("embedding", {}))
    vector_config = dict(config.get("vector_db", {}))
    if args.vector_db_type:
        vector_config["type"] = args.vector_db_type
    config["vector_db"] = vector_config

    model_path = str(embedding_config.get("model_path", "./models/dek21-v2"))
    vector_type = str(vector_config.get("type", "faiss"))
    collection_name = str(vector_config["collection_name"])
    batch_size = args.batch_size or int(embedding_config.get("batch_size", 32))
    total_chunks = count_chunks(args.chunks_dir, args.max_chunks)
    if total_chunks == 0:
        LOGGER.error("No chunk records found in %s", args.chunks_dir)
        return 1

    source_hash = source_fingerprint(args.chunks_dir, args.max_chunks)
    model_hash = calculate_model_hash(model_path, embedding_config)
    manifest_path = _vector_manifest_path(vector_config, vector_type)
    cache_identity = {
        "source_fingerprint": source_hash,
        "model_hash": model_hash,
        "vector_db_type": vector_type,
        "collection_name": collection_name,
        "max_chunks": args.max_chunks,
    }
    if not args.force:
        LOGGER.info("Checking corpus hash without loading it into RAM...")
        precheck_hash = RunningCorpusHash()
        precheck_errors: list[dict[str, Any]] = []
        for chunk in iter_chunks(
            args.chunks_dir,
            precheck_errors,
            max_chunks=args.max_chunks,
        ):
            precheck_hash.update(chunk.chunk_id, chunk.text)
        cache_identity["corpus_hash"] = precheck_hash.hexdigest()
        if is_current_manifest(manifest_path, cache_identity):
            print(f"Dense index is current: {manifest_path}")
            return 0

    LOGGER.info("Indexing %d chunks with %s", total_chunks, model_path)
    embedder = EmbeddingClient(
        model_path=model_path,
        device=str(embedding_config.get("device", "cpu")),
        batch_size=batch_size,
        max_length=int(embedding_config.get("max_length", 256)),
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
        output_dimension=embedding_config.get("output_dimension"),
        window_long_texts=bool(embedding_config.get("window_long_texts", False)),
        window_overlap_tokens=int(embedding_config.get("window_overlap_tokens", 32)),
    )
    vector_db = get_vector_db_adapter(config)

    from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

    faiss_db = vector_db if isinstance(vector_db, FaissAdapter) else None
    errors: list[dict[str, Any]] = []
    corpus_hash = RunningCorpusHash()
    processed = 0
    bulk_started = False
    started = time.perf_counter()
    progress = tqdm(total=total_chunks, desc="Embedding + indexing", unit="chunk")
    try:
        batches = iter_batches(
            iter_chunks(
                args.chunks_dir,
                errors,
                max_chunks=args.max_chunks,
            ),
            batch_size,
        )
        for batch in batches:
            embeddings = embedder.embed_documents(
                [chunk.text for chunk in batch],
                batch_size=batch_size,
            )
            if not bulk_started:
                vector_size = len(embeddings[0])
                if faiss_db is not None:
                    faiss_db.begin_bulk_replace(
                        collection_name,
                        vector_size,
                        str(vector_config.get("distance", "cosine")),
                    )
                else:
                    vector_db.delete_collection(collection_name)
                    vector_db.create_collection(
                        collection_name,
                        vector_size,
                        str(vector_config.get("distance", "cosine")),
                    )
                bulk_started = True

            if faiss_db is not None:
                faiss_db.append_bulk(batch, embeddings)
            else:
                vector_db.upsert(batch, embeddings)
            for chunk in batch:
                corpus_hash.update(chunk.chunk_id, chunk.text)
            processed += len(batch)
            progress.update(len(batch))
        if faiss_db is not None:
            faiss_db.commit_bulk()
    except BaseException:
        if faiss_db is not None:
            faiss_db.abort_bulk()
        raise
    finally:
        progress.close()

    if processed == 0:
        LOGGER.error("No valid chunks were indexed")
        return 1

    chunks_path = Path(args.chunks_dir)
    error_path = chunks_path.parent / "metadata" / "index_errors.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **cache_identity,
        "schema_version": 2,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_count": processed,
        "corpus_hash": corpus_hash.hexdigest(),
        "model_path": model_path,
        "model_id": embedding_config.get("model_id"),
        "embedding_dimension": (
            faiss_db.index.d
            if faiss_db is not None and faiss_db.index is not None
            else embedding_config.get("output_dimension")
        ),
        "git_commit": git_commit(),
        "download_manifest": _load_download_manifest(),
        "invalid_record_count": len(errors),
        "windowed_document_count": embedder.windowed_document_count,
        "encoded_window_count": embedder.encoded_window_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_manifest(manifest_path, manifest)

    if not args.skip_bm25:
        from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

        LOGGER.warning(
            "Building rank_bm25 in memory; use --skip-bm25 for the full BTC corpus"
        )
        bm25_path = args.bm25_index_path or config.get("sparse", {}).get(
            "bm25_index_path",
            "data/vector_store/bm25/legal_chunks.json",
        )
        bm25_errors: list[dict[str, Any]] = []
        chunks = list(
            iter_chunks(
                args.chunks_dir,
                bm25_errors,
                max_chunks=args.max_chunks,
            )
        )
        bm25 = BM25Retriever(str(bm25_path))
        bm25.build_index(chunks)
        bm25.save()

    print(f"Indexed chunks: {processed}")
    print(f"Invalid records: {len(errors)}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
