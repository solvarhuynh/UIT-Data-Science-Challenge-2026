"""Build a reproducible dense vector index from processed legal text chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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


def resolve_batch_size(
    cli_batch_size: int | None,
    embedding_config: dict[str, Any],
) -> int:
    """Resolve a validated CLI override or the configured embedding batch size."""
    batch_size = (
        cli_batch_size
        if cli_batch_size is not None
        else int(embedding_config.get("batch_size", 32))
    )
    if batch_size <= 0:
        raise ValueError("embedding batch_size must be greater than zero")
    return batch_size


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    parser.add_argument("--config-env", default="base")
    parser.add_argument("--vector-db-type", choices=("faiss",))
    parser.add_argument(
        "--error-report",
        type=Path,
        default=Path("data/reports/indexing/index_errors.json"),
        help=(
            "Operational indexing error report. It must stay outside the "
            "immutable processed corpus."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=_positive_int,
        help="Limit chunks included in an indexing pass.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a matching committed FAISS checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-chunks",
        type=_positive_int,
        default=100_000,
        help="Persist a resumable FAISS checkpoint after this many new chunks.",
    )
    parser.add_argument(
        "--commit-modal-volume",
        action="store_true",
        help="Commit the mounted Modal Volume after each durable checkpoint.",
    )
    parser.add_argument(
        "--modal-volume-name",
        default="udsc",
        help="Volume name used with --commit-modal-volume.",
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
    model_revision = config.get("revision")
    model_manifest = _load_download_manifest() or {}
    manifest_models = model_manifest.get("models", {})
    if isinstance(manifest_models, dict):
        for model in manifest_models.values():
            if not isinstance(model, dict):
                continue
            if model.get("repo_id") == config.get("model_id"):
                model_revision = model.get("resolved_revision", model_revision)
                break
    identity = {
        "model_path": model_path,
        "model_id": config.get("model_id"),
        "model_revision": model_revision,
        "embedding_implementation_sha256": _file_sha256(
            REPO_ROOT / "src/udsc2026/infrastructure/embedding/client.py"
        ),
        "max_length": config.get("max_length", 256),
        "query_max_length": config.get(
            "query_max_length", config.get("max_length", 256)
        ),
        "document_max_length": config.get(
            "document_max_length", config.get("max_length", 256)
        ),
        "output_dimension": config.get("output_dimension"),
        "normalize_embeddings": config.get("normalize_embeddings", True),
        "preprocessor": config.get("preprocessor", "pyvi"),
        "window_long_texts": config.get("window_long_texts", False),
        "window_overlap_tokens": config.get("window_overlap_tokens", 32),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash one implementation file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _combined_file_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _config_hash(environment: str) -> str:
    config_dir = Path(os.environ.get("UDSC2026_CONFIG_DIR", "configs"))
    paths = [config_dir / "base.yaml"]
    if environment != "base":
        paths.append(config_dir / f"{environment}.yaml")
    return _combined_file_hash(paths)


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
        "config_sha256",
        "indexing_implementation_sha256",
        "embedding_implementation_sha256",
        "preprocessor",
        "vector_db_type",
        "collection_name",
        "max_chunks",
        "batch_size",
        "complete",
    )
    if not all(actual.get(key) == expected.get(key) for key in keys):
        return False
    index_path = path.with_name("index.faiss")
    payload_path = path.with_name("payloads.json")
    return (
        index_path.is_file()
        and payload_path.is_file()
        and actual.get("index_sha256") == _file_sha256(index_path)
        and actual.get("payload_sha256") == _file_sha256(payload_path)
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _vector_manifest_path(config: dict[str, Any], vector_type: str) -> Path:
    collection = str(config["collection_name"])
    if vector_type == "faiss":
        return Path(config["faiss_index_path"]) / collection / "manifest.json"
    raise ValueError(f"unsupported vector database type: {vector_type}")


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
    from udsc2026.infrastructure.modal_volume import commit_mounted_volume_sync
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter

    config = load_config(args.config_env)
    embedding_config = dict(config.get("embedding", {}))
    vector_config = dict(config.get("vector_db", {}))
    if args.vector_db_type:
        vector_config["type"] = args.vector_db_type
    config["vector_db"] = vector_config

    model_path = str(
        embedding_config.get(
            "model_path", "./models/aiteam-vietnamese-embedding-v2"
        )
    )
    vector_type = str(vector_config.get("type", "faiss"))
    collection_name = str(vector_config["collection_name"])
    batch_size = resolve_batch_size(args.batch_size, embedding_config)
    if args.force and args.resume:
        raise ValueError("--force and --resume cannot be used together")
    if args.commit_modal_volume and vector_type != "faiss":
        raise ValueError("Modal checkpoints currently require FAISS")
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
        "config_sha256": _config_hash(args.config_env),
        "indexing_implementation_sha256": _combined_file_hash(
            [
                REPO_ROOT / "scripts/data_prep/index_chunks.py",
                REPO_ROOT / "src/udsc2026/infrastructure/vector_db/faiss_adapter.py",
            ]
        ),
        "batch_size": batch_size,
        "embedding_implementation_sha256": _file_sha256(
            REPO_ROOT / "src/udsc2026/infrastructure/embedding/client.py"
        ),
        "preprocessor": embedding_config.get("preprocessor", "pyvi"),
        "vector_db_type": vector_type,
        "collection_name": collection_name,
        "max_chunks": args.max_chunks,
        "complete": True,
    }
    manifest_path = _vector_manifest_path(vector_config, vector_type)
    checkpoint_path = manifest_path.with_name("checkpoint.json")
    if not args.force or args.resume:
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

    checkpoint: dict[str, Any] | None = None
    resume_count = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"matching incomplete index checkpoint is missing: {checkpoint_path}"
            )
        try:
            checkpoint_value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("index checkpoint is invalid") from exc
        if not isinstance(checkpoint_value, dict):
            raise ValueError("index checkpoint must contain a JSON object")
        checkpoint = checkpoint_value
        expected_checkpoint = {
            **cache_identity,
            "complete": False,
            "model_id": embedding_config.get("model_id"),
            "embedding_dimension": embedding_config.get("output_dimension"),
            "distance": vector_config.get("distance", "cosine"),
        }
        mismatches = {
            key: {"expected": value, "actual": checkpoint.get(key)}
            for key, value in expected_checkpoint.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"index checkpoint inputs changed: {mismatches}")
        raw_count = checkpoint.get("chunk_count")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
            or raw_count > total_chunks
        ):
            raise ValueError("index checkpoint has an invalid chunk count")
        resume_count = raw_count

    LOGGER.info("Indexing %d chunks with %s", total_chunks, model_path)
    embedder = EmbeddingClient(
        model_path=model_path,
        device=str(embedding_config.get("device", "cpu")),
        batch_size=batch_size,
        max_length=int(embedding_config.get("max_length", 256)),
        query_max_length=int(
            embedding_config.get(
                "query_max_length", embedding_config.get("max_length", 256)
            )
        ),
        document_max_length=int(
            embedding_config.get(
                "document_max_length", embedding_config.get("max_length", 256)
            )
        ),
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
        output_dimension=embedding_config.get("output_dimension"),
        preprocessor=str(embedding_config.get("preprocessor", "pyvi")),
        window_long_texts=bool(embedding_config.get("window_long_texts", False)),
        window_overlap_tokens=int(embedding_config.get("window_overlap_tokens", 32)),
    )
    vector_db = get_vector_db_adapter(config)

    from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

    faiss_db = vector_db if isinstance(vector_db, FaissAdapter) else None
    if args.resume and faiss_db is None:
        raise ValueError("resumable indexing requires the FAISS adapter")
    if args.resume and faiss_db is not None:
        assert checkpoint is not None
        if faiss_db.index is None or faiss_db.index.ntotal != resume_count:
            raise ValueError(
                "FAISS checkpoint vector count does not match its manifest"
            )
        faiss_db.begin_bulk_resume(
            collection_name,
            int(checkpoint["embedding_dimension"]),
            str(checkpoint["distance"]),
        )
    errors: list[dict[str, Any]] = []
    corpus_hash = RunningCorpusHash()
    processed = resume_count
    bulk_started = args.resume
    last_checkpoint = resume_count
    started = time.perf_counter()
    progress = tqdm(
        total=total_chunks,
        initial=resume_count,
        desc="Embedding + indexing",
        unit="chunk",
    )
    try:
        chunk_iterator = iter_chunks(
            args.chunks_dir,
            errors,
            max_chunks=args.max_chunks,
        )
        for _ in range(resume_count):
            try:
                chunk = next(chunk_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "input corpus ended before the saved checkpoint"
                ) from exc
            corpus_hash.update(chunk.chunk_id, chunk.text)
        if args.resume:
            assert checkpoint is not None
            if corpus_hash.hexdigest() != checkpoint.get("prefix_corpus_hash"):
                raise ValueError(
                    "input corpus prefix does not match the FAISS checkpoint"
                )

        batches = iter_batches(
            chunk_iterator,
            batch_size,
        )
        for batch in batches:
            embeddings = (
                embedder.embed_documents(
                    [chunk.text for chunk in batch],
                    batch_size=batch_size,
                )
                if batch
                else []
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
            if (
                faiss_db is not None
                and processed - last_checkpoint >= args.checkpoint_chunks
            ):
                faiss_db.checkpoint_bulk()
                write_manifest(
                    checkpoint_path,
                    {
                        **cache_identity,
                        "complete": False,
                        "model_id": embedding_config.get("model_id"),
                        "embedding_dimension": faiss_db.index.d,
                        "distance": vector_config.get("distance", "cosine"),
                        "chunk_count": processed,
                        "prefix_corpus_hash": corpus_hash.hexdigest(),
                    },
                )
                if args.commit_modal_volume:
                    commit_mounted_volume_sync(args.modal_volume_name)
                last_checkpoint = processed
        if processed != total_chunks or errors:
            error_path = args.error_report
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise ValueError(
                f"refusing to publish incomplete index: indexed={processed}, "
                f"source_records={total_chunks}, invalid={len(errors)}"
            )
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

    index_file = manifest_path.with_name("index.faiss")
    payload_file = manifest_path.with_name("payloads.json")

    error_path = args.error_report
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
        "batch_size": batch_size,
        "query_max_length": embedding_config.get(
            "query_max_length", embedding_config.get("max_length", 256)
        ),
        "document_max_length": embedding_config.get(
            "document_max_length", embedding_config.get("max_length", 256)
        ),
        "embedding_dimension": (
            faiss_db.index.d
            if faiss_db is not None and faiss_db.index is not None
            else embedding_config.get("output_dimension")
        ),
        "index_sha256": _file_sha256(index_file),
        "payload_sha256": _file_sha256(payload_file),
        "git_commit": git_commit(),
        "download_manifest": _load_download_manifest(),
        "invalid_record_count": len(errors),
        "windowed_document_count": embedder.windowed_document_count,
        "encoded_window_count": embedder.encoded_window_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_manifest(manifest_path, manifest)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if args.commit_modal_volume:
        commit_mounted_volume_sync(args.modal_volume_name)

    print(f"Indexed chunks: {processed}")
    print(f"Invalid records: {len(errors)}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
