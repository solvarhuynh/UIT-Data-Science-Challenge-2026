"""FAISS implementation with a JSON metadata side-store."""

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.vector_db.base import (
    VectorDBAdapter,
    chunk_payload,
    payload_to_hit,
    validate_collection_name,
    validate_query_vector,
    validate_top_k,
    validate_vector_size,
)

_INDEX_SCHEMA_VERSION = 1
_ARTIFACT_FILE_MODE = 0o644
_SUPPORTED_DISTANCES = {"cosine", "dot"}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


class FaissAdapter(VectorDBAdapter):
    """Store normalized vectors in FAISS and payloads beside the index on disk."""

    def __init__(self, index_path: str, collection_name: str) -> None:
        """Initialize a disk-backed FAISS collection."""
        if not isinstance(index_path, str) or not index_path.strip():
            raise ValueError("index_path must be a non-empty string")
        self._index_base = Path(index_path)
        self.collection_name = validate_collection_name(collection_name)
        self.root = self._index_base / self.collection_name
        self.index: faiss.Index | None = None
        self.payloads: dict[int, dict[str, Any]] = {}
        self.distance = "cosine"
        self._load()

    def _load(self) -> None:
        index_file = self.root / "index.faiss"
        payload_file = self.root / "payloads.json"
        if index_file.exists() != payload_file.exists():
            raise ValueError(
                f"incomplete FAISS collection at {self.root}: "
                "index.faiss and payloads.json must both exist"
            )
        if index_file.exists() and payload_file.exists():
            try:
                serialized_bytes = index_file.read_bytes()
                serialized = np.frombuffer(serialized_bytes, dtype=np.uint8)
                loaded_index = faiss.deserialize_index(serialized)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid FAISS index file: {index_file}") from exc
            try:
                raw_store: object = json.loads(
                    payload_file.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(f"invalid FAISS payload file: {payload_file}") from exc
            if not isinstance(raw_store, dict):
                raise ValueError("FAISS payload file must contain a JSON object")
            expected_fields = {
                "schema_version",
                "index_sha256",
                "distance",
                "vector_size",
                "vector_count",
                "payloads",
            }
            if set(raw_store) != expected_fields:
                raise ValueError(
                    "FAISS payload file has an unsupported or incomplete schema"
                )
            schema_version = raw_store["schema_version"]
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version != _INDEX_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"unsupported FAISS payload schema version: {schema_version!r}"
                )
            expected_checksum = hashlib.sha256(serialized_bytes).hexdigest()
            if raw_store["index_sha256"] != expected_checksum:
                raise ValueError(
                    "FAISS index and payload checksum do not match; "
                    "rebuild the collection"
                )
            distance = raw_store["distance"]
            if not isinstance(distance, str) or distance not in _SUPPORTED_DISTANCES:
                raise ValueError("FAISS payload distance must be 'cosine' or 'dot'")
            raw_vector_size = raw_store["vector_size"]
            if (
                isinstance(raw_vector_size, bool)
                or not isinstance(raw_vector_size, int)
                or raw_vector_size != loaded_index.d
            ):
                raise ValueError("FAISS payload vector_size does not match the index")
            raw_vector_count = raw_store["vector_count"]
            if (
                isinstance(raw_vector_count, bool)
                or not isinstance(raw_vector_count, int)
                or raw_vector_count != loaded_index.ntotal
            ):
                raise ValueError("FAISS payload vector_count does not match the index")
            raw_payloads = raw_store["payloads"]
            if not isinstance(raw_payloads, dict):
                raise ValueError("FAISS payloads must contain a JSON object")
            try:
                loaded_payloads = {
                    int(key): value
                    for key, value in raw_payloads.items()
                    if isinstance(key, str)
                    and key == str(int(key))
                    and isinstance(value, dict)
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid FAISS payload IDs in {payload_file}"
                ) from exc
            if len(loaded_payloads) != len(raw_payloads):
                raise ValueError(f"invalid FAISS payload records in {payload_file}")
            expected_ids = set(range(loaded_index.ntotal))
            if set(loaded_payloads) != expected_ids:
                raise ValueError(
                    "FAISS payload IDs must exactly match index vector positions"
                )
            chunk_ids: list[str] = []
            for item_id, payload in loaded_payloads.items():
                for field in ("chunk_id", "doc_id", "text"):
                    value = payload.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"FAISS payload {item_id} has invalid field {field!r}"
                        )
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"FAISS payload {item_id} metadata must be an object"
                    )
                chunk_ids.append(payload["chunk_id"])
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError("FAISS payload chunk_id values must be unique")
            self.index = loaded_index
            self.payloads = loaded_payloads
            self.distance = distance

    def create_collection(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """Create a persistent FAISS collection."""
        validated_name = validate_collection_name(name)
        validate_vector_size(vector_size)
        if distance not in _SUPPORTED_DISTANCES:
            raise ValueError("FAISS adapter supports cosine/dot distance only")
        previous_state = (
            self.root,
            self.collection_name,
            self.index,
            self.payloads,
            self.distance,
        )
        target_root = self._index_base / validated_name
        target_root_existed = target_root.exists()
        try:
            self.root = target_root
            self.collection_name = validated_name
            self.root.mkdir(parents=True, exist_ok=True)
            self.index = None
            self.payloads = {}
            self.distance = distance
            self._load()
            loaded_index = cast(faiss.Index | None, getattr(self, "index"))
            if loaded_index is not None:
                if loaded_index.d != vector_size or self.distance != distance:
                    raise ValueError(
                        "FAISS collection already exists with incompatible "
                        "vector size or distance"
                    )
                return
            self.index = faiss.IndexFlatIP(vector_size)
            self._persist()
        except BaseException:
            (
                self.root,
                self.collection_name,
                self.index,
                self.payloads,
                self.distance,
            ) = previous_state
            if not target_root_existed:
                try:
                    target_root.rmdir()
                except OSError:
                    pass
            raise

    def _persist(self) -> None:
        if self.index is None:
            return
        index_target = self.root / "index.faiss"
        payload_target = self.root / "payloads.json"
        serialized_index = faiss.serialize_index(self.index).tobytes()
        payload_store = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "index_sha256": hashlib.sha256(serialized_index).hexdigest(),
            "distance": self.distance,
            "vector_size": self.index.d,
            "vector_count": self.index.ntotal,
            "payloads": self.payloads,
        }
        serialized_payloads = (
            json.dumps(
                payload_store,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        root_existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        descriptors = [-1, -1]
        staged_paths: list[Path] = []
        backups: dict[Path, Path] = {}
        committed: list[Path] = []
        targets = (index_target, payload_target)
        preserve_backups = False
        persist_succeeded = False
        try:
            descriptors[0], index_temporary_name = tempfile.mkstemp(
                prefix=".index.",
                suffix=".tmp",
                dir=str(self.root),
            )
            staged_paths.append(Path(index_temporary_name))
            descriptors[1], payload_temporary_name = tempfile.mkstemp(
                prefix=".payloads.",
                suffix=".tmp",
                dir=str(self.root),
            )
            staged_paths.append(Path(payload_temporary_name))
            with os.fdopen(descriptors[0], "wb") as index_stream:
                descriptors[0] = -1
                index_stream.write(serialized_index)
                index_stream.flush()
                os.fsync(index_stream.fileno())
            with os.fdopen(
                descriptors[1],
                "w",
                encoding="utf-8",
                newline="\n",
            ) as payload_stream:
                descriptors[1] = -1
                payload_stream.write(serialized_payloads)
                payload_stream.flush()
                os.fsync(payload_stream.fileno())
            for staged_path in staged_paths:
                os.chmod(staged_path, _ARTIFACT_FILE_MODE)

            for target in targets:
                if target.exists():
                    backup_descriptor, backup_name = tempfile.mkstemp(
                        prefix=f".{target.name}.",
                        suffix=".backup",
                        dir=str(self.root),
                    )
                    os.close(backup_descriptor)
                    os.unlink(backup_name)
                    backup_path = Path(backup_name)
                    os.replace(target, backup_path)
                    backups[target] = backup_path

            for staged_path, target in zip(staged_paths, targets):
                os.replace(staged_path, target)
                committed.append(target)
            persist_succeeded = True
        except BaseException as persist_error:
            rollback_errors: list[BaseException] = []
            for target in reversed(committed):
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    rollback_errors.append(exc)
            for target in reversed(targets):
                existing_backup = backups.get(target)
                if existing_backup is not None:
                    try:
                        os.replace(existing_backup, target)
                    except OSError as exc:
                        rollback_errors.append(exc)
            if rollback_errors:
                preserve_backups = True
                raise RuntimeError(
                    "FAISS persistence failed and rollback was incomplete; "
                    "backup files were retained"
                ) from persist_error
            raise persist_error
        finally:
            for descriptor in descriptors:
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            for temporary_path in staged_paths:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            if not preserve_backups:
                for backup_path in backups.values():
                    try:
                        backup_path.unlink()
                    except OSError:
                        pass
            if not persist_succeeded and not root_existed:
                try:
                    self.root.rmdir()
                except OSError:
                    pass

    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks in the active FAISS collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        chunk_ids: list[str] = []
        for item_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(chunk, LegalChunk):
                raise TypeError(f"chunks[{item_index}] must be a LegalChunk")
            if not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{item_index}].chunk_id must not be blank")
            validate_query_vector(embedding)
            chunk_ids.append(chunk.chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("upsert chunks must have unique chunk_id values")

        batch_dimension = len(embeddings[0])
        if any(len(embedding) != batch_dimension for embedding in embeddings):
            raise ValueError(
                "all embeddings in an upsert batch must have the same dimension"
            )
        prepared_vectors = [
            self._prepare_vector(embedding, label=f"embeddings[{item_index}]")
            for item_index, embedding in enumerate(embeddings)
        ]
        index = self.index
        if index is None:
            index = faiss.IndexFlatIP(batch_dimension)
        if any(len(embedding) != index.d for embedding in embeddings):
            raise ValueError(f"all embeddings must have vector dimension {index.d}")

        if index.ntotal:
            existing_vectors = index.reconstruct_n(0, index.ntotal)
        else:
            existing_vectors = np.empty((0, index.d), dtype="float32")
        vectors = [vector.copy() for vector in existing_vectors]
        updated_payloads = dict(self.payloads)
        positions_by_chunk_id = {
            payload["chunk_id"]: position
            for position, payload in updated_payloads.items()
        }
        for chunk, vector in zip(chunks, prepared_vectors):
            position = positions_by_chunk_id.get(chunk.chunk_id)
            if position is None:
                position = len(vectors)
                positions_by_chunk_id[chunk.chunk_id] = position
                vectors.append(vector)
            else:
                vectors[position] = vector
            updated_payloads[position] = chunk_payload(chunk)

        updated_index = faiss.IndexFlatIP(index.d)
        updated_index.add(np.asarray(vectors, dtype="float32"))
        previous_index = self.index
        previous_payloads = self.payloads
        self.index = updated_index
        self.payloads = updated_payloads
        try:
            self._persist()
        except BaseException:
            self.index = previous_index
            self.payloads = previous_payloads
            raise

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return nearest FAISS hits that match optional filters."""
        validate_query_vector(query_vector)
        validate_top_k(top_k)
        if self.index is None or self.index.ntotal == 0:
            return []
        if len(query_vector) != self.index.d:
            raise ValueError(
                f"query vector must have dimension {self.index.d}, "
                f"got {len(query_vector)}"
            )
        prepared_query = self._prepare_vector(query_vector, label="query_vector")
        candidate_count = (
            self.index.ntotal if filters else min(top_k, self.index.ntotal)
        )
        scores, ids = self.index.search(
            np.asarray([prepared_query], dtype="float32"), candidate_count
        )
        hits: list[RetrievalHit] = []
        for score, item_id in zip(scores[0], ids[0]):
            payload = self.payloads.get(int(item_id))
            if payload and all(
                (
                    self._payload_value(payload, key) in value
                    if isinstance(value, list)
                    else self._payload_value(payload, key) == value
                )
                for key, value in (filters or {}).items()
            ):
                hits.append(payload_to_hit(payload, float(score)))
                if len(hits) == top_k:
                    break
        return hits

    def _prepare_vector(self, vector: list[float], *, label: str) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{label} cannot be represented as finite float values")
        if self.distance == "cosine":
            norm = float(np.linalg.norm(array))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError(f"{label} must have a non-zero finite norm")
            array = array / norm
        prepared = np.asarray(array, dtype=np.float32)
        if not np.isfinite(prepared).all():
            raise ValueError(f"{label} cannot be represented as finite float32 values")
        return prepared

    @staticmethod
    def _payload_value(payload: dict[str, Any], key: str) -> Any:
        if key in payload:
            return payload[key]
        metadata = payload.get("metadata")
        return metadata.get(key) if isinstance(metadata, dict) else None

    def delete_collection(self, name: str) -> None:
        """Delete a persistent FAISS collection."""
        validated_name = validate_collection_name(name)
        target = self._index_base / validated_name
        if target.exists():
            shutil.rmtree(target)
        if validated_name == self.collection_name:
            self.index = None
            self.payloads = {}
            self.distance = "cosine"
