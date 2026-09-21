"""Fail-closed Modal GPU runner for the bounded TOP5 BGE worklist.

This runner is preparation only.  It is intentionally separate from the
historical Step4 script and is never invoked during repository validation.

The worklist is uploaded once as a gzip-compressed argument by the local
entrypoint.  Questions and canonical chunks are read from the existing
``udsc-p13`` Volume; the Step4 Base and FT model directories are read from the
existing ``udsc-task1-modal`` Volume.  The scorer keeps separate FT and Base
scores and uses MAX aggregation for the subset artifact.  It never averages
the two models.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import modal


APP_NAME = "task1-bge-subset-top5"
CHECKPOINT_SCHEMA = "task1_bge_subset_checkpoint_v1"
OUTPUT_SCHEMA = "task1_bge_subset_scores_v1"
VOLUME_NAME = "udsc-p13"
MODEL_VOLUME_NAME = "udsc-task1-modal"
MOUNT = Path("/workspace/p13")
MODEL_MOUNT = Path("/data")
RUNTIME_ROOT = MOUNT / "runtime"
DEFAULT_RUNTIME_NAMESPACE = "task1_bge_subset_top5"
RUNTIME = RUNTIME_ROOT / DEFAULT_RUNTIME_NAMESPACE
CHECKPOINT_ROOT = RUNTIME / "checkpoints"
RESULT_ROOT = RUNTIME / "results"
MANIFEST_ROOT = RUNTIME / "manifests"
TRAIN_PATH = MOUNT / "runtime/data/raw/btc/LegalIR/train.json"
CHUNKS_ROOT = MOUNT / "runtime/data/processed_v3/chunks"
BASE_MODEL_PATH = MODEL_MOUNT / "models/reranker"
FT_MODEL_PATH = MODEL_MOUNT / "bge_ft/bge_m3_finetuned"

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BASE_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FT_MODEL_ID = "BGE-M3 fine-tuned Step4 Model A"
FT_WEIGHT_SHA256 = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
FT_CONFIG_SHA256 = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
FT_ARTIFACT_ID = f"sha256:{FT_WEIGHT_SHA256}"
MAX_LENGTH = 512
SELECTOR = "true_s2_bm25_within_document_v2"
AGGREGATION = "MAX"
SUPPORTED_BATCH_SIZES = (1, 4, 8, 16, 32)
CANARY_QDOCS = 256
CANARY_MAX_UNITS = 768
DEFAULT_CHECKPOINT_EVERY = 256

_SOURCE_PATH = Path(__file__).resolve()
LOCAL_ROOT_SENTINEL = Path("/__modal_local_source_unavailable__")


def resolve_local_root(source_path: Path) -> Path:
    """Resolve the repository root locally without breaking Modal imports."""
    return source_path.parents[2] if len(source_path.parents) >= 3 else LOCAL_ROOT_SENTINEL


LOCAL_ROOT = resolve_local_root(_SOURCE_PATH)
DEFAULT_WORKLIST = LOCAL_ROOT / "artifacts/task1/qwen_to_bge_minimal/top5_missing_bge.jsonl"
LOCAL_SELECTOR_ROOT = LOCAL_ROOT / "scripts/beam/task1_v2"


def validate_runtime_namespace(value: str) -> str:
    """Allow only one safe namespace component below /workspace/p13/runtime."""
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise RuntimeError(f"invalid runtime namespace: {value!r}")
    return value


def runtime_paths(namespace: str) -> tuple[Path, Path, Path]:
    namespace = validate_runtime_namespace(namespace)
    runtime = RUNTIME_ROOT / namespace
    return runtime / "checkpoints", runtime / "results", runtime / "manifests"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==5.0.0",
    "sentence-transformers==5.4.1",
    "accelerate>=1.1",
    "numpy",
    "pyvi==0.1.1",
)
if LOCAL_SELECTOR_ROOT.is_dir():
    image = image.add_local_dir(str(LOCAL_SELECTOR_ROOT), remote_path="/root/scripts/beam/task1_v2")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    return atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _one_label(config: dict[str, Any]) -> int:
    return int(config.get("num_labels", len(config.get("id2label", {})) or -1))


def validate_bge_directory(path: Path, label: str) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"{label}_MODEL_PATH_MISSING: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"{label}_MODEL_CONFIG_MISSING: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "xlm-roberta" or _one_label(config) != 1:
        raise RuntimeError(f"{label}_MODEL_CONFIG_NOT_SINGLE_LABEL_BGE")
    if "qwen" in str(path).casefold() or "qwen" in json.dumps(config).casefold():
        raise RuntimeError("UNEXPECTED_QWEN_RUNTIME_PATH")
    config["_resolved_num_labels"] = 1
    return config


def resolve_revision(path: Path, *, required: bool, default: str | None = None) -> str:
    """Resolve a durable model revision without guessing from a directory name."""
    candidates: list[str] = []
    config_path = path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("revision", "commit_hash", "_commit_hash", "model_revision"):
            if config.get(key):
                candidates.append(str(config[key]))
    for metadata_name in ("revision.json", "metadata.json", "model_metadata.json"):
        metadata_path = path / metadata_name
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
            if isinstance(metadata, dict):
                for key in ("revision", "commit_hash", "_commit_hash", "model_revision"):
                    if metadata.get(key):
                        candidates.append(str(metadata[key]))
    for relative in ("revision.txt", "REVISION", ".cache/huggingface/refs/main"):
        metadata_path = path / relative
        if metadata_path.is_file():
            value = metadata_path.read_text(encoding="utf-8").strip()
            if value:
                candidates.append(value)
    trees = path / ".cache/huggingface/trees"
    if trees.is_dir():
        for item in trees.iterdir():
            if re.fullmatch(r"[0-9a-f]{40}", item.name):
                candidates.append(item.name)
            elif item.is_file() and re.fullmatch(r"[0-9a-f]{40}\.json", item.name):
                candidates.append(item.stem)
    for candidate in candidates:
        if re.fullmatch(r"[0-9a-f]{40}", candidate):
            return candidate
    if default is not None:
        return default
    if required:
        raise RuntimeError(f"MODEL_REVISION_PROVENANCE_MISSING: no exact revision metadata under {path}")
    return "UNPROVEN"


def validate_ft_artifact(path: Path) -> dict[str, Any]:
    """Validate the fine-tuned checkpoint by immutable file hashes."""
    config = validate_bge_directory(path, "BGE_FT")
    weight_path = path / "model.safetensors"
    if not weight_path.is_file():
        raise RuntimeError(f"BGE_FT_WEIGHT_MISSING: {weight_path}")
    actual_weight_sha = sha256_file(weight_path)
    if actual_weight_sha != FT_WEIGHT_SHA256:
        raise RuntimeError(f"BGE_FT_WEIGHT_SHA256_MISMATCH: {actual_weight_sha}")
    actual_config_sha = sha256_file(path / "config.json")
    if actual_config_sha != FT_CONFIG_SHA256:
        raise RuntimeError(f"BGE_FT_CONFIG_SHA256_MISMATCH: {actual_config_sha}")
    return {
        "model_id": FT_MODEL_ID,
        "path": str(path),
        "artifact_id": FT_ARTIFACT_ID,
        "weight_sha256": actual_weight_sha,
        "config_sha256": actual_config_sha,
        "num_labels": config["_resolved_num_labels"],
    }


def validate_model_contracts() -> dict[str, Any]:
    base_config = validate_bge_directory(BASE_MODEL_PATH, "BGE_BASE")
    ft_metadata = validate_ft_artifact(FT_MODEL_PATH)
    base_revision = resolve_revision(BASE_MODEL_PATH, required=True)
    if base_revision != BASE_MODEL_REVISION:
        raise RuntimeError(f"BGE_BASE_REVISION_MISMATCH: {base_revision}")
    return {
        "base": {
            "model_id": MODEL_ID,
            "path": str(BASE_MODEL_PATH),
            "revision": base_revision,
            "config_sha256": sha256_file(BASE_MODEL_PATH / "config.json"),
            "num_labels": base_config["_resolved_num_labels"],
        },
        "ft": ft_metadata,
    }


def validate_model_preflight_contracts() -> dict[str, Any]:
    """Validate mounted model metadata without hashing or loading weights."""
    base_config = validate_bge_directory(BASE_MODEL_PATH, "BGE_BASE")
    base_weight_path = BASE_MODEL_PATH / "model.safetensors"
    if not base_weight_path.is_file():
        raise RuntimeError(f"BGE_BASE_WEIGHT_MISSING: {base_weight_path}")
    base_revision = resolve_revision(BASE_MODEL_PATH, required=True)
    if base_revision != BASE_MODEL_REVISION:
        raise RuntimeError(f"BGE_BASE_REVISION_MISMATCH: {base_revision}")

    ft_config = validate_bge_directory(FT_MODEL_PATH, "BGE_FT")
    ft_weight_path = FT_MODEL_PATH / "model.safetensors"
    if not ft_weight_path.is_file():
        raise RuntimeError(f"BGE_FT_WEIGHT_MISSING: {ft_weight_path}")
    ft_config_sha = sha256_file(FT_MODEL_PATH / "config.json")
    if ft_config_sha != FT_CONFIG_SHA256:
        raise RuntimeError(f"BGE_FT_CONFIG_SHA256_MISMATCH: {ft_config_sha}")
    return {
        "base_path": str(BASE_MODEL_PATH),
        "base_revision": base_revision,
        "base_revision_match": base_revision == BASE_MODEL_REVISION,
        "base_config_sha256": sha256_file(BASE_MODEL_PATH / "config.json"),
        "base_weight_bytes": base_weight_path.stat().st_size,
        "ft_path": str(FT_MODEL_PATH),
        "ft_config_sha256": ft_config_sha,
        "ft_config_match": ft_config_sha == FT_CONFIG_SHA256,
        "ft_weight_bytes": ft_weight_path.stat().st_size,
        "base_num_labels": base_config["_resolved_num_labels"],
        "ft_num_labels": ft_config["_resolved_num_labels"],
    }


def contract_sha256() -> str:
    contract = {
        "schema": OUTPUT_SCHEMA,
        "selector": SELECTOR,
        "aggregation": AGGREGATION,
        "max_length": MAX_LENGTH,
        "activation": "CrossEncoder default single-label activation; no manual sigmoid",
        "dtype": "model default on CUDA; no fp16 override",
        "question_source": str(TRAIN_PATH),
        "document_source": str(CHUNKS_ROOT),
    }
    return sha256_bytes(json.dumps(contract, sort_keys=True).encode())


def load_worklist(payload: bytes, limit: int | None) -> tuple[list[dict[str, Any]], str, str]:
    raw = gzip.decompress(payload)
    full_sha = sha256_bytes(raw)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = (str(row.get("query_id", "")), str(row.get("document_id", "")))
        if not all(identity):
            raise RuntimeError(f"invalid worklist identity at line {line_number}")
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError("empty worklist")
    selected_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    return rows, full_sha, sha256_bytes(selected_bytes)


def load_questions(query_ids: set[str], questions_payload: bytes | None = None) -> dict[str, str]:
    """Load question text from the explicit run source or validation default."""
    if questions_payload is None:
        payload = json.loads(TRAIN_PATH.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(gzip.decompress(questions_payload).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError("question source must be a JSON object keyed by query_id")
    result: dict[str, str] = {}
    for query_id in query_ids:
        record = payload.get(query_id)
        if not isinstance(record, dict) or not str(record.get("question", "")).strip():
            raise RuntimeError(f"canonical question missing: {query_id}")
        result[query_id] = str(record["question"])
    if set(result) != query_ids:
        raise RuntimeError("question coverage mismatch")
    return result


def load_document(doc_id: str, chunks_root: Path = CHUNKS_ROOT) -> list[dict[str, Any]]:
    path = chunks_root / f"{doc_id}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"canonical document missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id", ""))
            raw_text = str(item.get("raw_chunk_text", item.get("text", "")))
            if not chunk_id.startswith(doc_id + "_") or not raw_text.strip():
                raise RuntimeError(f"invalid chunk {path}:{line_number}")
            item["chunk_id"] = chunk_id
            item["raw_chunk_text"] = raw_text
            rows.append(item)
    if not rows or len({item["chunk_id"] for item in rows}) != len(rows):
        raise RuntimeError(f"invalid/duplicate chunks: {path}")
    return rows


def resolve_rows(
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    prepared_cache: dict[str, Any] | None = None,
    chunks_root: Path = CHUNKS_ROOT,
) -> tuple[list[dict[str, Any]], int]:
    from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared

    prepared = prepared_cache if prepared_cache is not None else {}
    for doc_id in sorted({str(row["document_id"]) for row in rows}):
        if doc_id not in prepared:
            prepared[doc_id] = prepare_document(load_document(doc_id, chunks_root))
    resolved: list[dict[str, Any]] = []
    total_units = 0
    for row in rows:
        query_id = str(row["query_id"])
        doc_id = str(row["document_id"])
        selected = select_true_s2_prepared(questions[query_id], prepared[doc_id], topk=3)
        expected = row.get("expected_inference_units")
        if expected is not None and len(selected) != int(expected):
            raise RuntimeError(f"selected unit mismatch: {query_id}/{doc_id}")
        selected_ids = [str(item["chunk_id"]) for item in selected]
        selected_texts = [str(item["raw_chunk_text"]) for item in selected]
        expected_ids = row.get("selected_chunk_ids")
        if expected_ids is not None:
            expected_ids = [str(value) for value in expected_ids]
            if selected_ids != expected_ids:
                raise RuntimeError(f"selected chunk identity mismatch: {query_id}/{doc_id}")
        if not (1 <= len(selected) <= 3) or len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError(f"TOP3_UP_TO_AVAILABLE violation: {query_id}/{doc_id}")
        if any(not chunk_id.startswith(doc_id + "_") or not text.strip() for chunk_id, text in zip(selected_ids, selected_texts)):
            raise RuntimeError(f"invalid selected chunk: {query_id}/{doc_id}")
        resolved.append({
            "query_id": query_id,
            "document_id": doc_id,
            "question": questions[query_id],
            "selected_chunk_ids": selected_ids,
            "selected_texts": selected_texts,
        })
        total_units += len(selected)
    return resolved, total_units


def resolve_frozen_rows(
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    document_cache: dict[str, dict[str, str]] | None = None,
    chunks_root: Path = CHUNKS_ROOT,
) -> tuple[list[dict[str, Any]], int]:
    """Resolve text for the selector output already frozen in the worklist.

    The worklist is the immutable selector boundary: its selected chunk IDs
    were produced by ``true_s2_bm25_within_document_v2`` during CPU
    preparation.  This path deliberately does not rerun BM25/tokenization in
    the GPU function.  It validates the frozen IDs and reads each document at
    most once per run, preserving the exact scored chunk identity/order.
    """
    cache = document_cache if document_cache is not None else {}
    for doc_id in sorted({str(row["document_id"]) for row in rows}):
        if doc_id not in cache:
            cache[doc_id] = {
                str(item["chunk_id"]): str(item["raw_chunk_text"])
                for item in load_document(doc_id, chunks_root)
            }
    resolved: list[dict[str, Any]] = []
    total_units = 0
    for row in rows:
        query_id = str(row["query_id"])
        doc_id = str(row["document_id"])
        selected_ids = [str(value) for value in row.get("selected_chunk_ids", [])]
        expected = int(row.get("expected_inference_units", len(selected_ids)))
        if not (1 <= len(selected_ids) <= 3) or len(selected_ids) != expected:
            raise RuntimeError(f"frozen selector unit mismatch: {query_id}/{doc_id}")
        if len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError(f"duplicate frozen selected chunk: {query_id}/{doc_id}")
        if any(not chunk_id.startswith(doc_id + "_") for chunk_id in selected_ids):
            raise RuntimeError(f"invalid frozen selected chunk: {query_id}/{doc_id}")
        try:
            selected_texts = [cache[doc_id][chunk_id] for chunk_id in selected_ids]
        except KeyError as exc:
            raise RuntimeError(f"frozen selected chunk missing: {query_id}/{doc_id}") from exc
        if any(not text.strip() for text in selected_texts):
            raise RuntimeError(f"empty frozen selected chunk: {query_id}/{doc_id}")
        resolved.append({
            "query_id": query_id,
            "document_id": doc_id,
            "question": questions[query_id],
            "selected_chunk_ids": selected_ids,
            "selected_texts": selected_texts,
        })
        total_units += len(selected_ids)
    return resolved, total_units


class DurableBgeStore:
    """Two-model generation checkpoint with fail-closed contract identity."""

    def __init__(self, path: Path, identity: dict[str, str], volume_commit: Callable[[], None]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.volume_commit = volume_commit
        try:
            self.db = sqlite3.connect(path)
            integrity = str(self.db.execute("PRAGMA integrity_check").fetchone()[0])
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("CHECKPOINT_CORRUPT") from exc
        if integrity != "ok":
            raise RuntimeError("CHECKPOINT_CORRUPT")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunk_scores(
              query_id TEXT NOT NULL, document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
              chunk_id TEXT NOT NULL, ft_score REAL NOT NULL, base_score REAL NOT NULL,
              generation INTEGER NOT NULL, PRIMARY KEY(query_id,document_id,chunk_index));
            CREATE TABLE IF NOT EXISTS document_results(
              query_id TEXT NOT NULL, document_id TEXT NOT NULL, ft_score REAL NOT NULL,
              base_score REAL NOT NULL, chunk_count INTEGER NOT NULL, generation INTEGER NOT NULL,
              PRIMARY KEY(query_id,document_id));
            """
        )
        current = dict(self.db.execute("SELECT key,value FROM metadata"))
        required = {"schema_version": CHECKPOINT_SCHEMA, **identity}
        if current:
            for key, value in required.items():
                if current.get(key) != value:
                    self.db.close()
                    raise RuntimeError(f"STALE_CHECKPOINT_REJECTED:{key}")
        else:
            self.db.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", list(required.items()) + [("durable_generation", "0")])
            self.db.commit()
        metadata = dict(self.db.execute("SELECT key,value FROM metadata"))
        self.durable_generation = int(metadata["durable_generation"])
        self.db.execute("DELETE FROM chunk_scores WHERE generation > ?", (self.durable_generation,))
        self.db.execute("DELETE FROM document_results WHERE generation > ?", (self.durable_generation,))
        self.db.commit()

    def completed_identities(self) -> set[tuple[str, str]]:
        return {(str(q), str(d)) for q, d in self.db.execute(
            "SELECT query_id,document_id FROM document_results WHERE generation <= ?", (self.durable_generation,)
        )}

    def commit_batch(self, batch: list[dict[str, Any]]) -> int:
        if not batch:
            return self.durable_generation
        generation = self.durable_generation + 1
        try:
            for item in batch:
                qid = item["query_id"]
                did = item["document_id"]
                ids = item["selected_chunk_ids"]
                ft_scores = item["ft_scores"]
                base_scores = item["base_scores"]
                if not ids or len(ids) != len(set(ids)) or len(ids) != len(ft_scores) or len(ids) != len(base_scores):
                    raise RuntimeError(f"invalid checkpoint chunk accounting: {qid}/{did}")
                if not all(math.isfinite(float(x)) for x in [*ft_scores, *base_scores]):
                    raise RuntimeError(f"non-finite score: {qid}/{did}")
                self.db.executemany(
                    "INSERT INTO chunk_scores VALUES (?,?,?,?,?,?,?) ON CONFLICT(query_id,document_id,chunk_index) DO UPDATE SET chunk_id=excluded.chunk_id,ft_score=excluded.ft_score,base_score=excluded.base_score,generation=excluded.generation",
                    [(qid, did, index, chunk_id, float(ft_scores[index]), float(base_scores[index]), generation) for index, chunk_id in enumerate(ids)],
                )
                self.db.execute(
                    "INSERT INTO document_results VALUES (?,?,?,?,?,?) ON CONFLICT(query_id,document_id) DO UPDATE SET ft_score=excluded.ft_score,base_score=excluded.base_score,chunk_count=excluded.chunk_count,generation=excluded.generation",
                    (qid, did, max(ft_scores), max(base_scores), len(ids), generation),
                )
            self.db.commit()
            self.volume_commit()
            self.db.execute("UPDATE metadata SET value=? WHERE key='durable_generation'", (str(generation),))
            self.db.commit()
            self.volume_commit()
        except Exception:
            self.db.rollback()
            raise
        self.durable_generation = generation
        return generation

    def validate_complete(self, expected: dict[tuple[str, str], int]) -> None:
        actual = {
            (str(q), str(d)): int(count)
            for q, d, count in self.db.execute(
                "SELECT query_id,document_id,chunk_count FROM document_results WHERE generation <= ?",
                (self.durable_generation,),
            )
        }
        if actual != expected:
            raise RuntimeError(f"CHECKPOINT_INCOMPLETE:{len(actual)}/{len(expected)}")
        for qid, did, ft_score, base_score, count, _ in self.db.execute(
            "SELECT query_id,document_id,ft_score,base_score,chunk_count,generation FROM document_results WHERE generation <= ?",
            (self.durable_generation,),
        ):
            chunk_rows = list(self.db.execute(
                "SELECT ft_score,base_score FROM chunk_scores WHERE query_id=? AND document_id=? AND generation <= ? ORDER BY chunk_index",
                (qid, did, self.durable_generation),
            ))
            if len(chunk_rows) != int(count) or not all(math.isfinite(float(x)) for pair in chunk_rows for x in pair):
                raise RuntimeError(f"CHECKPOINT_CHUNK_INVALID:{qid}/{did}")
            if float(ft_score) != max(float(x[0]) for x in chunk_rows) or float(base_score) != max(float(x[1]) for x in chunk_rows):
                raise RuntimeError(f"CHECKPOINT_MAX_MISMATCH:{qid}/{did}")

    def result_for(self, query_id: str, document_id: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT ft_score,base_score,chunk_count FROM document_results WHERE query_id=? AND document_id=? AND generation <= ?",
            (query_id, document_id, self.durable_generation),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing result: {query_id}/{document_id}")
        chunks = list(self.db.execute(
            "SELECT chunk_id,ft_score,base_score FROM chunk_scores WHERE query_id=? AND document_id=? AND generation <= ? ORDER BY chunk_index",
            (query_id, document_id, self.durable_generation),
        ))
        return {
            "bge_ft_score": float(row[0]),
            "bge_base_score": float(row[1]),
            "selected_chunk_ids": [str(x[0]) for x in chunks],
            "ft_chunk_scores": [float(x[1]) for x in chunks],
            "base_chunk_scores": [float(x[2]) for x in chunks],
        }

    def close(self) -> None:
        self.db.close()


def score_subset(
    worklist_payload: bytes,
    mode: str,
    limit: int | None,
    batch_size: int,
    checkpoint_every: int,
    run_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
    questions_payload: bytes | None = None,
    chunks_root: str = "",
) -> dict[str, Any]:
    if mode not in {"canary", "production"}:
        raise RuntimeError("mode must be canary or production")
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise RuntimeError(f"batch_size must be one of {SUPPORTED_BATCH_SIZES}")
    if checkpoint_every <= 0:
        raise RuntimeError("checkpoint_every must be positive")
    if mode == "canary" and limit != CANARY_QDOCS:
        raise RuntimeError(f"canary requires --limit {CANARY_QDOCS}")
    run_namespace = validate_runtime_namespace(run_namespace)
    resolved_chunks_root = Path(chunks_root) if chunks_root else CHUNKS_ROOT
    checkpoint_root, result_root, manifest_root = runtime_paths(run_namespace)

    volume.reload()
    model_volume.reload()
    started = time.perf_counter()
    worklist_started = time.perf_counter()
    rows, full_worklist_sha, selected_worklist_sha = load_worklist(worklist_payload, limit)
    worklist_load_seconds = time.perf_counter() - worklist_started
    if mode == "canary" and len(rows) != CANARY_QDOCS:
        raise RuntimeError(f"canary q-doc count mismatch: {len(rows)}/{CANARY_QDOCS}")

    expected: dict[tuple[str, str], int] = {}
    for row in rows:
        identity = (str(row["query_id"]), str(row["document_id"]))
        if identity in expected:
            raise RuntimeError(f"duplicate q-doc identity: {identity[0]}/{identity[1]}")
        raw_expected = row.get("expected_inference_units")
        if raw_expected is None:
            raise RuntimeError(f"missing expected inference units: {identity[0]}/{identity[1]}")
        count = int(raw_expected)
        selected_ids = row.get("selected_chunk_ids")
        if str(row.get("selector", "")) != SELECTOR:
            raise RuntimeError(f"selector contract mismatch: {identity[0]}/{identity[1]}")
        if str(row.get("aggregation", "")) != AGGREGATION:
            raise RuntimeError(f"aggregation contract mismatch: {identity[0]}/{identity[1]}")
        if int(row.get("max_length", MAX_LENGTH)) != MAX_LENGTH:
            raise RuntimeError(f"max_length contract mismatch: {identity[0]}/{identity[1]}")
        if not (1 <= count <= 3) or not isinstance(selected_ids, list) or len(selected_ids) != count:
            raise RuntimeError(f"invalid frozen selector contract: {identity[0]}/{identity[1]}")
        if len({str(value) for value in selected_ids}) != count:
            raise RuntimeError(f"duplicate frozen selected chunk: {identity[0]}/{identity[1]}")
        expected[identity] = count
    total_units = sum(expected.values())
    if mode == "canary" and total_units > CANARY_MAX_UNITS:
        raise RuntimeError(f"canary unit bound exceeded: {total_units}/{CANARY_MAX_UNITS}")

    contract_started = time.perf_counter()
    models = validate_model_contracts()
    model_contract_seconds = time.perf_counter() - contract_started

    identity = {
        "mode": mode,
        "worklist_sha256": full_worklist_sha,
        "selected_worklist_sha256": selected_worklist_sha,
        "q_doc_count": str(len(rows)),
        "inference_unit_count": str(total_units),
        "contract_sha256": contract_sha256(),
        "selector": SELECTOR,
        "aggregation": AGGREGATION,
        "max_length": str(MAX_LENGTH),
        "base_model_id": models["base"]["model_id"],
        "base_model_revision": models["base"]["revision"],
        "ft_model_id": models["ft"]["model_id"],
        "ft_artifact_id": models["ft"]["artifact_id"],
        "ft_weight_sha256": models["ft"]["weight_sha256"],
        "ft_config_sha256": models["ft"]["config_sha256"],
        "runtime_namespace": run_namespace,
    }
    checkpoint_path = checkpoint_root / mode / "subset.sqlite3"
    result_path = result_root / mode / "scores.jsonl"
    manifest_path = manifest_root / f"{mode}.json"
    checkpoint_open_started = time.perf_counter()
    store = DurableBgeStore(checkpoint_path, identity, volume.commit)
    checkpoint_open_seconds = time.perf_counter() - checkpoint_open_started
    completed = store.completed_identities()
    if not completed.issubset(expected):
        raise RuntimeError("CHECKPOINT_HAS_UNKNOWN_IDENTITIES")
    remaining = [
        row for row in rows
        if (str(row["query_id"]), str(row["document_id"])) not in completed
    ]
    completed_units = sum(expected.get(identity_key, 0) for identity_key in completed)
    questions = load_questions(
        {str(row["query_id"]) for row in remaining}, questions_payload
    ) if remaining else {}
    actual_gpu = "unknown"
    model_load_seconds = 0.0
    inference_seconds = 0.0
    inference_batches = 0
    inference_forward_calls = 0
    preprocessing_seconds = 0.0
    checkpoint_seconds = 0.0
    document_cache: dict[str, dict[str, str]] = {}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        actual_gpu = str(torch.cuda.get_device_name(0))
        if "L4" not in actual_gpu.upper():
            raise RuntimeError(f"UNEXPECTED_GPU:{actual_gpu}")

        if remaining:
            from sentence_transformers import CrossEncoder

            load_started = time.perf_counter()
            ft_model = CrossEncoder(str(FT_MODEL_PATH), num_labels=1, max_length=MAX_LENGTH, device="cuda")
            base_model = CrossEncoder(str(BASE_MODEL_PATH), num_labels=1, max_length=MAX_LENGTH, device="cuda")
            ft_model.model.eval()
            base_model.model.eval()
            model_load_seconds = time.perf_counter() - load_started
            with torch.inference_mode():
                for offset in range(0, len(remaining), checkpoint_every):
                    current_rows = remaining[offset : offset + checkpoint_every]
                    preprocess_started = time.perf_counter()
                    current, current_units = resolve_frozen_rows(
                        current_rows, questions, document_cache, resolved_chunks_root
                    )
                    preprocessing_seconds += time.perf_counter() - preprocess_started
                    expected_current_units = sum(
                        expected[(str(row["query_id"]), str(row["document_id"]))]
                        for row in current_rows
                    )
                    if current_units != expected_current_units:
                        raise RuntimeError(f"block unit mismatch: {current_units}/{expected_current_units}")
                    pairs: list[list[str]] = []
                    offsets: list[tuple[int, int]] = []
                    for item in current:
                        start = len(pairs)
                        pairs.extend([[item["question"], text] for text in item["selected_texts"]])
                        offsets.append((start, len(pairs)))
                    score_started = time.perf_counter()
                    ft_raw = ft_model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                    base_raw = base_model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                    inference_seconds += time.perf_counter() - score_started
                    inference_batches += 2
                    inference_forward_calls += 2 * math.ceil(len(pairs) / batch_size)
                    checkpoint_batch: list[dict[str, Any]] = []
                    for item, (start, end), ft_values, base_values in zip(current, offsets, [ft_raw[i:j] for i, j in offsets], [base_raw[i:j] for i, j in offsets]):
                        ft_scores = [float(x) for x in ft_values]
                        base_scores = [float(x) for x in base_values]
                        checkpoint_batch.append({**item, "ft_scores": ft_scores, "base_scores": base_scores})
                    checkpoint_started = time.perf_counter()
                    store.commit_batch(checkpoint_batch)
                    checkpoint_seconds += time.perf_counter() - checkpoint_started
                    done = len(completed) + min(offset + len(current_rows), len(remaining))
                    done_units = completed_units + sum(
                        expected[(str(item["query_id"]), str(item["document_id"]))]
                        for item in remaining[: offset + len(current_rows)]
                    )
                    print(f"BGE_PROGRESS completed={done}/{len(rows)} units={done_units}/{total_units} elapsed={time.perf_counter()-started:.1f}", flush=True)
                    del current, current_rows, pairs, offsets, checkpoint_batch
            del ft_model, base_model
            torch.cuda.empty_cache()
        store.validate_complete(expected)
        output_rows: list[dict[str, Any]] = []
        for item in rows:
            query_id = str(item["query_id"])
            document_id = str(item["document_id"])
            result = store.result_for(query_id, document_id)
            output_rows.append({
                "query_id": query_id,
                "document_id": document_id,
                **result,
                "model_id": MODEL_ID,
                "base_model_revision": models["base"]["revision"],
                "ft_model_id": models["ft"]["model_id"],
                "ft_artifact_id": models["ft"]["artifact_id"],
                "ft_weight_sha256": models["ft"]["weight_sha256"],
                "ft_config_sha256": models["ft"]["config_sha256"],
                "aggregation": AGGREGATION,
                "selector": SELECTOR,
            })
        output_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows)
        output_sha = atomic_write_text(result_path, output_text)
        manifest = {
            "status": "COMPLETE",
            "schema_version": OUTPUT_SCHEMA,
            "mode": mode,
            "q_doc_count": len(rows),
            "inference_unit_count": total_units,
            "worklist_sha256": full_worklist_sha,
            "selected_worklist_sha256": selected_worklist_sha,
            "contract_sha256": identity["contract_sha256"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_generation": store.durable_generation,
            "output": str(result_path),
            "output_sha256": output_sha,
            "models": models,
            "actual_gpu": actual_gpu,
            "batch_size": batch_size,
            "checkpoint_every": checkpoint_every,
            "model_download_seconds": 0.0,
            "worklist_load_seconds": worklist_load_seconds,
            "model_contract_seconds": model_contract_seconds,
            "checkpoint_open_seconds": checkpoint_open_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "inference_batches": inference_batches,
            "inference_forward_calls_expected": inference_forward_calls,
            "actual_forward_batch_size": batch_size,
            "inference_mode": True,
            "document_cache_count": len(document_cache),
            "checkpoint_seconds": checkpoint_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "gpu_run": True,
        }
        atomic_json(manifest_path, manifest)
        volume.commit()
        return manifest
    finally:
        store.close()


@app.function(
    image=image,
    volumes={str(MOUNT): volume, str(MODEL_MOUNT): model_volume},
    timeout=10 * 60,
    cpu=2,
    memory=4096,
    retries=0,
)
def remote_preflight(
    worklist_payload: bytes,
    limit: int | None,
    questions_payload: bytes | None = None,
    chunks_root: str = "",
) -> dict[str, Any]:
    """Check the real Modal container without GPU, model loading, or inference."""
    if limit != CANARY_QDOCS:
        raise RuntimeError(f"preflight requires --limit {CANARY_QDOCS}")

    volume.reload()
    model_volume.reload()
    resolved_chunks_root = Path(chunks_root) if chunks_root else CHUNKS_ROOT
    rows, full_worklist_sha, selected_worklist_sha = load_worklist(worklist_payload, limit)
    if len(rows) != CANARY_QDOCS:
        raise RuntimeError(f"preflight q-doc count mismatch: {len(rows)}/{CANARY_QDOCS}")

    unique_documents = sorted({str(row["document_id"]) for row in rows})
    missing_documents = [
        doc_id for doc_id in unique_documents if not (resolved_chunks_root / f"{doc_id}.jsonl").is_file()
    ]
    if missing_documents:
        raise RuntimeError(f"PREFLIGHT_MISSING_DOCUMENTS:{len(missing_documents)}")

    questions = load_questions({str(row["query_id"]) for row in rows}, questions_payload)
    resolved, total_units = resolve_rows(rows, questions, chunks_root=resolved_chunks_root)
    if len(resolved) != CANARY_QDOCS:
        raise RuntimeError(f"PREFLIGHT_RESOLUTION_MISMATCH:{len(resolved)}/{CANARY_QDOCS}")

    models = validate_model_preflight_contracts()
    return {
        "status": "REMOTE_PREFLIGHT_PASS",
        "remote_source_path": str(_SOURCE_PATH),
        "local_root_mode": "MODAL_SENTINEL" if LOCAL_ROOT == LOCAL_ROOT_SENTINEL else "LOCAL_REPO",
        "train_exists": TRAIN_PATH.is_file(),
        "chunks_root_exists": resolved_chunks_root.is_dir(),
        "canary_qdocs_checked": len(rows),
        "resolved_qdocs": len(resolved),
        "unique_documents": len(unique_documents),
        "missing_canary_documents": len(missing_documents),
        "inference_units_resolved": total_units,
        "worklist_sha256": full_worklist_sha,
        "selected_worklist_sha256": selected_worklist_sha,
        "base_path_exists": BASE_MODEL_PATH.is_dir(),
        "base_revision": models["base_revision"],
        "base_revision_match": models["base_revision_match"],
        "ft_path_exists": FT_MODEL_PATH.is_dir(),
        "ft_config_sha256": models["ft_config_sha256"],
        "ft_config_match": models["ft_config_match"],
        "gpu_requested": "NO",
        "models_loaded": "NO",
        "inference_pairs": 0,
        "volume_committed": "NO",
    }


@app.function(image=image, gpu="L4", volumes={str(MOUNT): volume, str(MODEL_MOUNT): model_volume}, timeout=6 * 60 * 60, cpu=8, memory=32768, retries=0)
def run_subset(
    worklist_payload: bytes,
    mode: str,
    limit: int | None,
    batch_size: int,
    checkpoint_every: int,
    run_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
    questions_payload: bytes | None = None,
    chunks_root: str = "",
) -> dict[str, Any]:
    return score_subset(
        worklist_payload,
        mode,
        limit,
        batch_size,
        checkpoint_every,
        run_namespace,
        questions_payload,
        chunks_root,
    )


def production_call_is_active() -> bool:
    """Fail closed if this Modal function already has any active call."""
    stats = run_subset.get_current_stats()
    running_inputs = int(getattr(stats, "num_running_inputs", 0) or 0)
    backlog = int(getattr(stats, "backlog", 0) or 0)
    if running_inputs or backlog:
        print("PRODUCTION_CALL_ALREADY_ACTIVE", flush=True)
        print(f"backlog={backlog}", flush=True)
        print(f"running_inputs={running_inputs}", flush=True)
        return True
    return False


def spawn_production_call(
    payload: bytes,
    batch_size: int,
    checkpoint_every: int,
    run_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
    questions_payload: bytes | None = None,
    chunks_root: str = "",
) -> Any | None:
    """Spawn exactly one detached-capable production call, never wait for it."""
    if production_call_is_active():
        return None
    args = (
        payload,
        "production",
        None,
        batch_size,
        checkpoint_every,
        run_namespace,
        questions_payload,
        chunks_root,
    )
    return run_subset.spawn(*args)


def reattach_call(call_id: str) -> Any:
    """Reattach to an existing FunctionCall without creating a new call."""
    if not call_id.startswith("fc-"):
        raise ValueError("call_id must be an existing fc-* ID")
    return modal.FunctionCall.from_id(call_id).get()


@app.local_entrypoint()
def main(
    worklist: str = str(DEFAULT_WORKLIST),
    questions_file: str = "",
    mode: str = "canary",
    limit: int | None = None,
    batch_size: int = 16,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    call_id: str = "",
    run_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
    chunks_root: str = "",
) -> None:
    if mode in {"status", "reattach"}:
        result = reattach_call(call_id)
        if isinstance(result, (dict, list)):
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        else:
            print(result, flush=True)
        return

    path = Path(worklist)
    if not path.is_file():
        raise SystemExit(f"worklist not found: {path}")
    payload = gzip.compress(path.read_bytes(), compresslevel=9)
    questions_payload: bytes | None = None
    if questions_file:
        questions_path = Path(questions_file)
        if not questions_path.is_file():
            raise SystemExit(f"questions file not found: {questions_path}")
        questions_payload = gzip.compress(questions_path.read_bytes(), compresslevel=9)
    if mode == "preflight":
        result = remote_preflight.remote(payload, limit, questions_payload, chunks_root)
    elif mode == "production":
        if limit is not None:
            raise SystemExit("production must not use --limit")
        call = spawn_production_call(
            payload, batch_size, checkpoint_every, run_namespace, questions_payload, chunks_root
        )
        if call is None:
            return
        app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
        print("PRODUCTION_CALL_SPAWNED", flush=True)
        print(f"app_id={app_id}", flush=True)
        print(f"call_id={call.object_id}", flush=True)
        return
    elif mode == "canary":
        result = run_subset.remote(
            payload,
            mode,
            limit,
            batch_size,
            checkpoint_every,
            run_namespace,
            questions_payload,
            chunks_root,
        )
    else:
        raise SystemExit("mode must be preflight, canary, production, status, or reattach")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
