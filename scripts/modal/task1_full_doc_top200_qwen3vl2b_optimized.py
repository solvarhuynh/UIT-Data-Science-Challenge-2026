"""Optimized FULLDOC TOP200 Qwen runner.

Immutable-input loading/preparation and inference batching are optimized.  The
scientific model, prompt, tokenizer, selector, aggregation, and score formula
are delegated to the existing historical scorer unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import modal

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import task1_b2a_qwen3vl2b as historical
from task1_full_doc_top200_durability import DurableShardStore, atomic_jsonl, sha256_file
from task1_full_doc_top200_qwen_preprocessing import OptimizedPreprocessor, frozen_selected_chunks


APP_NAME = "task1-full-doc-top200-qwen3vl2b-optimized"
VOLUME_NAME = "udsc-p13"
MOUNT = Path("/workspace/p13")
RUNTIME = MOUNT / "runtime"
OUTPUT_ROOT = RUNTIME / "full_doc_top200_qwen_optimized"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
SHARD_ROOT = OUTPUT_ROOT / "shards"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
PROBE_OUTPUT_ROOT = RUNTIME / "full_doc_top200_qwen_optimized_probe"
PRESELECTED_ROOT = OUTPUT_ROOT / "preselected"
PROBE_QDOC_LIMIT = 256
PROBE_UNIQUE_DOCS = 252
PROBE_EXPECTED_UNITS = 768
PRODUCTION_INFERENCE_BATCH_SIZE = 1
FULL_SHARD_QDOCS = 18693
FULL_SHARD_UNITS = 56079
FULL_SHARD_UNIQUE_DOCS = 5308
PROBE_SCORER_RATE_UNITS_PER_SECOND = 38.2346115061
PROBE_SELECTOR_SECONDS = 1.001659973
PROBE_SELECTOR_QDOCS = 256
PROBE_DURABILITY_FLUSH_SECONDS = 2.894354534
PROBE_DURABILITY_UNITS = 768
PROBE_MODEL_STARTUP_SECONDS = 23.44370736 + 12.125527293
REMOTE_WORKLIST_ROOT = Path("/opt/full_doc_top200_qwen_worklists")
REMOTE_UNIVERSE_MANIFEST = Path("/opt/full_doc_top200_qwen_future_universe_manifest.json")
REMOTE_CURRENT_SAMPLE = Path("/opt/full_doc_top200_qwen_current_canary_16.jsonl")

MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODEL_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
HISTORICAL_SCORER_SHA256 = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
PROMPT_SCORING_SHA256 = "60feb6aca921a1f6e87f08dbebb0cef202e49d09daa4e727592b756de042f55a"
UNIVERSE_SHA256 = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"
CONTRACT_SHA256 = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
AUTHORITATIVE_MANIFEST_SHA256 = "1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722"
OLD_CURRENT_CANARY_OUTPUT_SHA256 = "fa2dca9d9753701300381955fa972b09cfcb0ff8a17dff57cc8b0e4a401b21cc"
RESULT_BUFFER_QDOCS = 256
RESULT_BUFFER_CHUNK_ROWS = 1024
PRESELECTED_DURABILITY_QDOCS = 4096
PRESELECTED_DURABILITY_CHUNK_ROWS = 12288
SUPPORTED_INFERENCE_BATCHES = (1, 4, 8, 16)
SUPPORTED_DURABILITY_QDOCS = (256, 512, PRESELECTED_DURABILITY_QDOCS)
SUPPORTED_CURRENT_CANARY_KINDS = ("optimized_current_canary_b1", "optimized_current_canary_b8")
PRESELECTED_SCHEMA_VERSION = "full_doc_qwen_preselected_shard_v1"
PRESELECTED_SELECTOR_SEMANTICS = (
    "select_true_s2_prepared(topk=3); TOP3_UP_TO_AVAILABLE; "
    "BM25 descending, chunk_id ascending"
)

_SOURCE_PATH = Path(__file__).resolve()
LOCAL_ROOT = _SOURCE_PATH.parents[2] if len(_SOURCE_PATH.parents) >= 3 else Path("/__modal_local_source_unavailable__")
LOCAL_REPORT_ROOT = LOCAL_ROOT / "reports/task1/full_document_legal_field_retrieval"
LOCAL_WORKLIST_ROOT = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_worklists"
LOCAL_UNIVERSE_MANIFEST = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_future_universe_manifest.json"
LOCAL_CURRENT_SAMPLE = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl"

if LOCAL_WORKLIST_ROOT.is_dir() and LOCAL_UNIVERSE_MANIFEST.is_file():
    image = (
        historical.image
        .pip_install("pyvi==0.1.1")
        .add_local_dir(str(LOCAL_WORKLIST_ROOT), remote_path="/opt/full_doc_top200_qwen_worklists")
        .add_local_file(str(LOCAL_UNIVERSE_MANIFEST), remote_path="/opt/full_doc_top200_qwen_future_universe_manifest.json")
        .add_local_file(str(LOCAL_CURRENT_SAMPLE), remote_path="/opt/full_doc_top200_qwen_current_canary_16.jsonl")
        .add_local_file(str(LOCAL_ROOT / "scripts/modal/task1_b2a_qwen3vl2b.py"), remote_path="/root/task1_b2a_qwen3vl2b.py")
        .add_local_file(str(LOCAL_ROOT / "scripts/modal/task1_full_doc_top200_durability.py"), remote_path="/root/task1_full_doc_top200_durability.py")
        .add_local_file(str(LOCAL_ROOT / "scripts/modal/task1_full_doc_top200_qwen_preprocessing.py"), remote_path="/root/task1_full_doc_top200_qwen_preprocessing.py")
        .add_local_file(str(LOCAL_ROOT / "scripts/beam/task1_v2/evidence.py"), remote_path="/root/scripts/beam/task1_v2/evidence.py")
    )
else:
    image = historical.image.pip_install("pyvi==0.1.1")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def fsync_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def frozen_universe() -> dict[str, Any]:
    value = json.loads(REMOTE_UNIVERSE_MANIFEST.read_text(encoding="utf-8"))
    if value.get("schema_version") != "full_doc_top200_qwen_future_universe_v1":
        raise RuntimeError("future universe schema mismatch")
    if int(value.get("q_doc_count", -1)) != 598192 or int(value.get("inference_unit_count", -1)) != 1794571:
        raise RuntimeError("future universe cardinality mismatch")
    if str(value.get("universe_sha256")) != UNIVERSE_SHA256:
        raise RuntimeError("future universe SHA mismatch")
    return value


def validate_production_shard(shard_id: str, universe: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(shard_id, str) or not re.fullmatch(r"(?:0[0-9]|[12][0-9]|3[01])", shard_id):
        raise ValueError("shard_id must be a zero-padded value from 00 through 31")
    if sha256_file(REMOTE_UNIVERSE_MANIFEST) != AUTHORITATIVE_MANIFEST_SHA256:
        raise RuntimeError("authoritative production manifest SHA mismatch")
    entry = next((item for item in universe.get("shards", []) if int(item["shard_id"]) == int(shard_id)), None)
    if entry is None:
        raise RuntimeError(f"production shard missing from manifest: {shard_id}")
    source_path = REMOTE_WORKLIST_ROOT / f"shard_{shard_id}.jsonl"
    worklist_sha = sha256_file(source_path)
    if worklist_sha != str(entry["worklist_sha256"]):
        raise RuntimeError(f"production worklist SHA mismatch: {shard_id}")
    rows = list(jsonl(source_path))
    identities = [(str(row["query_id"]), str(row["document_id"])) for row in rows]
    if len(rows) != int(entry["q_doc_count"]) or len(identities) != len(set(identities)):
        raise RuntimeError(f"production shard q-doc coverage mismatch: {shard_id}")
    units = sum(int(row["expected_inference_units"]) for row in rows)
    if units != int(entry["inference_unit_count"]):
        raise RuntimeError(f"production shard unit coverage mismatch: {shard_id}")
    return rows, worklist_sha


def verify_scoring_contract() -> None:
    source = Path(historical.__file__).resolve()
    if sha256_file(source) != HISTORICAL_SCORER_SHA256:
        raise RuntimeError("historical scientific scorer SHA mismatch")
    if historical.MODEL_ID != MODEL_ID or historical.MODEL_REVISION != MODEL_REVISION:
        raise RuntimeError("model contract regression")
    if historical.MAX_LENGTH != 8192 or historical.INFERENCE_BATCH_SIZE != 1:
        raise RuntimeError("sequence/batch contract regression")


def _timed_commit(store: DurableShardStore, pending: list[tuple[str, str, list[tuple[str, float]]]], timing: dict[str, float]) -> int:
    if not pending:
        return store.durable_generation
    volume_seconds = 0.0

    def commit_volume() -> None:
        nonlocal volume_seconds
        start = time.perf_counter()
        volume.commit()
        volume_seconds += time.perf_counter() - start

    start = time.perf_counter()
    generation = store.commit_qdocs(pending, commit_volume)
    timing["durability_flush_seconds"] += time.perf_counter() - start
    timing["volume_commit_seconds"] += volume_seconds
    return generation


def _print_progress(
    store: DurableShardStore,
    shard_id: str,
    expected_rows: int,
    expected_units: int,
    generation: int,
    started: float,
    prefix: str = "PROGRESS",
) -> None:
    completed_units = sum(int(row[0]) for row in store.db.execute("SELECT chunk_count FROM document_results"))
    print(
        f"{prefix} shard={shard_id} completed={len(store.completed_identities())}/{expected_rows} "
        f"units={completed_units}/{expected_units} generation={generation} elapsed={time.perf_counter() - started:.1f}",
        flush=True,
    )


def _validate_runtime_options(inference_batch_size: int, durability_qdocs: int) -> tuple[int, int, int]:
    inference_batch_size = int(inference_batch_size)
    durability_qdocs = int(durability_qdocs)
    if inference_batch_size not in SUPPORTED_INFERENCE_BATCHES:
        raise ValueError(f"inference_batch_size must be one of {SUPPORTED_INFERENCE_BATCHES}")
    if durability_qdocs not in SUPPORTED_DURABILITY_QDOCS:
        raise ValueError(f"durability_qdocs must be one of {SUPPORTED_DURABILITY_QDOCS}")
    # Keep the existing bounds for the forensic/canary paths.  The
    # preselected production path is explicitly paired with 4,096 q-docs and
    # 12,288 chunks (TOP3_UP_TO_AVAILABLE upper bound).
    if durability_qdocs == PRESELECTED_DURABILITY_QDOCS:
        chunk_limit = PRESELECTED_DURABILITY_CHUNK_ROWS
    elif durability_qdocs == 512:
        chunk_limit = 1536
    else:
        chunk_limit = RESULT_BUFFER_CHUNK_ROWS
    return inference_batch_size, durability_qdocs, chunk_limit


def _iter_selected_qdocs(
    rows: list[dict[str, Any]],
    preprocessor: OptimizedPreprocessor,
    completed: set[tuple[str, str]],
) -> Iterable[tuple[str, str, str, list[dict[str, Any]]]]:
    """Yield frozen selection results in canonical q-doc order."""
    for row in rows:
        query, doc = str(row["query_id"]), str(row["document_id"])
        if (query, doc) in completed:
            continue
        if "selected_chunk_ids" in row:
            question = preprocessor.questions[query]
            chunks = frozen_selected_chunks(preprocessor.documents.chunks_root, query, doc, list(row["selected_chunk_ids"]))
        else:
            question, chunks = preprocessor.select(row)
        yield query, doc, question, chunks


def _preselected_paths(shard_id: str) -> tuple[Path, Path]:
    return (
        PRESELECTED_ROOT / f"shard_{shard_id}.jsonl",
        PRESELECTED_ROOT / f"shard_{shard_id}.manifest.json",
    )


def _validate_preselected_rows(
    worklist_rows: list[dict[str, Any]],
    preselected_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], int], int]:
    """Validate a frozen selector artifact against its exact worklist order."""
    if len(preselected_rows) != len(worklist_rows):
        raise RuntimeError(
            f"preselected q-doc count mismatch: {len(preselected_rows)}/{len(worklist_rows)}"
        )
    expected: dict[tuple[str, str], int] = {}
    expected_identities: list[tuple[str, str]] = []
    for source in worklist_rows:
        identity = (str(source["query_id"]), str(source["document_id"]))
        if identity in expected:
            raise RuntimeError(f"duplicate worklist identity: {identity[0]}/{identity[1]}")
        expected[identity] = int(source["expected_inference_units"])
        expected_identities.append(identity)

    actual_identities: list[tuple[str, str]] = []
    total_units = 0
    for row in preselected_rows:
        query = str(row.get("query_id", ""))
        doc = str(row.get("document_id", ""))
        identity = (query, doc)
        actual_identities.append(identity)
        if identity not in expected:
            raise RuntimeError(f"unexpected preselected identity: {query}/{doc}")
        if not str(row.get("question", "")).strip():
            raise RuntimeError(f"empty preselected question: {query}/{doc}")
        chunks = row.get("selected_chunks")
        if not isinstance(chunks, list):
            raise RuntimeError(f"selected_chunks is not a list: {query}/{doc}")
        expected_units = expected[identity]
        if not (1 <= len(chunks) <= 3) or len(chunks) != expected_units:
            raise RuntimeError(f"preselected unit-count mismatch: {query}/{doc}")
        chunk_ids: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise RuntimeError(f"invalid preselected chunk record: {query}/{doc}")
            chunk_id = str(chunk.get("chunk_id", ""))
            raw_text = str(chunk.get("raw_chunk_text", ""))
            chunk_ids.append(chunk_id)
            if not chunk_id.startswith(doc + "_") or not raw_text.strip():
                raise RuntimeError(f"invalid preselected chunk payload: {query}/{doc}/{chunk_id}")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError(f"duplicate preselected chunk IDs: {query}/{doc}")
        total_units += len(chunks)

    if actual_identities != expected_identities:
        raise RuntimeError("preselected q-doc identity/order mismatch")
    if len(actual_identities) != len(set(actual_identities)):
        raise RuntimeError("duplicate preselected q-doc identities")
    expected_units_total = sum(expected.values())
    if total_units != expected_units_total:
        raise RuntimeError(f"preselected total-unit mismatch: {total_units}/{expected_units_total}")
    return expected, total_units


def _read_preselected_artifact(
    shard_id: str,
    worklist_rows: list[dict[str, Any]],
    worklist_sha: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int], dict[str, Any], float]:
    artifact_path, manifest_path = _preselected_paths(shard_id)
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"preselected artifact/manifest missing for shard {shard_id}")
    read_started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": PRESELECTED_SCHEMA_VERSION,
        "shard_id": shard_id,
        "worklist_sha256": worklist_sha,
        "universe_sha256": UNIVERSE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "q_doc_count": len(worklist_rows),
        "inference_unit_count": sum(int(row["expected_inference_units"]) for row in worklist_rows),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selector_semantics": PRESELECTED_SELECTOR_SEMANTICS,
        "status": "COMPLETE",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"preselected manifest mismatch: {key}")
    if manifest.get("preselected_jsonl_sha256") != sha256_file(artifact_path):
        raise RuntimeError("preselected artifact SHA mismatch")
    preselected_rows = list(jsonl(artifact_path))
    expected, _ = _validate_preselected_rows(worklist_rows, preselected_rows)
    return preselected_rows, expected, manifest, time.perf_counter() - read_started


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=30 * 60, cpu=8, memory=32768, retries=0)
def prepare_optimized_production_shard_selection(shard_id: str) -> dict[str, Any]:
    """Materialize exact selector inputs on CPU before any GPU allocation."""
    if not isinstance(shard_id, str) or not re.fullmatch(r"(?:0[0-9]|[12][0-9]|3[01])", shard_id):
        raise ValueError("shard_id must be a zero-padded value from 00 through 31")
    if int(shard_id) < 9:
        raise ValueError("production preselection is reserved for shards 09 through 31")
    volume.reload()
    universe = frozen_universe()
    worklist_rows, worklist_sha = validate_production_shard(shard_id, universe)
    artifact_path, manifest_path = _preselected_paths(shard_id)
    if artifact_path.exists() != manifest_path.exists():
        raise RuntimeError(f"partial preselected artifact state for shard {shard_id}; refusing overwrite")
    if artifact_path.is_file() and manifest_path.is_file():
        preselected_rows, expected, manifest, read_seconds = _read_preselected_artifact(
            shard_id, worklist_rows, worklist_sha
        )
        return {
            "status": "COMPLETE_EXISTING",
            "shard_id": shard_id,
            "preselected_input": True,
            "preselected": str(artifact_path),
            "manifest": str(manifest_path),
            "q_doc_count": len(preselected_rows),
            "inference_unit_count": sum(expected.values()),
            "worklist_sha256": worklist_sha,
            "universe_sha256": UNIVERSE_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "artifact_read_seconds": read_seconds,
            "volume_commit_count": 0,
            "gpu_run": False,
            "model_loaded": False,
            "manifest_status": manifest["status"],
        }

    query_ids = {str(row["query_id"]) for row in worklist_rows}
    query_started = time.perf_counter()
    questions = historical.load_questions(query_ids)
    query_seconds = time.perf_counter() - query_started
    preprocessor = OptimizedPreprocessor(questions, historical.CHUNKS)
    unique_docs = {str(row["document_id"]) for row in worklist_rows}
    preload_started = time.perf_counter()
    preprocessor.documents.preload(unique_docs, workers=8)
    preload_seconds = time.perf_counter() - preload_started

    preselected_rows: list[dict[str, Any]] = []
    selection_started = time.perf_counter()
    for source in worklist_rows:
        query = str(source["query_id"])
        doc = str(source["document_id"])
        question, selected = preprocessor.select(source)
        preselected_rows.append({
            "query_id": query,
            "document_id": doc,
            "question": question,
            "expected_inference_units": int(source["expected_inference_units"]),
            "selected_chunks": [
                {"chunk_id": str(chunk["chunk_id"]), "raw_chunk_text": str(chunk["raw_chunk_text"])}
                for chunk in selected
            ],
        })
    selection_seconds = time.perf_counter() - selection_started
    expected, total_units = _validate_preselected_rows(worklist_rows, preselected_rows)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_sha = atomic_jsonl(artifact_path, preselected_rows)
    manifest = {
        "schema_version": PRESELECTED_SCHEMA_VERSION,
        "shard_id": shard_id,
        "worklist_sha256": worklist_sha,
        "universe_sha256": UNIVERSE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "q_doc_count": len(preselected_rows),
        "inference_unit_count": total_units,
        "preselected_jsonl_sha256": artifact_sha,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selector_semantics": PRESELECTED_SELECTOR_SEMANTICS,
        "status": "COMPLETE",
    }
    fsync_json(manifest_path, manifest)
    commit_started = time.perf_counter()
    volume.commit()
    commit_seconds = time.perf_counter() - commit_started
    return {
        "status": "COMPLETE",
        "shard_id": shard_id,
        "preselected_input": True,
        "preselected": str(artifact_path),
        "manifest": str(manifest_path),
        "q_doc_count": len(preselected_rows),
        "inference_unit_count": total_units,
        "worklist_sha256": worklist_sha,
        "universe_sha256": UNIVERSE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "document_files_loaded": preprocessor.documents.files_loaded,
        "prepared_documents": len(preprocessor.documents.prepared),
        "document_pipeline_seconds": preprocessor.documents.document_pipeline_seconds,
        "document_path_resolution_seconds": preprocessor.documents.document_path_resolution_seconds,
        "selector_seconds": preprocessor.selector_seconds,
        "timing": {
            "query_preload_seconds": query_seconds,
            "preload_wall_seconds": preload_seconds,
            "selection_wall_seconds": selection_seconds,
            "volume_commit_seconds": commit_seconds,
            "total_cpu_stage_seconds": time.perf_counter() - query_started,
        },
        "document_cache_hits": preprocessor.documents.cache_hits,
        "document_cache_misses": preprocessor.documents.cache_misses,
        "worker_load_seconds_sum": preprocessor.documents.worker_load_seconds_sum,
        "worker_prepare_seconds_sum": preprocessor.documents.worker_prepare_seconds_sum,
        "volume_commit_count": 1,
        "gpu_run": False,
        "model_loaded": False,
    }


def score_optimized(
    kind: str,
    shard_id: str,
    rows: list[dict[str, Any]],
    complete: bool,
    inference_batch_size: int = 8,
    durability_qdocs: int = RESULT_BUFFER_QDOCS,
    output_root: Path = OUTPUT_ROOT,
    progress_prefix: str = "PROGRESS",
    expected_gpu: str | None = None,
    preparation_workers: int = 8,
) -> dict[str, Any]:
    if kind == "optimized_production" and int(inference_batch_size) != PRODUCTION_INFERENCE_BATCH_SIZE:
        raise ValueError(
            "optimized production is hard-locked to inference_batch_size=1; "
            "batch4/8/16 are canary-only"
        )
    inference_batch_size, durability_qdocs, durability_chunk_rows = _validate_runtime_options(
        inference_batch_size, durability_qdocs
    )
    wall_started = time.perf_counter()
    checkpoint_root = output_root / "checkpoints"
    shard_root = output_root / "shards"
    manifest_root = output_root / "manifests"
    volume.reload()
    verify_scoring_contract()
    universe = frozen_universe()
    source_path = REMOTE_CURRENT_SAMPLE if kind.startswith("optimized_current_canary") else REMOTE_WORKLIST_ROOT / f"shard_{int(shard_id):02d}.jsonl"
    worklist_sha = sha256_file(source_path)
    store_path = checkpoint_root / kind / f"shard_{shard_id}.sqlite3"
    store_identity = {
        "contract_sha256": CONTRACT_SHA256,
        "universe_sha256": UNIVERSE_SHA256,
        "shard_sha256": worklist_sha,
        "shard_id": shard_id,
        "run_kind": kind,
    }
    if kind == "optimized_production":
        store_identity["input_mode"] = "live_preprocessing_v1"
    store = DurableShardStore(store_path, store_identity)
    completed_before = store.completed_identities()
    timing: dict[str, float] = {
        "query_preload_seconds": 0.0,
        "model_download_seconds": 0.0,
        "model_load_seconds": 0.0,
        "selector_seconds": 0.0,
        "document_index_seconds": 0.0,
        "document_preparation_seconds": 0.0,
        "scorer_total_seconds": 0.0,
        "durability_flush_seconds": 0.0,
        "volume_commit_seconds": 0.0,
        "inference_batches": 0,
        "inference_units": 0,
    }
    query_start = time.perf_counter()
    questions = historical.load_questions({str(row["query_id"]) for row in rows})
    timing["query_preload_seconds"] = time.perf_counter() - query_start
    preprocessor = OptimizedPreprocessor(questions, historical.CHUNKS)
    pending_doc_ids = sorted({
        str(row["document_id"])
        for row in rows
        if (str(row["query_id"]), str(row["document_id"])) not in completed_before
        and "selected_chunk_ids" not in row
    })
    preprocessor.documents.preload(pending_doc_ids, workers=preparation_workers)
    download_start = time.perf_counter()
    model_path, resolved_revision, provenance, yes, no = historical.download_locked_snapshot()
    timing["model_download_seconds"] = time.perf_counter() - download_start
    load_start = time.perf_counter()
    import torch
    import transformers
    processor, model, _, gpu_name, _, _ = historical.load_model(torch, transformers, model_path, yes, no)
    timing["model_load_seconds"] = time.perf_counter() - load_start
    if expected_gpu is not None and str(gpu_name) != expected_gpu:
        raise RuntimeError(f"actual GPU mismatch: {gpu_name!r} != {expected_gpu!r}")
    buffer: list[tuple[str, str, list[tuple[str, float]]]] = []
    buffer_chunks = 0
    scored_now = 0
    started = time.perf_counter()
    expected: dict[tuple[str, str], int] = {
        (str(row["query_id"]), str(row["document_id"])): int(row["expected_inference_units"])
        for row in rows
    }
    expected_units_total = sum(expected.values())

    # Each queue entry is one independent (query, selected-chunk) unit.  A
    # state object keeps q-doc aggregation and exact chunk order while units
    # from adjacent q-docs share one model forward call.
    unit_queue: list[tuple[dict[str, Any], str, str, str]] = []

    def flush_units(force: bool = False) -> None:
        nonlocal buffer, buffer_chunks, scored_now, unit_queue
        while unit_queue and (force or len(unit_queue) >= inference_batch_size):
            take = inference_batch_size if len(unit_queue) >= inference_batch_size else len(unit_queue)
            batch = unit_queue[:take]
            del unit_queue[:take]
            score_start = time.perf_counter()
            scores, _, _, _ = historical.score_batch(
                processor,
                model,
                torch,
                [item[3] for item in batch],
                [item[2] for item in batch],
            )
            timing["scorer_total_seconds"] += time.perf_counter() - score_start
            timing["inference_batches"] += 1
            timing["inference_units"] += len(batch)
            if len(scores) != len(batch):
                raise RuntimeError("batched scorer output length mismatch")
            for item, score in zip(batch, scores):
                state, chunk_id, _, _ = item
                value = float(score)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite scorer output: {state['query']}/{state['doc']}")
                state["values"].append((chunk_id, value))
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    buffer.append((state["query"], state["doc"], state["values"]))
                    buffer_chunks += len(state["values"])
                    scored_now += 1
                    if len(buffer) >= durability_qdocs or buffer_chunks >= durability_chunk_rows:
                        generation = _timed_commit(store, buffer, timing)
                        buffer = []
                        buffer_chunks = 0
                        _print_progress(store, shard_id, len(rows), expected_units_total, generation, started, progress_prefix)

    for query, doc, question, chunks in _iter_selected_qdocs(rows, preprocessor, completed_before):
        state: dict[str, Any] = {"query": query, "doc": doc, "remaining": len(chunks), "values": []}
        for chunk in chunks:
            unit_queue.append((state, str(chunk["chunk_id"]), str(chunk["raw_chunk_text"]), question))
        flush_units()
    flush_units(force=True)
    if unit_queue:
        raise RuntimeError("batched scorer queue did not drain")
    if buffer:
        generation = _timed_commit(store, buffer, timing)
        _print_progress(store, shard_id, len(rows), expected_units_total, generation, started, progress_prefix)
    timing["selector_seconds"] = preprocessor.selector_seconds
    timing["document_index_seconds"] = preprocessor.documents.document_index_seconds
    timing["document_preparation_seconds"] = preprocessor.documents.document_preparation_seconds
    timing["post_model_load_elapsed_seconds"] = time.perf_counter() - started
    timing["post_model_load_units_per_second"] = (
        timing["inference_units"] / timing["post_model_load_elapsed_seconds"]
        if timing["post_model_load_elapsed_seconds"] > 0
        else None
    )
    timing["units_per_second_scorer_only"] = (
        timing["inference_units"] / timing["scorer_total_seconds"]
        if timing["scorer_total_seconds"] > 0
        else None
    )
    result: dict[str, Any] = {
        "kind": kind,
        "shard_id": shard_id,
        "expected": len(rows),
        "expected_units": expected_units_total,
        "completed_before": len(completed_before),
        "scored_now": scored_now,
        "completed_after": len(store.completed_identities()),
        "contract_sha256": CONTRACT_SHA256,
        "universe_sha256": UNIVERSE_SHA256,
        "worklist_sha256": worklist_sha,
        "checkpoint": str(store_path),
        "model_cache_path": str(model_path),
        "actual_gpu": str(gpu_name),
        "resolved_revision": resolved_revision,
        "inference_batch_size": inference_batch_size,
        "durability_qdocs": durability_qdocs,
        "durability_chunk_rows": durability_chunk_rows,
        "timing": timing,
        "document_files_loaded": preprocessor.documents.files_loaded,
        "prepared_documents": len(preprocessor.documents.prepared),
        "document_path_resolution_seconds": preprocessor.documents.document_path_resolution_seconds,
        "document_pipeline_seconds": preprocessor.documents.document_pipeline_seconds,
        "worker_load_seconds_sum": preprocessor.documents.worker_load_seconds_sum,
        "worker_prepare_seconds_sum": preprocessor.documents.worker_prepare_seconds_sum,
        "filesystem_scans": preprocessor.documents.filesystem_scans,
        "document_cache_hits": preprocessor.documents.cache_hits,
        "document_cache_misses": preprocessor.documents.cache_misses,
        "prepare_document_calls": preprocessor.documents.prepare_document_calls,
        "preparation_workers": int(preparation_workers),
        "tokenizer_processor_seconds": None,
        "model_forward_seconds": None,
        "status": "INCOMPLETE",
    }
    if complete:
        output_rows = store.validate_complete(expected)
        output_path = shard_root / kind / f"shard_{shard_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_sha = atomic_jsonl(output_path, output_rows)
        volume.commit()
        manifest = {
            "schema_version": "full_doc_qwen_shard_manifest_v1",
            "shard_id": shard_id,
            "universe_sha256": UNIVERSE_SHA256,
            "shard_worklist_sha256": worklist_sha,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "scorer_sha256": HISTORICAL_SCORER_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "q_doc_count": len(output_rows),
            "inference_unit_count": sum(expected.values()),
            "output_sha256": output_sha,
            "status": "COMPLETE",
            "timing": timing,
            "snapshot_provenance_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        }
        manifest_path = manifest_root / kind / f"shard_{shard_id}.json"
        fsync_json(manifest_path, manifest)
        volume.commit()
        result.update({"output": str(output_path), "output_sha256": output_sha, "manifest": str(manifest_path), "status": "COMPLETE"})
    timing["total_wall_seconds"] = time.perf_counter() - wall_started
    result["timing"] = timing
    store.close()
    return result


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60 * 60, cpu=8, memory=32768, retries=0)
def optimized_production_shard_preselected(
    shard_id: str,
    inference_batch_size: int = PRODUCTION_INFERENCE_BATCH_SIZE,
    durability_qdocs: int = PRESELECTED_DURABILITY_QDOCS,
) -> dict[str, Any]:
    """Score frozen CPU-produced inputs without reopening document sources."""
    if int(inference_batch_size) != PRODUCTION_INFERENCE_BATCH_SIZE:
        raise ValueError("preselected production is hard-locked to inference_batch_size=1")
    if not isinstance(shard_id, str) or not re.fullmatch(r"(?:0[0-9]|[12][0-9]|3[01])", shard_id):
        raise ValueError("shard_id must be a zero-padded value from 00 through 31")
    if int(shard_id) < 9:
        raise ValueError("preselected production path is reserved for shards 09 through 31")
    if int(durability_qdocs) != PRESELECTED_DURABILITY_QDOCS:
        raise ValueError("preselected production requires durability_qdocs=4096")

    wall_started = time.perf_counter()
    volume.reload()
    verify_scoring_contract()
    universe = frozen_universe()
    worklist_rows, worklist_sha = validate_production_shard(shard_id, universe)
    preselected_rows, expected, preselected_manifest, artifact_read_seconds = _read_preselected_artifact(
        shard_id, worklist_rows, worklist_sha
    )
    _, _, durability_chunk_rows = _validate_runtime_options(1, durability_qdocs)
    store_path = CHECKPOINT_ROOT / "optimized_production" / f"shard_{shard_id}.sqlite3"
    output_path = SHARD_ROOT / "optimized_production" / f"shard_{shard_id}.jsonl"
    manifest_path = MANIFEST_ROOT / "optimized_production" / f"shard_{shard_id}.json"
    if not store_path.exists() and (output_path.exists() or manifest_path.exists()):
        raise RuntimeError(
            f"orphan optimized production output for shard {shard_id}; refusing overwrite without checkpoint"
        )
    store = DurableShardStore(store_path, {
        "contract_sha256": CONTRACT_SHA256,
        "universe_sha256": UNIVERSE_SHA256,
        "shard_sha256": worklist_sha,
        "shard_id": shard_id,
        "run_kind": "optimized_production",
        "input_mode": "preselected_v1",
    })
    completed_before = store.completed_identities()
    timing: dict[str, float] = {
        "preselected_artifact_read_seconds": artifact_read_seconds,
        "model_download_seconds": 0.0,
        "model_load_seconds": 0.0,
        "selector_seconds": 0.0,
        "document_pipeline_seconds": 0.0,
        "scorer_total_seconds": 0.0,
        "durability_flush_seconds": 0.0,
        "volume_commit_seconds": 0.0,
        "inference_batches": 0,
        "inference_units": 0,
    }

    download_start = time.perf_counter()
    model_path, resolved_revision, provenance, yes, no = historical.download_locked_snapshot()
    timing["model_download_seconds"] = time.perf_counter() - download_start
    load_start = time.perf_counter()
    import torch
    import transformers
    processor, model, _, gpu_name, _, _ = historical.load_model(torch, transformers, model_path, yes, no)
    timing["model_load_seconds"] = time.perf_counter() - load_start
    if str(gpu_name) != "NVIDIA A10":
        raise RuntimeError(f"actual GPU mismatch: {gpu_name!r} != 'NVIDIA A10'")

    buffer: list[tuple[str, str, list[tuple[str, float]]]] = []
    buffer_chunks = 0
    scored_now = 0
    started = time.perf_counter()
    unit_queue: list[tuple[dict[str, Any], str, str, str]] = []

    def flush_units(force: bool = False) -> None:
        nonlocal buffer, buffer_chunks, scored_now, unit_queue
        while unit_queue and (force or len(unit_queue) >= inference_batch_size):
            take = inference_batch_size if len(unit_queue) >= inference_batch_size else len(unit_queue)
            batch = unit_queue[:take]
            del unit_queue[:take]
            score_start = time.perf_counter()
            scores, _, _, _ = historical.score_batch(
                processor,
                model,
                torch,
                [item[3] for item in batch],
                [item[2] for item in batch],
            )
            timing["scorer_total_seconds"] += time.perf_counter() - score_start
            timing["inference_batches"] += 1
            timing["inference_units"] += len(batch)
            if len(scores) != len(batch):
                raise RuntimeError("preselected scorer output length mismatch")
            for item, score in zip(batch, scores):
                state, chunk_id, _, _ = item
                value = float(score)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite scorer output: {state['query']}/{state['doc']}")
                state["values"].append((chunk_id, value))
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    buffer.append((state["query"], state["doc"], state["values"]))
                    buffer_chunks += len(state["values"])
                    scored_now += 1
                    if len(buffer) >= durability_qdocs or buffer_chunks >= durability_chunk_rows:
                        generation = _timed_commit(store, buffer, timing)
                        buffer = []
                        buffer_chunks = 0
                        _print_progress(
                            store,
                            shard_id,
                            len(worklist_rows),
                            sum(expected.values()),
                            generation,
                            started,
                            "PRESELECTED_PROGRESS",
                        )

    for frozen in preselected_rows:
        identity = (str(frozen["query_id"]), str(frozen["document_id"]))
        if identity in completed_before:
            continue
        state: dict[str, Any] = {
            "query": identity[0],
            "doc": identity[1],
            "remaining": len(frozen["selected_chunks"]),
            "values": [],
        }
        question = str(frozen["question"])
        for chunk in frozen["selected_chunks"]:
            unit_queue.append((state, str(chunk["chunk_id"]), str(chunk["raw_chunk_text"]), question))
        flush_units()
    flush_units(force=True)
    if unit_queue:
        raise RuntimeError("preselected scorer queue did not drain")
    if buffer:
        generation = _timed_commit(store, buffer, timing)
        _print_progress(
            store,
            shard_id,
            len(worklist_rows),
            sum(expected.values()),
            generation,
            started,
            "PRESELECTED_PROGRESS",
        )

    timing["post_model_load_elapsed_seconds"] = time.perf_counter() - started
    timing["units_per_second_scorer_only"] = (
        timing["inference_units"] / timing["scorer_total_seconds"]
        if timing["scorer_total_seconds"] > 0
        else None
    )
    output_rows = store.validate_complete(expected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_sha = atomic_jsonl(output_path, output_rows)
    output_commit_start = time.perf_counter()
    volume.commit()
    timing["volume_commit_seconds"] += time.perf_counter() - output_commit_start
    manifest = {
        "schema_version": "full_doc_qwen_shard_manifest_v1",
        "shard_id": shard_id,
        "universe_sha256": UNIVERSE_SHA256,
        "shard_worklist_sha256": worklist_sha,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "scorer_sha256": HISTORICAL_SCORER_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "q_doc_count": len(output_rows),
        "inference_unit_count": sum(expected.values()),
        "output_sha256": output_sha,
        "status": "COMPLETE",
        "preselected_input": True,
        "preselected_jsonl_sha256": preselected_manifest["preselected_jsonl_sha256"],
        "preselected_manifest_sha256": sha256_file(_preselected_paths(shard_id)[1]),
        "durability_qdocs": durability_qdocs,
        "durability_chunk_rows": durability_chunk_rows,
        "timing": timing,
        "snapshot_provenance_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
    }
    fsync_json(manifest_path, manifest)
    manifest_commit_start = time.perf_counter()
    volume.commit()
    timing["volume_commit_seconds"] += time.perf_counter() - manifest_commit_start
    timing["total_wall_seconds"] = time.perf_counter() - wall_started
    result = {
        "kind": "optimized_production",
        "shard_id": shard_id,
        "expected": len(worklist_rows),
        "expected_units": sum(expected.values()),
        "completed_before": len(completed_before),
        "scored_now": scored_now,
        "completed_after": len(store.completed_identities()),
        "contract_sha256": CONTRACT_SHA256,
        "universe_sha256": UNIVERSE_SHA256,
        "worklist_sha256": worklist_sha,
        "checkpoint": str(store_path),
        "model_cache_path": str(model_path),
        "actual_gpu": str(gpu_name),
        "resolved_revision": resolved_revision,
        "inference_batch_size": inference_batch_size,
        "durability_qdocs": durability_qdocs,
        "durability_chunk_rows": durability_chunk_rows,
        "preselected_input": True,
        "preselected_artifact_read_seconds": artifact_read_seconds,
        "document_pipeline_seconds": 0.0,
        "selector_seconds": 0.0,
        "model_download_seconds": timing["model_download_seconds"],
        "model_load_seconds": timing["model_load_seconds"],
        "scorer_total_seconds": timing["scorer_total_seconds"],
        "durability_flush_seconds": timing["durability_flush_seconds"],
        "volume_commit_seconds": timing["volume_commit_seconds"],
        "total_wall_seconds": timing["total_wall_seconds"],
        "timing": timing,
        "inference_units": timing["inference_units"],
        "units_per_second_scorer_only": timing["units_per_second_scorer_only"],
        "output": str(output_path),
        "output_sha256": output_sha,
        "manifest": str(manifest_path),
        "status": "COMPLETE",
    }
    store.close()
    return result


def _current_canary_expected() -> tuple[list[dict[str, Any]], dict[tuple[str, str], int], str]:
    rows = list(jsonl(REMOTE_CURRENT_SAMPLE))
    identities = [(str(row["query_id"]), str(row["document_id"])) for row in rows]
    if len(rows) != 16 or len(identities) != len(set(identities)):
        raise RuntimeError("current canary q-doc coverage mismatch")
    expected = {
        (str(row["query_id"]), str(row["document_id"])): int(row["expected_inference_units"])
        for row in rows
    }
    if sum(expected.values()) != 43:
        raise RuntimeError("current canary inference-unit coverage mismatch")
    return rows, expected, sha256_file(REMOTE_CURRENT_SAMPLE)


def _validate_current_canary_kind(kind: str) -> str:
    if kind not in SUPPORTED_CURRENT_CANARY_KINDS:
        raise ValueError(f"current canary kind must be one of {SUPPORTED_CURRENT_CANARY_KINDS}")
    return kind


def _rss_mb() -> float | None:
    """Return current Linux RSS without adding a runtime dependency."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    try:
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 if os.name == "posix" else (1024.0 * 1024.0))
    except (ImportError, OSError):
        return None


def _shard_runtime_estimate(document_seconds: float) -> dict[str, Any]:
    document = document_seconds / PROBE_UNIQUE_DOCS * FULL_SHARD_UNIQUE_DOCS
    scorer = FULL_SHARD_UNITS / PROBE_SCORER_RATE_UNITS_PER_SECOND
    selector = PROBE_SELECTOR_SECONDS / PROBE_SELECTOR_QDOCS * FULL_SHARD_QDOCS
    durability = PROBE_DURABILITY_FLUSH_SECONDS / PROBE_DURABILITY_UNITS * FULL_SHARD_UNITS
    central = PROBE_MODEL_STARTUP_SECONDS + document + scorer + selector + durability
    low = (
        PROBE_MODEL_STARTUP_SECONDS * 0.5
        + document * 0.75
        + scorer * 0.85
        + selector * 0.8
        + durability * 0.8
    )
    high = (
        60.0
        + document * 1.5
        + scorer * 1.15
        + selector * 1.2
        + durability * 1.2
    )
    return {
        "estimated_shard00_document_minutes": document / 60.0,
        "estimated_shard00_scorer_minutes": scorer / 60.0,
        "estimated_shard00_selector_minutes": selector / 60.0,
        "estimated_shard00_durability_minutes": durability / 60.0,
        "estimated_shard00_startup_minutes": PROBE_MODEL_STARTUP_SECONDS / 60.0,
        "estimated_shard00_central_minutes": central / 60.0,
        "estimated_shard00_low_high_minutes": [low / 60.0, high / 60.0],
        "timeout_margin_minutes": 60.0 - high / 60.0,
        "runtime_estimate_is_extrapolation": True,
    }


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=10 * 60, cpu=2, memory=4096, retries=0)
def optimized_current_canary_finalize(kind: str = "optimized_current_canary_b8") -> dict[str, Any]:
    """Finalize a complete canary checkpoint without loading Qwen or scoring."""
    volume.reload()
    frozen_universe()
    verify_scoring_contract()
    rows, expected, worklist_sha = _current_canary_expected()
    kind = _validate_current_canary_kind(kind)
    store_path = CHECKPOINT_ROOT / kind / "shard_00.sqlite3"
    store = DurableShardStore(store_path, {
        "contract_sha256": CONTRACT_SHA256,
        "universe_sha256": UNIVERSE_SHA256,
        "shard_sha256": worklist_sha,
        "shard_id": "00",
        "run_kind": kind,
    })
    try:
        completed = store.completed_identities()
        if len(completed) != len(expected) or store.durable_generation != 1:
            raise RuntimeError(
                f"checkpoint incomplete: generation={store.durable_generation} "
                f"qdocs={len(completed)}/{len(expected)}"
            )
        output_rows = store.validate_complete(expected)
        output_units = sum(int(row["chunk_count"]) for row in output_rows)
        if output_units != 43:
            raise RuntimeError(f"checkpoint unit coverage mismatch: {output_units}/43")
        output_path = SHARD_ROOT / kind / "shard_00.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_sha = atomic_jsonl(output_path, output_rows)
        volume.commit()
        manifest = {
            "schema_version": "full_doc_qwen_shard_manifest_v1",
            "shard_id": "00",
            "universe_sha256": UNIVERSE_SHA256,
            "shard_worklist_sha256": worklist_sha,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "scorer_sha256": HISTORICAL_SCORER_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "q_doc_count": len(output_rows),
            "inference_unit_count": output_units,
            "output_sha256": output_sha,
            "status": "COMPLETE",
            "finalized_from_checkpoint": True,
            "durable_generation": store.durable_generation,
            "snapshot_provenance_sha256": None,
        }
        manifest_path = MANIFEST_ROOT / kind / "shard_00.json"
        fsync_json(manifest_path, manifest)
        volume.commit()
        return {
            "status": "COMPLETE",
            "checkpoint": str(store_path),
            "checkpoint_generation": store.durable_generation,
            "checkpoint_qdocs": len(completed),
            "checkpoint_units": output_units,
            "output": str(output_path),
            "output_sha256": output_sha,
            "manifest": str(manifest_path),
            "worklist_sha256": worklist_sha,
            "contract_sha256": CONTRACT_SHA256,
            "model_loaded": False,
            "gpu_run": False,
        }
    finally:
        store.close()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return list(jsonl(path))


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=10 * 60, cpu=2, memory=4096, retries=0)
def optimized_current_canary_compare(kind: str = "optimized_current_canary_b8") -> dict[str, Any]:
    """CPU-only comparison of old and finalized canary artifacts."""
    volume.reload()
    kind = _validate_current_canary_kind(kind)
    old_output = RUNTIME / "full_doc_top200_qwen/shards/current_canary/shard_00.jsonl"
    new_output = SHARD_ROOT / kind / "shard_00.jsonl"
    old_db = RUNTIME / "full_doc_top200_qwen/checkpoints/current_canary/shard_00.sqlite3"
    new_db = CHECKPOINT_ROOT / kind / "shard_00.sqlite3"
    if not all(path.is_file() for path in (old_output, new_output, old_db, new_db)):
        raise RuntimeError("old/new canary comparison artifact missing")
    old_output_sha = sha256_file(old_output)
    if old_output_sha != OLD_CURRENT_CANARY_OUTPUT_SHA256:
        raise RuntimeError("old current-canary output SHA mismatch")

    old_rows = _read_jsonl_rows(old_output)
    new_rows = _read_jsonl_rows(new_output)
    old_ids = [(str(row["query_id"]), str(row["document_id"])) for row in old_rows]
    new_ids = [(str(row["query_id"]), str(row["document_id"])) for row in new_rows]
    old_doc = {(str(row["query_id"]), str(row["document_id"])): float(row["score"]) for row in old_rows}
    new_doc = {(str(row["query_id"]), str(row["document_id"])): float(row["score"]) for row in new_rows}

    def chunk_scores(path: Path) -> dict[tuple[str, str, str], float]:
        import sqlite3
        with sqlite3.connect(path) as db:
            return {
                (str(query), str(doc), str(chunk)): float(score)
                for query, doc, chunk, score in db.execute(
                    "SELECT query_id,document_id,chunk_id,score FROM chunk_scores ORDER BY query_id,document_id,chunk_id"
                )
            }

    old_chunk = chunk_scores(old_db)
    new_chunk = chunk_scores(new_db)
    old_chunk_ids, new_chunk_ids = set(old_chunk), set(new_chunk)
    shared_chunks = sorted(old_chunk_ids & new_chunk_ids)
    deltas = [abs(new_chunk[key] - old_chunk[key]) for key in shared_chunks]
    old_nonfinite = sum(not math.isfinite(value) for value in old_chunk.values())
    new_nonfinite = sum(not math.isfinite(value) for value in new_chunk.values())
    nonfinite = old_nonfinite + new_nonfinite
    shared_docs = sorted(set(old_doc) & set(new_doc))
    doc_deltas = [abs(new_doc[key] - old_doc[key]) for key in shared_docs]
    old_doc_nonfinite = sum(not math.isfinite(value) for value in old_doc.values())
    new_doc_nonfinite = sum(not math.isfinite(value) for value in new_doc.values())
    old_rank = {}
    new_rank = {}
    for query in sorted({key[0] for key in shared_docs}):
        old_rank[query] = [doc for _, doc in sorted(((old_doc[(query, doc)], doc) for q, doc in shared_docs if q == query), key=lambda item: (-item[0], item[1]))]
        new_rank[query] = [doc for _, doc in sorted(((new_doc[(query, doc)], doc) for q, doc in shared_docs if q == query), key=lambda item: (-item[0], item[1]))]
    ordering_mismatches = sum(old_rank[query] != new_rank[query] for query in old_rank)
    return {
        "old_output_sha256": old_output_sha,
        "new_output_sha256": sha256_file(new_output),
        "kind": kind,
        "old_qdocs": len(old_ids),
        "new_qdocs": len(new_ids),
        "qdoc_identity_mismatches": len(set(old_ids) ^ set(new_ids)),
        "missing_qdocs_in_new": len(set(old_ids) - set(new_ids)),
        "unexpected_qdocs_in_new": len(set(new_ids) - set(old_ids)),
        "old_qdoc_duplicates": len(old_ids) - len(set(old_ids)),
        "new_qdoc_duplicates": len(new_ids) - len(set(new_ids)),
        "chunk_scores_compared": len(shared_chunks),
        "chunk_identity_mismatches": len(old_chunk_ids ^ new_chunk_ids),
        "missing_chunks_in_new": len(old_chunk_ids - new_chunk_ids),
        "unexpected_chunks_in_new": len(new_chunk_ids - old_chunk_ids),
        "exact_chunk_score_matches": sum(delta == 0.0 for delta in deltas),
        "max_abs_chunk_score_diff": max(deltas) if deltas else None,
        "mean_abs_chunk_score_diff": sum(deltas) / len(deltas) if deltas else None,
        "document_score_mismatches": sum(delta != 0.0 for delta in doc_deltas),
        "max_abs_document_score_diff": max(doc_deltas) if doc_deltas else None,
        "ordering_mismatches": ordering_mismatches,
        "nonfinite_scores": nonfinite,
        "old_nonfinite_chunk_scores": old_nonfinite,
        "new_nonfinite_chunk_scores": new_nonfinite,
        "old_nonfinite_document_scores": old_doc_nonfinite,
        "new_nonfinite_document_scores": new_doc_nonfinite,
        "numeric_parity": "PASS" if (
            len(old_ids) == len(new_ids) == 16
            and len(set(old_ids) ^ set(new_ids)) == 0
            and len(old_chunk_ids) == len(new_chunk_ids) == 43
            and not (old_chunk_ids ^ new_chunk_ids)
            and not nonfinite
            and not deltas
            and not doc_deltas
            and ordering_mismatches == 0
        ) else "FAIL",
    }


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=10 * 60, cpu=8, memory=32768, retries=0)
def optimized_current_canary(inference_batch_size: int = 8, durability_qdocs: int = RESULT_BUFFER_QDOCS) -> dict[str, Any]:
    rows = list(jsonl(REMOTE_CURRENT_SAMPLE))
    kind = f"optimized_current_canary_b{int(inference_batch_size)}"
    return score_optimized(kind, "00", rows, complete=True, inference_batch_size=inference_batch_size, durability_qdocs=durability_qdocs)


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=10 * 60, cpu=8, memory=32768, retries=0)
def optimized_production_probe() -> dict[str, Any]:
    """Bounded batch-1 probe over an isolated canonical shard-00 prefix."""
    probe_started = time.perf_counter()
    volume.reload()
    verify_scoring_contract()
    universe = frozen_universe()
    rows, worklist_sha = validate_production_shard("00", universe)
    full_units = sum(int(row["expected_inference_units"]) for row in rows)
    if len(rows) < PROBE_QDOC_LIMIT:
        probe_rows = rows
        if not probe_rows:
            raise RuntimeError("production shard has no q-docs available for probe")
    else:
        probe_rows = rows[:PROBE_QDOC_LIMIT]
    probe_ids = [(str(row["query_id"]), str(row["document_id"])) for row in probe_rows]
    if len(probe_rows) > PROBE_QDOC_LIMIT or len(probe_ids) != len(set(probe_ids)):
        raise RuntimeError("bounded probe q-doc coverage mismatch")
    probe_units = sum(int(row["expected_inference_units"]) for row in probe_rows)
    if probe_units <= 0:
        raise RuntimeError("bounded probe has no inference units")
    print(
        f"PROBE_PLAN shard=00 qdocs={len(probe_rows)}/{PROBE_QDOC_LIMIT} "
        f"units={probe_units} worklist_sha256={worklist_sha}",
        flush=True,
    )
    result = score_optimized(
        "optimized_production_probe",
        "00",
        probe_rows,
        complete=True,
        inference_batch_size=1,
        durability_qdocs=RESULT_BUFFER_QDOCS,
        output_root=PROBE_OUTPUT_ROOT,
        progress_prefix="PROBE_PROGRESS",
        expected_gpu="NVIDIA A10",
    )
    result.update({
        "probe_qdocs": len(probe_rows),
        "probe_expected_units": probe_units,
        "probe_limit": PROBE_QDOC_LIMIT,
        "probe_namespace": str(PROBE_OUTPUT_ROOT),
        "probe_worklist_sha256": worklist_sha,
        "full_shard_qdocs": len(rows),
        "full_shard_expected_units": full_units,
        "probe_function_wall_seconds": time.perf_counter() - probe_started,
        "bounded_probe": True,
    })
    return result


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60 * 60, cpu=8, memory=32768, retries=0)
def optimized_production_shard(
    shard_id: str,
    inference_batch_size: int = PRODUCTION_INFERENCE_BATCH_SIZE,
    durability_qdocs: int = RESULT_BUFFER_QDOCS,
) -> dict[str, Any]:
    if int(inference_batch_size) != PRODUCTION_INFERENCE_BATCH_SIZE:
        raise ValueError(
            "optimized production is hard-locked to inference_batch_size=1; "
            "batch4/8/16 are canary-only"
        )
    volume.reload()
    universe = frozen_universe()
    rows, _ = validate_production_shard(shard_id, universe)
    return score_optimized(
        "optimized_production",
        shard_id,
        rows,
        complete=True,
        inference_batch_size=inference_batch_size,
        durability_qdocs=durability_qdocs,
    )


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=10 * 60, cpu=8, memory=32768, retries=0)
def optimized_document_pipeline_modal_cpu_probe() -> dict[str, Any]:
    """Read-only Modal CPU probe for the bounded production shard-00 prefix.

    This function intentionally has no GPU and never opens a checkpoint or
    output path.  It measures the actual Modal document preload/selection
    path that production batch-1 would use, then returns an extrapolation
    based on those measured timings.
    """
    probe_started = time.perf_counter()
    volume.reload()
    verify_scoring_contract()
    universe = frozen_universe()
    rows, worklist_sha = validate_production_shard("00", universe)
    probe_rows = rows[:PROBE_QDOC_LIMIT]
    probe_ids = [(str(row["query_id"]), str(row["document_id"])) for row in probe_rows]
    unique_docs = {doc for _, doc in probe_ids}
    expected_units = sum(int(row["expected_inference_units"]) for row in probe_rows)
    if len(probe_rows) != PROBE_QDOC_LIMIT:
        raise RuntimeError(f"CPU probe requires {PROBE_QDOC_LIMIT} q-docs, got {len(probe_rows)}")
    if len(probe_ids) != len(set(probe_ids)) or len(unique_docs) != PROBE_UNIQUE_DOCS:
        raise RuntimeError(
            f"CPU probe identity coverage mismatch: qdocs={len(probe_ids)} "
            f"unique_docs={len(unique_docs)}"
        )
    if expected_units != PROBE_EXPECTED_UNITS:
        raise RuntimeError(
            f"CPU probe expected-unit coverage mismatch: {expected_units}/{PROBE_EXPECTED_UNITS}"
        )
    print(
        f"CPU_PROBE_PLAN shard=00 qdocs={len(probe_rows)}/{PROBE_QDOC_LIMIT} "
        f"unique_docs={len(unique_docs)}/{PROBE_UNIQUE_DOCS} "
        f"units={expected_units}/{PROBE_EXPECTED_UNITS} "
        f"worklist_sha256={worklist_sha}",
        flush=True,
    )
    query_ids = {query for query, _ in probe_ids}
    query_started = time.perf_counter()
    questions = historical.load_questions(query_ids)
    query_seconds = time.perf_counter() - query_started
    preprocessor = OptimizedPreprocessor(questions, historical.CHUNKS)

    rss_before = _rss_mb()
    preload_started = time.perf_counter()
    preprocessor.documents.preload(unique_docs, workers=8)
    preload_seconds = time.perf_counter() - preload_started
    rss_after_preload = _rss_mb()
    if len(preprocessor.documents.prepared) != PROBE_UNIQUE_DOCS:
        raise RuntimeError(
            f"CPU probe prepared-document coverage mismatch: "
            f"{len(preprocessor.documents.prepared)}/{PROBE_UNIQUE_DOCS}"
        )

    selected_units = 0
    selected_qdocs = 0
    for row in probe_rows:
        query = str(row["query_id"])
        doc = str(row["document_id"])
        question, selected = preprocessor.select(row)
        expected = int(row["expected_inference_units"])
        ids = [str(item["chunk_id"]) for item in selected]
        if (
            len(selected) != expected
            or len(ids) != len(set(ids))
            or any(
                not str(item["chunk_id"]).startswith(doc + "_")
                or not str(item["raw_chunk_text"]).strip()
                for item in selected
            )
        ):
            raise RuntimeError(f"CPU probe selected-unit mismatch: {query}/{doc}")
        selected_units += len(selected)
        selected_qdocs += 1
    rss_after_selection = _rss_mb()
    if selected_qdocs != PROBE_QDOC_LIMIT or selected_units != PROBE_EXPECTED_UNITS:
        raise RuntimeError(
            f"CPU probe selection coverage mismatch: qdocs={selected_qdocs}/{PROBE_QDOC_LIMIT} "
            f"units={selected_units}/{PROBE_EXPECTED_UNITS}"
        )

    pickle_sizes = {
        str(doc): len(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
        for doc, prepared in sorted(preprocessor.documents.prepared.items())
    }
    if len(pickle_sizes) != PROBE_UNIQUE_DOCS:
        raise RuntimeError(
            f"CPU probe pickle coverage mismatch: {len(pickle_sizes)}/{PROBE_UNIQUE_DOCS}"
        )
    total_pickle_bytes = sum(pickle_sizes.values())
    result = {
        "status": "PASS",
        "probe_mode": "optimized_document_pipeline_modal_cpu_probe",
        "shard_id": "00",
        "probe_qdocs": len(probe_rows),
        "probe_unique_documents": len(unique_docs),
        "probe_expected_units": expected_units,
        "selected_qdocs": selected_qdocs,
        "selected_units": selected_units,
        "worklist_sha256": worklist_sha,
        "universe_sha256": UNIVERSE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu_run": False,
        "model_loaded": False,
        "volume_commit": False,
        "persistent_writes": False,
        "document_path_resolution_seconds": preprocessor.documents.document_path_resolution_seconds,
        "document_pipeline_seconds": preprocessor.documents.document_pipeline_seconds,
        "worker_load_seconds_sum": preprocessor.documents.worker_load_seconds_sum,
        "worker_prepare_seconds_sum": preprocessor.documents.worker_prepare_seconds_sum,
        "document_cache_hits": preprocessor.documents.cache_hits,
        "document_cache_misses": preprocessor.documents.cache_misses,
        "prepared_cache_serialized_bytes_252": total_pickle_bytes,
        "prepared_cache_serialized_mb_252": total_pickle_bytes / 1024**2,
        "mean_serialized_bytes_per_doc": total_pickle_bytes / PROBE_UNIQUE_DOCS,
        "min_serialized_bytes_per_doc": min(pickle_sizes.values()),
        "max_serialized_bytes_per_doc": max(pickle_sizes.values()),
        "estimated_serialized_cache_mb_shard00": (
            total_pickle_bytes / PROBE_UNIQUE_DOCS * FULL_SHARD_UNIQUE_DOCS / 1024**2
        ),
        "serialized_cache_estimate_is_rss": False,
        "serialized_cache_estimate_basis": "pickle serialized size extrapolation, not RSS",
        "timing": {
            "query_preload_seconds": query_seconds,
            "document_index_seconds": preprocessor.documents.document_index_seconds,
            "document_preparation_seconds": preprocessor.documents.document_preparation_seconds,
            "document_pipeline_seconds": preprocessor.documents.document_pipeline_seconds,
            "selector_seconds": preprocessor.selector_seconds,
            "preload_wall_seconds": preload_seconds,
            "post_preload_selection_seconds": time.perf_counter() - preload_started - preload_seconds,
            "modal_cpu_probe_wall_seconds": time.perf_counter() - probe_started,
        },
        "rss_mb": {
            "before_preload": rss_before,
            "after_preload": rss_after_preload,
            "after_selection": rss_after_selection,
        },
        "prepared_documents": len(preprocessor.documents.prepared),
        "document_files_loaded": preprocessor.documents.files_loaded,
        "filesystem_scans": preprocessor.documents.filesystem_scans,
        "prepare_document_calls": preprocessor.documents.prepare_document_calls,
        "preparation_workers": 8,
    }
    result["shard00_runtime_estimate"] = _shard_runtime_estimate(
        preprocessor.documents.document_pipeline_seconds
    )
    return result


@app.local_entrypoint()
def main(
    mode: str = "optimized-current-canary",
    shard_id: str = "00",
    inference_batch_size: int = PRODUCTION_INFERENCE_BATCH_SIZE,
    durability_qdocs: int = RESULT_BUFFER_QDOCS,
    kind: str = "optimized_current_canary_b8",
) -> None:
    if mode == "optimized-current-canary-finalize":
        result = optimized_current_canary_finalize.remote(kind)
    elif mode == "optimized-current-canary-compare":
        result = optimized_current_canary_compare.remote(kind)
    elif mode == "optimized-current-canary":
        result = optimized_current_canary.remote(inference_batch_size, durability_qdocs)
    elif mode == "optimized-production-probe":
        result = optimized_production_probe.remote()
    elif mode == "prepare-optimized-production-shard-selection":
        result = prepare_optimized_production_shard_selection.remote(shard_id)
    elif mode == "optimized-document-pipeline-modal-cpu-probe":
        result = optimized_document_pipeline_modal_cpu_probe.remote()
    elif mode == "optimized-production-shard-preselected":
        result = optimized_production_shard_preselected.remote(shard_id, inference_batch_size, durability_qdocs)
    elif mode == "optimized-production-shard":
        result = optimized_production_shard.remote(shard_id, inference_batch_size, durability_qdocs)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    print("OPTIMIZED_QWEN_RESULT=" + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
