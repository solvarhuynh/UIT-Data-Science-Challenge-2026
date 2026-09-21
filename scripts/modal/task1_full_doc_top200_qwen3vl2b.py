"""Durable, resumable FULLDOC_TOP200 Qwen runner and bounded canaries.

The numerical scorer is imported unchanged from task1_b2a_qwen3vl2b.py.
There is intentionally no local-entrypoint mode that launches all 32 shards.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import modal

import task1_b2a_qwen3vl2b as historical
from task1_full_doc_top200_durability import (
    SCHEMA_VERSION,
    CheckpointError,
    DurableShardStore,
    atomic_jsonl,
    sha256_file,
)
from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared


APP_NAME = "task1-full-doc-top200-qwen3vl2b"
VOLUME_NAME = "udsc-p13"
MOUNT = Path("/workspace/p13")
RUNTIME = MOUNT / "runtime"
OUTPUT_ROOT = RUNTIME / "full_doc_top200_qwen"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
SHARD_ROOT = OUTPUT_ROOT / "shards"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
LOG_ROOT = OUTPUT_ROOT / "logs"
FINAL_ROOT = OUTPUT_ROOT / "final"

# The source is a nested repository file locally but Modal 1.x imports the
# packaged runner directly from /root.  The local-root paths are deployment
# inputs only; use an inert non-existent root during remote re-import.
_SOURCE_PATH = Path(__file__).resolve()
LOCAL_ROOT = _SOURCE_PATH.parents[2] if len(_SOURCE_PATH.parents) >= 3 else Path("/__modal_local_source_unavailable__")
LOCAL_REPORT_ROOT = LOCAL_ROOT / "reports/task1/full_document_legal_field_retrieval"
LOCAL_WORKLIST_ROOT = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_worklists"
LOCAL_UNIVERSE_MANIFEST = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_future_universe_manifest.json"
LOCAL_PARITY_SAMPLE = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_historical_parity_256.jsonl"
LOCAL_CURRENT_SAMPLE = LOCAL_REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl"
REMOTE_WORKLIST_ROOT = Path("/opt/full_doc_top200_qwen_worklists")
REMOTE_UNIVERSE_MANIFEST = Path("/opt/full_doc_top200_qwen_future_universe_manifest.json")
REMOTE_PARITY_SAMPLE = Path("/opt/full_doc_top200_qwen_historical_parity_256.jsonl")
REMOTE_CURRENT_SAMPLE = Path("/opt/full_doc_top200_qwen_current_canary_16.jsonl")

MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODEL_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
PROMPT_SCORING_SHA256 = "60feb6aca921a1f6e87f08dbebb0cef202e49d09daa4e727592b756de042f55a"
QUERY_SOURCE_SHA256 = "c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b7"
HISTORICAL_SCORER_SHA256 = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
RESULT_BUFFER_QDOCS = 256
RESULT_BUFFER_CHUNK_ROWS = 1024
INFERENCE_BATCH_SIZE = 1
MAX_LENGTH = 8192
DTYPE = "torch.bfloat16"
SHARD_COUNT = 32
AUTHORITATIVE_MANIFEST_SHA256 = "1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722"


if LOCAL_WORKLIST_ROOT.is_dir() and LOCAL_UNIVERSE_MANIFEST.is_file():
    image = (
        historical.image
        .pip_install("pyvi==0.1.1")
        .add_local_dir(str(LOCAL_WORKLIST_ROOT), remote_path="/opt/full_doc_top200_qwen_worklists")
        .add_local_file(str(LOCAL_UNIVERSE_MANIFEST), remote_path="/opt/full_doc_top200_qwen_future_universe_manifest.json")
        .add_local_file(str(LOCAL_PARITY_SAMPLE), remote_path="/opt/full_doc_top200_qwen_historical_parity_256.jsonl")
        .add_local_file(str(LOCAL_CURRENT_SAMPLE), remote_path="/opt/full_doc_top200_qwen_current_canary_16.jsonl")
        # Modal 1.x does not implicitly ship sibling local modules.  Preserve
        # their historical import names without duplicating scorer code.
        .add_local_file(str(LOCAL_ROOT / "scripts/modal/task1_b2a_qwen3vl2b.py"), remote_path="/root/task1_b2a_qwen3vl2b.py")
        .add_local_file(str(LOCAL_ROOT / "scripts/modal/task1_full_doc_top200_durability.py"), remote_path="/root/task1_full_doc_top200_durability.py")
        .add_local_file(str(LOCAL_ROOT / "scripts/beam/task1_v2/evidence.py"), remote_path="/root/scripts/beam/task1_v2/evidence.py")
    )
else:
    # Keeps source importable for static inspection before the builder runs.
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
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def frozen_universe() -> dict[str, Any]:
    value = json.loads(REMOTE_UNIVERSE_MANIFEST.read_text(encoding="utf-8"))
    if value.get("schema_version") != "full_doc_top200_qwen_future_universe_v1":
        raise RuntimeError("future universe schema mismatch")
    if int(value.get("q_doc_count", -1)) != 598192 or int(value.get("inference_unit_count", -1)) != 1794571:
        raise RuntimeError("future universe cardinality mismatch")
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
    if identities and [identities[0][0], identities[0][1]] != [str(x) for x in entry["first_identity"]]:
        raise RuntimeError(f"production shard first identity mismatch: {shard_id}")
    if identities and [identities[-1][0], identities[-1][1]] != [str(x) for x in entry["last_identity"]]:
        raise RuntimeError(f"production shard last identity mismatch: {shard_id}")
    return rows, worklist_sha


def contract_values(universe: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_sha256": historical.json.loads((historical.RUNTIME / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json").read_text(encoding="utf-8"))["snapshot_provenance_sha256"],
        "prompt_scoring_sha256": PROMPT_SCORING_SHA256,
        "max_length": MAX_LENGTH,
        "dtype": DTYPE,
        "batch": INFERENCE_BATCH_SIZE,
        "chunk_selector": "select_true_s2_prepared; BM25 descending; chunk_id ascending; ordered[:3]",
        "aggregation": "MAX(chunk_scores)",
        "query_source_sha256": QUERY_SOURCE_SHA256,
        "canonical_chunk_source_sha256": universe["canonical_chunk_source_sha256"],
    }


def contract_sha256(universe: dict[str, Any]) -> str:
    raw = json.dumps(contract_values(universe), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_scientific_source() -> None:
    source = Path(historical.__file__).resolve()
    if sha256_file(source) != HISTORICAL_SCORER_SHA256:
        raise RuntimeError("historical scientific scorer SHA mismatch")
    if historical.MODEL_ID != MODEL_ID or historical.MODEL_REVISION != MODEL_REVISION:
        raise RuntimeError("model contract regression")
    if historical.MAX_LENGTH != MAX_LENGTH or historical.INFERENCE_BATCH_SIZE != INFERENCE_BATCH_SIZE:
        raise RuntimeError("sequence/batch contract regression")


def load_document_rows(doc: str) -> list[dict[str, Any]]:
    path = historical.CHUNKS / f"{doc}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"canonical document missing: {doc}")
    rows = list(jsonl(path))
    ids = [str(row.get("chunk_id", "")) for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not item or not item.startswith(doc + "_") for item in ids):
        raise RuntimeError(f"canonical document mapping invalid: {doc}")
    return rows


def selected_chunks(query: str, doc: str, question: str, expected_units: int, expected_ids: list[str] | None) -> list[dict[str, Any]]:
    rows = load_document_rows(doc)
    selected = select_true_s2_prepared(question, prepare_document(rows), topk=3)
    ids = [str(item["chunk_id"]) for item in selected]
    if not (1 <= len(selected) <= 3) or len(ids) != len(set(ids)) or len(selected) != expected_units:
        raise RuntimeError(f"TOP3_UP_TO_AVAILABLE violation: {query}/{doc}")
    if expected_ids is not None and ids != expected_ids:
        raise RuntimeError(f"frozen chunk-selection mismatch: {query}/{doc}")
    if any(not item["raw_chunk_text"].strip() or not str(item["chunk_id"]).startswith(doc + "_") for item in selected):
        raise RuntimeError(f"synthetic/empty/cross-document chunk: {query}/{doc}")
    return selected


def frozen_parity_chunks(query: str, doc: str, frozen_ids: list[str]) -> list[dict[str, Any]]:
    """Resolve, in frozen order, the historical scorer inputs for parity only."""
    if not 1 <= len(frozen_ids) <= 3 or len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError(f"invalid frozen parity IDs: {query}/{doc}")
    by_id = {str(item.get("chunk_id", "")): item for item in load_document_rows(doc)}
    selected: list[dict[str, Any]] = []
    for chunk_id in frozen_ids:
        item = by_id.get(str(chunk_id))
        text = str(item.get("text", "")) if item is not None else ""
        if item is None or not str(chunk_id).startswith(doc + "_") or not text.strip():
            raise RuntimeError(f"unresolvable frozen parity chunk: {query}/{doc}/{chunk_id}")
        selected.append({"chunk_id": str(chunk_id), "raw_chunk_text": text})
    return selected


def validated_questions(query_ids: set[str]) -> dict[str, str]:
    questions = historical.load_questions(query_ids)
    for query, question in questions.items():
        if not isinstance(question, str) or not question.strip() or question == str(query):
            raise RuntimeError(f"query-text tripwire failure: {query}")
    return questions


def validate_historical_parity_inputs(rows: list[dict[str, Any]]) -> dict[str, str]:
    if len(rows) != 256 or sorted(Counter(len(row["selected_chunk_ids"]) for row in rows).items()) != [(1, 32), (2, 32), (3, 192)]:
        raise RuntimeError("historical parity sample contract mismatch")
    questions = validated_questions({str(row["query_id"]) for row in rows})
    units = 0
    for row in rows:
        frozen_ids = list(row["selected_chunk_ids"])
        frozen_parity_chunks(str(row["query_id"]), str(row["document_id"]), frozen_ids)
        units += len(frozen_ids)
    if units != 672:
        raise RuntimeError(f"historical parity inference-unit mismatch: {units}")
    return questions


def tripwire(path: Path, row: dict[str, Any], question: str, chunks: list[dict[str, Any]]) -> None:
    fsync_json(path, {
        "query_id": row["query_id"],
        "query_text_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "query_text_preview": question[:100],
        "document_id": row["document_id"],
        "selected_chunk_ids": [item["chunk_id"] for item in chunks],
        "selected_chunk_count": len(chunks),
        "model_revision": MODEL_REVISION,
    })
    volume.commit()


def model_bundle(query_ids: set[str], questions: dict[str, str] | None = None) -> tuple[dict[str, str], Any, Any, Any, dict[str, str], str, str]:
    import torch
    import transformers
    questions = questions if questions is not None else validated_questions(query_ids)
    model_path, revision, provenance, yes, no = historical.download_locked_snapshot()
    if revision != MODEL_REVISION:
        raise RuntimeError("resolved model revision mismatch")
    processor, model, _, gpu_name, _, _ = historical.load_model(torch, transformers, model_path, yes, no)
    return questions, processor, model, torch, provenance, str(model_path), str(gpu_name)


def identity(kind: str, universe: dict[str, Any], worklist_sha: str, shard_id: str) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256(universe),
        "universe_sha256": str(universe["universe_sha256"]),
        "shard_sha256": worklist_sha,
        "shard_id": shard_id,
        "run_kind": kind,
    }


def score_rows(kind: str, shard_id: str, rows: list[dict[str, Any]], complete: bool, validated_parity_questions: dict[str, str] | None = None) -> dict[str, Any]:
    volume.reload()
    verify_scientific_source(); universe = frozen_universe()
    source_path = REMOTE_PARITY_SAMPLE if kind == "historical_parity" else REMOTE_CURRENT_SAMPLE if kind == "current_canary" else REMOTE_WORKLIST_ROOT / f"shard_{int(shard_id):02d}.jsonl"
    worklist_sha = sha256_file(source_path)
    store_path = CHECKPOINT_ROOT / kind / f"shard_{shard_id}.sqlite3"
    store = DurableShardStore(store_path, identity(kind, universe, worklist_sha, shard_id))
    completed_before = store.completed_identities()
    questions, processor, model, torch, provenance, model_path, gpu_name = model_bundle({str(row["query_id"]) for row in rows}, validated_parity_questions)
    buffer: list[tuple[str, str, list[tuple[str, float]]]] = []
    buffer_chunks = 0; scored_now = 0; first_pending = True
    expected: dict[tuple[str, str], int] = {}
    for row in rows:
        query, doc = str(row["query_id"]), str(row["document_id"])
        expected_units = len(row["selected_chunk_ids"]) if "selected_chunk_ids" in row else int(row["expected_inference_units"])
        expected[(query, doc)] = expected_units
        if (query, doc) in completed_before:
            continue
        question = questions[query]
        if question != historical.load_questions({query})[query]:
            raise RuntimeError(f"canonical query equality failure: {query}")
        chunks = frozen_parity_chunks(query, doc, list(row["selected_chunk_ids"])) if kind == "historical_parity" else selected_chunks(query, doc, question, expected_units, row.get("selected_chunk_ids"))
        if first_pending:
            tripwire(LOG_ROOT / kind / f"shard_{shard_id}_first_qdoc.json", row, question, chunks)
            first_pending = False
        values: list[tuple[str, float]] = []
        for chunk in chunks:
            scores, _, _, _ = historical.score_batch(processor, model, torch, [question], [chunk["raw_chunk_text"]])
            if len(scores) != 1 or not math.isfinite(float(scores[0])):
                raise RuntimeError(f"non-finite scorer output: {query}/{doc}")
            values.append((str(chunk["chunk_id"]), float(scores[0])))
        buffer.append((query, doc, values)); buffer_chunks += len(values); scored_now += 1
        if len(buffer) >= RESULT_BUFFER_QDOCS or buffer_chunks >= RESULT_BUFFER_CHUNK_ROWS:
            store.commit_qdocs(buffer, volume.commit); buffer = []; buffer_chunks = 0
    store.commit_qdocs(buffer, volume.commit)
    result: dict[str, Any] = {
        "kind": kind, "shard_id": shard_id, "expected": len(rows),
        "completed_before": len(completed_before), "scored_now": scored_now,
        "completed_after": len(store.completed_identities()), "contract_sha256": contract_sha256(universe),
        "universe_sha256": universe["universe_sha256"], "worklist_sha256": worklist_sha,
        "checkpoint": str(store_path), "model_cache_path": model_path, "actual_gpu": gpu_name,
        "resolved_revision": MODEL_REVISION,
    }
    if complete:
        output_rows = store.validate_complete(expected)
        output_path = SHARD_ROOT / kind / f"shard_{shard_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_sha = atomic_jsonl(output_path, output_rows); volume.commit()
        manifest = {
            "schema_version": "full_doc_qwen_shard_manifest_v1", "shard_id": shard_id,
            "universe_sha256": universe["universe_sha256"], "shard_worklist_sha256": worklist_sha,
            "model": MODEL_ID, "revision": MODEL_REVISION, "scorer_sha256": HISTORICAL_SCORER_SHA256,
            "contract_sha256": contract_sha256(universe), "q_doc_count": len(output_rows),
            "inference_unit_count": sum(expected.values()), "output_sha256": output_sha,
            "first_identity": [rows[0]["query_id"], rows[0]["document_id"]],
            "last_identity": [rows[-1]["query_id"], rows[-1]["document_id"]],
            "completed_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "COMPLETE",
            "snapshot_provenance_sha256": provenance,
        }
        manifest_path = MANIFEST_ROOT / kind / f"shard_{shard_id}.json"
        fsync_json(manifest_path, manifest); volume.commit()
        result.update({"output": str(output_path), "output_sha256": output_sha, "manifest": str(manifest_path), "status": "COMPLETE"})
    store.close()
    return result


def parity_remote() -> dict[str, Any]:
    rows = list(jsonl(REMOTE_PARITY_SAMPLE))
    questions = validate_historical_parity_inputs(rows)
    result = score_rows("historical_parity", "00", rows, complete=True, validated_parity_questions=questions)
    fresh = {(row["query_id"], row["document_id"]): row["score"] for row in jsonl(Path(result["manifest"]).parents[2] / "shards/historical_parity/shard_00.jsonl")}
    deltas = [abs(fresh[(row["query_id"], row["document_id"])] - float(row["expected_document_score"])) for row in rows]
    exact = sum(delta == 0.0 for delta in deltas)
    # Ordering is assessed only within sampled queries containing >=2 docs.
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows: by_query.setdefault(str(row["query_id"]), []).append(row)
    contradictions = 0
    for values in by_query.values():
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                historical_order = math.copysign(1, float(left["expected_document_score"]) - float(right["expected_document_score"])) if left["expected_document_score"] != right["expected_document_score"] else 0
                fresh_order = math.copysign(1, fresh[(left["query_id"], left["document_id"])] - fresh[(right["query_id"], right["document_id"])]) if fresh[(left["query_id"], left["document_id"])] != fresh[(right["query_id"], right["document_id"])] else 0
                contradictions += historical_order != fresh_order
    ordered_deltas = sorted(deltas)
    parity = {"rows": len(rows), "produced": len(fresh), "exact_matches": exact, "max_abs_diff": max(deltas), "mean_abs_diff": sum(deltas)/len(deltas), "median_abs_diff": ordered_deltas[len(ordered_deltas)//2], "p95_abs_diff": ordered_deltas[math.ceil(0.95 * len(ordered_deltas)) - 1], "nan_inf": 0, "ordering_mismatches": contradictions}
    parity["gate"] = "PASS" if len(fresh) == 256 and max(deltas) <= 1e-4 and contradictions == 0 else "FAIL"
    fsync_json(MANIFEST_ROOT / "historical_parity_result.json", parity); volume.commit()
    return {**result, "parity": parity}


def current_part1_remote() -> dict[str, Any]:
    rows = list(jsonl(REMOTE_CURRENT_SAMPLE))
    return score_rows("current_canary", "00", rows[:8], complete=False)


def current_part2_remote() -> dict[str, Any]:
    rows = list(jsonl(REMOTE_CURRENT_SAMPLE))
    result = score_rows("current_canary", "00", rows, complete=True)
    result["duplicates"] = 0; result["missing"] = 0
    result["resume_tested"] = result["completed_before"] == 8 and result["scored_now"] == 8
    result["gate"] = "PASS" if result["resume_tested"] else "FAIL"
    fsync_json(MANIFEST_ROOT / "current_canary_result.json", result); volume.commit()
    return result


def production_shard_remote(shard_id: str) -> dict[str, Any]:
    volume.reload(); verify_scientific_source(); universe = frozen_universe()
    rows, worklist_sha = validate_production_shard(shard_id, universe)
    print(json.dumps({
        "kind": "production", "shard_id": shard_id, "expected": len(rows),
        "expected_units": sum(int(row["expected_inference_units"]) for row in rows),
        "universe_sha256": universe["universe_sha256"], "worklist_sha256": worklist_sha,
        "contract_sha256": contract_sha256(universe),
    }, sort_keys=True), flush=True)
    return score_rows("production", shard_id, rows, complete=True)


def volume_test_write_remote() -> dict[str, Any]:
    path = OUTPUT_ROOT / "volume_test/payload.json"
    payload = {"schema_version": "volume_test_v1", "content": "durable-cross-container-test"}
    fsync_json(path, payload); digest = sha256_file(path); volume.commit()
    return {"path": str(path), "sha256": digest}


def volume_test_verify_remote(expected_sha256: str) -> dict[str, Any]:
    volume.reload(); path = OUTPUT_ROOT / "volume_test/payload.json"
    actual = sha256_file(path) if path.is_file() else "MISSING"
    if actual != expected_sha256:
        raise RuntimeError(f"cross-container Volume persistence mismatch: {actual}")
    return {"status": "PASS", "sha256": actual}


@app.function(image=image, timeout=10*60, cpu=1, retries=0)
def import_smoke_remote() -> str:
    """CPU-only packaging validation: no Volume, model, or inference access."""
    import task1_b2a_qwen3vl2b as remote_historical
    import task1_full_doc_top200_durability as remote_durability
    import scripts.beam.task1_v2.evidence as remote_evidence
    modules = {
        "task1_full_doc_top200_qwen3vl2b": __file__,
        "task1_b2a_qwen3vl2b": remote_historical.__file__,
        "task1_full_doc_top200_durability": remote_durability.__file__,
        "scripts.beam.task1_v2.evidence": remote_evidence.__file__,
    }
    embedded = [
        REMOTE_WORKLIST_ROOT.is_dir(), REMOTE_UNIVERSE_MANIFEST.is_file(),
        REMOTE_PARITY_SAMPLE.is_file(), REMOTE_CURRENT_SAMPLE.is_file(),
    ]
    return json.dumps({
        "status": "REMOTE_IMPORT_PACKAGING_PASS" if all(embedded) else "REMOTE_IMPORT_PACKAGING_FAIL",
        "modules": modules,
        "historical_scorer_sha256": sha256_file(Path(remote_historical.__file__)),
        "image_embedded_inputs_present": all(embedded),
        "qwen_inference_executed": False,
    }, sort_keys=True)


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60*60, cpu=8, memory=32768, retries=0)
def historical_parity() -> dict[str, Any]: return parity_remote()

@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60*60, cpu=8, memory=32768, retries=0)
def current_part1() -> dict[str, Any]: return current_part1_remote()

@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60*60, cpu=8, memory=32768, retries=0)
def current_part2() -> dict[str, Any]: return current_part2_remote()

@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60*60, cpu=8, memory=32768, retries=0)
def production_shard(shard_id: str) -> dict[str, Any]: return production_shard_remote(shard_id)

@app.function(image=image, volumes={str(MOUNT): volume}, timeout=10*60, cpu=1, retries=0)
def volume_test_write() -> dict[str, Any]: return volume_test_write_remote()

@app.function(image=image, volumes={str(MOUNT): volume}, timeout=10*60, cpu=1, retries=0)
def volume_test_verify(expected_sha256: str) -> dict[str, Any]: return volume_test_verify_remote(expected_sha256)


@app.local_entrypoint()
def main(mode: str = "volume-test", expected_sha256: str = "", shard_id: str = "") -> str:
    if mode == "import-smoke": result = import_smoke_remote.remote()
    elif mode == "volume-test-write": result = volume_test_write.remote()
    elif mode == "volume-test-verify": result = volume_test_verify.remote(expected_sha256)
    elif mode == "historical-parity": result = historical_parity.remote()
    elif mode == "current-part1": result = current_part1.remote()
    elif mode == "current-part2": result = current_part2.remote()
    elif mode == "production-shard": result = production_shard.remote(shard_id)
    else: raise ValueError("mode must be import-smoke, volume-test-write, volume-test-verify, historical-parity, current-part1, current-part2, or production-shard")
    serialized = result if isinstance(result, str) else json.dumps(result, sort_keys=True)
    print("FULLDOC_TOP200_RESULT=" + serialized, flush=True)
    return serialized
