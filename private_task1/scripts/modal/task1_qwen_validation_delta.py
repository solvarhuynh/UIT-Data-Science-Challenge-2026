"""Bounded corrected-Qwen validation-delta runner.

The GPU function reuses the frozen historical scorer's model loader and
``score_batch``.  It accepts only the preflighted 184-q-doc delta and writes to
an isolated namespace.  No labels, Private input, or rescue policy is read.
This file is prepared for a user-authorized Modal run; importing/compiling it
does not load a model or contact Modal.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import modal


_SOURCE_PATH = Path(__file__).resolve()
_LOCAL_ROOT = _SOURCE_PATH.parents[3] if len(_SOURCE_PATH.parents) >= 4 else None
HISTORICAL_LOCAL = (
    _LOCAL_ROOT / "scripts/modal/task1_b2a_qwen3vl2b.py"
    if _LOCAL_ROOT is not None
    else None
)
DELTA_LOCAL = (
    _LOCAL_ROOT / "private_task1/experiments/qwen_single_rescue/validation_delta_worklist.jsonl"
    if _LOCAL_ROOT is not None
    else None
)
EXPECTED_WORKLIST_SHA = "c9384fde1338e9a6b850e473876a4ff819c8f8dbfe7d90d536493a5d1d9fc428"
EXPECTED_QDOCS = 184
EXPECTED_UNITS = 552
MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODEL_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
SCORER_SHA = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
CONTRACT_SHA = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
UNIVERSE_SHA = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"
SELECTOR = "true_s2_bm25_within_document_v2"
INFERENCE_BATCH_SIZE = 1
OUTPUT_ROOT = Path("/workspace/p13/runtime/private_task1/experiments/qwen_single_rescue/validation_delta_gpu")
# These are Linux-container paths. Keep them as POSIX strings at the Modal
# image-packaging boundary; Windows pathlib must not be involved in
# ``remote_path=`` validation.
REMOTE_DELTA = "/opt/qwen_validation_delta.jsonl"
REMOTE_SCORER = "/root/task1_b2a_qwen3vl2b.py"
REMOTE_CHUNKS = Path("/workspace/p13/runtime/data/processed_pv1/chunks")

# Local CLI import uses the repository copy.  Modal's packaged module is
# imported directly from /root and must not infer a repository root.
if HISTORICAL_LOCAL is not None and HISTORICAL_LOCAL.is_file():
    if str(HISTORICAL_LOCAL.parent) not in sys.path:
        sys.path.insert(0, str(HISTORICAL_LOCAL.parent))
elif Path(REMOTE_SCORER).is_file():
    if str(Path(REMOTE_SCORER).parent) not in sys.path:
        sys.path.insert(0, str(Path(REMOTE_SCORER).parent))
import task1_b2a_qwen3vl2b as historical  # noqa: E402


if HISTORICAL_LOCAL is not None and DELTA_LOCAL is not None and HISTORICAL_LOCAL.is_file() and DELTA_LOCAL.is_file():
    image = (
        historical.image
        .add_local_file(str(HISTORICAL_LOCAL), remote_path=REMOTE_SCORER)
        .add_local_file(str(DELTA_LOCAL), remote_path=REMOTE_DELTA)
    )
else:
    # Remote import branch: both files were embedded by the local image
    # definition and are already visible at their absolute container paths.
    image = historical.image
volume = modal.Volume.from_name("udsc-p13", create_if_missing=False)
app = modal.App("task1-qwen-validation-delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"non-object worklist row: {path}:{line_no}")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_chunks(document_id: str, text_audit: dict[str, int]) -> dict[str, str]:
    path = REMOTE_CHUNKS / f"{document_id}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing PV1 chunk file: {document_id}")
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        chunk_id = str(row.get("chunk_id", ""))
        has_raw = "raw_chunk_text" in row
        has_text = "text" in row
        if has_raw and has_text:
            text_audit["both_present"] += 1
            if str(row.get("raw_chunk_text")) != str(row.get("text")):
                text_audit["mismatch"] += 1
        # Canonical scorer source order: text -> chunk_text.  raw_chunk_text
        # is deliberately never used as a fallback.
        text = str(row.get("text", row.get("chunk_text", "")))
        if chunk_id in result:
            raise RuntimeError(f"duplicate chunk ID in {document_id}: {chunk_id}")
        result[chunk_id] = text
    return result


def validate_worklist(rows: list[dict[str, Any]]) -> tuple[str, int]:
    if sha256_file(Path(REMOTE_DELTA)) != EXPECTED_WORKLIST_SHA:
        raise RuntimeError("delta worklist SHA mismatch")
    identities = [(str(row["query_id"]), str(row["document_id"])) for row in rows]
    if len(rows) != EXPECTED_QDOCS or len(identities) != len(set(identities)):
        raise RuntimeError("delta q-doc identity coverage mismatch")
    units = 0
    for row in rows:
        selected = [str(value) for value in row.get("selected_chunk_ids", [])]
        if row.get("selector") != SELECTOR or row.get("aggregation") != "MAX":
            raise RuntimeError("delta selector/aggregation contract mismatch")
        if row.get("model") != MODEL_ID or row.get("revision") != MODEL_REVISION:
            raise RuntimeError("delta model/revision contract mismatch")
        if len(selected) != int(row.get("expected_inference_units", -1)) or len(selected) != 3:
            raise RuntimeError("delta selected-unit mismatch")
        if len(selected) != len(set(selected)):
            raise RuntimeError("delta duplicate selected chunk ID")
        units += len(selected)
    if units != EXPECTED_UNITS:
        raise RuntimeError(f"delta unit coverage mismatch: {units}/{EXPECTED_UNITS}")
    return sha256_file(Path(REMOTE_DELTA)), units


def _load_and_validate_remote_inputs() -> tuple[list[dict[str, Any]], str, int, dict[str, int], dict[str, dict[str, str]], dict[str, Any]]:
    rows = read_jsonl(Path(REMOTE_DELTA))
    worklist_sha, expected_units = validate_worklist(rows)
    query_ids = {str(row["query_id"]) for row in rows}
    questions = historical.load_questions(query_ids)
    for row in rows:
        if str(row.get("question", "")) != questions[str(row["query_id"])]:
            raise RuntimeError(f"delta question text mismatch: {row['query_id']}/{row['document_id']}")

    text_audit = {"both_present": 0, "mismatch": 0}
    docs: dict[str, dict[str, str]] = {}
    for row in rows:
        docs.setdefault(str(row["document_id"]), load_chunks(str(row["document_id"]), text_audit))
    chunk_owner: dict[str, str] = {}
    cross_document_ids = 0
    missing_selected_chunks = 0
    empty_selected_texts = 0
    for row in rows:
        did = str(row["document_id"])
        for chunk_id_value in row["selected_chunk_ids"]:
            chunk_id = str(chunk_id_value)
            owner = chunk_owner.setdefault(chunk_id, did)
            if owner != did:
                cross_document_ids += 1
            text = docs[did].get(chunk_id)
            if text is None:
                missing_selected_chunks += 1
            elif not text.strip():
                empty_selected_texts += 1
            if not chunk_id.startswith(did + "_"):
                cross_document_ids += 1
    input_checks = {
        "remote_document_files": len(docs),
        "expected_document_files": len(docs),
        "resolved_qdocs": len(rows),
        "selected_chunks": sum(len(row["selected_chunk_ids"]) for row in rows),
        "missing_selected_chunks": missing_selected_chunks,
        "empty_selected_texts": empty_selected_texts,
        "cross_document_ids": cross_document_ids,
        "question_text_missing": 0,
        "canonical_text_field_gate": text_audit["mismatch"] == 0 and missing_selected_chunks == 0 and empty_selected_texts == 0,
        "remote_input_gate": (
            len(docs) == 64
            and len(rows) == EXPECTED_QDOCS
            and sum(len(row["selected_chunk_ids"]) for row in rows) == EXPECTED_UNITS
            and missing_selected_chunks == 0
            and empty_selected_texts == 0
            and cross_document_ids == 0
            and text_audit["mismatch"] == 0
        ),
    }
    return rows, worklist_sha, expected_units, text_audit, docs, input_checks


@app.function(image=image, volumes={"/workspace/p13": volume}, timeout=10 * 60, cpu=2, memory=4096, retries=0)
def validation_delta_cpu_preflight() -> dict[str, Any]:
    """Inspect the mounted PV1 chunks without loading Qwen or using a GPU."""
    volume.reload()
    _rows, worklist_sha, expected_units, text_audit, _docs, checks = _load_and_validate_remote_inputs()
    result = {
        "status": "PASS" if checks["remote_input_gate"] else "FAIL",
        "canonical_text_field_gate": "PASS" if checks["canonical_text_field_gate"] else "FAIL",
        "raw_text_vs_text_both_present": text_audit["both_present"],
        "raw_text_vs_text_mismatch": text_audit["mismatch"],
        "remote_document_files": f"{checks['remote_document_files']}/64",
        "remote_qdocs_resolvable": f"{checks['resolved_qdocs']}/{EXPECTED_QDOCS}",
        "remote_selected_chunks": f"{checks['selected_chunks']}/{EXPECTED_UNITS}",
        "remote_chunk_texts_nonempty": f"{checks['selected_chunks'] - checks['empty_selected_texts']}/{EXPECTED_UNITS}",
        "cross_document_ids": checks["cross_document_ids"],
        "remote_input_gate": "PASS" if checks["remote_input_gate"] else "FAIL",
        "worklist_sha256": worklist_sha,
        "batch": INFERENCE_BATCH_SIZE,
        "gpu_run": False,
        "qwen_model_loaded": False,
        "volume_commit": False,
    }
    print(json.dumps({
        "CANONICAL_TEXT_FIELD_GATE": result["canonical_text_field_gate"],
        "RAW_TEXT_VS_TEXT_BOTH_PRESENT": result["raw_text_vs_text_both_present"],
        "RAW_TEXT_VS_TEXT_MISMATCH": result["raw_text_vs_text_mismatch"],
        "REMOTE_DOCUMENT_FILES": result["remote_document_files"],
        "REMOTE_SELECTED_CHUNKS": result["remote_selected_chunks"],
        "REMOTE_INPUT_GATE": result["remote_input_gate"],
        "BATCH": INFERENCE_BATCH_SIZE,
        "GPU_TIMEOUT_SECONDS": 900,
        "GPU_RUNS": 0,
    }, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(image=image, gpu="A10", volumes={"/workspace/p13": volume}, timeout=15 * 60, cpu=8, memory=32768, retries=0)
def run_validation_delta() -> dict[str, Any]:
    """Score exactly the frozen delta, once, with canonical batch-1 semantics."""
    volume.reload()
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"isolated output namespace already exists: {OUTPUT_ROOT}")
    rows, worklist_sha, expected_units, text_audit, docs, input_checks = _load_and_validate_remote_inputs()
    if not input_checks["remote_input_gate"]:
        raise RuntimeError(f"remote PV1 input gate failed: {input_checks}; text_audit={text_audit}")
    query_ids = {str(row["query_id"]) for row in rows}
    questions = historical.load_questions(query_ids)

    import torch
    import transformers

    scorer_sha = sha256_file(Path(REMOTE_SCORER))
    if scorer_sha != SCORER_SHA:
        raise RuntimeError(f"canonical scorer SHA mismatch: {scorer_sha}")
    download_started = time.perf_counter()
    model_path, resolved_revision, snapshot_provenance, true_token_id, false_token_id = historical.download_locked_snapshot()
    model_download_seconds = time.perf_counter() - download_started
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(f"resolved revision mismatch: {resolved_revision}")
    processor, model, model_load_seconds, gpu_name, _, _ = historical.load_model(
        torch, transformers, model_path, true_token_id, false_token_id
    )
    if not torch.cuda.is_available() or not str(gpu_name).strip():
        raise RuntimeError(f"actual GPU gate failed: {gpu_name}")

    started = time.perf_counter()
    raw_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    processed_units = 0
    for index, row in enumerate(rows, 1):
        qid = str(row["query_id"])
        did = str(row["document_id"])
        question = questions[qid]
        chunk_scores: list[float] = []
        for chunk_id in row["selected_chunk_ids"]:
            text = docs[did][str(chunk_id)]
            values, _, _, _ = historical.score_batch(processor, model, torch, [question], [text])
            value = float(values[0])
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite score: {qid}/{did}/{chunk_id}")
            chunk_scores.append(value)
            raw_rows.append({"query_id": qid, "document_id": did, "chunk_id": str(chunk_id), "score": value})
            processed_units += 1
        document_rows.append({
            "query_id": qid,
            "document_id": did,
            "selected_chunk_ids": [str(value) for value in row["selected_chunk_ids"]],
            "chunk_scores": chunk_scores,
            "score": max(chunk_scores),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "scorer_sha256": scorer_sha,
            "selector": SELECTOR,
            "aggregation": "MAX",
        })
        if index == 1 or index % 32 == 0 or index == len(rows):
            print(f"DELTA_PROGRESS completed={index}/{len(rows)} units={processed_units}/{expected_units}", flush=True)

    if len(document_rows) != EXPECTED_QDOCS or processed_units != EXPECTED_UNITS:
        raise RuntimeError("delta output cardinality mismatch")
    if len({(row["query_id"], row["document_id"]) for row in document_rows}) != EXPECTED_QDOCS:
        raise RuntimeError("delta output duplicate q-doc identity")

    raw_path = OUTPUT_ROOT / "chunk_scores.jsonl"
    document_path = OUTPUT_ROOT / "qdoc_scores.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(document_path, document_rows)
    manifest = {
        "schema_version": "corrected_qwen_validation_delta_gpu_v1",
        "status": "COMPLETE",
        "q_doc_count": EXPECTED_QDOCS,
        "inference_unit_count": EXPECTED_UNITS,
        "query_count": len(query_ids),
        "worklist_sha256": worklist_sha,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "scorer_sha256": scorer_sha,
        "contract_sha256": CONTRACT_SHA,
        "universe_sha256": UNIVERSE_SHA,
        "selector": SELECTOR,
        "aggregation": "MAX",
        "score_direction": "higher = more relevant",
        "actual_gpu": gpu_name,
        "model_download_seconds": model_download_seconds,
        "model_load_seconds": model_load_seconds,
        "batch_size": INFERENCE_BATCH_SIZE,
        "raw_text_vs_text_both_present": text_audit["both_present"],
        "raw_text_vs_text_mismatch": text_audit["mismatch"],
        "inference_seconds": time.perf_counter() - started,
        "delta_errors": len(errors),
        "chunk_score_rows": len(raw_rows),
        "qdoc_output": str(document_path),
        "qdoc_output_sha256": sha256_file(document_path),
        "chunk_output": str(raw_path),
        "chunk_output_sha256": sha256_file(raw_path),
        "finite_scores": True,
        "labels_loaded": False,
        "private_qwen_run": False,
    }
    manifest_path = OUTPUT_ROOT / "execution_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


@app.local_entrypoint()
def main(mode: str = "validation-delta") -> None:
    if mode == "cpu-preflight":
        result = validation_delta_cpu_preflight.remote()
    elif mode == "validation-delta":
        result = run_validation_delta.remote()
    else:
        raise ValueError("mode must be cpu-preflight or validation-delta")
    print("QWEN_VALIDATION_DELTA_RESULT=" + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
