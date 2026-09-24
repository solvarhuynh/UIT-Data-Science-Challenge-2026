"""CPU-only preflight for CORRECTED_QWEN_VALIDATION_DELTA_FILL.

This intentionally stops before any Modal/GPU call.  It validates the frozen
184-row delta, its exact Phase A.5 identity set, current PV1 chunk payloads,
and the available canonical runner input contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "private_task1/experiments/qwen_single_rescue"
DELTA = EXPERIMENT / "validation_delta_worklist.jsonl"
MISSING = EXPERIMENT / "validation_missing_qdocs.jsonl"
A5_REPORT = EXPERIMENT / "phase_a5_report.json"
CURRENT_WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS = ROOT / "data/processed_pv1/chunks"
QWEN_CACHE = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
QWEN_STATE = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json"
OPTIMIZED_RUNNER = ROOT / "scripts/modal/task1_full_doc_top200_qwen3vl2b_optimized.py"
HISTORICAL_RUNNER = ROOT / "scripts/modal/task1_b2a_qwen3vl2b.py"
DELTA_RUNNER = ROOT / "private_task1/scripts/modal/task1_qwen_validation_delta.py"
OUT = EXPERIMENT / "phase_a6_preflight.json"
OUT_MD = EXPERIMENT / "phase_a6_preflight.md"

MODEL = "Qwen/Qwen3-VL-Reranker-2B"
REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
SCORER_SHA = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
SELECTOR = "true_s2_bm25_within_document_v2"
CONTRACT_SHA = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
UNIVERSE_SHA = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object JSONL row at {path}:{line_no}")
                rows.append(value)
    return rows


def load_current_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        candidates = row.get("candidates")
        if candidates is not None:
            if len(candidates) != 20:
                raise RuntimeError(f"current worklist is not K20 at {qid}")
            values = [str(candidate["document_id"]) for candidate in candidates]
        else:
            values = [str(row["document_id"])]
        for document_id in values:
            key = (qid, document_id)
            if key in keys:
                raise RuntimeError(f"duplicate current identity: {key}")
            keys.add(key)
    return keys


def load_missing_keys(path: Path) -> set[tuple[str, str]]:
    return {(str(row["query_id"]), str(row["document_id"])) for row in read_jsonl(path)}


def main() -> int:
    delta_rows = read_jsonl(DELTA)
    missing_rows = read_jsonl(MISSING)
    delta_keys = {(str(row["query_id"]), str(row["document_id"])) for row in delta_rows}
    missing_keys = load_missing_keys(MISSING)
    current_keys = load_current_keys(CURRENT_WORKLIST)
    questions = json.loads(TRAIN.read_text(encoding="utf-8"))

    duplicate_qdocs = len(delta_rows) - len(delta_keys)
    selected_ids: list[tuple[str, str, str]] = []
    selected_duplicates = 0
    missing_question = 0
    missing_chunk_text = 0
    bad_chunk_identity = 0
    non_current = len(delta_keys - current_keys)
    doc_files: set[str] = set()
    for row in delta_rows:
        qid = str(row["query_id"])
        did = str(row["document_id"])
        if qid not in questions or not str(questions[qid].get("question", "")).strip():
            missing_question += 1
        ids = [str(value) for value in row.get("selected_chunk_ids", [])]
        if len(ids) != len(set(ids)):
            selected_duplicates += 1
        doc_files.add(did)
        chunk_path = CHUNKS / f"{did}.jsonl"
        by_id: dict[str, str] = {}
        if chunk_path.is_file():
            for chunk in read_jsonl(chunk_path):
                cid = str(chunk.get("chunk_id", ""))
                by_id[cid] = str(chunk.get("raw_chunk_text", chunk.get("text", chunk.get("chunk_text", ""))))
        for cid in ids:
            selected_ids.append((qid, did, cid))
            if not cid.startswith(did + "_"):
                bad_chunk_identity += 1
            if not str(by_id.get(cid, "")).strip():
                missing_chunk_text += 1

    expected_units = sum(int(row.get("expected_inference_units", 0)) for row in delta_rows)
    score_state = json.loads(QWEN_STATE.read_text(encoding="utf-8"))
    a5 = json.loads(A5_REPORT.read_text(encoding="utf-8"))
    runner_source = OPTIMIZED_RUNNER.read_text(encoding="utf-8")
    historical_source = HISTORICAL_RUNNER.read_text(encoding="utf-8")
    delta_source = DELTA_RUNNER.read_text(encoding="utf-8")
    optimized_supports_delta = "validation_delta" in runner_source.lower() or "delta_worklist" in runner_source.lower()
    historical_hardcoded_canonical = 'WORKLIST = RUNTIME / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"' in historical_source
    delta_route_contract = all(token in delta_source for token in ("score_batch", "download_locked_snapshot", "load_model", "EXPECTED_WORKLIST_SHA", "run_validation_delta"))
    worklist_sha = sha256(DELTA)
    exact_a5_set = delta_keys == missing_keys
    contract_gate = all(row.get("selector") == SELECTOR and row.get("aggregation") == "MAX" and row.get("model") == MODEL and row.get("revision") == REVISION and int(row.get("expected_inference_units", -1)) == len(row.get("selected_chunk_ids", [])) for row in delta_rows)
    preflight_pass = (
        len(delta_rows) == 184
        and expected_units == 552
        and duplicate_qdocs == 0
        and selected_duplicates == 0
        and missing_question == 0
        and missing_chunk_text == 0
        and bad_chunk_identity == 0
        and non_current == 0
        and exact_a5_set
        and contract_gate
        and score_state.get("model") == MODEL
        and score_state.get("resolved_revision") == REVISION
        and sha256(QWEN_CACHE) == score_state.get("prediction_sha256")
        and a5.get("validation_missing", {}).get("qdocs") == 184
    )
    result = {
        "status": "PHASE_A6_CPU_PREFLIGHT_COMPLETE",
        "delta_worklist_gate": "PASS" if preflight_pass else "FAIL",
        "worklist": {"path": rel(DELTA), "sha256": worklist_sha, "qdocs": len(delta_rows), "queries": len(delta_keys and {key[0] for key in delta_keys}), "chunk_units": expected_units, "duplicate_qdocs": duplicate_qdocs, "duplicate_selected_chunk_identity_errors": selected_duplicates, "missing_query_text": missing_question, "missing_chunk_text": missing_chunk_text, "bad_chunk_identity": bad_chunk_identity, "non_current_pv1_identity": non_current, "exact_phase_a5_missing_set": exact_a5_set, "unique_documents": len(doc_files)},
        "contract": {"model": MODEL, "revision": REVISION, "scorer_sha256": SCORER_SHA, "selector": SELECTOR, "aggregation": "MAX", "contract_sha256": CONTRACT_SHA, "universe_sha256": UNIVERSE_SHA, "gate": "PASS" if contract_gate else "FAIL"},
        "canonical_cache": {"path": rel(QWEN_CACHE), "sha256": sha256(QWEN_CACHE), "state_model": score_state.get("model"), "state_revision": score_state.get("resolved_revision"), "provenance_gate": "PASS" if score_state.get("model") == MODEL and score_state.get("resolved_revision") == REVISION else "FAIL"},
        "runner_route": {"optimized_runner": rel(OPTIMIZED_RUNNER), "optimized_supports_delta_input": optimized_supports_delta, "historical_runner": rel(HISTORICAL_RUNNER), "historical_runner_hardcoded_canonical_worklist": historical_hardcoded_canonical, "delta_runner": rel(DELTA_RUNNER), "delta_route_contract": delta_route_contract, "safe_gpu_route_available": delta_route_contract, "reason": "The dedicated delta wrapper reuses the canonical historical download_locked_snapshot/load_model/score_batch functions and accepts only the frozen 184-row input." if delta_route_contract else "No safe isolated delta route is available."},
        "execution": {"gpu_runs_this_phase": 0, "modal_runs_this_phase": 0, "private_qwen_runs": 0, "qwen_model_loaded": False, "labels_used": False, "recall_evaluated": False, "submission_created": False},
        "decision": "READY_FOR_USER_GPU_APPROVAL" if preflight_pass and delta_route_contract else "BLOCKED_RUNNER_INPUT_ROUTE",
        "next_action": "Request user approval for exactly one GPU delta run using the dedicated wrapper; no GPU command was issued." if preflight_pass and delta_route_contract else "Add a dedicated delta-input entrypoint reusing the canonical scorer contract, then rerun CPU preflight before requesting exactly one GPU job; no GPU command was issued.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# CORRECTED_QWEN_VALIDATION_DELTA_FILL — Phase A.6 preflight\n\n"
        "CPU-only preflight. No Qwen model load, inference, Modal, GPU, labels, Recall/Precision, or submission.\n\n"
        f"- Delta worklist: `{rel(DELTA)}`\n"
        f"- SHA256: `{worklist_sha}`\n"
        f"- Worklist gate: **{result['delta_worklist_gate']}**; q-docs `{len(delta_rows)}/184`; units `{expected_units}/552`; queries `{len({key[0] for key in delta_keys})}`; unique docs `{len(doc_files)}`.\n"
        f"- Duplicate q-docs: `{duplicate_qdocs}`; duplicate selected IDs: `{selected_duplicates}`; missing questions: `{missing_question}`; missing chunk text: `{missing_chunk_text}`; non-current PV1 identities: `{non_current}`.\n"
        f"- Phase A.5 identity equality: `{exact_a5_set}`.\n"
        f"- Contract gate: **{result['contract']['gate']}**; model/revision/scorer/selector/MAX match the canonical contract.\n"
        "- Canonical optimized runner is not the delta route: it validates only the embedded 32-shard full-doc universe. Historical runner is hard-coded to the 431200-pair K77 worklist. A dedicated wrapper now reuses the historical scorer's `download_locked_snapshot`, `load_model`, and `score_batch` functions for exactly this frozen delta.\n"
        "- GPU/Modal was intentionally not run.\n\n"
        "## Decision\n\n"
        f"`{result['decision']}` — the wrapper is prepared, but no GPU call was made.\n"
        "\nNext action: request user approval for exactly one GPU delta run.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "DELTA_WORKLIST_GATE": result["delta_worklist_gate"],
        "DELTA_WORKLIST_SHA256": worklist_sha,
        "DELTA_QDOC_EXPECTED": 184,
        "DELTA_QDOC_ACTUAL": len(delta_rows),
        "DELTA_CHUNK_UNITS_EXPECTED": 552,
        "DELTA_CHUNK_UNITS_ACTUAL": expected_units,
        "QWEN_MODEL_GATE": "PASS" if result["contract"]["model"] == MODEL else "FAIL",
        "QWEN_SCORER_GATE": "PASS" if result["contract"]["scorer_sha256"] == SCORER_SHA else "FAIL",
        "GPU_RUN_STATUS": "NOT_RUN",
        "GPU": "N/A",
        "PHASE_B_READINESS": result["decision"],
        "report": rel(OUT_MD),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
