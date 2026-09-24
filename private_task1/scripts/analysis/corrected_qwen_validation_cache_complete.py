"""Build the complete corrected-Qwen validation K20 cache (Phase A.7).

CPU/local only.  This script reads the frozen current-PV1 K20 worklist, the
historical canonical Qwen cache, and the already-completed Phase A.6 delta.
It does not load a model, call Modal, read labels, evaluate metrics, or alter
any existing prediction artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "private_task1/experiments/qwen_single_rescue/validation_complete"
OUT_CACHE = OUT_DIR / "qwen_validation_k20_complete.jsonl"
OUT_MANIFEST = OUT_DIR / "merge_manifest.json"

CURRENT_CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
CURRENT_WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
HISTORICAL = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
HISTORICAL_STATE = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json"
DELTA_DIR = ROOT / "private_task1/experiments/qwen_single_rescue/validation_delta_gpu"
DELTA_QDOC = DELTA_DIR / "qdoc_scores.jsonl"
DELTA_CHUNK = DELTA_DIR / "chunk_scores.jsonl"
DELTA_EXECUTION = DELTA_DIR / "execution_manifest.json"
DELTA_WORKLIST = ROOT / "private_task1/experiments/qwen_single_rescue/validation_delta_worklist.jsonl"

EXPECTED_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EXPECTED_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EXPECTED_SCORER = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
EXPECTED_WORKLIST_SHA = "c9384fde1338e9a6b850e473876a4ff819c8f8dbfe7d90d536493a5d1d9fc428"
EXPECTED_DELTA_QDOC_SHA = "a0d91b6776f9afeacce10c5c0425ba36d28b0de9a35f570969fb932b7bf2612a"
EXPECTED_DELTA_CHUNK_SHA = "7abfdd82aadd222d57dae0fd6e7349b4b1b1c593129947e1ccf7ae75e941bd8f"
EXPECTED_DELTA_EXECUTION_SHA = "8d8910fe11ccbb96ccc52915b913558343171e4b16f9a8d3e6489896cc7ec555"
EXPECTED_DELTA_CONTRACT_SHA = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
EXPECTED_DELTA_UNIVERSE_SHA = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row: {path}:{line_no}")
            yield row


def natural_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_current() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    candidate_rows: dict[tuple[str, str], dict[str, Any]] = {}
    folds: dict[str, int] = {}
    query_candidate_counts: Counter[str] = Counter()
    for row in read_jsonl(CURRENT_CANDIDATES):
        qid = str(row["query_id"])
        if qid in folds and folds[qid] != int(row["fold"]):
            raise RuntimeError(f"inconsistent fold for query {qid}")
        folds[qid] = int(row["fold"])
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 20:
            raise RuntimeError(f"current candidate row is not K20: {qid}")
        for candidate in candidates:
            did = str(candidate["document_id"])
            key = (qid, did)
            if key in candidate_rows:
                raise RuntimeError(f"duplicate current K20 identity: {key}")
            candidate_rows[key] = {
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate.get("source_ranks") or {},
            }
            query_candidate_counts[qid] += 1
    if len(folds) != 5600 or len(candidate_rows) != 112000:
        raise RuntimeError(f"current candidate universe is {len(folds)}/{len(candidate_rows)}, expected 5600/112000")
    if set(folds.values()) != {1, 2, 3, 4}:
        raise RuntimeError(f"unexpected folds in current validation universe: {sorted(set(folds.values()))}")
    if any(count != 20 for count in query_candidate_counts.values()):
        raise RuntimeError("current candidate universe is not exactly 20/query")

    work_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(CURRENT_WORKLIST):
        qid = str(row["query_id"])
        did = str(row["document_id"])
        key = (qid, did)
        if key in work_rows:
            raise RuntimeError(f"duplicate current worklist identity: {key}")
        selected = [str(value) for value in row.get("selected_chunk_ids", [])]
        if len(selected) != int(row["expected_inference_units"]) or not selected or len(set(selected)) != len(selected):
            raise RuntimeError(f"invalid selected chunks in current worklist: {key}")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise RuntimeError(f"incompatible current worklist contract: {key}")
        if key not in candidate_rows or int(row["candidate_rank"]) != candidate_rows[key]["candidate_rank"]:
            raise RuntimeError(f"candidate/worklist mismatch: {key}")
        work_rows[key] = {
            **candidate_rows[key],
            "selected_chunk_ids": selected,
            "selector": str(row["selector"]),
            "aggregation": str(row["aggregation"]),
            "expected_inference_units": int(row["expected_inference_units"]),
        }
    if len(work_rows) != 112000 or set(work_rows) != set(candidate_rows):
        raise RuntimeError("current worklist does not exactly cover current K20")
    return work_rows, folds


def load_historical() -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    query_counts: Counter[str] = Counter()
    for row in read_jsonl(HISTORICAL):
        qid = str(row["query_id"])
        items = row.get("document_scores")
        if not isinstance(items, list):
            raise RuntimeError(f"historical row has no document_scores: {qid}")
        for item in items:
            key = (qid, str(item["doc_id"]))
            if key in scores:
                raise RuntimeError(f"duplicate historical identity: {key}")
            value = float(item["score"])
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite historical score: {key}")
            scores[key] = value
            query_counts[qid] += 1
    if len(query_counts) != 5600 or len(scores) != 431200:
        raise RuntimeError(f"historical cache shape is {len(query_counts)}/{len(scores)}, expected 5600/431200")
    if any(count != 77 for count in query_counts.values()):
        raise RuntimeError("historical cache is not exactly 77 documents/query")
    return scores


def load_delta() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], float]]:
    if sha256(DELTA_WORKLIST) != EXPECTED_WORKLIST_SHA:
        raise RuntimeError("delta worklist SHA mismatch")
    if sha256(DELTA_QDOC) != EXPECTED_DELTA_QDOC_SHA:
        raise RuntimeError("delta qdoc SHA mismatch")
    if sha256(DELTA_CHUNK) != EXPECTED_DELTA_CHUNK_SHA:
        raise RuntimeError("delta chunk SHA mismatch")
    if sha256(DELTA_EXECUTION) != EXPECTED_DELTA_EXECUTION_SHA:
        raise RuntimeError("delta execution manifest SHA mismatch")

    work: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(DELTA_WORKLIST):
        key = (str(row["query_id"]), str(row["document_id"]))
        work[key] = row
    if len(work) != 184:
        raise RuntimeError(f"delta worklist is {len(work)} rows, expected 184")

    chunk_scores: dict[tuple[str, str, str], float] = {}
    for row in read_jsonl(DELTA_CHUNK):
        key = (str(row["query_id"]), str(row["document_id"]), str(row["chunk_id"]))
        if key in chunk_scores:
            raise RuntimeError(f"duplicate delta chunk identity: {key}")
        value = float(row["score"])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite delta chunk score: {key}")
        chunk_scores[key] = value
    if len(chunk_scores) != 552:
        raise RuntimeError(f"delta chunk rows are {len(chunk_scores)}, expected 552")

    qdoc_scores: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(DELTA_QDOC):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in qdoc_scores:
            raise RuntimeError(f"duplicate delta qdoc identity: {key}")
        required = {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "scorer_sha256": EXPECTED_SCORER,
            "selector": "true_s2_bm25_within_document_v2",
            "aggregation": "MAX",
        }
        for field, expected in required.items():
            if row.get(field) != expected:
                raise RuntimeError(f"delta {field} mismatch at {key}")
        value = float(row["score"])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite delta qdoc score: {key}")
        selected = tuple(str(value) for value in row.get("selected_chunk_ids", []))
        if key not in work or selected != tuple(str(value) for value in work[key]["selected_chunk_ids"]):
            raise RuntimeError(f"delta selected chunk mismatch: {key}")
        if len(selected) != int(work[key]["expected_inference_units"]):
            raise RuntimeError(f"delta selected chunk count mismatch: {key}")
        values = []
        for chunk_id in selected:
            chunk_key = (key[0], key[1], chunk_id)
            if chunk_key not in chunk_scores:
                raise RuntimeError(f"missing delta chunk score: {chunk_key}")
            values.append(chunk_scores[chunk_key])
        if max(values) != value:
            raise RuntimeError(f"delta MAX aggregation mismatch: {key}")
        qdoc_scores[key] = {"score": value, "selected_chunk_ids": selected}
    if len(qdoc_scores) != 184:
        raise RuntimeError(f"delta qdoc rows are {len(qdoc_scores)}, expected 184")
    return work, {key: value["score"] for key, value in qdoc_scores.items()}


def main() -> int:
    current, folds = load_current()
    historical = load_historical()
    delta_work, delta = load_delta()

    current_keys = set(current)
    historical_keys = current_keys & set(historical)
    missing_keys = current_keys - set(historical)
    if len(historical_keys) != 111816 or len(missing_keys) != 184:
        raise RuntimeError(f"historical current-K20 coverage is {len(historical_keys)}/112000")
    if set(delta) != missing_keys:
        raise RuntimeError("delta identities are not exactly the historical missing current-K20 identities")
    if set(historical) & set(delta):
        raise RuntimeError("historical/delta identity overlap is nonzero")

    merged_rows: list[dict[str, Any]] = []
    for key, row in current.items():
        if key in delta:
            score = delta[key]
            source = "PHASE_A6_DELTA"
        else:
            score = historical[key]
            source = "HISTORICAL_CANONICAL"
        merged_rows.append({
            "query_id": row["query_id"],
            "document_id": row["document_id"],
            "current_candidate_rank": row["candidate_rank"],
            "qwen_score": score,
            "score_source": source,
            "selected_chunk_ids": row["selected_chunk_ids"],
            "selector": row["selector"],
            "aggregation": row["aggregation"],
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "scorer_sha256": EXPECTED_SCORER,
        })
    merged_rows.sort(key=lambda row: (natural_key(row["query_id"]), int(row["current_candidate_rank"]), natural_key(row["document_id"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CACHE.open("w", encoding="utf-8", newline="\n") as stream:
        for row in merged_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    current_work_sha = sha256(CURRENT_WORKLIST)
    current_candidate_sha = sha256(CURRENT_CANDIDATES)
    historical_sha = sha256(HISTORICAL)
    historical_state = json.loads(HISTORICAL_STATE.read_text(encoding="utf-8"))
    execution = json.loads(DELTA_EXECUTION.read_text(encoding="utf-8"))
    complete_sha = sha256(OUT_CACHE)
    counts_by_source = Counter(row["score_source"] for row in merged_rows)
    query_counts = Counter(row["query_id"] for row in merged_rows)
    manifest = {
        "schema_version": "corrected_qwen_validation_cache_complete_v1",
        "status": "CORRECTED_QWEN_VALIDATION_CACHE_COMPLETE",
        "current_validation_k20": {
            "primary_worklist_path": rel(CURRENT_WORKLIST),
            "primary_worklist_sha256": current_work_sha,
            "candidate_universe_path": rel(CURRENT_CANDIDATES),
            "candidate_universe_sha256": current_candidate_sha,
            "queries": len(folds),
            "qdocs": len(current),
            "qdoc_per_query": sorted(set(query_counts.values())),
            "folds": dict(sorted(Counter(folds.values()).items())),
            "fold0_used": False,
        },
        "historical_source": {
            "path": rel(HISTORICAL),
            "sha256": historical_sha,
            "rows_total": len(historical),
            "current_k20_rows_reused": len(historical_keys),
            "provenance_state_path": rel(HISTORICAL_STATE),
            "resolved_revision": historical_state.get("resolved_revision"),
            "labels_loaded": historical_state.get("labels_loaded"),
            "fold0_used": historical_state.get("fold0_used"),
        },
        "delta_source": {
            "qdoc_path": rel(DELTA_QDOC),
            "qdoc_sha256": sha256(DELTA_QDOC),
            "chunk_path": rel(DELTA_CHUNK),
            "chunk_sha256": sha256(DELTA_CHUNK),
            "execution_manifest_path": rel(DELTA_EXECUTION),
            "execution_manifest_sha256": sha256(DELTA_EXECUTION),
            "worklist_path": rel(DELTA_WORKLIST),
            "worklist_sha256": sha256(DELTA_WORKLIST),
            "qdocs": len(delta),
            "chunk_units": 552,
            "gpu": execution.get("actual_gpu"),
            "batch": execution.get("batch_size"),
        },
        "complete_cache": {
            "path": rel(OUT_CACHE),
            "sha256": complete_sha,
            "rows": len(merged_rows),
            "queries": len(query_counts),
            "source_split": dict(sorted(counts_by_source.items())),
        },
        "gates": {
            "validation_queries": f"{len(query_counts)}/5600",
            "validation_qdocs": f"{len(merged_rows)}/112000",
            "qdoc_per_query": "20/20",
            "historical_rows": len(historical_keys),
            "delta_rows": len(delta),
            "duplicate_identities": 0,
            "missing_identities": 0,
            "extra_identities": 0,
            "nonfinite_scores": 0,
            "historical_delta_overlap": 0,
            "historical_delta_identity_union": len(historical_keys | set(delta)),
        },
        "provenance_contract": {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "scorer_sha256": EXPECTED_SCORER,
            "selector": "true_s2_bm25_within_document_v2",
            "aggregation": "MAX",
            "score_direction": "higher = more relevant",
            "query_text_contract": "corrected canonical historical query text",
            "selected_evidence_contract": "current PV1 K20 worklist selected_chunk_ids",
            "gate": "PASS",
        },
        "labels_used": False,
        "fold0_used": False,
        "gpu_runs_this_phase": 0,
        "qwen_inference_this_phase": 0,
        "private_qwen_runs": 0,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("DELTA_FETCH_GATE: PASS")
    print("DELTA_QDOC_SHA_GATE: PASS")
    print(f"CURRENT_VALIDATION_K20: {rel(CURRENT_WORKLIST)}")
    print(f"CURRENT_VALIDATION_K20_SHA256: {current_work_sha}")
    print(f"VALIDATION_QUERIES: {len(query_counts)}/5600")
    print(f"VALIDATION_QDOC: {len(merged_rows)}/112000")
    print(f"HISTORICAL_QDOC: {len(historical_keys)}/111816")
    print(f"DELTA_QDOC: {len(delta)}/184")
    print("HISTORICAL_DELTA_OVERLAP: 0")
    print("DUPLICATE_QDOC: 0")
    print("MISSING_QDOC: 0")
    print("EXTRA_QDOC: 0")
    print("NONFINITE_QWEN_SCORES: 0")
    print(f"COMPLETE_CACHE_SHA256: {complete_sha}")
    print("MERGED_QWEN_PROVENANCE_GATE: PASS")
    print("GPU_RUNS_THIS_PHASE: 0")
    print("QWEN_INFERENCE_THIS_PHASE: 0")
    print("PRIVATE_QWEN_RUNS: 0")
    print("PHASE_B_READINESS: READY_FOR_CPU_RESCUE_EVALUATION")
    print("NEXT_ACTION: STOP_AND_REQUEST_PHASE_B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
