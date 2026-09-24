"""CPU-only Phase A.5 audit for corrected-Qwen single-rescue.

This materializes only bounded worklists and provenance reports.  It never
loads Qwen, calls Modal/GPU, reads labels, evaluates Recall, or selects a
rescue policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "private_task1/experiments/qwen_single_rescue"
OUT_MISSING = OUT_DIR / "validation_missing_qdocs.jsonl"
OUT_VALIDATION_DELTA = OUT_DIR / "validation_delta_worklist.jsonl"
OUT_PRIVATE = OUT_DIR / "private_qwen_worklist.jsonl"
OUT_JSON = OUT_DIR / "phase_a5_report.json"
OUT_MD = OUT_DIR / "phase_a5_report.md"

VALIDATION_CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
VALIDATION_WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
VALIDATION_BGE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
PRIVATE_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIVATE_WORKLIST = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_bge_full_worklist.jsonl"
PRIVATE_BGE = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
PRIVATE_INPUT = ROOT / "private_task1/input/private-official.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"

HISTORICAL_QWEN = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
HISTORICAL_STATE = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json"
HISTORICAL_OLD = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/acc_c_full_old/predictions.jsonl"
FULLDOC_QWEN = ROOT / "artifacts/task1/full_doc_qwen_optimized_merged/full_doc_top200_qwen_optimized_32shards.jsonl"
FULLDOC_MANIFEST = ROOT / "artifacts/task1/full_doc_qwen_optimized_merged/full_doc_top200_qwen_optimized_32shards_manifest.json"
PARITY_WORKLIST = ROOT / "reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_historical_parity_256.jsonl"
PARITY_OUTPUT = ROOT / "reports/task1/full_document_legal_field_retrieval/modal_historical_parity_output.jsonl"
PARITY_RESULT = ROOT / "reports/task1/full_document_legal_field_retrieval/modal_historical_qwen_parity_256_replay_result.json"

EXPECTED_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EXPECTED_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EXPECTED_SCORER_SHA = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
EXPECTED_CONTRACT_SHA = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
EXPECTED_UNIVERSE_SHA = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"
EXPECTED_PARITY_WORKLIST_SHA = "11cc2f98e821a0e968d1c73da90c047a350f23a796827fafd4114fd26bd7cbfa"
HISTORICAL_THROUGHPUT = 36.351306519275624
HISTORICAL_RUNTIME_SECONDS = 35575.007443385

REPAIRED_DOCUMENTS = {
    "10533", "131890", "149317", "177151", "181693", "187338", "191261",
    "196918", "208668", "210808", "232489", "255762", "263763", "288457",
    "34810", "55497", "56098", "57978", "67660", "71014",
}


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
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"non-object JSONL row at {path}:{line_no}")
                yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = {}
    for qid, record in payload.items():
        if not isinstance(record, dict) or not isinstance(record.get("question"), str):
            raise RuntimeError(f"invalid question record: {path}:{qid}")
        questions[str(qid)] = record["question"]
    return questions


def load_candidates(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    query_ids: set[str] = set()
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        query_ids.add(qid)
        candidates = row.get("candidates", [])
        if len(candidates) != 20:
            raise RuntimeError(f"expected K20 at {path}:{qid}, got {len(candidates)}")
        for candidate in candidates:
            did = str(candidate["document_id"])
            key = (qid, did)
            if key in rows:
                raise RuntimeError(f"duplicate candidate identity: {path}:{key}")
            rows[key] = {
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": {str(k): int(v) for k, v in (candidate.get("source_ranks") or {}).items()},
            }
    if len(rows) != len(query_ids) * 20:
        raise RuntimeError(f"candidate K20 count mismatch in {path}")
    return rows


def load_worklist(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    query_ids: set[str] = set()
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        did = str(row["document_id"])
        key = (qid, did)
        if key in rows:
            raise RuntimeError(f"duplicate worklist identity: {path}:{key}")
        selected = [str(value) for value in row.get("selected_chunk_ids", [])]
        expected = int(row.get("expected_inference_units", len(selected)))
        if expected != len(selected) or len(selected) == 0 or len(set(selected)) != len(selected):
            raise RuntimeError(f"invalid selected chunks at {path}:{key}")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise RuntimeError(f"incompatible selector contract at {path}:{key}")
        query_ids.add(qid)
        rows[key] = {
            "query_id": qid,
            "document_id": did,
            "candidate_rank": int(row["candidate_rank"]),
            "source_ranks": {str(k): int(v) for k, v in (row.get("source_ranks") or {}).items()},
            "selected_chunk_ids": selected,
            "expected_inference_units": expected,
            "selector": str(row["selector"]),
            "aggregation": str(row["aggregation"]),
            "max_length": int(row.get("max_length", 8192)),
        }
    return rows, query_ids


def load_bge_ranks(path: Path) -> dict[tuple[str, str], int]:
    by_query: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in seen:
            raise RuntimeError(f"duplicate BGE identity: {path}:{key}")
        seen.add(key)
        by_query[key[0]].append(row)
    ranks: dict[tuple[str, str], int] = {}
    for qid, rows in by_query.items():
        ordered = sorted(rows, key=lambda row: (
            -float(row["bge_ft_score"]),
            -float(row["bge_base_score"]),
            str(row["document_id"]),
        ))
        for index, row in enumerate(ordered, 1):
            ranks[(qid, str(row["document_id"]))] = index
    return ranks


def load_historical(path: Path) -> tuple[dict[tuple[str, str], float], dict[str, set[str]]]:
    scores: dict[tuple[str, str], float] = {}
    docs_by_query: defaultdict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        for item in row.get("document_scores", []):
            did = str(item["doc_id"])
            key = (qid, did)
            if key in scores:
                raise RuntimeError(f"duplicate historical Qwen identity: {key}")
            score = float(item["score"])
            if not math.isfinite(score):
                raise RuntimeError(f"nonfinite historical score: {key}")
            scores[key] = score
            docs_by_query[qid].add(did)
    return scores, dict(docs_by_query)


def load_flat_scores(path: Path, score_fields: tuple[str, ...]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in read_jsonl(path):
        if "query_id" not in row or "document_id" not in row:
            continue
        field = next((name for name in score_fields if name in row), None)
        if field is None:
            continue
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in scores:
            raise RuntimeError(f"duplicate flat cache identity: {path}:{key}")
        value = float(row[field])
        if math.isfinite(value):
            scores[key] = value
    return scores


def load_selected_map(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    result = {}
    for row in read_jsonl(path):
        key = (str(row["query_id"]), str(row["document_id"]))
        result[key] = tuple(str(value) for value in row.get("selected_chunk_ids", []))
    return result


def natural_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def workload_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    units = [int(row["expected_inference_units"]) for row in rows]
    return {
        "qdocs": len(units),
        "chunk_units": sum(units),
        "min_chunks_per_qdoc": min(units) if units else 0,
        "median_chunks_per_qdoc": statistics.median(units) if units else 0,
        "max_chunks_per_qdoc": max(units) if units else 0,
    }


def compatible_parity_reuse(
    missing: list[tuple[str, str]],
    work: dict[tuple[str, str], dict[str, Any]],
    historical_scores: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Use only the completed same-contract 256-row parity artifact.

    The local parity manifest contains selected chunk IDs separately from the
    score output.  A score is reusable only when those IDs exactly equal the
    current PV1 worklist row.  The larger full-doc artifact is intentionally
    not used because it has no selected-chunk provenance for these rows.
    """
    replay = json.loads(PARITY_RESULT.read_text(encoding="utf-8"))
    parity_work = load_selected_map(PARITY_WORKLIST)
    parity_scores = load_flat_scores(PARITY_OUTPUT, ("score", "document_score", "expected_document_score"))
    contract_ok = (
        replay.get("status") == "COMPLETE"
        and replay.get("contract_sha256") == EXPECTED_CONTRACT_SHA
        and replay.get("resolved_revision") == EXPECTED_REVISION
        and replay.get("universe_sha256") == EXPECTED_UNIVERSE_SHA
        and replay.get("parity", {}).get("gate") == "PASS"
        and sha256(PARITY_WORKLIST) == EXPECTED_PARITY_WORKLIST_SHA
    )
    recovered: dict[tuple[str, str], float] = {}
    for key in missing:
        if not contract_ok or key not in parity_scores or key not in parity_work:
            continue
        if parity_work[key] == tuple(work[key]["selected_chunk_ids"]):
            recovered[key] = parity_scores[key]
    return recovered, {
        "path": rel(PARITY_OUTPUT),
        "worklist_path": rel(PARITY_WORKLIST),
        "contract_compatible": contract_ok,
        "score_rows": len(parity_scores),
        "matching_missing_rows": len(recovered),
        "status": "SAFE_EXACT_COMPATIBLE_SUBSET" if contract_ok else "INCOMPATIBLE",
    }


def make_worklist_row(row: dict[str, Any], question: str) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "document_id": row["document_id"],
        "question": question,
        "candidate_rank": int(row["candidate_rank"]),
        "source_ranks": row["source_ranks"],
        "selected_chunk_ids": list(row["selected_chunk_ids"]),
        "expected_inference_units": int(row["expected_inference_units"]),
        "selector": "true_s2_bm25_within_document_v2",
        "aggregation": "MAX",
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "max_length": 8192,
        "dtype": "bfloat16",
        "labels_used": False,
    }


def main() -> int:
    validation_candidates = load_candidates(VALIDATION_CANDIDATES)
    private_candidates = load_candidates(PRIVATE_CANDIDATES)
    validation_work, validation_qids = load_worklist(VALIDATION_WORKLIST)
    private_work, private_qids = load_worklist(PRIVATE_WORKLIST)
    validation_bge_rank = load_bge_ranks(VALIDATION_BGE)
    private_bge_rank = load_bge_ranks(PRIVATE_BGE)
    validation_questions = load_questions(TRAIN)
    private_questions = load_questions(PRIVATE_INPUT)
    historical_scores, historical_docs = load_historical(HISTORICAL_QWEN)

    if len(validation_candidates) != 112000 or len(validation_work) != 112000 or len(validation_qids) != 5600:
        raise RuntimeError("validation PV1 K20 universe is not exactly 5600/112000")
    if len(private_candidates) != 41600 or len(private_work) != 41600 or len(private_qids) != 2080:
        raise RuntimeError("Private PV1 K20 universe is not exactly 2080/41600")
    if not validation_qids.issubset(validation_questions):
        raise RuntimeError("validation question text is missing for one or more qids")
    if not private_qids.issubset(private_questions):
        raise RuntimeError("Private question text is missing for one or more qids")

    missing_keys = sorted(
        [key for key in validation_work if key not in historical_scores],
        key=lambda key: (natural_key(key[0]), int(validation_work[key]["candidate_rank"]), natural_key(key[1])),
    )
    if len(missing_keys) != 184 or len({qid for qid, _ in missing_keys}) != 157:
        raise RuntimeError(f"expected exactly 184 missing qdocs/157 queries, got {len(missing_keys)}/{len({qid for qid, _ in missing_keys})}")

    missing_records = []
    root_causes = Counter()
    for key in missing_keys:
        qid, did = key
        row = validation_work[key]
        candidate = validation_candidates[key]
        repaired = did in REPAIRED_DOCUMENTS
        if repaired:
            root_cause = "PV1_NEW_OR_REPAIRED_DOC"
        elif qid not in historical_docs or did not in historical_docs[qid]:
            root_cause = "CANDIDATE_UNIVERSE_DRIFT"
        else:
            root_cause = "HISTORICAL_QWEN_K77_MISS"
        root_causes[root_cause] += 1
        missing_records.append({
            "query_id": qid,
            "document_id": did,
            "question": validation_questions[qid],
            "current_candidate_rank": int(row["candidate_rank"]),
            "is_repaired_pv1_document": repaired,
            "root_cause": root_cause,
            "dense_rank": row["source_ranks"].get("dense"),
            "bm25_rank": row["source_ranks"].get("bm25"),
            "knn_rank": row["source_ranks"].get("knn_word"),
            "current_bge_rank": validation_bge_rank.get(key),
            "selected_chunk_ids": list(row["selected_chunk_ids"]),
            "selected_chunk_count": int(row["expected_inference_units"]),
            "selector": row["selector"],
            "aggregation": row["aggregation"],
        })

    parity_recovered, parity_audit = compatible_parity_reuse(missing_keys, validation_work, historical_scores)
    remaining_validation = [key for key in missing_keys if key not in parity_recovered]
    validation_delta_rows = [
        make_worklist_row(validation_work[key], validation_questions[key[0]])
        for key in remaining_validation
    ]
    validation_delta_rows.sort(key=lambda row: (natural_key(row["query_id"]), int(row["candidate_rank"]), natural_key(row["document_id"])))

    private_question_to_validation_qids: defaultdict[str, list[str]] = defaultdict(list)
    for qid in validation_qids:
        private_question_to_validation_qids[validation_questions[qid]].append(qid)

    private_reusable: dict[tuple[str, str], float] = {}
    private_reuse_sources: Counter[str] = Counter()
    private_question_overlap = 0
    for key, row in private_work.items():
        qid, did = key
        source_qids = private_question_to_validation_qids.get(private_questions[qid], [])
        if source_qids:
            private_question_overlap += 1
        candidates: list[tuple[str, float]] = []
        for source_qid in source_qids:
            source_key = (source_qid, did)
            if source_key not in historical_scores or source_key not in validation_work:
                continue
            if tuple(validation_work[source_key]["selected_chunk_ids"]) != tuple(row["selected_chunk_ids"]):
                continue
            candidates.append(("canonical_historical", historical_scores[source_key]))
        if candidates and len({value for _, value in candidates}) == 1:
            private_reusable[key] = candidates[0][1]
            private_reuse_sources[candidates[0][0]] += 1

    private_rows = [
        make_worklist_row(row, private_questions[row["query_id"]])
        for row in private_work.values()
    ]
    private_rows.sort(key=lambda row: (natural_key(row["query_id"]), int(row["candidate_rank"]), natural_key(row["document_id"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_MISSING, missing_records)
    write_jsonl(OUT_VALIDATION_DELTA, validation_delta_rows)
    write_jsonl(OUT_PRIVATE, private_rows)

    validation_stats = workload_stats(validation_delta_rows)
    private_new_rows = [row for row in private_rows if (str(row["query_id"]), str(row["document_id"])) not in private_reusable]
    private_new_stats = workload_stats(private_new_rows)
    private_all_stats = workload_stats(private_rows)
    historical_state = json.loads(HISTORICAL_STATE.read_text(encoding="utf-8"))
    full_manifest = json.loads(FULLDOC_MANIFEST.read_text(encoding="utf-8"))
    old_flat = load_flat_scores(HISTORICAL_OLD, ("score", "document_score"))
    fulldoc_flat = load_flat_scores(FULLDOC_QWEN, ("score", "document_score"))
    missing_old_hits = sum(key in old_flat for key in missing_keys)
    missing_fulldoc_hits = sum(key in fulldoc_flat for key in missing_keys)

    provenance_pass = (
        historical_state.get("status") == "PASS_PREDICTIONS_FROZEN"
        and historical_state.get("resolved_revision") == EXPECTED_REVISION
        and historical_state.get("model_revision_verified") is True
        and historical_state.get("prediction_sha256") == sha256(HISTORICAL_QWEN)
        and historical_state.get("labels_loaded") is False
        and historical_state.get("fold0_used") is False
        and full_manifest.get("status") == "COMPLETE"
    )
    if not provenance_pass:
        readiness = "BLOCKED_PROVENANCE"
    elif not remaining_validation:
        readiness = "VALIDATION_CACHE_COMPLETE"
    else:
        readiness = "READY_FOR_VALIDATION_DELTA_GPU"

    rate = HISTORICAL_THROUGHPUT
    result = {
        "status": "CORRECTED_QWEN_SINGLE_RESCUE_PHASE_A5_COMPLETE",
        "execution": {"gpu_runs": 0, "modal_runs": 0, "qwen_inference_runs": 0, "labels_used": False, "recall_evaluated": False, "submission_created": False},
        "canonical_qwen": {"path": rel(HISTORICAL_QWEN), "sha256": sha256(HISTORICAL_QWEN), "model": EXPECTED_MODEL, "revision": EXPECTED_REVISION, "scorer_sha256": EXPECTED_SCORER_SHA, "selector": "true_s2_bm25_within_document_v2", "aggregation": "MAX", "provenance_gate": "PASS" if provenance_pass else "FAIL"},
        "validation_missing": {"qdocs": len(missing_keys), "queries": len({qid for qid, _ in missing_keys}), "root_causes": dict(sorted(root_causes.items())), "recovered_from_safe_cache": len(parity_recovered), "remaining_need_inference": len(remaining_validation), "remaining_worklist": rel(OUT_VALIDATION_DELTA), "workload": validation_stats},
        "cache_search": {"same_contract_parity": parity_audit, "historical_old_rows_matching_missing": missing_old_hits, "full_doc_rows_matching_missing": missing_fulldoc_hits, "full_doc_status": "NOT_REUSABLE_WITHOUT_SELECTED_CHUNK_PROVENANCE", "historical_old_status": "NOT_REUSED_PROVENANCE_UNPROVEN"},
        "private": {"qdocs": len(private_work), "queries": len(private_qids), "worklist": rel(OUT_PRIVATE), "workload_all": private_all_stats, "exact_question_overlap_qdocs": private_question_overlap, "exact_reusable_qdocs": len(private_reusable), "exact_reuse_sources": dict(private_reuse_sources), "need_new_inference_qdocs": private_new_stats["qdocs"], "need_new_inference_units": private_new_stats["chunk_units"], "workload_new": private_new_stats},
        "runtime_estimate_only": {"historical_gpu": "NVIDIA A10", "historical_full_chunks": historical_state.get("full_chunks_scored"), "historical_runtime_seconds": HISTORICAL_RUNTIME_SECONDS, "historical_chunks_per_second": rate, "validation_delta_seconds": validation_stats["chunk_units"] / rate if rate else None, "private_new_seconds": private_new_stats["chunk_units"] / rate if rate else None, "basis": "same Qwen model/revision/selector/MAX/batch-1 corrected run; excludes fresh startup, checkpoint and Volume commit overhead", "cost": "NOT_ESTIMATED"},
        "phase_b_readiness": readiness,
        "next_action": "Run only the bounded validation delta Qwen inference after explicit GPU approval; do not score Private or open rescue policy yet." if readiness == "READY_FOR_VALIDATION_DELTA_GPU" else "Open Phase B only after validation cache reaches 112000/112000." if readiness == "VALIDATION_CACHE_COMPLETE" else "Repair provenance before any Phase B.",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = f"""# Corrected Qwen single-rescue — Phase A.5

CPU-only materialization/provenance audit. No Qwen inference, GPU, Modal, labels, Recall/Precision, rescue selection, thresholding, or submission creation.

## Canonical cache

- `{rel(HISTORICAL_QWEN)}`
- SHA256 `{sha256(HISTORICAL_QWEN)}`
- Model/revision `{EXPECTED_MODEL}` / `{EXPECTED_REVISION}`
- Selector `true_s2_bm25_within_document_v2`; aggregation `MAX`; provenance `{result['canonical_qwen']['provenance_gate']}`

## Validation missing gap

- Missing: `{len(missing_keys)}` q-docs across `{len({qid for qid, _ in missing_keys})}` queries.
- Root causes: `{dict(sorted(root_causes.items()))}`.
- Safely recovered from compatible cache: `{len(parity_recovered)}/{len(missing_keys)}`.
- Remaining inference: `{len(remaining_validation)}` q-docs / `{validation_stats['chunk_units']}` chunk units.
- Chunk range: `{validation_stats['min_chunks_per_qdoc']}` / `{validation_stats['median_chunks_per_qdoc']}` / `{validation_stats['max_chunks_per_qdoc']}` min/median/max.
- Worklist: `{rel(OUT_VALIDATION_DELTA)}`.

## Private future workload

- Current PV1 K20: `{len(private_qids)}` queries / `{len(private_work)}` q-docs.
- Worklist: `{rel(OUT_PRIVATE)}`.
- All work: `{private_all_stats['chunk_units']}` chunk units; `{private_all_stats['min_chunks_per_qdoc']}` / `{private_all_stats['median_chunks_per_qdoc']}` / `{private_all_stats['max_chunks_per_qdoc']}` min/median/max.
- Exact question-text + document + selected-chunk reusable q-docs: `{len(private_reusable)}/{len(private_work)}`.
- New inference required: `{private_new_stats['qdocs']}` q-docs / `{private_new_stats['chunk_units']}` chunk units.

## Runtime estimate only

- Comparable corrected run: NVIDIA A10, `{HISTORICAL_RUNTIME_SECONDS}` seconds, `{rate}` selected chunks/sec.
- Validation delta estimate: `{validation_stats['chunk_units'] / rate:.2f}` seconds, scorer-only extrapolation.
- Private new-work estimate: `{private_new_stats['chunk_units'] / rate:.2f}` seconds, scorer-only extrapolation.
- These estimates exclude startup, checkpoint, and Volume commit overhead. No dollar cost estimated.

## Gate

`PHASE_B_READINESS = {readiness}`

Next action: {result['next_action']}
"""
    OUT_MD.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "VALIDATION_MISSING_QDOC": len(missing_keys),
        "VALIDATION_MISSING_QUERIES": len({qid for qid, _ in missing_keys}),
        "PV1_NEW_OR_REPAIRED_DOC": root_causes.get("PV1_NEW_OR_REPAIRED_DOC", 0),
        "CANDIDATE_UNIVERSE_DRIFT": root_causes.get("CANDIDATE_UNIVERSE_DRIFT", 0),
        "HISTORICAL_QWEN_K77_MISS": root_causes.get("HISTORICAL_QWEN_K77_MISS", 0),
        "IDENTITY_JOIN_ISSUE": root_causes.get("IDENTITY_JOIN_ISSUE", 0),
        "VALIDATION_MISSING_RECOVERED_FROM_CACHE": len(parity_recovered),
        "VALIDATION_REMAINING_NEED_INFERENCE": len(remaining_validation),
        "VALIDATION_REMAINING_CHUNK_UNITS": validation_stats["chunk_units"],
        "PRIVATE_QDOC_WORK": len(private_work),
        "PRIVATE_CHUNK_UNITS": private_all_stats["chunk_units"],
        "PRIVATE_EXACT_REUSABLE_QDOC": len(private_reusable),
        "PRIVATE_NEED_NEW_INFERENCE_QDOC": private_new_stats["qdocs"],
        "PRIVATE_NEED_NEW_INFERENCE_UNITS": private_new_stats["chunk_units"],
        "VALIDATION_DELTA_RUNTIME_ESTIMATE": validation_stats["chunk_units"] / rate,
        "PRIVATE_RUNTIME_ESTIMATE": private_new_stats["chunk_units"] / rate,
        "GPU_RUNS": 0,
        "MODAL_RUNS": 0,
        "PHASE_B_READINESS": readiness,
        "report": rel(OUT_MD),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
