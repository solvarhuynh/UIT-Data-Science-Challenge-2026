"""Phase A audit for cached corrected-Qwen single-rescue feasibility.

Read-only with respect to scoring artifacts: no labels, inference, GPU, Modal,
Recall evaluation, thresholding, reranking, or submission generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

CANONICAL_QWEN = ROOT / "artifacts/task1/full_doc_qwen_optimized_merged/full_doc_top200_qwen_optimized_32shards.jsonl"
CANONICAL_MANIFEST = ROOT / "artifacts/task1/full_doc_qwen_optimized_merged/full_doc_top200_qwen_optimized_32shards_manifest.json"
CANONICAL_SHARD_MANIFEST_DIR = ROOT / "artifacts/task1/full_doc_qwen_optimized_manifests/optimized_production"
CANONICAL_SHARD_OUTPUT_DIR = ROOT / "artifacts/task1/full_doc_qwen_optimized_outputs/optimized_production"
HISTORICAL_QWEN = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
HISTORICAL_STATE = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json"
VALIDATION_CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
VALIDATION_BGE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
PRIVATE_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIVATE_INCUMBENT = ROOT / "private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.json"
OUT_JSON = ROOT / "private_task1/reports/task1/corrected_qwen_single_rescue_phase_a.json"
OUT_MD = ROOT / "private_task1/reports/task1/corrected_qwen_single_rescue_phase_a.md"

EXPECTED_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EXPECTED_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EXPECTED_SCORER_SHA = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
EXPECTED_CONTRACT_SHA = "ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e"
EXPECTED_UNIVERSE_SHA = "28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                yield json.loads(line), line_no


def load_current_candidates(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row, line_no in read_jsonl(path):
        qid = str(row["query_id"])
        docs = []
        for candidate in row.get("candidates", []):
            did = str(candidate["document_id"])
            key = (qid, did)
            if key in by_key:
                raise RuntimeError(f"duplicate candidate identity at {path}:{line_no}: {key}")
            source_ranks = {str(k): int(v) for k, v in (candidate.get("source_ranks") or {}).items()}
            item = {
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": source_ranks,
                "source_support": len(source_ranks),
            }
            docs.append(item)
            by_key[key] = item
        if qid in by_query:
            raise RuntimeError(f"duplicate query row at {path}:{line_no}: {qid}")
        by_query[qid] = docs
    return by_query, by_key


def load_scores(path: Path) -> tuple[dict[tuple[str, str], float], dict[str, int], dict[str, Any]]:
    scores: dict[tuple[str, str], float] = {}
    qdoc_counts: Counter[str] = Counter()
    score_values: list[float] = []
    duplicate = 0
    nonfinite = 0
    fields: set[str] = set()
    for row, line_no in read_jsonl(path):
        fields.update(row.keys())
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in scores:
            duplicate += 1
            continue
        value = row.get("score")
        if not finite(value):
            nonfinite += 1
        else:
            score_values.append(float(value))
        scores[key] = float(value) if finite(value) else float("nan")
        qdoc_counts[key[0]] += 1
    return scores, dict(qdoc_counts), {
        "rows": len(scores),
        "query_count": len(qdoc_counts),
        "duplicate_keys": duplicate,
        "nonfinite_scores": nonfinite,
        "fields": sorted(fields),
        "score_min": min(score_values) if score_values else None,
        "score_median": statistics.median(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
    }


def scan_historical(path: Path) -> dict[str, Any]:
    keys: set[tuple[str, str]] = set()
    qids: set[str] = set()
    scores: list[float] = []
    duplicates = 0
    nonfinite = 0
    depths: set[int] = set()
    for row, _line_no in read_jsonl(path):
        qid = str(row["query_id"])
        qids.add(qid)
        document_scores = row.get("document_scores", [])
        depths.add(len(document_scores))
        for item in document_scores:
            key = (qid, str(item["doc_id"]))
            if key in keys:
                duplicates += 1
            keys.add(key)
            value = item.get("score")
            if not finite(value):
                nonfinite += 1
            else:
                scores.append(float(value))
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "query_count": len(qids),
        "qdoc_count": len(keys),
        "candidate_depths": sorted(depths),
        "score_field": "document_scores[].score",
        "duplicate_keys": duplicates,
        "nonfinite_scores": nonfinite,
        "score_min": min(scores),
        "score_median": statistics.median(scores),
        "score_max": max(scores),
        "scope": "F1-F4 corrected historical K77; labels_loaded=false; fold0_used=false",
    }


def load_historical_score_map(path: Path) -> tuple[dict[tuple[str, str], float], dict[str, int], dict[str, Any]]:
    """Load the nested historical K77 document scores for exact-key joins."""

    scores: dict[tuple[str, str], float] = {}
    qdoc_counts: Counter[str] = Counter()
    values: list[float] = []
    duplicates = 0
    nonfinite = 0
    for row, _line_no in read_jsonl(path):
        qid = str(row["query_id"])
        for item in row.get("document_scores", []):
            key = (qid, str(item["doc_id"]))
            if key in scores:
                duplicates += 1
                continue
            value = item.get("score")
            if not finite(value):
                nonfinite += 1
                scores[key] = float("nan")
            else:
                scores[key] = float(value)
                values.append(float(value))
            qdoc_counts[qid] += 1
    return scores, dict(qdoc_counts), {
        "rows": len(scores),
        "query_count": len(qdoc_counts),
        "duplicate_keys": duplicates,
        "nonfinite_scores": nonfinite,
        "score_min": min(values),
        "score_median": statistics.median(values),
        "score_max": max(values),
    }


def verify_canonical_manifest() -> dict[str, Any]:
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    checked_shards = []
    for entry in manifest.get("shards", []):
        output = ROOT / str(entry["output_path"])
        shard_manifest = ROOT / str(entry["manifest_path"])
        output_sha = sha256(output) if output.is_file() else None
        shard_payload = json.loads(shard_manifest.read_text(encoding="utf-8")) if shard_manifest.is_file() else {}
        checked_shards.append({
            "shard_id": entry.get("shard_id"),
            "status": entry.get("status"),
            "output_exists": output.is_file(),
            "manifest_exists": shard_manifest.is_file(),
            "output_sha_match": output_sha == entry.get("output_sha256") == entry.get("manifest_output_sha256"),
            "model": shard_payload.get("model"),
            "revision": shard_payload.get("revision"),
            "scorer_sha256": shard_payload.get("scorer_sha256"),
            "contract_sha256": shard_payload.get("contract_sha256"),
            "universe_sha256": shard_payload.get("universe_sha256"),
        })
    model_pass = all(item["model"] == EXPECTED_MODEL for item in checked_shards)
    revision_pass = all(item["revision"] == EXPECTED_REVISION for item in checked_shards)
    scorer_pass = all(item["scorer_sha256"] == EXPECTED_SCORER_SHA for item in checked_shards)
    contract_pass = all(item["contract_sha256"] == EXPECTED_CONTRACT_SHA for item in checked_shards)
    universe_pass = all(item["universe_sha256"] == EXPECTED_UNIVERSE_SHA for item in checked_shards)
    output_pass = all(item["output_sha_match"] for item in checked_shards)
    merged_sha = sha256(CANONICAL_QWEN)
    pass_gate = (
        manifest.get("status") == "COMPLETE"
        and manifest.get("shard_count") == 32
        and manifest.get("duplicate_identities") == 0
        and manifest.get("nonfinite_scores") == 0
        and manifest.get("merged_sha256") == merged_sha
        and len(checked_shards) == 32
        and output_pass
        and model_pass
        and revision_pass
        and scorer_pass
        and contract_pass
        and universe_pass
    )
    return {
        "manifest_path": rel(CANONICAL_MANIFEST),
        "manifest_sha256": sha256(CANONICAL_MANIFEST),
        "manifest_status": manifest.get("status"),
        "merged_sha256_recorded": manifest.get("merged_sha256"),
        "merged_sha256_actual": merged_sha,
        "shard_count": manifest.get("shard_count"),
        "merged_q_doc_rows": manifest.get("merged_q_doc_rows"),
        "unique_query_document_identities": manifest.get("unique_query_document_identities"),
        "duplicate_identities": manifest.get("duplicate_identities"),
        "nonfinite_scores": manifest.get("nonfinite_scores"),
        "all_shards_output_sha_match": output_pass,
        "model_pass": model_pass,
        "revision_pass": revision_pass,
        "scorer_pass": scorer_pass,
        "contract_pass": contract_pass,
        "universe_pass": universe_pass,
        "shards": checked_shards,
        "gate": "PASS" if pass_gate else "FAIL",
    }


def rank_current_rrf(docs: list[dict[str, Any]], bge_scores: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    joined = []
    for meta in docs:
        key = (meta["query_id"], meta["document_id"])
        score = bge_scores[key]
        joined.append({**meta, **score})
    bge_order = [str(row["document_id"]) for row in sorted(
        joined,
        key=lambda row: (-float(row["bge_ft_score"]), -float(row["bge_base_score"]), int(row["candidate_rank"]), str(row["document_id"])),
    )]
    bge_rank = {did: index for index, did in enumerate(bge_order, 1)}
    weights = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
    rank_maps = {}
    rrf = {}
    for row in joined:
        did = str(row["document_id"])
        source_ranks = {str(k): int(v) for k, v in (row.get("source_ranks") or {}).items()}
        source_ranks["bge"] = bge_rank[did]
        rank_maps[did] = source_ranks
        rrf[did] = sum(weight / (2 + source_ranks[source]) for source, weight in weights.items() if source in source_ranks)
    return sorted(rrf, key=lambda did: (-rrf[did], *(rank_maps[did].get(source, 10**9) for source in weights), str(did)))


def rescue_feasibility(
    current: dict[str, list[dict[str, Any]]],
    qwen_keys: set[tuple[str, str]],
    incumbent: dict[str, list[str]],
) -> dict[str, Any]:
    per_query_covered = 0
    full_rescue = 0
    partial_rescue = 0
    zero_rescue = 0
    for qid, docs in current.items():
        top5 = set(incumbent[qid])
        eligible = [
            doc for doc in docs
            if 6 <= int(doc["candidate_rank"]) <= 20
            and doc["document_id"] not in top5
            and (qid, doc["document_id"]) in qwen_keys
        ]
        if eligible:
            per_query_covered += 1
    return {
        "queries_with_rescue_candidate": per_query_covered,
        "definition": "candidate_rank 6-20, outside incumbent Top5, finite cached Qwen score present",
    }


def coverage(current: dict[str, list[dict[str, Any]]], qwen_keys: set[tuple[str, str]]) -> dict[str, Any]:
    total = sum(len(docs) for docs in current.values())
    covered = sum((qid, str(doc["document_id"])) in qwen_keys for qid, docs in current.items() for doc in docs)
    by_query = {
        qid: sum((qid, str(doc["document_id"])) in qwen_keys for doc in docs)
        for qid, docs in current.items()
    }
    bands = {}
    for low, high, label in ((1, 5, "rank_1_5"), (6, 10, "rank_6_10"), (11, 20, "rank_11_20")):
        band_docs = [doc for docs in current.values() for doc in docs if low <= int(doc["candidate_rank"]) <= high]
        band_covered = sum((str(doc["query_id"]), str(doc["document_id"])) in qwen_keys for doc in band_docs)
        bands[label] = {
            "total": len(band_docs),
            "covered": band_covered,
            "missing": len(band_docs) - band_covered,
            "coverage_pct": 100.0 * band_covered / len(band_docs) if band_docs else 0.0,
        }
    return {
        "qdoc_total": total,
        "qwen_covered": covered,
        "qwen_missing": total - covered,
        "coverage_pct": 100.0 * covered / total if total else 0.0,
        "queries_total": len(current),
        "queries_fully_covered": sum(count == len(current[qid]) for qid, count in by_query.items()),
        "queries_partially_covered": sum(0 < count < len(current[qid]) for qid, count in by_query.items()),
        "queries_zero_coverage": sum(count == 0 for count in by_query.values()),
        "rank_bands": bands,
    }


def main() -> int:
    manifest_audit = verify_canonical_manifest()
    current_full_scores, current_full_qcounts, current_full_stats = load_scores(CANONICAL_QWEN)
    historical_scores, historical_qcounts, historical_stats = load_historical_score_map(HISTORICAL_QWEN)
    validation, validation_keys = load_current_candidates(VALIDATION_CANDIDATES)
    private, private_keys = load_current_candidates(PRIVATE_CANDIDATES)

    # Validation incumbent is reconstructed from frozen current BGE scores only;
    # this is structural ranking context, not label evaluation.
    bge_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row, _line_no in read_jsonl(VALIDATION_BGE):
        bge_rows[(str(row["query_id"]), str(row["document_id"]))] = row
    validation_incumbent = {
        qid: rank_current_rrf(docs, bge_rows)[:5]
        for qid, docs in validation.items()
    }
    private_payload = json.loads(PRIVATE_INCUMBENT.read_text(encoding="utf-8"))
    private_incumbent = {str(qid): [str(value) for value in row["answer"]] for qid, row in private_payload.items()}

    # The corrected K77 cache is the usable rescue artifact for current PV1.
    # The newer full-doc residual artifact is audited separately but is not
    # silently substituted: its universe intentionally excludes the PV1 pool.
    validation_cov = coverage(validation, set(historical_scores))
    private_cov = coverage(private, set(historical_scores))
    validation_rescue = rescue_feasibility(validation, set(historical_scores), validation_incumbent)
    private_rescue = rescue_feasibility(private, set(historical_scores), private_incumbent)

    historical = scan_historical(HISTORICAL_QWEN)
    historical_state = json.loads(HISTORICAL_STATE.read_text(encoding="utf-8"))
    historical["model"] = historical_state.get("model")
    historical["revision"] = historical_state.get("resolved_revision")
    historical["scorer_sha256"] = EXPECTED_SCORER_SHA
    historical["provenance_status"] = "VALID_BUT_STALE_K77"

    qwen_query_ids = set(historical_qcounts)
    current_full_validation_cov = coverage(validation, set(current_full_scores))
    current_full_private_cov = coverage(private, set(current_full_scores))
    validation_query_ids = set(validation)
    private_query_ids = set(private)
    provenance_pass = (
        historical_state.get("status") == "PASS_PREDICTIONS_FROZEN"
        and historical_state.get("model_revision_verified") is True
        and historical_state.get("resolved_revision") == EXPECTED_REVISION
        and historical_state.get("prediction_sha256") == sha256(HISTORICAL_QWEN)
        and historical_state.get("labels_loaded") is False
        and historical_state.get("fold0_used") is False
        and historical_state.get("errors") == 0
        and historical_stats["duplicate_keys"] == 0
        and historical_stats["nonfinite_scores"] == 0
    )
    private_status = "PRESENT" if private_cov["qwen_covered"] == private_cov["qdoc_total"] else "PARTIAL" if private_cov["qwen_covered"] else "ABSENT"
    if provenance_pass and validation_cov["qwen_covered"] == validation_cov["qdoc_total"] and private_status == "PRESENT":
        deployability = "CACHE_READY_BOTH"
        next_action = "Prepare Phase B one-slot rank5 Qwen rescue with strict OOF, without running it in this phase."
    elif provenance_pass and validation_cov["qwen_covered"] == validation_cov["qdoc_total"] and private_status == "ABSENT":
        deployability = "CACHE_READY_VALIDATION_ONLY"
        next_action = "Run Phase B validation-only strict OOF rescue; request new Private Qwen inference only after a validation PASS."
    elif provenance_pass:
        deployability = "CACHE_PARTIAL"
        next_action = "Do not open rescue policy; identify and fill the missing cached Qwen identities before any Phase B."
    else:
        deployability = "CACHE_INVALID"
        next_action = "Close the corrected-Qwen rescue branch because provenance or score integrity failed."

    result = {
        "status": "CORRECTED_QWEN_SINGLE_RESCUE_PHASE_A_COMPLETE",
        "canonical_artifact": {
            "path": rel(HISTORICAL_QWEN),
            "sha256": sha256(HISTORICAL_QWEN),
            "query_count": len(qwen_query_ids),
            "qdoc_count": historical_stats["rows"],
            "candidate_depth": "K77 (77 document scores/query)",
            "score_field": "document_scores[].score",
            "fields": ["query_id", "predicted_doc_ids", "document_scores"],
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "scorer_sha256": EXPECTED_SCORER_SHA,
            "selector": "true_s2_bm25_within_document_v2 / select_true_s2_prepared",
            "aggregation": "MAX over selected chunks",
            "score_direction": "higher = more relevant",
            "scope": "canonical corrected historical artifact for current F1-F4 PV1; no Private qids",
            "provenance_status": "VALID",
            "run_state": historical_state,
            "score_stats": historical_stats,
        },
        "other_corrected_qwen_candidates": [
            historical,
            {
                "path": rel(CANONICAL_QWEN),
                "sha256": sha256(CANONICAL_QWEN),
                "scope": "complete 32-shard full-document rank<=200 residual; current PV1 overlap is not its intended universe",
                "status": "VALID_BUT_NOT_CURRENT_K20_CANONICAL",
                "manifest_audit": manifest_audit,
                "current_validation_coverage": current_full_validation_cov,
                "current_private_coverage": current_full_private_cov,
                "score_stats": current_full_stats,
            },
            {
                "path": "reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_historical_parity_256.jsonl",
                "sha256": sha256(ROOT / "reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_historical_parity_256.jsonl"),
                "scope": "256-row parity subset",
                "status": "PARTIAL",
            },
            {
                "path": "reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_current_canary_16.jsonl",
                "sha256": sha256(ROOT / "reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_current_canary_16.jsonl"),
                "scope": "16-q-doc canary subset",
                "status": "PARTIAL",
            },
        ],
        "join_key": {
            "fields": ["query_id", "document_id"],
            "row_order_join_used": False,
            "validation_duplicate_keys": 0,
            "private_duplicate_keys": 0,
            "qwen_duplicate_keys": historical_stats["duplicate_keys"],
        },
        "validation_coverage": validation_cov,
        "private_coverage": private_cov,
        "private_qwen_cache_status": private_status,
        "validation_query_ids_equal_qwen_query_ids": validation_query_ids == qwen_query_ids,
        "private_query_ids_overlap_qwen": len(private_query_ids & qwen_query_ids),
        "rescue_feasibility": {
            "validation_queries_with_rescue_candidate": validation_rescue["queries_with_rescue_candidate"],
            "private_queries_with_rescue_candidate": private_rescue["queries_with_rescue_candidate"],
            "definition": validation_rescue["definition"],
            "candidate_selection_performed": False,
            "threshold_applied": False,
            "labels_used": False,
        },
        "signal_compatibility": {
            "direction": "higher = more relevant",
            "evidence": "historical corrected run_state and full-doc provenance contract",
            "normalization_applied": False,
            "score_min": historical_stats["score_min"],
            "score_median": historical_stats["score_median"],
            "score_max": historical_stats["score_max"],
            "finite_scores": historical_stats["nonfinite_scores"] == 0,
        },
        "provenance_gate": "PASS" if provenance_pass else "FAIL",
        "deployability_class": deployability,
        "execution": {
            "gpu_runs": 0,
            "modal_runs": 0,
            "inference_runs": 0,
            "recall_evaluated": False,
            "precision_evaluated": False,
            "labels_used": False,
            "submission_created": False,
            "incumbent_changed": False,
        },
        "next_action": next_action,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# Corrected Qwen single-rescue — Phase A",
        "",
        "CPU-only structural/cache audit. No labels, Recall, Precision, inference, GPU, Modal, policy selection, or submission creation.",
        "",
        "## Canonical artifact",
        "",
        f"- Path: `{rel(HISTORICAL_QWEN)}`.",
        f"- SHA256: `{sha256(HISTORICAL_QWEN)}`; provenance gate: **{result['provenance_gate']}**.",
        f"- Scope: `{len(qwen_query_ids)}` queries, `{historical_stats['rows']}` q-doc rows, K77 (77/query); score field `document_scores[].score`.",
        f"- Model/revision: `{EXPECTED_MODEL}` / `{EXPECTED_REVISION}`; selector `true_s2_bm25_within_document_v2`; aggregation `MAX`.",
        "- The newer 32-shard full-doc residual artifact is complete but intentionally does not cover the current PV1 K20 universe; it is not silently used for rescue joins.",
        "",
        "## Coverage by exact `(query_id, document_id)` join",
        "",
        f"- Validation: `{validation_cov['qwen_covered']}/{validation_cov['qdoc_total']}` ({validation_cov['coverage_pct']:.6f}%), full queries `{validation_cov['queries_fully_covered']}`, partial `{validation_cov['queries_partially_covered']}`, zero `{validation_cov['queries_zero_coverage']}`.",
        f"- Private: `{private_cov['qwen_covered']}/{private_cov['qdoc_total']}` ({private_cov['coverage_pct']:.6f}%), full queries `{private_cov['queries_fully_covered']}`, partial `{private_cov['queries_partially_covered']}`, zero `{private_cov['queries_zero_coverage']}`; cache **{private_status}**.",
        f"- Validation rank bands: `{validation_cov['rank_bands']}`.",
        f"- Private rank bands: `{private_cov['rank_bands']}`.",
        "",
        "## Structural rescue feasibility",
        "",
        f"- Validation queries with at least one eligible rank6–20 cached-Qwen candidate outside incumbent Top5: `{validation_rescue['queries_with_rescue_candidate']}`.",
        f"- Private queries with at least one eligible candidate: `{private_rescue['queries_with_rescue_candidate']}`.",
        "- No candidate was selected and no threshold/policy was applied.",
        "",
        f"## Decision: `{deployability}`",
        "",
        f"Next action: {next_action}",
    ]
    OUT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "CANONICAL_QWEN_ARTIFACT": rel(HISTORICAL_QWEN),
        "CANONICAL_QWEN_SHA256": sha256(HISTORICAL_QWEN),
        "QWEN_PROVENANCE_GATE": result["provenance_gate"],
        "VALIDATION_K20_QDOC": validation_cov["qdoc_total"],
        "VALIDATION_QWEN_COVERED": validation_cov["qwen_covered"],
        "VALIDATION_COVERAGE_PCT": validation_cov["coverage_pct"],
        "PRIVATE_K20_QDOC": private_cov["qdoc_total"],
        "PRIVATE_QWEN_COVERED": private_cov["qwen_covered"],
        "PRIVATE_COVERAGE_PCT": private_cov["coverage_pct"],
        "PRIVATE_QWEN_CACHE_STATUS": private_status,
        "VALIDATION_QUERIES_WITH_RESCUE_CANDIDATE": validation_rescue["queries_with_rescue_candidate"],
        "PRIVATE_QUERIES_WITH_RESCUE_CANDIDATE": private_rescue["queries_with_rescue_candidate"],
        "DEPLOYABILITY_CLASS": deployability,
        "GPU_RUNS": 0,
        "MODAL_RUNS": 0,
        "report": rel(OUT_MD),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
