"""CPU-only forensic audit for the Private PV1 regression.

This is deliberately an offline, read-only audit of frozen artifacts.  It never
reads Private answers, never writes either corpus, and never calls Modal/GPU.
The only new files are the forensic outputs under the dedicated experiment
directory (plus optional validation working artifacts below that directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "private_task1/experiments/pv1_regression_forensics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

REPAIRED = {
    "10533", "131890", "149317", "177151", "181693", "187338", "191261",
    "196918", "208668", "210808", "232489", "255762", "263763", "288457",
    "34810", "55497", "56098", "57978", "67660", "71014",
}
WEIGHTS = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
RRF_K = 2
PRIVATE_QUERY_COUNT = 2080
VALIDATION_FOLDS = (1, 2, 3, 4)

PRIVATE = ROOT / "private_task1/input/private-official.json"
OLD_SUBMISSION = ROOT / "private_task1/submissions/final/first-time/submission_private.json"
PV1_SUBMISSION = ROOT / "private_task1/submissions/pv1_final/submission_private_pv1.json"
OLD_CANDIDATES = ROOT / "private_task1/retrieval/candidates/private_rrf_k20_candidates.jsonl"
PV1_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
OLD_WORKLIST = ROOT / "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl"
PV1_WORKLIST = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_bge_full_worklist.jsonl"
OLD_SCORES = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl"
PV1_SCORES = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
OLD_PROD_MANIFEST = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production.json"
OLD_ADDENDUM = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production_provenance_addendum.json"
PV1_FINAL_MANIFEST = ROOT / "private_task1/submissions/pv1_final/pv1_final_manifest.json"
PV1_CORPUS_MANIFEST = ROOT / "data/processed_pv1/metadata/pv1_corpus_manifest.json"
PV1_DENSE_MANIFEST = ROOT / "private_task1/experiments/private_pv1/dense/private_pv1_dense_manifest.json"
PV1_BM25_MANIFEST = ROOT / "private_task1/experiments/private_pv1/bm25/private_pv1_bm25_manifest.json"
PV1_KNN_MANIFEST = ROOT / "private_task1/experiments/private_pv1/knn/private_pv1_word_tfidf_knn_manifest.json"
PV1_BGE_MANIFEST = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_merged_manifest.json"
PV1_DELTA_METRICS = ROOT / "private_task1/experiments/private_pv1/bge/local_cpu_fallback/delta_metrics.json"
PV1_DELTA_SCORES = ROOT / "private_task1/experiments/private_pv1/bge/local_cpu_fallback/delta_scores.jsonl"

V3_VALIDATION_CANDIDATES = ROOT / "private_task1/experiments/private_safe_baseline/candidate_union_k20.jsonl"
V3_VALIDATION_WORKLIST = ROOT / "private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl"
V3_VALIDATION_SCORES = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl"
V3_VALIDATION_MANIFEST = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/manifests/production.json"
V3_DENSE_UNION = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/f1_f4_candidate_union.jsonl"
LEXICAL_CHECKPOINTS = ROOT / "artifacts/task1/recovery_096/candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
PV1_DOCS = ROOT / "data/processed_pv1/documents"
PV1_CHUNKS = ROOT / "data/processed_pv1/chunks"
PV1 = ROOT / "data/processed_pv1"
PV1_INDEX = ROOT / "data/processed_pv1/dense_faiss/index.faiss"
PV1_MAPPING = ROOT / "data/processed_pv1/dense_faiss/chunk_vector_mapping.jsonl"
PV1_DENSE_OUTPUT = OUT / "validation_pv1_dense_predictions.jsonl"
PV1_BM25_OUTPUT = OUT / "validation_pv1_bm25.jsonl"
PV1_CANDIDATES_OUTPUT = OUT / "validation_pv1_candidate_union_k20.jsonl"
PV1_WORKLIST_OUTPUT = OUT / "validation_pv1_worklist_k20.jsonl"
PV1_BGE_DELTA_WORKLIST = OUT / "validation_pv1_bge_delta_worklist.jsonl"
PV1_BGE_DELTA_SCORES = OUT / "validation_pv1_bge_delta_scores.jsonl"
PV1_BGE_CHECKPOINT = OUT / "validation_pv1_bge_delta.sqlite3"
PV1_BGE_DELTA_METRICS = OUT / "validation_pv1_bge_delta_metrics.json"
PRIVATE_BGE_REPLAY_SCORES = OUT / "private_bge_delta_replay_scores.jsonl"
PRIVATE_BGE_REPLAY_METRICS = OUT / "private_bge_delta_replay_metrics.json"
PV1_VALIDATION_BM25 = OUT / "validation_pv1_bm25.jsonl"
PV1_VALIDATION_BM25_MANIFEST = OUT / "validation_pv1_bm25_manifest.json"
PV1_VALIDATION_DENSE = OUT / "validation_pv1_dense_predictions.jsonl"
PV1_VALIDATION_DENSE_MANIFEST = OUT / "validation_pv1_dense_manifest.json"
PV1_VALIDATION_CANDIDATES = OUT / "validation_pv1_candidate_union_k20.jsonl"
PV1_VALIDATION_WORKLIST = OUT / "validation_pv1_worklist_k20.jsonl"
PV1_VALIDATION_BGE_DELTA_WORKLIST = OUT / "validation_pv1_bge_delta_worklist.jsonl"
PV1_VALIDATION_BGE_DELTA_SCORES = OUT / "validation_pv1_bge_delta_scores.jsonl"
PV1_VALIDATION_BGE_CHECKPOINT = OUT / "validation_pv1_bge_delta.sqlite3"
PV1_VALIDATION_BGE_METRICS = OUT / "validation_pv1_bge_delta_metrics.json"

MODEL_BASE = ROOT / "models/reranker"
MODEL_FT = ROOT / "outputs/task1/bge_ft_model/bge_m3_finetuned"
BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FT_WEIGHT_SHA = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
FT_CONFIG_SHA = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
SCORE_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scan_chunk_hashes(root: Path, wanted: set[str]) -> dict[str, str]:
    """Hash only the requested chunk texts for cross-corpus reuse checks."""

    found: dict[str, str] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = str(row["chunk_id"])
                if chunk_id in wanted:
                    found[chunk_id] = hashlib.sha256(
                        str(row["text"]).encode("utf-8")
                    ).hexdigest()
    return found


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256(path)


def numeric_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["query_id"]), str(row["document_id"])


def load_submission(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    result: dict[str, list[str]] = {}
    for qid, record in payload.items():
        if not isinstance(record, dict) or not isinstance(record.get("answer"), list):
            raise RuntimeError(f"invalid submission schema: {path}:{qid}")
        result[str(qid)] = [str(value) for value in record["answer"]]
    return result


def load_candidate_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(path)
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = str(row["query_id"])
        candidates = []
        for item in row["candidates"]:
            candidates.append({
                "document_id": str(item["document_id"]),
                "candidate_rank": int(item.get("candidate_rank", len(candidates) + 1)),
                "source_ranks": {str(k): int(v) for k, v in (item.get("source_ranks") or {}).items()},
            })
        result[qid] = candidates
    return result


def load_worklist_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise RuntimeError(f"duplicate worklist identity in {path}: {key}")
        result[key] = row
    return result


def load_score_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise RuntimeError(f"duplicate score identity in {path}: {key}")
        result[key] = row
    return result


def canonical_source_order() -> tuple[str, ...]:
    return ("dense", "bge", "knn_word", "bm25")


def rank_query(candidates: list[dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]], qid: str) -> list[dict[str, Any]]:
    bge_rows = sorted(
        candidates,
        key=lambda row: (
            -float(scores[(qid, str(row["document_id"]))]["bge_ft_score"]),
            -float(scores[(qid, str(row["document_id"]))]["bge_base_score"]),
            int(row["candidate_rank"]),
            numeric_id(str(row["document_id"])),
        ),
    )
    bge_rank = {str(row["document_id"]): index for index, row in enumerate(bge_rows, 1)}
    scored: list[dict[str, Any]] = []
    for row in candidates:
        did = str(row["document_id"])
        source_ranks = dict(row.get("source_ranks") or {})
        source_ranks["bge"] = bge_rank[did]
        value = sum(
            WEIGHTS[source] / (RRF_K + int(source_ranks[source]))
            for source in WEIGHTS
            if source in source_ranks
        )
        scored.append({
            "document_id": did,
            "rrf_score": value,
            "source_ranks": source_ranks,
            "candidate_rank": int(row["candidate_rank"]),
            "bge_score": float(scores[(qid, did)]["bge_ft_score"]),
        })
    return sorted(
        scored,
        key=lambda item: (
            -item["rrf_score"],
            *(int(item["source_ranks"].get(source, 10**9)) for source in canonical_source_order()),
            numeric_id(str(item["document_id"])),
        ),
    )


def replay_predictions(candidates: dict[str, list[dict[str, Any]]], scores: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, list[str]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for qid in sorted(candidates, key=numeric_id):
        ranked = rank_query(candidates[qid], scores, qid)
        predictions[qid] = [str(row["document_id"]) for row in ranked[:5]]
        traces[qid] = ranked
    return predictions, traces


def semantic_sha(predictions: dict[str, list[str]]) -> str:
    payload = "".join(
        json.dumps({"query_id": qid, "top5": predictions[qid]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for qid in sorted(predictions, key=numeric_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_hashes() -> dict[str, Any]:
    paths = {
        "private_input": PRIVATE,
        "old_submission_json": OLD_SUBMISSION,
        "old_submission_zip": OLD_SUBMISSION.with_suffix(".zip"),
        "pv1_submission_json": PV1_SUBMISSION,
        "pv1_submission_zip": PV1_SUBMISSION.with_suffix(".zip"),
        "old_candidates": OLD_CANDIDATES,
        "pv1_candidates": PV1_CANDIDATES,
        "old_worklist": OLD_WORKLIST,
        "pv1_worklist": PV1_WORKLIST,
        "old_scores": OLD_SCORES,
        "pv1_scores": PV1_SCORES,
        "old_production_manifest": OLD_PROD_MANIFEST,
        "old_provenance_addendum": OLD_ADDENDUM,
        "pv1_final_manifest": PV1_FINAL_MANIFEST,
        "pv1_corpus_manifest": PV1_CORPUS_MANIFEST,
        "pv1_dense_manifest": PV1_DENSE_MANIFEST,
        "pv1_bm25_manifest": PV1_BM25_MANIFEST,
        "pv1_knn_manifest": PV1_KNN_MANIFEST,
        "pv1_bge_manifest": PV1_BGE_MANIFEST,
        "pv1_delta_metrics": PV1_DELTA_METRICS,
        "pv1_delta_scores": PV1_DELTA_SCORES,
        "v3_validation_candidates": V3_VALIDATION_CANDIDATES,
        "v3_validation_worklist": V3_VALIDATION_WORKLIST,
        "v3_validation_scores": V3_VALIDATION_SCORES,
        "v3_validation_manifest": V3_VALIDATION_MANIFEST,
        "train": TRAIN,
        "folds": FOLDS,
    }
    result: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in paths.items():
        if path.is_file():
            result[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
        else:
            result[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "missing": True}
            missing.append(name)
    result["missing"] = missing
    return result


def stage0_and_private() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = artifact_hashes()
    if hashes["missing"]:
        raise RuntimeError(f"FORENSIC_INPUT_GATE_FAIL missing={hashes['missing']}")
    private_payload = read_json(PRIVATE)
    private_ids = {str(qid) for qid, record in private_payload.items() if isinstance(record, dict) and isinstance(record.get("question"), str)}
    if len(private_ids) != PRIVATE_QUERY_COUNT:
        raise RuntimeError("Private query universe is not 2080")
    # The input is used only for query identity/text provenance.  The answer
    # field is intentionally not read or inspected.
    if any(not isinstance(record, dict) or "question" not in record for record in private_payload.values()):
        raise RuntimeError("invalid private question input")

    old_candidates = load_candidate_map(OLD_CANDIDATES)
    pv1_candidates = load_candidate_map(PV1_CANDIDATES)
    old_work = load_worklist_map(OLD_WORKLIST)
    pv1_work = load_worklist_map(PV1_WORKLIST)
    old_scores = load_score_map(OLD_SCORES)
    pv1_scores = load_score_map(PV1_SCORES)
    old_submission = load_submission(OLD_SUBMISSION)
    pv1_submission = load_submission(PV1_SUBMISSION)
    if set(old_candidates) != private_ids or set(pv1_candidates) != private_ids:
        raise RuntimeError("private candidate query universe mismatch")
    if set(old_submission) != private_ids or set(pv1_submission) != private_ids:
        raise RuntimeError("private submission query universe mismatch")

    old_replay, old_trace = replay_predictions(old_candidates, old_scores)
    pv1_replay, pv1_trace = replay_predictions(pv1_candidates, pv1_scores)
    old_submission_match = old_replay == old_submission
    pv1_submission_match = pv1_replay == pv1_submission
    if not old_submission_match or not pv1_submission_match:
        raise RuntimeError(f"implementation replay mismatch old={old_submission_match} pv1={pv1_submission_match}")
    replay1 = replay_predictions(pv1_candidates, pv1_scores)[0]
    replay2 = replay_predictions(pv1_candidates, pv1_scores)[0]
    disagreements = sorted([qid for qid in private_ids if replay1[qid] != replay2[qid]], key=numeric_id)

    private_diff_rows: list[dict[str, Any]] = []
    source_change_counts: Counter[str] = Counter()
    top5_changed = 0
    unchanged = 0
    same_set_reordered = 0
    additions = 0
    removals = 0
    k20_changed = 0
    k20_added = 0
    k20_removed = 0
    repaired_top5_entries = 0
    repaired_top5_docs: set[str] = set()
    repaired_k20_queries: dict[str, set[str]] = defaultdict(set)
    repaired_final_queries: dict[str, set[str]] = defaultdict(set)
    source_categories: Counter[str] = Counter()

    for qid in sorted(private_ids, key=numeric_id):
        old_docs = [str(row["document_id"]) for row in old_candidates[qid]]
        new_docs = [str(row["document_id"]) for row in pv1_candidates[qid]]
        old_set, new_set = set(old_docs), set(new_docs)
        added = sorted(new_set - old_set, key=numeric_id)
        removed = sorted(old_set - new_set, key=numeric_id)
        candidate_changed = old_set != new_set
        if candidate_changed:
            k20_changed += 1
            k20_added += len(added)
            k20_removed += len(removed)
        old_top = old_submission[qid]
        new_top = pv1_submission[qid]
        changed = old_top != new_top
        if changed:
            top5_changed += 1
        else:
            unchanged += 1
        if set(old_top) == set(new_top) and old_top != new_top:
            same_set_reordered += 1
        additions += len(set(new_top) - set(old_top))
        removals += len(set(old_top) - set(new_top))
        repaired_new_k20 = new_set & REPAIRED
        repaired_new_top = set(new_top) & REPAIRED
        for did in repaired_new_k20:
            repaired_k20_queries[did].add(qid)
        for did in repaired_new_top:
            repaired_top5_docs.add(did)
            repaired_final_queries[did].add(qid)
            repaired_top5_entries += 1
        if changed:
            source_fields: set[str] = set()
            old_by_doc = {str(row["document_id"]): row for row in old_candidates[qid]}
            new_by_doc = {str(row["document_id"]): row for row in pv1_candidates[qid]}
            for did in sorted(old_set | new_set, key=numeric_id):
                old_sources = old_by_doc.get(did, {}).get("source_ranks", {})
                new_sources = new_by_doc.get(did, {}).get("source_ranks", {})
                for source in set(old_sources) | set(new_sources):
                    if old_sources.get(source) != new_sources.get(source):
                        source_fields.add(source)
            if source_fields == {"dense"}:
                source_category = "DENSE_ONLY_CHANGE"
            elif source_fields == {"bm25"}:
                source_category = "BM25_ONLY_CHANGE"
            elif source_fields == {"knn_word"}:
                source_category = "KNN_ONLY_CHANGE"
            elif source_fields:
                source_category = "MULTI_SOURCE_CHANGE"
            else:
                source_category = "BGE_OR_FINAL_ORDER_CHANGE"
            source_categories[source_category] += 1
            source_change_counts.update(source_fields)

        old_ranked = {str(row["document_id"]): {**row, "final_rank": index} for index, row in enumerate(old_trace[qid], 1)}
        new_ranked = {str(row["document_id"]): {**row, "final_rank": index} for index, row in enumerate(pv1_trace[qid], 1)}
        changed_docs: list[dict[str, Any]] = []
        for did in sorted(old_set | new_set, key=numeric_id):
            old_row = next((row for row in old_candidates[qid] if str(row["document_id"]) == did), None)
            new_row = next((row for row in pv1_candidates[qid] if str(row["document_id"]) == did), None)
            old_r = old_ranked.get(did, {})
            new_r = new_ranked.get(did, {})
            changed_docs.append({
                "document_id": did,
                "old_k20": old_row is not None,
                "new_k20": new_row is not None,
                "old_candidate_rank": old_row.get("candidate_rank") if old_row else None,
                "new_candidate_rank": new_row.get("candidate_rank") if new_row else None,
                "old_source_ranks": old_row.get("source_ranks") if old_row else None,
                "new_source_ranks": new_row.get("source_ranks") if new_row else None,
                "old_bge_score": old_r.get("bge_score"),
                "new_bge_score": new_r.get("bge_score"),
                "old_bge_rank": old_r.get("source_ranks", {}).get("bge"),
                "new_bge_rank": new_r.get("source_ranks", {}).get("bge"),
                "old_rrf_score": old_r.get("rrf_score"),
                "new_rrf_score": new_r.get("rrf_score"),
                "old_final_rank": old_r.get("final_rank"),
                "new_final_rank": new_r.get("final_rank"),
            })
        if changed:
            private_diff_rows.append({
                "query_id": qid,
                "old_top5": old_top,
                "new_top5": new_top,
                "added_top5_documents": sorted(set(new_top) - set(old_top), key=numeric_id),
                "removed_top5_documents": sorted(set(old_top) - set(new_top), key=numeric_id),
                "k20_changed": candidate_changed,
                "k20_added": added,
                "k20_removed": removed,
                "repaired_docs_in_old_k20": sorted(old_set & REPAIRED, key=numeric_id),
                "repaired_docs_in_new_k20": sorted(new_set & REPAIRED, key=numeric_id),
                "repaired_docs_in_old_top5": sorted(set(old_top) & REPAIRED, key=numeric_id),
                "repaired_docs_in_new_top5": sorted(set(new_top) & REPAIRED, key=numeric_id),
                "source_change_category": source_category,
                "changed_document_provenance": changed_docs,
            })

    write_jsonl(OUT / "private_old_vs_pv1_diff.jsonl", private_diff_rows)

    impact: dict[str, Any] = {}
    old_dense = load_dense_hits(ROOT / "private_task1/retrieval/candidates/private_dense_predictions.jsonl")
    pv1_dense = load_dense_hits(ROOT / "private_task1/experiments/private_pv1/dense/private_pv1_dense_predictions.jsonl")
    pv1_bm25 = load_simple_rankings(ROOT / "private_task1/experiments/private_pv1/bm25/private_pv1_bm25.jsonl")
    for did in sorted(REPAIRED, key=numeric_id):
        dense_q = {qid for qid, docs in pv1_dense.items() if did in docs}
        bm25_q = {qid for qid, docs in pv1_bm25.items() if did in docs}
        old_k20_q = {qid for qid, docs in ((q, {str(r["document_id"]) for r in rows}) for q, rows in old_candidates.items()) if did in docs}
        new_k20_q = {qid for qid, docs in ((q, {str(r["document_id"]) for r in rows}) for q, rows in pv1_candidates.items()) if did in docs}
        old_top_q = {qid for qid, docs in old_submission.items() if did in docs}
        new_top_q = {qid for qid, docs in pv1_submission.items() if did in docs}
        impact[did] = {
            "pv1_dense_top200_queries": len(dense_q),
            "pv1_bm25_top200_queries": len(bm25_q),
            "old_v3_k20_queries": len(old_k20_q),
            "pv1_k20_queries": len(new_k20_q),
            "old_v3_top5_queries": len(old_top_q),
            "pv1_top5_queries": len(new_top_q),
            "pv1_k20_query_ids": sorted(new_k20_q, key=numeric_id),
            "pv1_top5_query_ids": sorted(new_top_q, key=numeric_id),
            "bge_rank_distribution": Counter(
                str(row["source_ranks"].get("bge"))
                for qid, rows in pv1_trace.items()
                for row in rows
                if str(row["document_id"]) == did
            ),
            "rrf_rank_distribution": Counter(
                str(index)
                for qid, rows in pv1_trace.items()
                for index, row in enumerate(rows, 1)
                if str(row["document_id"]) == did
            ),
            "top5_displacements": [
                {"query_id": qid, "new_document_id": did, "old_top5_dropped": sorted(set(old_submission[qid]) - set(pv1_submission[qid]), key=numeric_id)}
                for qid in sorted(new_top_q - old_top_q, key=numeric_id)
            ],
        }
        impact[did]["bge_rank_distribution"] = dict(impact[did]["bge_rank_distribution"])
        impact[did]["rrf_rank_distribution"] = dict(impact[did]["rrf_rank_distribution"])

    changed_query_class: Counter[str] = Counter()
    for row in private_diff_rows:
        qid = str(row["query_id"])
        old_k = set(row["repaired_docs_in_old_k20"])
        new_k = set(row["repaired_docs_in_new_k20"])
        old_top = set(row["repaired_docs_in_old_top5"])
        new_top = set(row["repaired_docs_in_new_top5"])
        if new_top - old_top:
            category = "A_DIRECT_REPAIRED_DOC_ENTRY"
        elif new_k and not new_top:
            category = "B_REPAIRED_DOC_IN_K20_NOT_TOP5"
        elif not old_k and not new_k and row["k20_changed"]:
            category = "C_NO_REPAIRED_DOC_IN_K20_GLOBAL_RETRIEVAL_SHIFT"
        else:
            category = "D_SCORE_OR_ORDER_CHANGE"
        changed_query_class[category] += 1

    repaired_impact = {
        "repaired_document_count": len(REPAIRED),
        "documents": impact,
        "changed_private_query_classification": dict(changed_query_class),
        "top5_entries_from_repaired_docs": repaired_top5_entries,
        "repaired_docs_entering_top5": len(repaired_top5_docs),
        "repaired_doc_ids_entering_top5": sorted(repaired_top5_docs, key=numeric_id),
        "repaired_docs_in_any_pv1_k20": len({did for did in REPAIRED if any(did in {str(row["document_id"]) for row in rows} for rows in pv1_candidates.values())}),
    }
    write_json(OUT / "repaired_doc_impact.json", repaired_impact)

    quality = audit_quality()
    write_json(OUT / "data_quality_20docs.json", quality)

    determinism = {
        "status": "PASS" if not disagreements else "FAIL",
        "input_gate": "PASS",
        "private_query_count": len(private_ids),
        "old_replay_matches_submission": old_submission_match,
        "pv1_replay_matches_submission": pv1_submission_match,
        "replay1_semantic_sha256": semantic_sha(replay1),
        "replay2_semantic_sha256": semantic_sha(replay2),
        "top5_disagreement_queries": len(disagreements),
        "top5_disagreement_query_ids": disagreements[:100],
        "old_submission_sha256": sha256(OLD_SUBMISSION),
        "pv1_submission_sha256": sha256(PV1_SUBMISSION),
        "frozen_artifact_hashes": hashes,
        "policy": {"name": "RETRIEVAL_RRF_NO_LABEL", "weights": WEIGHTS, "rrf_k": RRF_K},
        "private_answers_read": False,
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    write_json(OUT / "determinism_report.json", determinism)

    result = {
        "input_gate": "PASS",
        "determinism_gate": determinism["status"],
        "bge_determinism_gate": "PENDING_EXACT_CPU_REPLAY",
        "private_top5_changed_queries": top5_changed,
        "private_top5_unchanged_queries": unchanged,
        "same_set_reordered_queries": same_set_reordered,
        "top5_document_additions": additions,
        "top5_document_removals": removals,
        "k20_changed_queries": k20_changed,
        "k20_qdoc_additions": k20_added,
        "k20_qdoc_removals": k20_removed,
        "source_change_query_counts": dict(source_categories),
        "source_field_change_counts": dict(source_change_counts),
        "repaired_impact": repaired_impact,
        "quality": quality,
        "hashes": hashes,
    }
    write_json(OUT / "private_stage_summary.json", result)
    return result


def load_simple_rankings(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("query_id", row.get("question_id")))
        if "candidates" in row:
            docs = [str(item.get("document_id", item.get("doc_id"))) for item in row["candidates"]]
        else:
            docs = [str(item.get("doc_id")) for item in row.get("hits", [])]
        result[qid] = docs
    return result


def load_dense_hits(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("query_id", row.get("question_id")))
        docs: list[str] = []
        seen: set[str] = set()
        for item in row.get("hits", []):
            did = str(item.get("doc_id", item.get("document_id")))
            if did not in seen:
                seen.add(did)
                docs.append(did)
        result[qid] = docs
    return result


def _norm_text(value: str) -> str:
    return " ".join(value.casefold().split())


def audit_quality() -> dict[str, Any]:
    documents: dict[str, Any] = {}
    objective_candidates: list[dict[str, Any]] = []
    for did in sorted(REPAIRED, key=numeric_id):
        path = PV1_CHUNKS / f"{did}.jsonl"
        rows = read_jsonl(path)
        texts = [str(row.get("text", "")) for row in rows]
        normalized = [_norm_text(text) for text in texts]
        hashes: dict[str, list[str]] = defaultdict(list)
        for row, text in zip(rows, texts):
            hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(str(row["chunk_id"]))
        exact_groups = [ids for ids in hashes.values() if len(ids) > 1]
        near_pairs: list[dict[str, Any]] = []
        # Structural near-duplicate evidence uses cheap character shingles.
        # Ordinary legal repetition is left alone; this avoids an expensive
        # quadratic sequence-alignment pass over long legal chunks.
        shingles = [
            {value[index : index + 5] for index in range(max(0, len(value) - 4))}
            for value in normalized
        ]
        for i in range(len(texts)):
            if not texts[i].strip():
                continue
            for j in range(i + 1, len(texts)):
                if normalized[i] == normalized[j]:
                    continue
                if abs(len(texts[i]) - len(texts[j])) > max(80, int(max(len(texts[i]), len(texts[j])) * 0.12)):
                    continue
                union = shingles[i] | shingles[j]
                ratio = len(shingles[i] & shingles[j]) / len(union) if union else 0.0
                if ratio >= 0.94:
                    near_pairs.append({"chunk_id_a": str(rows[i]["chunk_id"]), "chunk_id_b": str(rows[j]["chunk_id"]), "shingle_jaccard": ratio})
        short = [
            {"chunk_id": str(row["chunk_id"]), "length": len(text), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "snippet": text[:240]}
            for row, text in zip(rows, texts) if len(text.strip()) < 40
        ]
        replacement = [
            {"chunk_id": str(row["chunk_id"]), "replacement_count": text.count("�"), "snippet": text[:240]}
            for row, text in zip(rows, texts) if "�" in text
        ]
        page_number = [
            {"chunk_id": str(row["chunk_id"]), "snippet": text[:240]}
            for row, text in zip(rows, texts)
            if re.fullmatch(r"\s*(?:page\s*)?\d{1,4}\s*", text, re.I)
        ]
        boilerplate = [
            {"chunk_id": str(row["chunk_id"]), "snippet": text[:240]}
            for row, text in zip(rows, texts)
            if sum(marker in text.casefold() for marker in ("logo", "watermark", "www.", "http://", "https://")) >= 2
        ]
        # Repeated header/footer is evidence only when the same short text is
        # repeated in non-adjacent chunks; legal clauses are not auto-labelled.
        occurrences: dict[str, list[int]] = defaultdict(list)
        for index, text in enumerate(normalized):
            if 5 <= len(text) <= 180:
                occurrences[text].append(index)
        repeated_short = [
            {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "occurrences": [str(rows[k]["chunk_id"]) for k in indexes], "snippet": texts[indexes[0]][:240]}
            for text, indexes in occurrences.items() if len(indexes) >= 3 and (max(indexes) - min(indexes) > 2)
        ]
        exact_duplicate_evidence: list[dict[str, Any]] = []
        row_by_id = {str(row["chunk_id"]): row for row in rows}
        for group in exact_groups:
            contexts = [
                {
                    "chunk_id": cid,
                    "parent_id": str(row_by_id[cid].get("parent_id")),
                    "article": row_by_id[cid].get("article"),
                    "clause": row_by_id[cid].get("clause"),
                    "point": row_by_id[cid].get("point"),
                }
                for cid in group
            ]
            legitimate = len({(item["parent_id"], item["article"], item["clause"], item["point"]) for item in contexts}) > 1
            evidence = {
                "document_id": did,
                "type": "EXACT_DUPLICATE_CHUNK",
                "chunk_ids": group,
                "contexts": contexts,
                "classification": "LEGITIMATE_REPEATED_LEGAL_LANGUAGE" if legitimate else "POSSIBLE_DUPLICATE_EXTRACTION",
                "structural_fix_candidate": not legitimate,
            }
            exact_duplicate_evidence.append(evidence)
            if evidence["structural_fix_candidate"]:
                objective_candidates.append(evidence)
        for item in boilerplate:
            objective_candidates.append({"document_id": did, "type": "BOILERPLATE_MARKER", **item, "structural_fix_candidate": True})
        documents[did] = {
            "chunk_count": len(rows),
            "min_chunk_length": min(map(len, texts), default=0),
            "median_chunk_length": statistics.median(map(len, texts)) if texts else 0,
            "max_chunk_length": max(map(len, texts), default=0),
            "exact_duplicate_groups": exact_groups,
            "exact_duplicate_evidence": exact_duplicate_evidence,
            "near_duplicate_pairs": near_pairs,
            "short_chunks": short,
            "replacement_character_chunks": replacement,
            "isolated_page_number_chunks": page_number,
            "boilerplate_marker_chunks": boilerplate,
            "repeated_short_texts": repeated_short,
            "manifest_quality_flag_count": next((item.get("obvious_ocr_or_boilerplate_flags") for item in read_json(PV1_CORPUS_MANIFEST).get("quality_flags", []) if str(item.get("document_id")) == did), None),
            "content_hashes": {
                "chunk_file_sha256": sha256(path),
                "text_sha256_set_sha256": hashlib.sha256("\n".join(sorted(hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts)).encode()).hexdigest(),
            },
        }
    return {
        "corpus": "data/processed_pv1",
        "document_count": len(documents),
        "documents": documents,
        "objective_structural_fix_candidates": objective_candidates,
        "objective_data_defect_count": len(objective_candidates),
        "interpretation": "Markers and very short chunks are evidence flags; no legal text was deleted or rewritten. Exact duplicate evidence is separated from legitimate repeated legal language.",
    }


def load_validation_population() -> tuple[dict[str, str], dict[str, int], dict[str, set[str]]]:
    """Load the held-out F1-F4 population for evaluation only."""

    train = read_json(TRAIN)
    if not isinstance(train, dict) or len(train) != 7000:
        raise RuntimeError("validation train universe is not 7000 queries")
    folds_payload = read_json(FOLDS)
    fold_rows = folds_payload.get("folds") if isinstance(folds_payload, dict) else folds_payload
    if not isinstance(fold_rows, list) or len(fold_rows) != 5:
        raise RuntimeError("strict folds are not the expected five-fold contract")
    query_fold: dict[str, int] = {}
    fold_ids: dict[int, set[str]] = {}
    for item in fold_rows:
        fold = int(item["fold"])
        ids = {str(value) for value in item["validation_ids"]}
        if len(ids) != 1400:
            raise RuntimeError(f"validation fold {fold} does not contain 1400 queries")
        fold_ids[fold] = ids
        for query_id in ids:
            if query_id in query_fold:
                raise RuntimeError(f"query appears in multiple folds: {query_id}")
            query_fold[query_id] = fold
    if set(query_fold) != {str(key) for key in train}:
        raise RuntimeError("validation fold/query universe mismatch")
    target_ids = set().union(*(fold_ids[fold] for fold in VALIDATION_FOLDS))
    if len(target_ids) != 5600:
        raise RuntimeError("F1-F4 population is not exactly 5600 queries")
    questions: dict[str, str] = {}
    gold: dict[str, set[str]] = {}
    for query_id in target_ids:
        record = train[query_id]
        if not isinstance(record, dict) or not isinstance(record.get("question"), str):
            raise RuntimeError(f"invalid validation question: {query_id}")
        answers = record.get("answer")
        if not isinstance(answers, list) or not answers:
            raise RuntimeError(f"missing validation gold: {query_id}")
        questions[query_id] = str(record["question"])
        gold[query_id] = {str(value) for value in answers}
    return questions, query_fold, gold


def load_fold_checkpoint_rankings(
    source: str, fold: int, expected_ids: set[str]
) -> dict[str, list[str]]:
    path = LEXICAL_CHECKPOINTS / f"{source}_fold{fold}.json"
    payload = read_json(path)
    if payload.get("source") != source or int(payload.get("fold")) != fold:
        raise RuntimeError(f"invalid lexical checkpoint identity: {path}")
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict) or {str(key) for key in rankings} != expected_ids:
        raise RuntimeError(f"{path}: validation query coverage mismatch")
    result: dict[str, list[str]] = {}
    for query_id, values in rankings.items():
        if not isinstance(values, list):
            raise RuntimeError(f"{path}: ranking is not a list for {query_id}")
        docs = [str(value) for value in values]
        if len(docs) != len(set(docs)):
            raise RuntimeError(f"{path}: duplicate document for {query_id}")
        result[str(query_id)] = docs
    return result


def build_pv1_validation_dense(questions: dict[str, str]) -> dict[str, list[str]]:
    """Run the existing local PV1 dense/FAISS contract on F1-F4 queries."""

    if PV1_VALIDATION_DENSE.is_file() and PV1_VALIDATION_DENSE_MANIFEST.is_file():
        manifest = read_json(PV1_VALIDATION_DENSE_MANIFEST)
        if manifest.get("output_sha256") == sha256(PV1_VALIDATION_DENSE):
            return load_dense_hits(PV1_VALIDATION_DENSE)

    import numpy as np
    import faiss

    from udsc2026.infrastructure.embedding.client import EmbeddingClient

    dense_manifest = read_json(PV1 / "metadata/pv1_dense_faiss_manifest.json")
    if dense_manifest.get("dense_faiss_gate") != "PASS":
        raise RuntimeError("PV1 dense FAISS manifest is not PASS")
    index = faiss.read_index(str(PV1_INDEX))
    if type(index).__name__ != "IndexFlatIP" or index.d != 768 or index.ntotal != 1_272_971:
        raise RuntimeError("PV1 FAISS contract mismatch")
    row_to_item: list[tuple[str, str]] = []
    seen_chunks: set[str] = set()
    with PV1_MAPPING.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            expected_row = len(row_to_item)
            if int(item["row"]) != expected_row:
                raise RuntimeError("PV1 mapping rows are not contiguous")
            chunk_id = str(item["chunk_id"])
            if chunk_id in seen_chunks:
                raise RuntimeError(f"duplicate PV1 mapping chunk: {chunk_id}")
            seen_chunks.add(chunk_id)
            row_to_item.append((chunk_id, str(item["document_id"])))
    if len(row_to_item) != index.ntotal:
        raise RuntimeError("PV1 mapping does not cover FAISS index")

    embedder = EmbeddingClient(
        model_path=str(ROOT / "models/dek21-v2"),
        device="cpu",
        batch_size=32,
        max_length=256,
        normalize_embeddings=True,
        output_dimension=768,
        window_long_texts=False,
        local_files_only=True,
    )
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    query_ids = sorted(questions, key=numeric_id)
    for offset in range(0, len(query_ids), 64):
        current_ids = query_ids[offset : offset + 64]
        vectors = np.asarray(
            embedder.embed_documents([questions[qid] for qid in current_ids], batch_size=64),
            dtype="float32",
        )
        scores, rows = index.search(vectors, 200)
        for query_id, query_scores, query_rows in zip(current_ids, scores, rows):
            hits: list[dict[str, Any]] = []
            seen_docs: set[str] = set()
            for rank, (score, row_number) in enumerate(zip(query_scores, query_rows), 1):
                row_number = int(row_number)
                if row_number < 0:
                    continue
                chunk_id, document_id = row_to_item[row_number]
                if document_id in seen_docs:
                    continue
                seen_docs.add(document_id)
                value = float(score)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite PV1 dense score: {query_id}")
                hits.append({"doc_id": document_id, "chunk_id": chunk_id, "dense_score": value, "rank": len(hits) + 1})
            output_rows.append({"query_id": query_id, "hits": hits})
        print(f"PV1_VALIDATION_DENSE_PROGRESS completed={len(output_rows)}/{len(query_ids)}", flush=True)
    if len(output_rows) != len(query_ids):
        raise RuntimeError("PV1 validation dense output coverage mismatch")
    output_sha = write_jsonl(PV1_VALIDATION_DENSE, output_rows)
    write_json(
        PV1_VALIDATION_DENSE_MANIFEST,
        {
            "status": "PASS",
            "run_mode": "CPU",
            "query_count": len(query_ids),
            "candidate_k_chunks": 200,
            "unique_document_depth": 200,
            "output_sha256": output_sha,
            "output": str(PV1_VALIDATION_DENSE.relative_to(ROOT)).replace("\\", "/"),
            "pv1_corpus_fingerprint": dense_manifest.get("pv1_corpus_fingerprint"),
            "faiss_sha256": sha256(PV1_INDEX),
            "mapping_sha256": sha256(PV1_MAPPING),
            "embedding_model": "huyydangg/DEk21_hcmute_embedding_v2",
            "embedding_model_path": "models/dek21-v2",
            "embedding_revision_status": "LOCAL_REVISION_UNPROVEN",
            "elapsed_seconds": time.perf_counter() - started,
            "labels_used": False,
        },
    )
    return {str(row["query_id"]): [str(hit["doc_id"]) for hit in row["hits"]] for row in output_rows}


def build_pv1_validation_bm25(
    questions: dict[str, str],
) -> dict[str, list[str]]:
    """Build PV1 BM25 rankings with the repository's canonical implementation."""

    if PV1_VALIDATION_BM25.is_file() and PV1_VALIDATION_BM25_MANIFEST.is_file():
        manifest = read_json(PV1_VALIDATION_BM25_MANIFEST)
        if manifest.get("output_sha256") == sha256(PV1_VALIDATION_BM25):
            return load_simple_rankings(PV1_VALIDATION_BM25)

    from udsc2026.evaluation.legal_ir_lexical import LegalContext, build_bm25f_rankings

    contexts: list[LegalContext] = []
    for path in sorted(PV1_DOCS.glob("*.json"), key=lambda item: numeric_id(item.stem)):
        record = read_json(path)
        document_id = str(record.get("doc_id", path.stem))
        passage = str(record.get("cleaned_text", ""))
        if not passage.strip():
            raise RuntimeError(f"empty PV1 document in validation BM25: {path}")
        title = str(record.get("title") or record.get("law_name") or "")
        contexts.append(LegalContext(document_id, passage, title))
    if len(contexts) != 8532 or len({item.doc_id for item in contexts}) != 8532:
        raise RuntimeError("PV1 validation BM25 document inventory mismatch")
    started = time.perf_counter()
    rankings = build_bm25f_rankings(questions, contexts, title_weight=0.0, top_k=200)
    if set(rankings) != set(questions) or any(len(values) != len(set(values)) for values in rankings.values()):
        raise RuntimeError("PV1 validation BM25 coverage/duplicate gate failed")
    rows = [
        {"query_id": query_id, "candidates": [{"document_id": did, "rank": rank} for rank, did in enumerate(rankings[query_id], 1)]}
        for query_id in sorted(questions, key=numeric_id)
    ]
    output_sha = write_jsonl(PV1_VALIDATION_BM25, rows)
    write_json(
        PV1_VALIDATION_BM25_MANIFEST,
        {
            "status": "PASS",
            "run_mode": "CPU",
            "query_count": len(questions),
            "document_count": len(contexts),
            "top_k": 200,
            "implementation": "udsc2026.evaluation.legal_ir_lexical.build_bm25f_rankings",
            "corpus": "data/processed_pv1/documents",
            "output": str(PV1_VALIDATION_BM25.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": output_sha,
            "elapsed_seconds": time.perf_counter() - started,
            "labels_used": False,
        },
    )
    return rankings


def load_validation_dense_rankings(path: Path, expected_ids: set[str]) -> dict[str, list[str]]:
    result = load_dense_hits(path)
    if set(result) != expected_ids:
        raise RuntimeError("dense validation output query coverage mismatch")
    if any(len(docs) != len(set(docs)) for docs in result.values()):
        raise RuntimeError("duplicate dense validation document")
    return result


def load_chunks_for_root(root: Path, document_id: str) -> list[dict[str, Any]]:
    path = root / f"{document_id}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing chunk file: {path}")
    rows = read_jsonl(path)
    if not rows or len({str(row.get("chunk_id")) for row in rows}) != len(rows):
        raise RuntimeError(f"invalid/duplicate chunks: {path}")
    return rows


def build_validation_worklist(
    candidates: list[dict[str, Any]],
    questions: dict[str, str],
) -> list[dict[str, Any]]:
    """Materialize the frozen selector boundary for PV1 validation candidates."""

    from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared

    by_document: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in candidates:
        query_id = str(row["query_id"])
        for candidate in row["candidates"]:
            by_document[str(candidate["document_id"])].append((query_id, candidate))
    worklist: list[dict[str, Any]] = []
    total = len(by_document)
    for index, (document_id, entries) in enumerate(sorted(by_document.items(), key=lambda pair: numeric_id(pair[0])), 1):
        prepared = prepare_document(load_chunks_for_root(PV1_CHUNKS, document_id))
        for query_id, candidate in entries:
            selected = select_true_s2_prepared(questions[query_id], prepared, topk=3)
            selected_ids = [str(item["chunk_id"]) for item in selected]
            selected_texts = [str(item["raw_chunk_text"]) for item in selected]
            if not 1 <= len(selected_ids) <= 3:
                raise RuntimeError(f"invalid selected chunk count: {query_id}/{document_id}")
            if len(selected_ids) != len(set(selected_ids)):
                raise RuntimeError(f"duplicate selected chunks: {query_id}/{document_id}")
            if any(not chunk_id.startswith(document_id + "_") or not text.strip() for chunk_id, text in zip(selected_ids, selected_texts)):
                raise RuntimeError(f"invalid selected chunk payload: {query_id}/{document_id}")
            worklist.append(
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "source_ranks": candidate.get("source_ranks", {}),
                    "expected_inference_units": len(selected_ids),
                    "selected_chunk_ids": selected_ids,
                    "selector": "true_s2_bm25_within_document_v2",
                    "aggregation": "MAX",
                    "max_length": 512,
                    "scoring_rule": "FT_PRIMARY_BASE_TIEBREAK",
                }
            )
        if index == 1 or index % 500 == 0 or index == total:
            print(f"PV1_VALIDATION_SELECTOR_PROGRESS documents={index}/{total} qdocs={len(worklist)}", flush=True)
    worklist.sort(key=lambda row: (numeric_id(str(row["query_id"])), int(row["candidate_rank"]), numeric_id(str(row["document_id"]))))
    identities = [(str(row["query_id"]), str(row["document_id"])) for row in worklist]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate PV1 validation q-doc identity")
    if len(worklist) != 112000 or any(len(row["selected_chunk_ids"]) != int(row["expected_inference_units"]) for row in worklist):
        raise RuntimeError("PV1 validation worklist is not exactly 5600 x 20")
    return worklist


def validation_bge_reuse_partition(
    new_worklist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Partition PV1 validation rows into exact old-score reuse and CPU delta."""

    old_manifest = read_json(V3_VALIDATION_MANIFEST)
    if old_manifest.get("status") != "COMPLETE":
        raise RuntimeError("old validation BGE manifest is not COMPLETE")
    actual_old_sha = sha256(V3_VALIDATION_SCORES)
    if actual_old_sha != old_manifest.get("output_sha256"):
        raise RuntimeError("old validation BGE local output SHA mismatch")
    old_work = load_worklist_map(V3_VALIDATION_WORKLIST)
    old_scores = load_score_map(V3_VALIDATION_SCORES)
    if len(old_work) != 112000 or len(old_scores) != 112000:
        raise RuntimeError("old validation BGE q-doc count is not 112000")
    expected_provenance = {
        "base_model_revision": BASE_REVISION,
        "ft_weight_sha256": FT_WEIGHT_SHA,
        "ft_config_sha256": FT_CONFIG_SHA,
        "selector": "true_s2_bm25_within_document_v2",
        "aggregation": "MAX",
    }
    models = old_manifest.get("models", {})
    provenance_ok = (
        models.get("base", {}).get("revision") == expected_provenance["base_model_revision"]
        and models.get("ft", {}).get("weight_sha256") == expected_provenance["ft_weight_sha256"]
        and models.get("ft", {}).get("config_sha256") == expected_provenance["ft_config_sha256"]
        and all(row.get("selector") == expected_provenance["selector"] and row.get("aggregation") == expected_provenance["aggregation"] for row in old_scores.values())
        and all(finite(row.get("bge_ft_score")) and finite(row.get("bge_base_score")) for row in old_scores.values())
    )
    old_selected = {str(cid) for row in old_work.values() for cid in row.get("selected_chunk_ids", [])}
    new_selected = {str(cid) for row in new_worklist for cid in row.get("selected_chunk_ids", [])}
    old_hashes = scan_chunk_hashes(ROOT / "data/processed_v3/chunks", old_selected | new_selected)
    new_hashes = scan_chunk_hashes(PV1_CHUNKS, old_selected | new_selected)
    rerun: list[dict[str, Any]] = []
    reusable = 0
    reasons: Counter[str] = Counter()
    new_keys = {(str(row["query_id"]), str(row["document_id"])) for row in new_worklist}
    for row in new_worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        old_row = old_work.get(key)
        reason: str | None = None
        if str(row["document_id"]) in REPAIRED:
            reason = "REPAIRED_DOCUMENT"
        elif old_row is None:
            reason = "NEW_QDOC"
        elif key not in old_scores:
            reason = "MISSING_OLD_SCORE"
        elif not provenance_ok:
            reason = "PROVENANCE_MISMATCH"
        elif [str(value) for value in old_row.get("selected_chunk_ids", [])] != [str(value) for value in row.get("selected_chunk_ids", [])]:
            reason = "SELECTED_CHUNK_CHANGED"
        else:
            selected = [str(value) for value in row["selected_chunk_ids"]]
            if any(not old_hashes.get(cid) or not new_hashes.get(cid) or old_hashes[cid] != new_hashes[cid] for cid in selected):
                reason = "SELECTED_CHUNK_CHANGED"
        if reason is None:
            reusable += 1
        else:
            reasons[reason] += 1
            rerun.append({**row, "rerun_reason": reason})
    if reusable + len(rerun) != len(new_worklist) or len({(str(row["query_id"]), str(row["document_id"])) for row in rerun}) != len(rerun):
        raise RuntimeError("PV1 validation BGE reuse partition is incomplete")
    summary = {
        "old_qdocs": len(old_work),
        "new_qdocs": len(new_keys),
        "reusable_qdocs": reusable,
        "rerun_qdocs": len(rerun),
        "reuse_rate": reusable / len(new_worklist) if new_worklist else 0.0,
        "rerun_reasons": {name: reasons[name] for name in ("NEW_QDOC", "REPAIRED_DOCUMENT", "SELECTED_CHUNK_CHANGED", "MISSING_OLD_SCORE", "PROVENANCE_MISMATCH")},
        "old_score_provenance_pass": provenance_ok,
        "old_manifest_sha256": sha256(V3_VALIDATION_MANIFEST),
        "old_scores_sha256": actual_old_sha,
        "old_worklist_sha256": sha256(V3_VALIDATION_WORKLIST),
    }
    return rerun, old_scores, summary


def run_validation_bge_delta(
    rerun: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Score only the PV1 validation rows whose selector/content changed."""

    write_jsonl(PV1_VALIDATION_BGE_DELTA_WORKLIST, rerun)
    if not rerun:
        write_jsonl(PV1_VALIDATION_BGE_DELTA_SCORES, [])
        metrics = {
            "status": "PASS",
            "q_doc_count": 0,
            "inference_unit_count": 0,
            "device": "cpu",
            "gpu_used": False,
            "modal_used": False,
            "execution_source": "LOCAL_CPU_FALLBACK_EMPTY_DELTA",
        }
        write_json(PV1_VALIDATION_BGE_METRICS, metrics)
        return {}, metrics
    if PV1_VALIDATION_BGE_DELTA_SCORES.is_file() and PV1_VALIDATION_BGE_METRICS.is_file():
        existing = read_json(PV1_VALIDATION_BGE_METRICS)
        if existing.get("status") == "PASS" and existing.get("output_sha256") == sha256(PV1_VALIDATION_BGE_DELTA_SCORES):
            rows = load_score_map(PV1_VALIDATION_BGE_DELTA_SCORES)
            if len(rows) == len(rerun):
                return rows, existing
    from scripts.evaluation.score_private_pv1_bge_delta_cpu import run_score

    metrics = run_score(
        rerun,
        source_worklist_sha=sha256(PV1_VALIDATION_BGE_DELTA_WORKLIST),
        questions_path=TRAIN,
        chunks_root=PV1_CHUNKS,
        base_path=MODEL_BASE,
        ft_path=MODEL_FT,
        batch_size=16,
        checkpoint_path=PV1_VALIDATION_BGE_CHECKPOINT,
        output_path=PV1_VALIDATION_BGE_DELTA_SCORES,
        execution_source="LOCAL_CPU_FALLBACK_VALIDATION",
    )
    metrics["output_sha256"] = sha256(PV1_VALIDATION_BGE_DELTA_SCORES)
    write_json(PV1_VALIDATION_BGE_METRICS, metrics)
    return load_score_map(PV1_VALIDATION_BGE_DELTA_SCORES), metrics


def validation_metric(
    predictions: dict[str, list[str]], gold: dict[str, set[str]]
) -> dict[str, Any]:
    recalls: list[float] = []
    precisions: list[float] = []
    for query_id in sorted(gold, key=numeric_id):
        relevant = gold[query_id]
        top = predictions.get(query_id, [])[:5]
        hits = len(relevant & set(top))
        recalls.append(hits / len(relevant) if relevant else 1.0)
        precisions.append(hits / 5.0)
    return {
        "recall_at_5": sum(recalls) / len(recalls) if recalls else 0.0,
        "precision_at_5": sum(precisions) / len(precisions) if precisions else 0.0,
        "query_count": len(recalls),
    }


def validation_oracle_metric(
    candidates: dict[str, list[dict[str, Any]]], gold: dict[str, set[str]]
) -> dict[str, Any]:
    values: list[float] = []
    for query_id in sorted(gold, key=numeric_id):
        docs = {str(row["document_id"]) for row in candidates[query_id]}
        values.append(len(docs & gold[query_id]) / len(gold[query_id]))
    return {"candidate_recall_at_20": sum(values) / len(values), "query_count": len(values)}


def validation_fold_metrics(
    predictions: dict[str, list[str]],
    gold: dict[str, set[str]],
    query_fold: dict[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        f"F{fold}": validation_metric(
            {query_id: predictions[query_id] for query_id in predictions if query_fold[query_id] == fold},
            {query_id: gold[query_id] for query_id in gold if query_fold[query_id] == fold},
        )
        for fold in VALIDATION_FOLDS
    }


def validation_candidate_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        values: list[dict[str, Any]] = []
        for item in row["candidates"]:
            values.append(
                {
                    "document_id": str(item["document_id"]),
                    "candidate_rank": int(item.get("candidate_rank", len(values) + 1)),
                    "source_ranks": {str(key): int(value) for key, value in (item.get("source_ranks") or {}).items()},
                }
            )
        if len(values) != 20 or len({str(item["document_id"]) for item in values}) != 20:
            raise RuntimeError(f"validation candidate cap/duplicate mismatch: {query_id}")
        result[query_id] = values
    return result


def validation_repaired_gold(
    candidates: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[str]],
    gold: dict[str, set[str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query_id in sorted(gold, key=numeric_id):
        repaired_gold = sorted(gold[query_id] & REPAIRED, key=numeric_id)
        if not repaired_gold:
            continue
        candidate_docs = {str(row["document_id"]) for row in candidates[query_id]}
        top_docs = set(predictions.get(query_id, [])[:5])
        rows.append(
            {
                "query_id": query_id,
                "repaired_gold_documents": repaired_gold,
                "in_k20": sorted(set(repaired_gold) & candidate_docs, key=numeric_id),
                "in_top5": sorted(set(repaired_gold) & top_docs, key=numeric_id),
            }
        )
    return {
        "repaired_documents_relevant_at_least_once": sorted({did for row in rows for did in row["repaired_gold_documents"]}, key=numeric_id),
        "validation_queries_with_repaired_gold": len(rows),
        "queries_with_repaired_gold_in_k20": sum(bool(row["in_k20"]) for row in rows),
        "queries_with_repaired_gold_in_top5": sum(bool(row["in_top5"]) for row in rows),
        "details": rows,
    }


def evaluate_validation(
    old_candidates: dict[str, list[dict[str, Any]]],
    pv1_candidates: dict[str, list[dict[str, Any]]],
    old_scores: dict[tuple[str, str], dict[str, Any]],
    pv1_scores: dict[tuple[str, str], dict[str, Any]],
    questions: dict[str, str],
    query_fold: dict[str, int],
    gold: dict[str, set[str]],
    reuse_summary: dict[str, Any],
    delta_metrics: dict[str, Any],
) -> dict[str, Any]:
    old_predictions, old_trace = replay_predictions(old_candidates, old_scores)
    pv1_predictions, pv1_trace = replay_predictions(pv1_candidates, pv1_scores)
    old_metric = validation_metric(old_predictions, gold)
    pv1_metric = validation_metric(pv1_predictions, gold)
    old_oracle = validation_oracle_metric(old_candidates, gold)
    pv1_oracle = validation_oracle_metric(pv1_candidates, gold)
    old_folds = validation_fold_metrics(old_predictions, gold, query_fold)
    pv1_folds = validation_fold_metrics(pv1_predictions, gold, query_fold)
    candidate_changed = [
        query_id for query_id in sorted(questions, key=numeric_id)
        if {str(row["document_id"]) for row in old_candidates[query_id]} != {str(row["document_id"]) for row in pv1_candidates[query_id]}
    ]
    old_set = {str(row["document_id"]) for rows in old_candidates.values() for row in rows}
    pv1_set = {str(row["document_id"]) for rows in pv1_candidates.values() for row in rows}
    added_qdocs = sum(
        len({str(row["document_id"]) for row in pv1_candidates[qid]} - {str(row["document_id"]) for row in old_candidates[qid]})
        for qid in questions
    )
    removed_qdocs = sum(
        len({str(row["document_id"]) for row in old_candidates[qid]} - {str(row["document_id"]) for row in pv1_candidates[qid]})
        for qid in questions
    )

    def no_repaired_candidates() -> dict[str, list[dict[str, Any]]]:
        return {
            query_id: [row for row in rows if str(row["document_id"]) not in REPAIRED]
            for query_id, rows in pv1_candidates.items()
        }

    ablation_no_repaired = no_repaired_candidates()
    no_repaired_predictions, _ = replay_predictions(ablation_no_repaired, pv1_scores)
    no_repaired_metric = validation_metric(no_repaired_predictions, gold)

    # Diagnostic only: keep the PV1 candidate membership and scores, but use the
    # old source ranks where a q-doc existed in the V3 candidate set.  This
    # isolates source-rank drift from corpus membership drift without changing
    # the production policy.
    old_source_rank_candidates: dict[str, list[dict[str, Any]]] = {}
    for query_id, rows in pv1_candidates.items():
        old_by_doc = {str(row["document_id"]): row for row in old_candidates[query_id]}
        old_source_rank_candidates[query_id] = [
            {
                **row,
                "source_ranks": dict(old_by_doc[str(row["document_id"])] ["source_ranks"])
                if str(row["document_id"]) in old_by_doc
                else dict(row.get("source_ranks") or {}),
            }
            for row in rows
        ]
    old_source_predictions, _ = replay_predictions(old_source_rank_candidates, pv1_scores)
    old_source_metric = validation_metric(old_source_predictions, gold)

    changed_query_rows = []
    for query_id in sorted(questions, key=numeric_id):
        old_top = old_predictions[query_id]
        new_top = pv1_predictions[query_id]
        if old_top != new_top:
            changed_query_rows.append(
                {
                    "query_id": query_id,
                    "fold": query_fold[query_id],
                    "old_top5": old_top,
                    "pv1_top5": new_top,
                    "gold": sorted(gold[query_id], key=numeric_id),
                    "old_hit_count": len(set(old_top) & gold[query_id]),
                    "pv1_hit_count": len(set(new_top) & gold[query_id]),
                }
            )

    repaired_gold = validation_repaired_gold(pv1_candidates, pv1_predictions, gold)
    ablations = {
        "pv1_normal": {"metrics": pv1_metric, "candidate_oracle": pv1_oracle},
        "pv1_without_repaired_documents": {
            "metrics": no_repaired_metric,
            "candidate_oracle": validation_oracle_metric(ablation_no_repaired, gold),
            "interpretation": "diagnostic only; repaired documents were removed from PV1 candidates, no production policy change",
        },
        "pv1_candidates_with_old_source_ranks_when_available": {
            "metrics": old_source_metric,
            "candidate_oracle": pv1_oracle,
            "interpretation": "diagnostic only; PV1 membership and PV1 BGE scores retained",
        },
        "same_candidate_sets_old_vs_new_source_ranks": {
            "status": "NOT_APPLICABLE_AS_GLOBAL_ABLATION",
            "reason": "PV1 and V3 candidate sets differ for the changed queries; the preceding diagnostic preserves PV1 membership and substitutes old ranks only where available",
        },
    }
    result = {
        "status": "PASS",
        "population": {
            "folds": list(VALIDATION_FOLDS),
            "queries": len(questions),
            "fold0_used": False,
            "public_labels_used": False,
            "labels_used_only_for_evaluation": True,
        },
        "policy": {"name": "RETRIEVAL_RRF_NO_LABEL", "weights": WEIGHTS, "rrf_k": RRF_K},
        "old_v3": {
            "metrics": old_metric,
            "metrics_by_fold": old_folds,
            "candidate_oracle": old_oracle,
            "oracle_to_top5_recall_gap": old_oracle["candidate_recall_at_20"] - old_metric["recall_at_5"],
            "candidate_path": str(V3_VALIDATION_CANDIDATES.relative_to(ROOT)).replace("\\", "/"),
            "worklist_path": str(V3_VALIDATION_WORKLIST.relative_to(ROOT)).replace("\\", "/"),
            "scores_path": str(V3_VALIDATION_SCORES.relative_to(ROOT)).replace("\\", "/"),
        },
        "pv1": {
            "metrics": pv1_metric,
            "metrics_by_fold": pv1_folds,
            "candidate_oracle": pv1_oracle,
            "oracle_to_top5_recall_gap": pv1_oracle["candidate_recall_at_20"] - pv1_metric["recall_at_5"],
            "candidate_path": str(PV1_VALIDATION_CANDIDATES.relative_to(ROOT)).replace("\\", "/"),
            "worklist_path": str(PV1_VALIDATION_WORKLIST.relative_to(ROOT)).replace("\\", "/"),
            "scores_path": str(PV1_VALIDATION_BGE_DELTA_SCORES.relative_to(ROOT)).replace("\\", "/"),
        },
        "delta_pv1_minus_v3": {
            "recall_at_5": pv1_metric["recall_at_5"] - old_metric["recall_at_5"],
            "precision_at_5": pv1_metric["precision_at_5"] - old_metric["precision_at_5"],
            "candidate_oracle_recall_at_20": pv1_oracle["candidate_recall_at_20"] - old_oracle["candidate_recall_at_20"],
            "by_fold": {
                fold: {
                    "recall_at_5": pv1_folds[fold]["recall_at_5"] - old_folds[fold]["recall_at_5"],
                    "precision_at_5": pv1_folds[fold]["precision_at_5"] - old_folds[fold]["precision_at_5"],
                }
                for fold in old_folds
            },
        },
        "candidate_shift": {
            "queries_changed": len(candidate_changed),
            "qdoc_additions": added_qdocs,
            "qdoc_removals": removed_qdocs,
            "old_unique_documents": len(old_set),
            "pv1_unique_documents": len(pv1_set),
            "repaired_qdocs_in_old_k20": sum(1 for rows in old_candidates.values() for row in rows if str(row["document_id"]) in REPAIRED),
            "repaired_qdocs_in_pv1_k20": sum(1 for rows in pv1_candidates.values() for row in rows if str(row["document_id"]) in REPAIRED),
            "changed_query_ids_sample": candidate_changed[:100],
        },
        "bge_reuse": reuse_summary,
        "bge_delta_metrics": delta_metrics,
        "changed_final_top5_queries": len(changed_query_rows),
        "changed_final_top5_query_rows": changed_query_rows,
        "repaired_doc_gold_intersection": repaired_gold,
        "ablations": ablations,
        "provenance": {
            "train_sha256": sha256(TRAIN),
            "folds_sha256": sha256(FOLDS),
            "v3_candidates_sha256": sha256(V3_VALIDATION_CANDIDATES),
            "v3_worklist_sha256": sha256(V3_VALIDATION_WORKLIST),
            "v3_scores_sha256": sha256(V3_VALIDATION_SCORES),
            "pv1_corpus_manifest_sha256": sha256(PV1_CORPUS_MANIFEST),
            "pv1_dense_manifest_sha256": sha256(PV1 / "metadata/pv1_dense_faiss_manifest.json"),
            "pv1_candidates_sha256": sha256(PV1_VALIDATION_CANDIDATES),
            "pv1_worklist_sha256": sha256(PV1_VALIDATION_WORKLIST),
        },
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    write_json(OUT / "validation_v3_vs_pv1.json", result)
    return result


def stage_validation() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    questions, query_fold, gold = load_validation_population()
    dense = build_pv1_validation_dense(questions)
    if set(dense) != set(questions):
        raise RuntimeError("PV1 validation dense query universe mismatch")
    bm25 = build_pv1_validation_bm25(questions)
    if set(bm25) != set(questions):
        raise RuntimeError("PV1 validation BM25 query universe mismatch")
    fold_ids = {fold: {query_id for query_id, value in query_fold.items() if value == fold} for fold in VALIDATION_FOLDS}
    knn = {fold: load_fold_checkpoint_rankings("knn_word", fold, fold_ids[fold]) for fold in VALIDATION_FOLDS}
    from scripts.analysis.prepare_private_safe_baseline import build_union

    query_ids = sorted(questions, key=numeric_id)
    bm25_by_fold = {fold: bm25 for fold in VALIDATION_FOLDS}
    candidate_rows = build_union(query_ids, query_fold, dense, bm25_by_fold, knn, 20)
    pv1_candidate_map = validation_candidate_map_from_rows(candidate_rows)
    if set(pv1_candidate_map) != set(questions):
        raise RuntimeError("PV1 validation candidate query universe mismatch")
    write_jsonl(PV1_VALIDATION_CANDIDATES, candidate_rows)
    worklist = build_validation_worklist(candidate_rows, questions)
    write_jsonl(PV1_VALIDATION_WORKLIST, worklist)
    rerun, old_scores, reuse_summary = validation_bge_reuse_partition(worklist)
    delta_scores, delta_metrics = run_validation_bge_delta(rerun)
    new_scores: dict[tuple[str, str], dict[str, Any]] = {}
    old_work = load_worklist_map(V3_VALIDATION_WORKLIST)
    old_score_keys = set(old_scores)
    for row in worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in delta_scores:
            new_scores[key] = delta_scores[key]
        elif key in old_score_keys:
            old_row = old_scores[key]
            new_scores[key] = old_row
        else:
            raise RuntimeError(f"PV1 validation score missing: {key}")
    if len(new_scores) != len(worklist):
        raise RuntimeError("PV1 validation score coverage mismatch")
    result = evaluate_validation(
        load_candidate_map(V3_VALIDATION_CANDIDATES),
        pv1_candidate_map,
        old_scores,
        new_scores,
        questions,
        query_fold,
        gold,
        reuse_summary,
        delta_metrics,
    )
    result["provenance"]["old_worklist_selected_sha256"] = sha256(V3_VALIDATION_WORKLIST)
    result["provenance"]["old_score_worklist_join"] = len(old_work)
    write_json(OUT / "validation_v3_vs_pv1.json", result)
    return result


def audit_private_bge_replay() -> dict[str, Any]:
    """Compare the exact CPU replay with the already-frozen Private delta."""

    if not PRIVATE_BGE_REPLAY_SCORES.is_file():
        result = {"status": "PENDING", "reason": "replay output is not complete"}
        write_json(OUT / "bge_determinism_report.json", result)
        return result
    from scripts.evaluation.score_private_pv1_bge_delta_cpu import compare_reference

    result = compare_reference(PRIVATE_BGE_REPLAY_SCORES, PV1_DELTA_SCORES)
    result["actual_path"] = str(PRIVATE_BGE_REPLAY_SCORES.relative_to(ROOT)).replace("\\", "/")
    result["reference_path"] = str(PV1_DELTA_SCORES.relative_to(ROOT)).replace("\\", "/")
    result["gpu_runs"] = 0
    result["modal_runs"] = 0
    write_json(OUT / "bge_determinism_report.json", result)
    if (OUT / "determinism_report.json").is_file():
        determinism = read_json(OUT / "determinism_report.json")
        determinism["bge_determinism_gate"] = result.get("gate", "FAIL")
        determinism["bge_replay"] = result
        write_json(OUT / "determinism_report.json", determinism)
    if (OUT / "private_stage_summary.json").is_file():
        summary = read_json(OUT / "private_stage_summary.json")
        summary["bge_determinism_gate"] = result.get("gate", "FAIL")
        summary["bge_replay"] = result
        write_json(OUT / "private_stage_summary.json", summary)
    return result


def audit_wf_b() -> dict[str, Any]:
    """Audit existing Workflow-B artifacts without rerunning any model."""

    paths = {
        "docs_root": ROOT / "docs/task1/workflow_b",
        "reports_root": ROOT / "reports/task1/workflow_b",
        "scripts_root": ROOT / "scripts/analysis/workflow_b",
        "artifact_root": ROOT / "artifacts/task1/workflow_b",
        "b1_report": ROOT / "reports/task1/workflow_b/tv4/b1/reports/workflow_b_b1_final_report.json",
        "b1_contract": ROOT / "reports/task1/workflow_b/tv4/b1/contracts/workflow_b_b1_preregistered_contract.json",
    }
    existing: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            existing[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            existing[name] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "file_count": len(files),
                "sample_files": [str(item.relative_to(ROOT)).replace("\\", "/") for item in files[:40]],
            }
        else:
            existing[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "missing": True}
    b1 = read_json(paths["b1_report"]) if paths["b1_report"].is_file() else {}
    result = {
        "status": "PASS",
        "execution": "READ_ONLY_ARTIFACT_AUDIT",
        "workflow_b_signal": "INSUFFICIENT_EVIDENCE",
        "reason": "The registered B1 Direct LTR branch is explicitly BLOCKED before any model fit, score, prediction, or scientific comparison. Existing TV2/Qwen artifacts are historical research artifacts and no compatible frozen F1-F4 wf_B fusion result was found for deployment.",
        "b1_status": b1.get("status"),
        "b1_block_code": b1.get("block_code"),
        "b1_models_fit": b1.get("models_fit"),
        "b1_outer_oof_queries_scored": b1.get("outer_oof_queries_scored"),
        "b1_comparator_evaluated": b1.get("p1_comparator_evaluated"),
        "qwen_rerun": False,
        "existing_artifacts": existing,
        "deployment_decision": "DO_NOT_DEPLOY_WF_B",
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    write_json(OUT / "wf_b_audit.json", result)
    return result


def audit_reranker_headroom() -> dict[str, Any]:
    """Summarize same-registered-BGE evidence already present in F1-F4 artifacts."""

    decision_path = ROOT / "private_task1/experiments/f1_f4/final_ab_decision.json"
    report_path = ROOT / "private_task1/experiments/f1_f4/final_ab_report.md"
    decision = read_json(decision_path) if decision_path.is_file() else {}
    best_b = decision.get("best_b") or {}
    result = {
        "status": "PASS" if decision else "INSUFFICIENT_EVIDENCE",
        "source_decision": str(decision_path.relative_to(ROOT)).replace("\\", "/"),
        "source_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "population": decision.get("population", 5600),
        "fold0_used": False,
        "public_labels_used": False,
        "registered_base_model": "BAAI/bge-reranker-v2-m3",
        "registered_base_revision": BASE_REVISION,
        "registered_ft_weight_sha256": FT_WEIGHT_SHA,
        "registered_ft_config_sha256": FT_CONFIG_SHA,
        "baseline_recall": decision.get("baseline_recall"),
        "baseline_precision": decision.get("baseline_precision"),
        "best_same_registered_bge_policy": best_b.get("policy"),
        "best_same_registered_bge_recall": (best_b.get("metrics") or {}).get("recall"),
        "best_same_registered_bge_precision": (best_b.get("metrics") or {}).get("precision"),
        "best_same_registered_bge_delta_recall": (best_b.get("delta_vs_baseline") or {}).get("recall"),
        "best_same_registered_bge_changed_queries": (best_b.get("changes") or {}).get("changed_queries"),
        "relevant_gained": (best_b.get("changes") or {}).get("relevant_docs_gained"),
        "relevant_lost": (best_b.get("changes") or {}).get("relevant_docs_lost"),
        "current_bge_headroom": "NO" if best_b and (best_b.get("delta_vs_baseline") or {}).get("recall") <= 0 else "UNCLEAR",
        "interpretation": "The preregistered same-model BGE A/B artifact found no positive held-out Recall delta; this does not prove that every future selector/fusion is impossible, but it provides no currently validated headroom.",
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    write_json(OUT / "reranker_headroom.json", result)
    return result


def audit_registration() -> dict[str, Any]:
    registry = ROOT / "docs/models/model_registry.md"
    model_docs = ROOT / "docs/models/README.md"
    workflow_rules = ROOT / "docs/task1/workflow_b/workflow_B_execution_rules.md"
    text = "\n".join(path.read_text(encoding="utf-8") for path in (registry, model_docs, workflow_rules) if path.is_file()).casefold()
    return {
        "registry_path": str(registry.relative_to(ROOT)).replace("\\", "/"),
        "model_docs_path": str(model_docs.relative_to(ROOT)).replace("\\", "/"),
        "workflow_rules_path": str(workflow_rules.relative_to(ROOT)).replace("\\", "/"),
        "registered_qwen_promoted": "qwen/qwen3-vl-reranker-2b" in text,
        "bge_deferred_in_registry": "baai/bge-m3" in text,
        "external_cloud_model_forbidden": "cloud" in text and "gemini api" in text,
        "registration_derivative_status": "UNCLEAR",
        "new_external_reranker_status": "FORBIDDEN",
        "reason": "Repository rules require an explicitly registered/open-source/pinned model and do not state that this BGE fine-tuned derivative is an approved competition registration derivative; a new external/cloud reranker is explicitly disallowed for the official pipeline.",
    }


def write_root_cause_report(
    validation: dict[str, Any] | None,
    private: dict[str, Any] | None,
    bge_replay: dict[str, Any],
    wf_b: dict[str, Any],
    headroom: dict[str, Any],
    registration: dict[str, Any],
) -> dict[str, Any]:
    quality = (private or {}).get("quality", {})
    objective_defects = int(quality.get("objective_data_defect_count", 0))
    private_changed = int((private or {}).get("private_top5_changed_queries", 0))
    k20_changed = int((private or {}).get("k20_changed_queries", 0))
    validation_delta = ((validation or {}).get("delta_pv1_minus_v3") or {}).get("recall_at_5")
    if bge_replay.get("gate") == "FAIL":
        primary = "NONDETERMINISM"
    elif validation is not None and validation_delta is not None and validation_delta < -1e-9 and objective_defects > 0:
        primary = "DATA_EXTRACTION_QUALITY"
    elif validation is not None and validation_delta is not None and validation_delta < -1e-9:
        primary = "CORPUS_DISTRIBUTION_SHIFT"
    elif k20_changed > 0:
        primary = "RETRIEVAL_SHIFT"
    elif headroom.get("current_bge_headroom") == "NO":
        primary = "RERANKER_LIMITATION"
    else:
        primary = "INSUFFICIENT_EVIDENCE"
    secondary = []
    if k20_changed:
        secondary.append("RETRIEVAL_SHIFT_PRESENT_IN_PRIVATE")
    if private_changed:
        secondary.append("PRIVATE_TOP5_CHANGED_AFTER_PV1")
    if objective_defects:
        secondary.append("OBJECTIVE_DATA_FLAGS_REQUIRE_REVIEW")
    if wf_b.get("workflow_b_signal") == "INSUFFICIENT_EVIDENCE":
        secondary.append("WF_B_NOT_SCIENTIFICALLY_EVALUATED")
    if registration.get("registration_derivative_status") == "UNCLEAR":
        secondary.append("MODEL_REGISTRATION_DERIVATIVE_UNCLEAR")
    report = {
        "status": "COMPLETE",
        "primary_root_cause": primary,
        "secondary_causes": secondary,
        "evidence": {
            "private_top5_changed_queries": private_changed,
            "private_k20_changed_queries": k20_changed,
            "private_bge_determinism": bge_replay.get("gate"),
            "validation_recall_delta_pv1_minus_v3": validation_delta,
            "objective_data_defect_count": objective_defects,
            "wf_b_signal": wf_b.get("workflow_b_signal"),
            "current_bge_headroom": headroom.get("current_bge_headroom"),
        },
        "interpretation": "The primary classification is evidence-gated. Private leaderboard movement alone was not used as a causal label; candidate/source shifts, exact replay, labeled F1-F4 evaluation, data-quality evidence, and registered-model artifacts are kept separate.",
        "recommended_next_action": "KEEP_PV1_AND_IMPROVE_SAME_BGE" if primary in {"CORPUS_DISTRIBUTION_SHIFT", "RERANKER_LIMITATION", "RETRIEVAL_SHIFT"} and objective_defects == 0 else "FIX_DATA_AND_REBUILD_DELTA" if objective_defects > 0 else "NO_ACTION_UNTIL_MODEL_RULE_CONFIRMED",
        "registration_derivative_status": registration.get("registration_derivative_status"),
        "new_external_reranker_status": registration.get("new_external_reranker_status"),
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    report_path = OUT / "root_cause_report.md"
    lines = [
        "# Private PV1 regression forensics — root cause report",
        "",
        f"- **Primary root cause:** `{report['primary_root_cause']}`",
        f"- **Secondary causes:** `{', '.join(report['secondary_causes']) or 'NONE'}`",
        f"- **Recommended next action:** `{report['recommended_next_action']}`",
        "",
        "## Evidence gates",
        "",
        f"- Private exact implementation/determinism report: `{bge_replay.get('gate', 'PENDING')}`",
        f"- Private Top5 changed: `{private_changed}/2080`; K20 changed: `{k20_changed}/2080`",
        f"- F1-F4 PV1 minus V3 Recall@5: `{validation_delta}`",
        f"- Objective data defects: `{objective_defects}`",
        f"- wf_B signal: `{wf_b.get('workflow_b_signal')}`",
        f"- Same registered BGE headroom: `{headroom.get('current_bge_headroom')}`",
        f"- Registration derivative status: `{registration.get('registration_derivative_status')}`",
        f"- New external reranker status: `{registration.get('new_external_reranker_status')}`",
        "",
        "No GPU, Modal, Qwen rerun, Private-answer read, corpus mutation, or submission overwrite was performed by this forensic task.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "root_cause_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("private", "bge_replay", "validation", "supporting_audits", "all"), default="private")
    args = parser.parse_args()
    private_result: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    if args.stage in {"private", "all"}:
        private_result = stage0_and_private()
        print(json.dumps({k: private_result[k] for k in ("input_gate", "determinism_gate", "private_top5_changed_queries", "k20_changed_queries", "repaired_impact", "quality")}, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "bge_replay":
        print(json.dumps(audit_private_bge_replay(), ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"validation", "all"}:
        validation_result = stage_validation()
        print(json.dumps({
            "status": validation_result["status"],
            "old_v3": validation_result["old_v3"]["metrics"],
            "pv1": validation_result["pv1"]["metrics"],
            "delta": validation_result["delta_pv1_minus_v3"],
            "bge_reuse": validation_result["bge_reuse"],
        }, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"supporting_audits", "all"}:
        bge_replay = audit_private_bge_replay()
        wf_b = audit_wf_b()
        headroom = audit_reranker_headroom()
        registration = audit_registration()
        if private_result is None and (OUT / "private_stage_summary.json").is_file():
            private_result = read_json(OUT / "private_stage_summary.json")
        if validation_result is None and (OUT / "validation_v3_vs_pv1.json").is_file():
            validation_result = read_json(OUT / "validation_v3_vs_pv1.json")
        root = write_root_cause_report(validation_result, private_result, bge_replay, wf_b, headroom, registration)
        print(json.dumps({"wf_b": wf_b, "headroom": headroom, "registration": registration, "root_cause": root}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
