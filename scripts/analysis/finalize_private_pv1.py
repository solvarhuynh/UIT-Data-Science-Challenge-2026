"""Merge the certified PV1 BGE delta and run the frozen private RRF policy."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from udsc2026.evaluation.legal_ir_submission import (  # noqa: E402
    write_legal_ir_submission_json,
    write_legal_ir_submission_zip,
)


PRIVATE = ROOT / "private_task1/input/private-official.json"
FULL_WORKLIST = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_bge_full_worklist.jsonl"
DELTA_WORKLIST = ROOT / "private_task1/experiments/private_pv1/bge/private_pv1_bge_delta_worklist.jsonl"
OLD_SCORES = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl"
OLD_MANIFEST = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production.json"
DELTA_SCORES = ROOT / "private_task1/experiments/private_pv1/bge/local_cpu_fallback/delta_scores.jsonl"
DELTA_METRICS = ROOT / "private_task1/experiments/private_pv1/bge/local_cpu_fallback/delta_metrics.json"
K20 = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
CORPUS = ROOT / "artifacts/task1/corpus_document_ids.json"
OUT_ROOT = ROOT / "private_task1/experiments/private_pv1"
MERGED = OUT_ROOT / "bge/merged/private_pv1_bge_scores_merged.jsonl"
MERGED_MANIFEST = OUT_ROOT / "bge/merged/private_pv1_bge_merged_manifest.json"
SUBMISSION_DIR = ROOT / "private_task1/submissions/pv1_final"
SUBMISSION_JSON = SUBMISSION_DIR / "submission_private_pv1.json"
SUBMISSION_ZIP = SUBMISSION_DIR / "submission_private_pv1.zip"
FINAL_MANIFEST = SUBMISSION_DIR / "pv1_final_manifest.json"
VALIDATOR = ROOT / "scripts/submission/validate_legal_ir_submission.py"

EXPECTED_PRIVATE_SHA = "9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc"
EXPECTED_OLD_WORKLIST_SHA = "cf3554e9dfa9e753896e7b5332275e12aa030269c747a59222a072ad4d6a5f86"
EXPECTED_DELTA_SHA = "2c5b560a2d69d089e8604902e8e42711ed8b7f7c8565cb629b1e33c1a3d0ebae"
EXPECTED_OLD_SCORES_SHA = "1688dcec0afbc97722780fa1e0c92a18025498375b778d5e9cc81ac53cb7fa91"
EXPECTED_FT_SHA = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
EXPECTED_FT_CONFIG_SHA = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
EXPECTED_BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXPECTED_QDOCS = 41_600
EXPECTED_UNITS = 124_798
EXPECTED_DELTA_QDOCS = 297
EXPECTED_DELTA_UNITS = 891
WEIGHTS = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
RRF_K = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return sha256(path)


def numeric_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_delta() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if sha256(DELTA_WORKLIST) != EXPECTED_DELTA_SHA:
        raise RuntimeError("DELTA_WORKLIST_SHA_MISMATCH")
    work = load_jsonl(DELTA_WORKLIST)
    metrics = json.loads(DELTA_METRICS.read_text(encoding="utf-8"))
    if metrics.get("status") != "PASS" or metrics.get("execution_source") != "LOCAL_CPU_FALLBACK":
        raise RuntimeError("LOCAL_CPU_DELTA_NOT_COMPLETE")
    if metrics.get("q_doc_count") != EXPECTED_DELTA_QDOCS or metrics.get("inference_unit_count") != EXPECTED_DELTA_UNITS:
        raise RuntimeError("LOCAL_CPU_DELTA_POPULATION_MISMATCH")
    scores = load_jsonl(DELTA_SCORES)
    if len(work) != EXPECTED_DELTA_QDOCS or len(scores) != EXPECTED_DELTA_QDOCS:
        raise RuntimeError("LOCAL_CPU_DELTA_QDOC_MISMATCH")
    work_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in work:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in work_by_key:
            raise RuntimeError("DELTA_WORKLIST_DUPLICATE")
        work_by_key[key] = row
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scores:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in score_by_key:
            raise RuntimeError("DELTA_SCORE_DUPLICATE")
        score_by_key[key] = row
        values = [row.get("bge_ft_score"), row.get("bge_base_score")]
        values.extend(row.get("ft_chunk_scores", []))
        values.extend(row.get("base_chunk_scores", []))
        if not all(finite(value) for value in values):
            raise RuntimeError("DELTA_NONFINITE_SCORE")
        if row.get("ft_weight_sha256") != EXPECTED_FT_SHA or row.get("ft_config_sha256") != EXPECTED_FT_CONFIG_SHA:
            raise RuntimeError("DELTA_FT_PROVENANCE_MISMATCH")
        if row.get("base_model_revision") != EXPECTED_BASE_REVISION:
            raise RuntimeError("DELTA_BASE_REVISION_MISMATCH")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise RuntimeError("DELTA_SCORING_CONTRACT_MISMATCH")
    if set(work_by_key) != set(score_by_key):
        raise RuntimeError("DELTA_SCORE_IDENTITY_MISMATCH")
    for key, row in work_by_key.items():
        score = score_by_key[key]
        if [str(x) for x in row["selected_chunk_ids"]] != [str(x) for x in score["selected_chunk_ids"]]:
            raise RuntimeError(f"DELTA_SELECTED_CHUNK_MISMATCH:{key[0]}/{key[1]}")
        if len(row["selected_chunk_ids"]) != int(row["expected_inference_units"]):
            raise RuntimeError("DELTA_UNIT_MISMATCH")
    if sum(int(row["expected_inference_units"]) for row in work) != EXPECTED_DELTA_UNITS:
        raise RuntimeError("DELTA_UNIT_TOTAL_MISMATCH")
    return work, score_by_key


def merge_bge() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    delta_work, delta_scores = validate_delta()
    delta_keys = set(delta_scores)
    full_work = load_jsonl(FULL_WORKLIST)
    old_scores = load_jsonl(OLD_SCORES)
    if sha256(OLD_SCORES) != EXPECTED_OLD_SCORES_SHA:
        raise RuntimeError("OLD_SCORE_SHA_MISMATCH")
    if len(full_work) != EXPECTED_QDOCS or len(old_scores) != EXPECTED_QDOCS:
        raise RuntimeError("PV1_MERGE_QDOC_MISMATCH")
    old_by_key = {(str(row["query_id"]), str(row["document_id"])): row for row in old_scores}
    if len(old_by_key) != len(old_scores):
        raise RuntimeError("OLD_SCORE_DUPLICATE")
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    new_count = 0
    reused_count = 0
    for work in full_work:
        key = (str(work["query_id"]), str(work["document_id"]))
        if key in seen:
            raise RuntimeError("FULL_WORKLIST_DUPLICATE")
        seen.add(key)
        score = delta_scores.get(key) if key in delta_keys else old_by_key.get(key)
        if score is None:
            raise RuntimeError(f"MERGED_SCORE_MISSING:{key[0]}/{key[1]}")
        if [str(x) for x in work["selected_chunk_ids"]] != [str(x) for x in score["selected_chunk_ids"]]:
            raise RuntimeError(f"MERGED_SELECTED_CHUNK_MISMATCH:{key[0]}/{key[1]}")
        values = [score.get("bge_ft_score"), score.get("bge_base_score")]
        values.extend(score.get("ft_chunk_scores", []))
        values.extend(score.get("base_chunk_scores", []))
        if not all(finite(value) for value in values):
            raise RuntimeError("MERGED_NONFINITE_SCORE")
        merged_row = dict(score)
        merged_row["execution_source"] = "LOCAL_CPU_FALLBACK" if key in delta_keys else "REUSED_HISTORICAL_PRODUCTION"
        merged.append(merged_row)
        new_count += int(key in delta_keys)
        reused_count += int(key not in delta_keys)
    if not delta_keys.issubset(seen):
        raise RuntimeError("DELTA_IDENTITY_NOT_IN_PV1_WORKLIST")
    if any(key not in old_by_key for key in seen - delta_keys):
        raise RuntimeError("REUSABLE_HISTORICAL_SCORE_MISSING")
    if new_count != EXPECTED_DELTA_QDOCS or reused_count != EXPECTED_QDOCS - EXPECTED_DELTA_QDOCS:
        raise RuntimeError("MERGED_PARTITION_MISMATCH")
    if sum(len(row["selected_chunk_ids"]) for row in full_work) != EXPECTED_UNITS:
        raise RuntimeError("MERGED_UNIT_TOTAL_MISMATCH")
    output_sha = write_jsonl(MERGED, merged)
    manifest = {
        "status": "PASS",
        "q_doc_count": len(merged),
        "inference_unit_count": EXPECTED_UNITS,
        "reused_historical_qdocs": reused_count,
        "new_local_cpu_qdocs": new_count,
        "delta_worklist_sha256": EXPECTED_DELTA_SHA,
        "old_scores_sha256": EXPECTED_OLD_SCORES_SHA,
        "merged_output": str(MERGED.relative_to(ROOT)).replace("\\", "/"),
        "merged_output_sha256": output_sha,
        "model_revision": EXPECTED_BASE_REVISION,
        "ft_weight_sha256": EXPECTED_FT_SHA,
        "ft_config_sha256": EXPECTED_FT_CONFIG_SHA,
        "selected_chunk_mismatches": 0,
        "duplicate_identities": 0,
        "nonfinite_scores": 0,
        "execution_route": "LOCAL_CPU_FALLBACK",
    }
    MERGED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MERGED_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged, manifest


def rank_query(rows: list[dict[str, Any]], score_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    qid = str(rows[0]["query_id"])
    bge_rows = sorted(
        rows,
        key=lambda row: (
            -float(score_by_key[(qid, str(row["document_id"]))]["bge_ft_score"]),
            -float(score_by_key[(qid, str(row["document_id"]))]["bge_base_score"]),
            int(row["candidate_rank"]),
            str(row["document_id"]),
        ),
    )
    bge_rank = {str(row["document_id"]): index for index, row in enumerate(bge_rows, 1)}
    scored: list[tuple[str, float, dict[str, int]]] = []
    for row in rows:
        did = str(row["document_id"])
        source_ranks = {str(key): int(value) for key, value in (row.get("source_ranks") or {}).items()}
        source_ranks["bge"] = bge_rank[did]
        score = sum(
            WEIGHTS[source] / (RRF_K + source_ranks[source])
            for source in WEIGHTS
            if source in source_ranks
        )
        scored.append((did, score, source_ranks))
    scored.sort(
        key=lambda item: (
            -item[1],
            *(item[2].get(source, 10**9) for source in WEIGHTS),
            numeric_id(item[0]),
        )
    )
    return [did for did, _, _ in scored]


def final_rrf(merged: list[dict[str, Any]], merged_manifest: dict[str, Any]) -> dict[str, Any]:
    full_work = load_jsonl(FULL_WORKLIST)
    candidates = load_jsonl(K20)
    score_by_key = {(str(row["query_id"]), str(row["document_id"])): row for row in merged}
    work_by_key = {(str(row["query_id"]), str(row["document_id"])): row for row in full_work}
    if len(score_by_key) != EXPECTED_QDOCS or len(candidates) != 2080:
        raise RuntimeError("FINAL_RRF_INPUT_COUNT_MISMATCH")
    predictions: dict[str, list[str]] = {}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        qid = str(row["query_id"])
        for candidate in row["candidates"]:
            key = (qid, str(candidate["document_id"]))
            if key not in work_by_key:
                raise RuntimeError(f"K20_SCORE_MISSING:{qid}/{key[1]}")
            by_query[qid].append({
                "query_id": qid,
                "document_id": key[1],
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate.get("source_ranks", {}),
            })
    for qid, rows in by_query.items():
        if len(rows) != 20 or len({str(row["document_id"]) for row in rows}) != 20:
            raise RuntimeError(f"K20_CANDIDATE_GATE:{qid}")
        top5 = rank_query(rows, score_by_key)[:5]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(f"TOP5_GATE:{qid}")
        predictions[qid] = top5
    private = json.loads(PRIVATE.read_text(encoding="utf-8-sig"))
    private_ids = {str(key) for key in private}
    if set(predictions) != private_ids or len(predictions) != 2080:
        raise RuntimeError("PRIVATE_QUERY_COVERAGE_MISMATCH")
    allowed = {str(value) for value in json.loads(CORPUS.read_text(encoding="utf-8"))}
    if any(did not in allowed for docs in predictions.values() for did in docs):
        raise RuntimeError("INVALID_DOCUMENT_ID")
    payload = {qid: {"answer": predictions[qid]} for qid in sorted(predictions, key=numeric_id)}
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    write_legal_ir_submission_json(payload, SUBMISSION_JSON, expected_question_ids=sorted(predictions, key=numeric_id), allowed_document_ids=allowed)
    write_legal_ir_submission_zip(payload, SUBMISSION_ZIP, expected_question_ids=sorted(predictions, key=numeric_id), allowed_document_ids=allowed)
    for artifact in (SUBMISSION_JSON, SUBMISSION_ZIP):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--input", str(artifact), "--corpus-manifest", str(CORPUS)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SUBMISSION_VALIDATOR_FAILED:{artifact}:{result.stdout}{result.stderr}")
    final = {
        "status": "READY_FOR_MANUAL_SUBMISSION",
        "execution_route": "LOCAL_CPU_FALLBACK",
        "policy": "RETRIEVAL_RRF_NO_LABEL",
        "weights": WEIGHTS,
        "rrf_k": RRF_K,
        "queries": len(predictions),
        "qdocs": EXPECTED_QDOCS,
        "inference_units": EXPECTED_UNITS,
        "private_answers_read": False,
        "labels_used": False,
        "modal_inference_used": False,
        "gpu_inference_used": False,
        "merged_bge_manifest": merged_manifest,
        "submission_json": str(SUBMISSION_JSON.relative_to(ROOT)).replace("\\", "/"),
        "submission_json_sha256": sha256(SUBMISSION_JSON),
        "submission_zip": str(SUBMISSION_ZIP.relative_to(ROOT)).replace("\\", "/"),
        "submission_zip_sha256": sha256(SUBMISSION_ZIP),
        "duplicate_predictions": 0,
        "invalid_document_ids": 0,
        "missing_queries": 0,
        "validator": "PASS",
    }
    FINAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MANIFEST.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final


def main() -> None:
    merged, manifest = merge_bge()
    final = final_rrf(merged, manifest)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
