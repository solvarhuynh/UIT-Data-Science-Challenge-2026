"""Finalize the frozen Private RRF K20 run from downloaded BGE scores.

CPU-only.  This script never loads a model, calls Modal, reads Private answer
labels, or reruns retrieval.  It verifies the provenance sidecar and the
downloaded score artifact before using the frozen validation convention.
"""

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

PRIVATE_INPUT = ROOT / "private_task1/input/private-official.json"
WORKLIST = ROOT / "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl"
MANIFEST = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production.json"
SCORES = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl"
ADDENDUM = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production_provenance_addendum.json"
CORPUS = ROOT / "artifacts/task1/corpus_document_ids.json"
VALIDATOR = ROOT / "scripts/submission/validate_legal_ir_submission.py"
OUT_DIR = ROOT / "private_task1/submissions/final"
OUT_JSON = OUT_DIR / "submission_private.json"
OUT_ZIP = OUT_DIR / "submission_private.zip"
OUT_MANIFEST = OUT_DIR / "submission_manifest.json"

EXPECTED_PRIVATE_SHA = "9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc"
EXPECTED_WORKLIST_SHA = "cf3554e9dfa9e753896e7b5332275e12aa030269c747a59222a072ad4d6a5f86"
EXPECTED_QDOCS = 41_600
EXPECTED_QUERIES = 2_080
EXPECTED_UNITS = 124_798
WEIGHTS = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
RRF_K = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_inputs() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    set[str],
    dict[str, Any],
]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    private = json.loads(PRIVATE_INPUT.read_text(encoding="utf-8-sig"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if sha256(PRIVATE_INPUT) != EXPECTED_PRIVATE_SHA:
        raise RuntimeError("FILE_CHANGED_SINCE_FROZEN_STATE: private-official.json")
    if sha256(WORKLIST) != EXPECTED_WORKLIST_SHA:
        raise RuntimeError("WORKLIST_SHA_MISMATCH")
    if manifest.get("status") != "COMPLETE" or manifest.get("mode") != "production":
        raise RuntimeError("GPU_MANIFEST_NOT_COMPLETE_PRODUCTION")
    if manifest.get("worklist_sha256") != EXPECTED_WORKLIST_SHA:
        raise RuntimeError("GPU_MANIFEST_WORKLIST_SHA_MISMATCH")
    if manifest.get("q_doc_count") != EXPECTED_QDOCS or manifest.get("inference_unit_count") != EXPECTED_UNITS:
        raise RuntimeError("GPU_MANIFEST_POPULATION_MISMATCH")
    if manifest.get("actual_gpu") != "NVIDIA L4":
        raise RuntimeError("GPU_CONTRACT_MISMATCH")
    if "task1_private_rrf_k20" not in str(manifest.get("checkpoint", "")):
        raise RuntimeError("GPU_NAMESPACE_MISMATCH")
    if "task1_private_rrf_k20" not in str(manifest.get("output", "")):
        raise RuntimeError("GPU_OUTPUT_NAMESPACE_MISMATCH")
    if addendum.get("original_manifest_modified") is not False:
        raise RuntimeError("PROVENANCE_ADDENDUM_INVALID")
    if addendum.get("question_source_sha256") != EXPECTED_PRIVATE_SHA:
        raise RuntimeError("PROVENANCE_PRIVATE_SHA_MISMATCH")
    if not addendum.get("evidence") or "questions-file" not in " ".join(
        str(item) for item in addendum["evidence"]
    ):
        raise RuntimeError("PROVENANCE_LAUNCH_EVIDENCE_MISSING")

    private_ids = {str(query_id) for query_id in private}
    if len(private_ids) != EXPECTED_QUERIES:
        raise RuntimeError("PRIVATE_QUERY_COUNT_MISMATCH")
    if any(not isinstance(row, dict) or row.get("answer") is not None for row in private.values()):
        raise RuntimeError("PRIVATE_ANSWER_FIELD_NOT_NULL")

    worklist = load_jsonl(WORKLIST)
    scores = load_jsonl(SCORES)
    work_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    work_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in work_by_key:
            raise RuntimeError("DUPLICATE_WORKLIST_IDENTITY")
        work_by_key[key] = row
        work_by_query[key[0]].append(row)
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scores:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in score_by_key:
            raise RuntimeError("DUPLICATE_SCORE_IDENTITY")
        score_by_key[key] = row
        values = [row.get("bge_ft_score"), row.get("bge_base_score")]
        values.extend(row.get("ft_chunk_scores", []))
        values.extend(row.get("base_chunk_scores", []))
        if not all(is_finite(value) for value in values):
            raise RuntimeError("NONFINITE_BGE_SCORE")
    if len(worklist) != EXPECTED_QDOCS or len(score_by_key) != EXPECTED_QDOCS:
        raise RuntimeError("QDOC_COUNT_MISMATCH")
    if sum(int(row["expected_inference_units"]) for row in worklist) != EXPECTED_UNITS:
        raise RuntimeError("WORKLIST_UNIT_COUNT_MISMATCH")
    work_keys = set(work_by_key)
    score_keys = set(score_by_key)
    if work_keys != score_keys:
        raise RuntimeError(f"SCORE_IDENTITY_MISMATCH:{len(work_keys-score_keys)}/{len(score_keys-work_keys)}")
    if set(work_by_query) != private_ids:
        raise RuntimeError("PRIVATE_WORKLIST_QUERY_SET_MISMATCH")
    for key, work_row in work_by_key.items():
        score_row = score_by_key[key]
        if [str(x) for x in work_row["selected_chunk_ids"]] != [str(x) for x in score_row["selected_chunk_ids"]]:
            raise RuntimeError(f"SELECTED_CHUNK_MISMATCH:{key[0]}/{key[1]}")
    corpus_ids = {str(value) for value in json.loads(CORPUS.read_text(encoding="utf-8"))}
    invalid = {key[1] for key in score_keys if key[1] not in corpus_ids}
    if invalid:
        raise RuntimeError(f"INVALID_DOCUMENT_IDS:{len(invalid)}")
    if manifest.get("output_sha256") != sha256(SCORES):
        raise RuntimeError("SCORES_SHA_MISMATCH")
    return manifest, score_by_key, work_by_key, work_by_query, corpus_ids, addendum


def rank_query(
    rows: list[dict[str, Any]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[str, float]]:
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
        value = sum(
            WEIGHTS[source] / (RRF_K + source_ranks[source])
            for source in WEIGHTS
            if source in source_ranks
        )
        scored.append((did, value, source_ranks))
    ordered = sorted(
        scored,
        key=lambda item: (
            -item[1],
            *(item[2].get(source, 10**9) for source in WEIGHTS),
            numeric_id(item[0]),
        ),
    )
    return [(did, value) for did, value, _ in ordered]


def main() -> None:
    manifest, score_by_key, work_by_key, work_by_query, corpus_ids, addendum = verify_inputs()
    predictions: dict[str, list[str]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    candidate_counts = []
    for qid in sorted(work_by_query, key=numeric_id):
        rows = work_by_query[qid]
        candidate_counts.append(len(rows))
        ranked = rank_query(rows, score_by_key)
        top5 = ranked[:5]
        if len(top5) != 5 or len({did for did, _ in top5}) != 5:
            raise RuntimeError(f"TOP5_NOT_UNIQUE:{qid}")
        predictions[qid] = [did for did, _ in top5]
        if len(samples) < 5:
            samples[qid] = [
                {"document_id": did, "rrf_score": score}
                for did, score in top5
            ]
    if len(predictions) != EXPECTED_QUERIES:
        raise RuntimeError("FINAL_QUERY_COVERAGE_MISMATCH")
    if min(candidate_counts) != 20 or max(candidate_counts) != 20:
        raise RuntimeError("CANDIDATE_COUNT_NOT_20")

    payload = {
        qid: {"answer": predictions[qid]}
        for qid in sorted(predictions, key=numeric_id)
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_legal_ir_submission_json(
        payload,
        OUT_JSON,
        expected_question_ids=sorted(predictions, key=numeric_id),
        allowed_document_ids=corpus_ids,
    )
    write_legal_ir_submission_zip(
        payload,
        OUT_ZIP,
        expected_question_ids=sorted(predictions, key=numeric_id),
        allowed_document_ids=corpus_ids,
    )
    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--input",
            str(OUT_JSON),
            "--corpus-manifest",
            str(CORPUS),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if validator.returncode != 0:
        raise RuntimeError(f"CANONICAL_VALIDATOR_FAILED:{validator.stdout}{validator.stderr}")
    validator_zip = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--input",
            str(OUT_ZIP),
            "--corpus-manifest",
            str(CORPUS),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if validator_zip.returncode != 0:
        raise RuntimeError(f"CANONICAL_ZIP_VALIDATOR_FAILED:{validator_zip.stdout}{validator_zip.stderr}")

    final_manifest = {
        "status": "READY_TO_SUBMIT",
        "pipeline": "Dense + BM25 + word-TFIDF KNN -> K20 -> current BGE -> weighted RRF -> Top5",
        "policy": "RETRIEVAL_RRF_NO_LABEL",
        "weights": WEIGHTS,
        "rrf_k": RRF_K,
        "formula": "0.2/(2+dense_rank) + 0.3/(2+bge_rank) + 0.2/(2+knn_word_rank) + 0.3/(2+bm25_rank); absent source contributes 0",
        "queries": len(predictions),
        "qdocs": len(work_by_key),
        "inference_units": EXPECTED_UNITS,
        "private_query_coverage_check": "PASS: exact 2080-ID set match against private-official.json; private answers were not read",
        "private_input": {"path": str(PRIVATE_INPUT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(PRIVATE_INPUT)},
        "worklist": {"path": str(WORKLIST.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(WORKLIST)},
        "gpu_production_manifest": {"path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(MANIFEST)},
        "gpu_scores": {"path": str(SCORES.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SCORES)},
        "provenance_addendum": {"path": str(ADDENDUM.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ADDENDUM)},
        "submission_json": {"path": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(OUT_JSON)},
        "submission_zip": {"path": str(OUT_ZIP.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(OUT_ZIP)},
        "gpu": manifest.get("actual_gpu"),
        "validator": {
            "json": {"status": "PASS", "stdout": validator.stdout.strip()},
            "zip": {"status": "PASS", "stdout": validator_zip.stdout.strip()},
        },
        "sanity": {
            "candidates_min_median_max": [min(candidate_counts), 20, max(candidate_counts)],
            "queries_exactly_20_candidates": sum(value == 20 for value in candidate_counts),
            "queries_exactly_5_predictions": len(predictions),
            "duplicate_predictions": 0,
            "invalid_document_ids": 0,
            "first_five_queries": samples,
        },
        "private_answers_read": False,
        "original_gpu_manifest_modified": False,
        "provenance": "PASS",
        "gpu_runs_this_task": 0,
        "modal_inference_this_task": 0,
    }
    OUT_MANIFEST.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
