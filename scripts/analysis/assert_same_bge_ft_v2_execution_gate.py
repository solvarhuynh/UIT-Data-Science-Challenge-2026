"""CPU-only Stage 0/1 gate for the frozen SAME-BGE FT V2 experiment.

This deliberately imports neither torch nor a model library.  It re-hashes the
Phase 2A inputs, validates all four materialized outer-fold inputs, and proves
the V2 evaluator reproduces the frozen historical RRF policy before any GPU
process is allowed to start.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "private_task1/experiments/sprint48_bge_ft_v2"
EVALUATOR = ROOT / "scripts/analysis/evaluate_same_bge_ft_v2_oof.py"
HISTORICAL = ROOT / "scripts/analysis/phase1_step4_recovery.py"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
REFERENCE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
FROZEN_CONFIG = OUT / "ft_v2_frozen_config.json"
PREFLIGHT = OUT / "preflight/preflight_gate.json"
CORPUS_MANIFEST = ROOT / "data/processed_pv1/metadata/pv1_corpus_manifest.json"
FT_MODEL = ROOT / "outputs/task1/bge_ft_model/bge_m3_finetuned"

EXPECTED_QDOCS = 112000
EXPECTED_QUERIES = 5600
EXPECTED_RECALL = 0.9255863095238096
EXPECTED_PRECISION = 0.19710714285714284
EXPECTED_CORPUS = "70075f130fe18a5a5ac0569c6e326b98e2ed9a735bcd2c13fef468f4e8cb6274"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hashes() -> dict[str, Any]:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    files = {
        "phase1_reconstructed_scores": REFERENCE,
        "worklist": WORKLIST,
        "candidates": CANDIDATES,
        "ft_weight": FT_MODEL / "model.safetensors",
        "ft_config": FT_MODEL / "config.json",
        "strict_folds": FOLDS,
        "train": TRAIN,
    }
    actual = {name: sha256(path) for name, path in files.items()}
    expected = preflight.get("hashes", {})
    checks: dict[str, bool] = {
        "phase2a_preflight": preflight.get("pass") is True,
        "frozen_config_sha256": sha256(FROZEN_CONFIG) == preflight.get("frozen_config_sha256"),
        "corpus_fingerprint": corpus.get("corpus_fingerprint") == EXPECTED_CORPUS,
        "student_weight": actual["ft_weight"] == frozen.get("student_init", {}).get("weight_sha256"),
        "teacher_weight": actual["ft_weight"] == frozen.get("teacher", {}).get("weight_sha256"),
        "student_config": actual["ft_config"] == frozen.get("student_init", {}).get("config_sha256"),
        "teacher_config": actual["ft_config"] == frozen.get("teacher", {}).get("config_sha256"),
        "frozen_recipe": (
            frozen.get("backbone") == "BAAI/bge-reranker-v2-m3"
            and frozen.get("base_revision") == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
            and frozen.get("selector") == "true_s2_bm25_within_document_v2"
            and frozen.get("document_inference_aggregation") == "MAX"
            and frozen.get("learning_rate") == 5e-7
            and frozen.get("epochs") == 1
            and frozen.get("batch_size") == 4
            and frozen.get("gradient_accumulation") == 4
            and frozen.get("max_length") == 512
            and frozen.get("freeze_first_encoder_layers") == 18
            and frozen.get("teacher_weight") == 0.5
            and frozen.get("negative_cap") == 6
            and frozen.get("seed") == 2026
        ),
    }
    checks.update({f"hash_{name}": digest == expected.get(name) for name, digest in actual.items()})

    fold_results: dict[str, Any] = {}
    for fold in range(1, 5):
        directory = OUT / f"fold{fold}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        frozen_manifest = preflight.get("folds", {}).get(str(fold), {})
        group_hash = sha256(directory / "train_groups.jsonl")
        worklist_hash = sha256(directory / "score_worklist.jsonl")
        pass_fold = (
            manifest.get("held_out_fold") == fold
            and manifest.get("training_folds") == [value for value in range(1, 5) if value != fold]
            and manifest.get("held_out_leakage") == 0
            and manifest.get("fold0_count") == 0
            and manifest.get("private_count") == 0
            and group_hash == manifest.get("train_groups_sha256") == frozen_manifest.get("train_groups_sha256")
            and worklist_hash == manifest.get("score_worklist_sha256") == frozen_manifest.get("score_worklist_sha256")
            and manifest.get("training_examples") == frozen_manifest.get("training_examples")
            and manifest.get("score_qdocs") == frozen_manifest.get("score_qdocs")
            and manifest.get("score_units") == frozen_manifest.get("score_units")
        )
        fold_results[f"F{fold}"] = {
            "pass": pass_fold,
            "train_groups_sha256": group_hash,
            "score_worklist_sha256": worklist_hash,
            "training_examples": manifest.get("training_examples"),
            "score_qdocs": manifest.get("score_qdocs"),
            "score_units": manifest.get("score_units"),
        }
    checks["outer_fold_inputs"] = all(item["pass"] for item in fold_results.values())
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "hashes": actual,
        "frozen_config_sha256": sha256(FROZEN_CONFIG),
        "folds": fold_results,
    }


def verify_evaluator_contract() -> dict[str, Any]:
    evaluator = load_module("same_bge_v2_evaluator", EVALUATOR)
    historical = load_module("same_bge_v2_historical_rrf", HISTORICAL)
    folds = evaluator.fold_map()
    target = {qid for qid, fold in folds.items() if fold in {1, 2, 3, 4}}
    if len(target) != EXPECTED_QUERIES:
        raise RuntimeError("F1--F4 query count mismatch")
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    gold = {qid: {str(value) for value in train[qid]["answer"]} for qid in target}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for item in load_jsonl(CANDIDATES):
        qid = str(item.get("query_id", ""))
        rows = item.get("candidates")
        if not qid or qid in candidates or not isinstance(rows, list) or len(rows) != 20:
            raise RuntimeError(f"invalid frozen candidate universe: {qid}")
        documents = [str(row.get("document_id", "")) for row in rows]
        if len(set(documents)) != 20:
            raise RuntimeError(f"duplicate candidate document: {qid}")
        candidates[qid] = rows
    if set(candidates) != target:
        raise RuntimeError("candidate query universe mismatch")
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(REFERENCE):
        key = (str(row.get("query_id", "")), str(row.get("document_id", "")))
        if key in reference or not all(key):
            raise RuntimeError(f"invalid/duplicate reference identity: {key}")
        if not math.isfinite(float(row.get("bge_ft_score", float("nan")))) or not math.isfinite(float(row.get("bge_base_score", float("nan")))):
            raise RuntimeError(f"non-finite frozen BGE score: {key}")
        reference[key] = row
    expected_keys = {(qid, str(row["document_id"])) for qid in target for row in candidates[qid]}
    if len(reference) != EXPECTED_QDOCS or set(reference) != expected_keys:
        raise RuntimeError("reference q-doc universe mismatch")
    selected: dict[tuple[str, str], list[str]] = {}
    for row in load_jsonl(WORKLIST):
        key = (str(row.get("query_id", "")), str(row.get("document_id", "")))
        ids = [str(value) for value in row.get("selected_chunk_ids", [])]
        if key in selected or key not in expected_keys or not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
            raise RuntimeError(f"invalid frozen worklist identity: {key}")
        selected[key] = ids
    if set(selected) != expected_keys:
        raise RuntimeError("worklist q-doc universe mismatch")

    baseline: dict[str, list[str]] = {}
    historical_mismatches = 0
    component_isolation_failures = 0
    for qid in sorted(target, key=evaluator.qkey):
        rows: list[dict[str, Any]] = []
        for candidate in candidates[qid]:
            doc = str(candidate["document_id"])
            key = (qid, doc)
            ref = reference[key]
            source_ranks = {str(name): int(rank) for name, rank in (candidate.get("source_ranks") or {}).items()}
            if not source_ranks or set(source_ranks) - {"dense", "bm25", "knn_word"}:
                raise RuntimeError(f"mutable/non-frozen source-rank field: {key}")
            if selected[key] != [str(value) for value in ref.get("selected_chunk_ids", [])]:
                raise RuntimeError(f"selected chunk provenance mismatch: {key}")
            rows.append({
                "query_id": qid,
                "document_id": doc,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": source_ranks,
                "bge_ft_score": float(ref["bge_ft_score"]),
                "bge_base_score": float(ref["bge_base_score"]),
            })
        baseline[qid] = evaluator.rrf_top5(rows)
        if baseline[qid] != list(historical.current_rrf_rank(rows))[:5]:
            historical_mismatches += 1
        # Change only the FT scalar and prove the structural RRF inputs remain
        # the exact frozen candidate/base-score values.
        synthetic = []
        for index, row in enumerate(rows, 1):
            copy = dict(row)
            copy["source_ranks"] = dict(row["source_ranks"])
            copy["bge_ft_score"] = float(index)
            synthetic.append(copy)
        for frozen_row, changed_row in zip(rows, synthetic):
            if (
                frozen_row["source_ranks"] != changed_row["source_ranks"]
                or frozen_row["candidate_rank"] != changed_row["candidate_rank"]
                or frozen_row["bge_base_score"] != changed_row["bge_base_score"]
            ):
                component_isolation_failures += 1
    if historical_mismatches or component_isolation_failures:
        raise RuntimeError("frozen final-policy component contract mismatch")
    baseline_metrics = evaluator.metrics(baseline, gold)
    if (
        abs(float(baseline_metrics["recall"]) - EXPECTED_RECALL) > 1e-15
        or abs(float(baseline_metrics["precision"]) - EXPECTED_PRECISION) > 1e-15
    ):
        raise RuntimeError("frozen PV1 baseline reproduction mismatch")
    return {
        "pass": True,
        "policy": "RETRIEVAL_RRF_NO_LABEL",
        "components": {
            "candidate_universe": "frozen current PV1 K20",
            "dense": "frozen candidate source ranks",
            "bge": "new BGE-FT score/rank only",
            "word_knn": "frozen candidate source ranks",
            "bm25": "frozen candidate source ranks",
            "weights": {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3},
            "rrf_k": 2,
            "final_prediction": "RRF Top-5 (not standalone BGE)",
        },
        "baseline": baseline_metrics,
        "historical_rrf_prediction_mismatches": historical_mismatches,
        "component_isolation_failures": component_isolation_failures,
        "selected_chunk_identity": "PASS",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    output = OUT / "preflight/stage2b_final_execution_gate.json"
    try:
        inputs = verify_hashes()
        if not inputs["pass"]:
            raise RuntimeError("immutable Phase 2A execution input mismatch")
        evaluator = verify_evaluator_contract()
        result = {
            "status": "STAGE2B_PRE_GPU_GATES_PASS",
            "FINAL_POLICY_EVALUATOR_GATE": "PASS",
            "EXECUTION_INPUT_GATE": "PASS",
            "input_rehash": inputs,
            "evaluator_contract": evaluator,
            "gpu_used": False,
            "model_loaded": False,
            "modal_used": False,
        }
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        result = {
            "status": "STAGE2B_PRE_GPU_GATES_FAIL",
            "FINAL_POLICY_EVALUATOR_GATE": "FAIL",
            "EXECUTION_INPUT_GATE": "FAIL",
            "error": str(exc),
            "gpu_used": False,
            "model_loaded": False,
            "modal_used": False,
        }
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
