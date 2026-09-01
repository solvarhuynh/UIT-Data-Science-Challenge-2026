"""Build the authorized all-NO_OP P5 deployment without public model scoring.

This is a deployment-only program.  It first proves the K77/36-feature
construction is unchanged on F1--F4, then reuses the proven public baseline.
It never loads a P1/P4 model, computes a public score, or reads public labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/task1"
ARTIFACTS = ROOT / "artifacts/task1"
OUTPUT = ARTIFACTS / "submission_p5_all_noop"
EQUIVALENCE_REPORT = REPORTS / "public_k77_feature_equivalence_report.json"
BASELINE_ZIP = ARTIFACTS / "recovery_096/public_anchor_093/submission_093.zip"
BASELINE_MANIFEST = ARTIFACTS / "recovery_096/public_anchor_093/producer_manifest.json"
V3A_POLICY_REPORT = ARTIFACTS / "recovery_096/final_public_v3/public_v3a_policy_report.json"
PUBLIC_QUESTIONS = ROOT / "data/raw/btc/LegalIR/public-official.json"
PUBLIC_ID_MANIFEST = ARTIFACTS / "recovery_096/public_anchor_093/public_question_ids.json"
CORPUS_MANIFEST = ARTIFACTS / "corpus_document_ids.json"
CANONICAL_BUILDER = ROOT / "scripts/analysis/step2_k77_oof_policy_realizability.py"
P1_CONTRACT = REPORTS / "step2p1_model_contract.json"

CONTRACT_FILES = [
    REPORTS / "production_p1_fit_config.json",
    REPORTS / "production_p4_fit_config.json",
    REPORTS / "production_model_finalfit_report.json",
    REPORTS / "production_threshold.json",
    REPORTS / "production_threshold_selection_report.json",
    REPORTS / "step4_workflow_a_finalization.md",
    REPORTS / "progress_log.md",
    ROOT / "docs/task1/workflow_A_new.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fail(code: str, details: dict[str, Any]) -> None:
    payload = {"status": code, **details}
    write_json(EQUIVALENCE_REPORT, payload)
    raise RuntimeError(code)


def preflight() -> dict[str, str]:
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in CONTRACT_FILES}
    p1 = read_json(REPORTS / "production_p1_fit_config.json")
    p4 = read_json(REPORTS / "production_p4_fit_config.json")
    finalfit = read_json(REPORTS / "production_model_finalfit_report.json")
    threshold = read_json(REPORTS / "production_threshold.json")
    selection = read_json(REPORTS / "production_threshold_selection_report.json")
    finalization = (REPORTS / "step4_workflow_a_finalization.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs/task1/workflow_A_new.md").read_text(encoding="utf-8")
    checks = {
        "p1_status": p1.get("status") == "PASS",
        "p4_status": p4.get("status") == "PASS",
        "finalfit_status": finalfit.get("status") == "PASS",
        "threshold_status": threshold.get("status") == "PASS" and selection.get("status") == "PASS",
        "threshold_is_infinity": math.isinf(float(threshold.get("threshold"))) and float(threshold["threshold"]) > 0,
        "K_77": p1.get("K") == p4.get("K") == threshold.get("K") == 77,
        "feature_count_36": p1.get("feature_count") == p4.get("feature_count") == threshold.get("feature_count") == 36,
        "beta_005": threshold.get("beta") == 0.05,
        "p4_target": p4.get("target") == threshold.get("p4_target") == "BENEFICIAL_vs_NEUTRAL_HARMFUL",
        "fold0_not_used": not any((p1, p4, finalfit, threshold, selection)[i].get(key, False) for i, key in ((0, "fold0_used"), (1, "fold0_used"), (2, "Fold0_used"), (3, "fold0_used"), (4, "Fold0_used"))),
        "public_labels_not_used": not any((p1, p4, finalfit, threshold, selection)[i].get(key, False) for i, key in ((0, "public_labels_used"), (1, "public_labels_used"), (2, "public_labels_used"), (3, "public_labels_used"), (4, "public_labels_used"))),
        "no_finite_substitution": math.isinf(float(selection.get("selected_threshold"))) and selection.get("tie_break_resolution") == "max(gain_sum, precision_delta, threshold): largest threshold selected",
        "workflow_a_closed": "Workflow A: **CLOSED**" in finalization and "WORKFLOW_A_CLOSED" in workflow,
    }
    if not all(checks.values()):
        fail(
            "BLOCKED_DEPLOYMENT_CONTRACT_HASH_OR_STATE_MISMATCH",
            {"operation": "K77_36_FEATURE_EQUIVALENCE", "preflight_checks": checks, "contract_sha256": hashes},
        )
    return hashes


def question_metadata(train_path: Path, targets: set[str]) -> dict[str, dict[str, str]]:
    """Read only target IDs and question text; answer/gold fields are never accessed."""
    raw = train_path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    i = 0
    while i < len(raw) and raw[i].isspace():
        i += 1
    if raw[i] != "{":
        raise ValueError("unexpected train metadata root")
    i += 1
    records: dict[str, dict[str, str]] = {}
    while True:
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] == "}":
            break
        key, i = decoder.raw_decode(raw, i)
        while raw[i].isspace():
            i += 1
        if raw[i] != ":":
            raise ValueError("invalid train metadata")
        i += 1
        while i < len(raw) and raw[i].isspace():
            i += 1
        value, i = decoder.raw_decode(raw, i)
        qid = str(key)
        if qid in targets:
            records[qid] = {"question": str(value.get("question", ""))}
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i < len(raw) and raw[i] == ",":
            i += 1
    if set(records) != targets:
        raise ValueError("question metadata coverage invalid")
    return records


def production_label_free_actions(
    parent: dict[str, Any],
    folds: dict[str, int],
    records: dict[str, dict[str, str]],
    baseline: dict[str, list[str]],
    candidates: dict[str, list[dict[str, Any]]],
    columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, int]]]]:
    """The canonical action construction with only truth-dependent fields removed."""
    rows: list[dict[str, Any]] = []
    candidate_orders: dict[str, list[tuple[str, int]]] = {}
    for query_id in sorted(folds, key=lambda value: int(value) if value.isdigit() else value):
        base = baseline[query_id]
        canonical_candidates = candidates[query_id]
        candidate_orders[query_id] = [(item["doc_id"], int(item["union_rank"])) for item in canonical_candidates]
        by_doc = {item["doc_id"]: item for item in canonical_candidates}
        question = records[query_id]["question"]
        qtokens, qchars = float(len(question.split())), float(len(question))
        incoming = [item for item in canonical_candidates if item["union_rank"] <= parent["K"] and item["doc_id"] not in set(base)]
        for item in incoming:
            for drop_rank in (4, 5):
                dropped_id = base[drop_rank - 1]
                dropped = by_doc.get(dropped_id, {"doc_id": dropped_id, "union_rank": 1e9, "source_ranks": {}, "source_support": 0})
                features: dict[str, float] = {}
                for name in parent["V3_BASE"]:
                    if f"incoming_{name}" in columns:
                        features[f"incoming_{name}"] = parent["doc_value"](item, name, qtokens, qchars, None)
                        features[f"dropped_{name}"] = parent["doc_value"](dropped, name, qtokens, qchars, drop_rank)
                for name in parent["V3_DIFF"]:
                    if f"diff_{name}" in columns:
                        features[f"diff_{name}"] = features[f"incoming_{name}"] - features[f"dropped_{name}"]
                features["dropped_baseline_rank"] = float(drop_rank)
                features["incoming_is_baseline_top5"] = 0.0
                rows.append(
                    {
                        "query_id": query_id,
                        "fold": folds[query_id],
                        "incoming_doc_id": item["doc_id"],
                        "incoming_union_rank": int(item["union_rank"]),
                        "drop_rank": drop_rank,
                        "dropped_doc_id": dropped_id,
                        "features": features,
                    }
                )
    return rows, candidate_orders


def equivalence_gate(contract_hashes: dict[str, str]) -> dict[str, Any]:
    parent = runpy.run_path(str(CANONICAL_BUILDER))
    folds = parent["target_folds"]()
    targets = set(folds)
    columns = read_json(P1_CONTRACT)["feature_columns"]
    if len(columns) != 36 or len(set(columns)) != 36 or parent["K"] != 77:
        fail("BLOCKED_K77_FEATURE_PIPELINE_EQUIVALENCE_FAILURE", {"reason": "canonical schema or K mismatch", "contract_sha256": contract_hashes})
    records = question_metadata(parent["TRAIN"], targets)
    baseline, _ = parent["load_baseline"](targets, folds)
    candidates, _ = parent["load_candidates"](targets, folds)

    # The unmodified canonical builder requires a gold mapping only to attach labels.
    # A synthetic non-gold sentinel leaves action/features unchanged and proves that
    # the comparison is fully structural rather than an evaluation.
    sentinel = {query_id: {baseline[query_id][0]} for query_id in targets}
    canonical_by_fold, incoming_count = parent["make_actions"](folds, records, sentinel, baseline, candidates, columns)
    canonical_rows = [row for fold in range(1, 5) for row in canonical_by_fold[fold]]
    production_rows, production_candidate_orders = production_label_free_actions(parent, folds, records, baseline, candidates, columns)
    canonical_candidate_orders = {query_id: [(item["doc_id"], int(item["union_rank"])) for item in candidates[query_id]] for query_id in targets}

    action_fields = ("query_id", "fold", "incoming_doc_id", "incoming_union_rank", "drop_rank", "dropped_doc_id")
    action_sort_key = lambda row: (int(row["query_id"]) if row["query_id"].isdigit() else row["query_id"], row["incoming_union_rank"], row["drop_rank"], row["incoming_doc_id"], row["dropped_doc_id"])
    canonical_rows.sort(key=action_sort_key)
    production_rows.sort(key=action_sort_key)
    mismatch_count = 0
    max_error = 0.0
    action_identity_match = len(canonical_rows) == len(production_rows)
    categorical_exact_match = True
    numeric_match = True
    if action_identity_match:
        for canonical, production in zip(canonical_rows, production_rows):
            if tuple(canonical[field] for field in action_fields) != tuple(production[field] for field in action_fields):
                action_identity_match = False
                mismatch_count += 1
                continue
            # Feature ordering is the frozen matrix-column contract (`columns`),
            # not incidental dictionary insertion order while action fields are built.
            if set(canonical["features"]) != set(columns) or set(production["features"]) != set(columns):
                mismatch_count += 1
                numeric_match = False
                continue
            for column in columns:
                left, right = canonical["features"][column], production["features"][column]
                error = abs(float(left) - float(right))
                max_error = max(max_error, error)
                if error > 1e-9:
                    numeric_match = False
                    mismatch_count += 1
                if column.endswith("_rank") or column.endswith("_missing") or "source_support" in column or column.endswith("_length") or column.endswith("top5"):
                    if left != right:
                        categorical_exact_match = False
                        mismatch_count += 1
    else:
        mismatch_count += abs(len(canonical_rows) - len(production_rows))
    candidate_identity_match = canonical_candidate_orders == production_candidate_orders
    candidate_order_match = candidate_identity_match
    report = {
        "status": "PASS" if all((len(folds) == 5600, incoming_count == 403322, len(canonical_rows) == 806644, len(production_rows) == 806644, candidate_identity_match, candidate_order_match, action_identity_match, columns == read_json(REPORTS / "production_p1_fit_config.json")["feature_names"], categorical_exact_match, numeric_match, max_error <= 1e-9, mismatch_count == 0)) else "BLOCKED_K77_FEATURE_PIPELINE_EQUIVALENCE_FAILURE",
        "operation": "K77_36_FEATURE_EQUIVALENCE",
        "K": 77,
        "feature_count": 36,
        "queries_checked": len(folds),
        "candidate_identity_match": candidate_identity_match,
        "candidate_order_match": candidate_order_match,
        "action_identity_match": action_identity_match,
        "feature_schema_match": columns == read_json(REPORTS / "production_p1_fit_config.json")["feature_names"],
        "feature_order_match": columns == read_json(REPORTS / "production_p4_fit_config.json")["feature_names"],
        "categorical_exact_match": categorical_exact_match,
        "numeric_tolerance": 1e-9,
        "max_numeric_abs_error": max_error,
        "mismatch_count": mismatch_count,
        "incoming_candidate_count": incoming_count,
        "action_count": len(production_rows),
        "gold_or_truth_loaded": False,
        "synthetic_non_gold_sentinel_used_only_for_canonical_reference_labels": True,
        "fold0_used": False,
        "public_labels_used": False,
        "canonical_sources": [
            "scripts/analysis/step2_k77_oof_policy_realizability.py",
            "reports/task1/step2p1_model_contract.json",
            "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
            "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl",
        ],
        "production_builder_sources": ["scripts/submission/build_frozen_p5_all_noop_submission.py"],
        "production_contract_sha256": contract_hashes,
    }
    write_json(EQUIVALENCE_REPORT, report)
    if report["status"] != "PASS":
        raise RuntimeError("BLOCKED_K77_FEATURE_PIPELINE_EQUIVALENCE_FAILURE")
    return report


def verified_baseline() -> tuple[bytes, dict[str, Any]]:
    manifest = read_json(BASELINE_MANIFEST)
    v3a = read_json(V3A_POLICY_REPORT)
    baseline_hash = sha256(BASELINE_ZIP)
    if not (
        manifest.get("status") == "EXACT_BYTE_REPRODUCED"
        and manifest.get("anchor", {}).get("sha256") == baseline_hash
        and manifest.get("anchor", {}).get("question_count") == 1000
        and manifest.get("anchor", {}).get("documents_per_question") == 5
        and manifest.get("constraints", {}).get("public_anchor_used_as_gold") is False
        and v3a.get("baseline_sha256") == baseline_hash
        and v3a.get("swapped_query_count", 0) > 0
        and v3a.get("no_public_labels_used") is True
        and v3a.get("no_fold_used") is True
    ):
        raise RuntimeError("BLOCKED_CANONICAL_PUBLIC_BASELINE_ARTIFACT_UNRESOLVED")
    public_question_ids = json.loads(PUBLIC_ID_MANIFEST.read_text(encoding="utf-8"))
    public_source = json.loads(PUBLIC_QUESTIONS.read_text(encoding="utf-8"))
    if not isinstance(public_question_ids, list) or not isinstance(public_source, dict) or set(public_question_ids) != set(public_source) or len(public_question_ids) != 1000:
        raise RuntimeError("BLOCKED_CANONICAL_PUBLIC_BASELINE_ARTIFACT_UNRESOLVED")
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        if archive.namelist() != ["submission.json"]:
            raise RuntimeError("BLOCKED_CANONICAL_PUBLIC_BASELINE_ARTIFACT_UNRESOLVED")
        payload = archive.read("submission.json")
    source = {
        "path": str(BASELINE_ZIP.relative_to(ROOT)).replace("\\", "/"),
        "sha256": baseline_hash,
        "public_query_count": 1000,
        "public_id_manifest": str(PUBLIC_ID_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "pre_action_proof": "public_v3a_policy_report.baseline_sha256 matches this source and V3A changed 328 queries from it",
        "public_not_fold0": True,
        "public_labels_used": False,
        "policy_actions_applied": False,
    }
    return payload, source


def validate_and_compare(submission: Path, baseline_bytes: bytes) -> tuple[dict[str, Any], str]:
    baseline_payload = json.loads(baseline_bytes.decode("utf-8-sig"))
    final_payload = json.loads(submission.read_text(encoding="utf-8-sig"))
    mismatches = 0
    if set(baseline_payload) != set(final_payload):
        mismatches = len(set(baseline_payload) ^ set(final_payload))
    else:
        for query_id in baseline_payload:
            if baseline_payload[query_id] != final_payload[query_id]:
                mismatches += 1
    if mismatches:
        raise RuntimeError("BLOCKED_ALL_NOOP_BASELINE_DIFFERENCE")
    command = [
        sys.executable,
        str(ROOT / "scripts/submission/validate_legal_ir_submission.py"),
        "--input", str(submission),
        "--questions", str(PUBLIC_ID_MANIFEST),
        "--corpus-manifest", str(CORPUS_MANIFEST),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
    validation_text = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"validator failed: {validation_text}")
    return {"query_mismatches": 0, "baseline_query_count": len(baseline_payload), "submission_query_count": len(final_payload)}, validation_text


def append_progress(baseline_source: str, submission: Path, submission_hash: str) -> None:
    progress = REPORTS / "progress_log.md"
    text = progress.read_text(encoding="utf-8")
    marker = "Task: Frozen P5 All-NO_OP Public Submission"
    if marker in text:
        raise RuntimeError("progress entry already exists; refusing duplicate deployment entry")
    entry = f"""
=== PROGRESS_LOG_ENTRY START ===
Task: Frozen P5 All-NO_OP Public Submission
Date: {date.today().isoformat()}
Technical status: PASS
Operation: DEPLOYMENT_OF_FROZEN_P5_WITH_ALL_NO_OP
K77/36-feature equivalence gate: PASS
Production threshold: Infinity
Finite-threshold substitution: false
All public queries NO_OP: true
Public P1 inference executed: false
Public P4 inference executed: false
P5 fusion inference executed: false
Baseline public artifact: {baseline_source}
Baseline/public final mismatch queries: 0
Fold0 used: false
Public labels used: false
Leaderboard feedback used: false
Tuning executed: false
Validator: PASS
Upload-ready submission: {submission.relative_to(ROOT).as_posix()}
Submission SHA256: {submission_hash}
Scientific improvement claimed: false
Workflow A reopened: false
Workflow B evidence created: false
=== PROGRESS_LOG_ENTRY END ===
"""
    progress.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")


def main() -> None:
    contract_hashes = preflight()
    equivalence = equivalence_gate(contract_hashes)
    baseline_bytes, baseline = verified_baseline()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "public_submission.json"
    submission.write_bytes(baseline_bytes)
    comparison, validation_text = validate_and_compare(submission, baseline_bytes)
    (OUTPUT / "submission_validation.txt").write_text(validation_text, encoding="utf-8")
    submission_hash = sha256(submission)
    manifest = {
        "status": "PASS", "workflow": "A", "policy": "P5",
        "deployment_classification": "DEPLOYMENT_OF_FROZEN_P5_WITH_ALL_NO_OP",
        "K": 77, "feature_count": 36, "beta": 0.05, "production_threshold": "Infinity",
        "all_queries_no_op": True, "public_model_scoring_executed": False,
        "baseline_public_source": baseline["path"], "baseline_source_sha256": baseline["sha256"],
        "equivalence_gate": "PASS", "equivalence_report": str(EQUIVALENCE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "baseline_equivalence_query_mismatches": comparison["query_mismatches"],
        "fold0_used": False, "public_labels_used": False, "leaderboard_feedback_used": False, "tuning_performed": False,
        "public_query_count": comparison["baseline_query_count"], "submission_query_count": comparison["submission_query_count"],
        "validator_status": "PASS", "upload_ready_submission": str(submission.relative_to(ROOT)).replace("\\", "/"),
        "submission_sha256": submission_hash, "scientific_improvement_claimed": False,
        "production_contract_sha256": contract_hashes,
    }
    write_json(OUTPUT / "submission_manifest.json", manifest)
    deployment = {
        "status": "PASS", "operation": "DEPLOYMENT_OF_FROZEN_P5_WITH_ALL_NO_OP",
        "infinity_accepted_by_professor": True, "finite_threshold_substitution_authorized": False,
        "all_fused_scores_assumed_finite_under_frozen_contract": True,
        "apply_count": 0, "no_op_count": comparison["submission_query_count"],
        "public_p1_inference_executed": False, "public_p4_inference_executed": False, "p5_fusion_computation_executed": False,
        "baseline_output_reused": True, "equivalence_gate": equivalence["status"], "baseline_provenance": "PASS",
        "baseline_vs_final_mismatch": 0, "public_labels_used": False, "fold0_used": False,
        "submission_validator": "PASS", "output_path": manifest["upload_ready_submission"], "output_sha256": submission_hash,
    }
    write_json(OUTPUT / "all_noop_deployment_report.json", deployment)
    append_progress(baseline["path"], submission, submission_hash)
    print(json.dumps({"status": "PASS", "submission": manifest["upload_ready_submission"], "sha256": submission_hash, "queries": comparison["submission_query_count"]}, allow_nan=False))


if __name__ == "__main__":
    main()
