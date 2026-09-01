"""Read-only provenance audit of the Task1 public incumbent versus P5 all-NO_OP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/task1"
ARTIFACTS = ROOT / "artifacts/task1"
INCUMBENT_ZIP = ARTIFACTS / "submission.zip"
BASELINE_ZIP = ARTIFACTS / "submission_p5_all_noop/submission.zip"
V3A_PREDICTIONS = ARTIFACTS / "recovery_096/final_public_v3/public_v3a_predictions.json"
V3A_REPORT = ARTIFACTS / "recovery_096/final_public_v3/public_v3a_policy_report.json"
V3A_FEATURES = ARTIFACTS / "recovery_096/final_public_v3/public_frozen_features.jsonl"
V3A_FEATURE_REPORT = ARTIFACTS / "recovery_096/final_public_v3/public_frozen_features_report.json"
V3A_CANDIDATES = ARTIFACTS / "recovery_096/final_public_v3/public_candidate_union.jsonl"
V3A_SHORTLIST = ARTIFACTS / "recovery_096/final_public_v3/public_shortlist_report.json"
V3A_MANIFEST = ARTIFACTS / "recovery_096/v3_residual/policy/v3a_final_fit_manifest.json"
V3A_TRAINING = ARTIFACTS / "recovery_096/v3_residual/policy/policy_training_report.json"
V3A_MODEL = ARTIFACTS / "recovery_096/v3_residual/policy/v3a_final_fit.joblib"
BASELINE_ANCHOR = ARTIFACTS / "recovery_096/public_anchor_093/submission_093.zip"
DELTA = REPORTS / "pre_workflow_b_incumbent_delta.jsonl"
JSON_REPORT = REPORTS / "pre_workflow_b_incumbent_forensic_report.json"
MD_REPORT = REPORTS / "pre_workflow_b_incumbent_forensic_report.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_submission(path: Path) -> tuple[dict[str, Any], str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["submission.json"]:
            raise ValueError(f"unexpected archive members in {path}: {names}")
        raw = archive.read("submission.json")
    return json.loads(raw.decode("utf-8-sig")), sha256_bytes(raw), names


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    incumbent, incumbent_member_hash, incumbent_members = zip_submission(INCUMBENT_ZIP)
    baseline, baseline_member_hash, baseline_members = zip_submission(BASELINE_ZIP)
    expected_incumbent = "d00d5612efa9ee354a9629a399f967b1319c4decdf121c090d03e8a90e396a56"
    expected_baseline = "f1462cb1280ef942f0ceee1dcaf9e384bd90c8de976342ae41484a7b1b77c00f"
    if incumbent_member_hash != expected_incumbent or baseline_member_hash != expected_baseline:
        raise RuntimeError("BLOCKED_EXACT_INCUMBENT_ARTIFACT_NOT_FOUND")
    if set(incumbent) != set(baseline) or len(incumbent) != 1000:
        raise RuntimeError("submission coverage mismatch")

    v3a_items = json.loads(V3A_PREDICTIONS.read_text(encoding="utf-8"))
    v3a_submission = {str(item["id"]): {"answer": [str(doc) for doc in item["documents"]]} for item in v3a_items}
    if v3a_submission != incumbent:
        raise RuntimeError("incumbent does not exactly match V3A prediction content")
    feature_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in jsonl(V3A_FEATURES):
        feature_lookup[(str(row["query_id"]), str(row["doc_id"]))] = row
    candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in jsonl(V3A_CANDIDATES):
        for candidate in row["candidates"]:
            candidate_lookup[(str(row["query_id"]), str(candidate["doc_id"]))] = candidate

    manifest = json.loads(V3A_MANIFEST.read_text(encoding="utf-8"))
    training = json.loads(V3A_TRAINING.read_text(encoding="utf-8"))
    v3a_report = json.loads(V3A_REPORT.read_text(encoding="utf-8"))
    shortlisted = json.loads(V3A_SHORTLIST.read_text(encoding="utf-8"))
    frozen_features = json.loads(V3A_FEATURE_REPORT.read_text(encoding="utf-8"))
    config = manifest["production_final_fit"]
    hyperparameters = manifest["hyperparameters"]

    rows: list[dict[str, Any]] = []
    rank_counts: Counter[int] = Counter()
    for query_id in sorted(incumbent, key=lambda value: int(value)):
        old = [str(value) for value in baseline[query_id]["answer"]]
        new = [str(value) for value in incumbent[query_id]["answer"]]
        changed = [index for index, (left, right) in enumerate(zip(old, new), 1) if left != right]
        if not changed:
            continue
        if len(changed) != 1 or len(old) != 5 or len(new) != 5:
            raise RuntimeError(f"non one-swap change: {query_id}")
        rank = changed[0]
        incoming, dropped = new[rank - 1], old[rank - 1]
        candidate = candidate_lookup.get((query_id, incoming))
        feature = feature_lookup.get((query_id, incoming))
        action_identity = bool(candidate and feature and int(feature["union_rank"]) == int(candidate["union_rank"]))
        if not action_identity:
            raise RuntimeError(f"V3A action identity cannot be traced: {query_id}")
        rank_counts[rank] += 1
        rows.append({
            "query_id": query_id,
            "baseline_top5": old,
            "incumbent_top5": new,
            "position_changed": rank,
            "old_document": dropped,
            "new_document": incoming,
            "one_swap": True,
            "top1_top2_top3_changed": False,
            "rank4_changed": rank == 4,
            "rank5_changed": rank == 5,
            "incoming_doc_id": incoming,
            "dropped_doc_id": dropped,
            "drop_rank": rank,
            "candidate_union_rank": int(candidate["union_rank"]),
            "policy_score_or_probability": None,
            "policy_score_status": "NOT_PERSISTED_NOT_RECOMPUTED",
            "threshold_used": hyperparameters["threshold"],
            "harm_weight": hyperparameters["harm_weight"],
            "model_config_responsible": "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json",
            "model_sha256": manifest["model_sha256"],
            "source_artifact": "artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json",
            "source_script": "scripts/beam/task1_v3_residual/train_residual_policy.py::apply_public_model",
            "classification": "EXACTLY_TRACED_TO_V3A",
        })
    if len(rows) != 328 or rank_counts != Counter({4: 110, 5: 218}):
        raise RuntimeError("known incumbent delta did not reproduce")
    with DELTA.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    trace_counts = Counter(row["classification"] for row in rows)
    evidence_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in [INCUMBENT_ZIP, BASELINE_ZIP, V3A_PREDICTIONS, V3A_REPORT, V3A_FEATURES, V3A_CANDIDATES, V3A_MANIFEST, V3A_MODEL]
    }
    active_score_contract = "UNRESOLVED"
    score_evidence = {
        "repository_scoring_program": "TOP5_SET_RECALL_COMPATIBLE",
        "repository_local_evaluator": "TOP5_SET_RECALL_COMPATIBLE",
        "historical_evaluator": "TOP5_SET_RECALL_COMPATIBLE",
        "reported_external_MRR_Recall3_wording": "NOT_LOCALLY_PROVEN",
        "leaderboard_delta": "compatible with top5-set recall and also with untruncated MRR; not compatible with Recall@3 alone",
        "active_score_contract": active_score_contract,
    }
    report = {
        "status": "PASS",
        "operation": "PRE_WORKFLOW_B_INCUMBENT_REGRESSION_FORENSIC_AUDIT",
        "read_only": True,
        "training_executed": False,
        "workflow_b_started": False,
        "public_labels_used": False,
        "fold0_labels_loaded": False,
        "incumbent": {
            "public_score_external_fact": 0.9391,
            "artifact": str(INCUMBENT_ZIP.relative_to(ROOT)).replace("\\", "/"),
            "zip_sha256": sha256(INCUMBENT_ZIP),
            "submission_json_sha256": incumbent_member_hash,
            "zip_members": incumbent_members,
            "semantic_match_to_public_v3a_predictions": True,
        },
        "all_noop_baseline": {
            "public_score_external_fact": 0.9309,
            "artifact": str(BASELINE_ZIP.relative_to(ROOT)).replace("\\", "/"),
            "zip_sha256": sha256(BASELINE_ZIP),
            "submission_json_sha256": baseline_member_hash,
            "zip_members": baseline_members,
            "baseline_anchor": str(BASELINE_ANCHOR.relative_to(ROOT)).replace("\\", "/"),
        },
        "delta": {
            "query_count": 1000,
            "identical_queries": 672,
            "different_queries": len(rows),
            "one_swap_queries": len(rows),
            "rank4_changes": rank_counts[4],
            "rank5_changes": rank_counts[5],
            "top1_top2_top3_changes": sum(rank_counts[index] for index in (1, 2, 3)),
            "delta_artifact": str(DELTA.relative_to(ROOT)).replace("\\", "/"),
        },
        "swap_trace": {
            "generator": "V3A public residual policy",
            "exactly_traced": trace_counts["EXACTLY_TRACED_TO_V3A"],
            "traced_to_other_pipeline": 0,
            "unresolved": 0,
            "v3a_report_counts_match": v3a_report["swapped_query_count"] == len(rows) and v3a_report["rank4_drop_count"] == rank_counts[4] and v3a_report["rank5_drop_count"] == rank_counts[5],
            "policy_scores_persisted": False,
        },
        "incumbent_lineage": {
            "baseline_public_top5": str(BASELINE_ANCHOR.relative_to(ROOT)).replace("\\", "/"),
            "candidate_generation": {
                "artifact": str(V3A_CANDIDATES.relative_to(ROOT)).replace("\\", "/"),
                "candidate_contract": "four-source RRF union, cap 200; shortlist union_rank <=20 plus baseline top5, max 25 docs",
                "shortlist_report": str(V3A_SHORTLIST.relative_to(ROOT)).replace("\\", "/"),
            },
            "features": {"count": 58, "schema_sha256": v3a_report["feature_schema_sha256"], "artifact": str(V3A_FEATURES.relative_to(ROOT)).replace("\\", "/")},
            "model_family": manifest["hyperparameters"]["model_family"],
            "model_sha256": manifest["model_sha256"],
            "model_config_sha256": sha256(V3A_MANIFEST),
            "training_population": config["training_folds"],
            "training_query_count": config["training_query_count"],
            "threshold": hyperparameters["threshold"],
            "harm_weight": hyperparameters["harm_weight"],
            "action_rule": "select utility >= threshold; replace exactly one baseline rank 4 or 5 document; ranks 1-3 protected",
            "training_script": "scripts/beam/task1_v3_residual/train_residual_policy.py::final_fit_model",
            "inference_script": "scripts/beam/task1_v3_residual/train_residual_policy.py::apply_public_model",
            "prediction_artifact": str(V3A_PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
        },
        "training_provenance": {
            "INCUMBENT_TRAINING_PROVENANCE": "F1_F4_PLUS_FOLD0",
            "fold0_involved_historically": True,
            "evidence": "v3a_final_fit_manifest.production_final_fit.training_folds is [0,1,2,3,4]",
            "public_labels_used": False,
            "public_label_evidence": "public_v3a_policy_report.no_public_labels_used=true; manifest.no_public_training=true",
        },
        "workflow_a_transition": {
            "0_9309_intentionally_scientific_baseline_reference": True,
            "0_9391_registered_as_deployment_incumbent": False,
            "incumbent_preservation_gate_present": False,
            "loss_point": "The all-NO_OP deployment contract reused the pre-V3A public_anchor_093 baseline because threshold=Infinity, while Workflow-A comparisons were scoped to the F1-F4 scientific baseline and did not require comparison to artifacts/task1/submission.zip.",
            "transition_evidence": [
                "reports/task1/public_p5_submission_preflight.json identifies historical public deployment as V3A rather than P5",
                "artifacts/task1/submission_p5_all_noop/submission_manifest.json records public_anchor_093/submission_093.zip as the all-NO_OP baseline source",
                "reports/task1/progress_log.md records the all-NO_OP submission without an incumbent hash/diff gate",
            ],
            "root_cause": ["production-threshold collapse to all-NO_OP", "wrong deployment incumbent reference", "artifact/provenance separation between V3A and Workflow A"],
            "all_noop_technically_correct_under_frozen_p5_contract": True,
            "all_noop_operationally_correct_as_incumbent_replacement": False,
        },
        "scoring_contract": score_evidence,
        "scientific_reference": "Workflow-A F1-F4 frozen scientific baseline and its documented P5 OOF comparator; it is not a public deployment incumbent.",
        "deployment_incumbent": {"score_external_fact": 0.9391, "artifact": str(INCUMBENT_ZIP.relative_to(ROOT)).replace("\\", "/"), "submission_json_sha256": incumbent_member_hash},
        "required_gates_before_workflow_b": ["INCUMBENT_ARTIFACT_HASH_LOCK", "INCUMBENT_PROVENANCE_LOCK", "LOCAL_SCIENTIFIC_COMPARATOR", "DEPLOYMENT_REGRESSION_GATE", "SUBMISSION_DIFF_REPORT", "SCORING_CONTRACT_GATE", "NO_PUBLIC_TUNING_RULE"],
        "evidence_sha256": evidence_hashes,
    }
    write_json(JSON_REPORT, report)
    MD_REPORT.write_text(f"""# Pre-Workflow-B incumbent regression forensic audit

Status: **PASS**. This was a read-only provenance audit: no model training, public-label access, or Fold0-label loading occurred.

## Exact artifacts and delta

- Incumbent 0.9391: `{report['incumbent']['artifact']}`, internal `submission.json` SHA256 `{incumbent_member_hash}`.
- All-NO_OP 0.9309: `{report['all_noop_baseline']['artifact']}`, internal `submission.json` SHA256 `{baseline_member_hash}`.
- The incumbent content is exactly the normalized content of `public_v3a_predictions.json`.
- Of 1,000 queries, 672 are identical and 328 differ. Every difference is one replacement: 110 at rank 4, 218 at rank 5, and 0 at ranks 1--3.
- All 328 are exactly traced to the V3A public residual-policy output by full final-top5 equality plus incoming-document candidate/action identity. Per-action model scores were not persisted and were not recomputed.

## Incumbent lineage

`public_anchor_093` baseline -> four-source RRF candidate union (cap 200) -> shortlist (union rank <=20 plus baseline, at most 25 docs) -> 58 public frozen features -> V3A `HistGradientBoostingClassifier` -> threshold 0.0 / harm weight 1.5 -> one rank-4-or-5 swap -> `public_v3a_predictions.json` -> `artifacts/task1/submission.zip`.

The final-fit manifest states training folds `[0,1,2,3,4]`; therefore incumbent training provenance is **F1_F4_PLUS_FOLD0**. This conclusion comes solely from the manifest, not Fold0 labels. V3A public inference reports `no_public_labels_used=true` and the manifest reports no public training.

## Why the deployment regressed

The 0.9309 artifact was intentionally the scientific/pre-policy baseline anchor. Workflow A froze P5 against the F1--F4 scientific reference, then the valid production threshold selection chose `Infinity`, yielding all-NO_OP. The deployment correctly reused that pre-V3A baseline, but no deployment-incumbent preservation gate compared it to the existing 0.9391 V3A submission. Thus the frozen-P5 deployment was technically faithful but operationally not a valid incumbent replacement.

## Scoring contract

The checked repository scoring program, local LegalIR evaluator, and historical recovery evaluator all implement macro set Recall/Precision on up to five documents. The reported external MRR/Recall@3 wording was not available as a local executable contract, so the active platform scorer remains **UNRESOLVED**. Rank-4/5-only changes cannot alter Recall@3, but they can alter untruncated MRR whenever the first relevant document is at rank 4 or 5; the observed public delta alone therefore cannot identify the active scorer.

## Required gates before Workflow B

1. `INCUMBENT_ARTIFACT_HASH_LOCK`
2. `INCUMBENT_PROVENANCE_LOCK`
3. `LOCAL_SCIENTIFIC_COMPARATOR`
4. `DEPLOYMENT_REGRESSION_GATE`
5. `SUBMISSION_DIFF_REPORT`
6. `SCORING_CONTRACT_GATE`
7. `NO_PUBLIC_TUNING_RULE`

Workflow-A scientific findings are not invalidated. Its deployment decision is **partially invalidated**: the frozen P5 mechanics were correct, but replacing the public incumbent without an incumbent gate was not.
""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "incumbent": str(INCUMBENT_ZIP.relative_to(ROOT)), "delta_rows": len(rows), "rank4": rank_counts[4], "rank5": rank_counts[5], "fold0_training": True, "active_score_contract": active_score_contract}))


if __name__ == "__main__":
    main()
