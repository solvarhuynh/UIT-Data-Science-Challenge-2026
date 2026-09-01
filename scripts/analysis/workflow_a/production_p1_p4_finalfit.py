"""Authorized deterministic F1-F4-only final fit for frozen Workflow-A P1/P4."""
from __future__ import annotations

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/task1"
P1_MODEL = REPORTS / "production_p1_model.pkl"
P1_CONFIG = REPORTS / "production_p1_fit_config.json"
P4_MODEL = REPORTS / "production_p4_model.pkl"
P4_CONFIG = REPORTS / "production_p4_fit_config.json"
REPORT = REPORTS / "production_model_finalfit_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def matrix(rows: list[dict], columns: list[str]) -> np.ndarray:
    return np.asarray([[row["features"][column] for column in columns] for row in rows], dtype=np.float32)


def pair_matrix(rows: list[dict], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    rank = {"BENEFICIAL": 2, "NEUTRAL": 1, "HARMFUL": 0}
    count = 0
    for items in grouped.values():
        classes = Counter(item["label"] for item in items)
        count += 2 * (classes["BENEFICIAL"] * classes["NEUTRAL"] + classes["BENEFICIAL"] * classes["HARMFUL"] + classes["NEUTRAL"] * classes["HARMFUL"])
    x = np.empty((count, len(columns)), dtype=np.float32)
    y = np.empty(count, dtype=np.uint8)
    position = 0
    for items in grouped.values():
        items = sorted(items, key=lambda item: (item["incoming_union_rank"], item["incoming_doc_id"], item["drop_rank"]))
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                if left["label"] == right["label"]:
                    continue
                high, low = (left, right) if rank[left["label"]] > rank[right["label"]] else (right, left)
                difference = np.asarray([high["features"][column] - low["features"][column] for column in columns], dtype=np.float32)
                x[position] = difference
                y[position] = 1
                x[position + 1] = -difference
                y[position + 1] = 0
                position += 2
    if position != count or int(y.sum()) * 2 != count:
        raise RuntimeError("P1 pair-construction contract mismatch")
    return x, y


def main() -> None:
    preflight = json.loads((REPORTS / "production_model_finalfit_preflight.json").read_text(encoding="utf-8"))
    p1_contract = json.loads((REPORTS / "step2p1_model_contract.json").read_text(encoding="utf-8"))
    p4_contract = json.loads((REPORTS / "step2p4_discriminability_report.json").read_text(encoding="utf-8"))
    if preflight["status"] != "PASS" or p1_contract["status"] != "FROZEN_PRE_FIT" or p4_contract["status"] != "PASS":
        raise RuntimeError("Production final-fit preflight contract mismatch")
    columns = p1_contract["feature_columns"]
    hp = p1_contract["hyperparameters"]
    if len(columns) != len(set(columns)) or len(columns) != 36 or p1_contract["fixed_K"] != 77:
        raise RuntimeError("Canonical P1 schema/K contract mismatch")
    if hp != p4_contract["model"]["hyperparameters"] or hp["random_state"] != p1_contract["random_seed"]:
        raise RuntimeError("Canonical P1/P4 hyperparameter contract mismatch")
    parent = runpy.run_path(str(ROOT / "scripts/analysis/step2_k77_oof_policy_realizability.py"), run_name="production_finalfit_parent")
    folds = parent["target_folds"]()
    if set(folds.values()) != {1, 2, 3, 4}:
        raise RuntimeError("Fold scope mismatch")
    records, gold = parent["load_gold"](set(folds))
    baseline, _ = parent["load_baseline"](set(folds), folds)
    candidates, _ = parent["load_candidates"](set(folds), folds)
    actions, incoming = parent["make_actions"](folds, records, gold, baseline, candidates, columns)
    rows = [row for fold in range(1, 5) for row in actions[fold]]
    query_ids = {row["query_id"] for row in rows}
    if incoming != 403322 or len(rows) != 806644 or len(query_ids) != 5600 or any(row["fold"] not in (1, 2, 3, 4) for row in rows):
        raise RuntimeError("K77 F1-F4 action-population contract mismatch")
    x_pair, y_pair = pair_matrix(rows, columns)
    if len(y_pair) != 3333290:
        raise RuntimeError("P1 all-fold pair-count contract mismatch")
    p1 = HistGradientBoostingClassifier(**hp)
    p1.fit(x_pair, y_pair, sample_weight=np.ones(len(y_pair), dtype=np.float64))
    dump({"model": p1, "artifact_type": "WORKFLOW_A_P1_PRODUCTION_FINAL_FIT", "ordered_feature_columns": columns, "effective_hgb_params": hp, "training_fold_ids": [1, 2, 3, 4], "K": 77, "pair_construction": p1_contract["pair_generation"], "pair_weighting": "uniform_1.0"}, P1_MODEL)
    del x_pair, y_pair, p1
    y_proxy = np.asarray([int(row["truth_delta"] > 1e-12) for row in rows], dtype=np.uint8)
    class_counts = {"BENEFICIAL": int(y_proxy.sum()), "NEUTRAL": sum(row["label"] == "NEUTRAL" for row in rows), "HARMFUL": sum(row["label"] == "HARMFUL" for row in rows)}
    if class_counts["BENEFICIAL"] != 897 or class_counts["NEUTRAL"] + class_counts["HARMFUL"] != 805747:
        raise RuntimeError("P4 target-class contract mismatch")
    p4 = HistGradientBoostingClassifier(**hp)
    p4.fit(matrix(rows, columns), y_proxy)
    dump({"model": p4, "artifact_type": "WORKFLOW_A_P4_PRODUCTION_FINAL_FIT", "ordered_feature_columns": columns, "effective_hgb_params": hp, "training_fold_ids": [1, 2, 3, 4], "K": 77, "target": "BENEFICIAL_vs_NEUTRAL_HARMFUL", "class_weighting": "NONE", "resampling": "NONE"}, P4_MODEL)
    del p4, y_proxy
    p1_hash, p4_hash = sha256(P1_MODEL), sha256(P4_MODEL)
    p1_loaded, p4_loaded = load(P1_MODEL), load(P4_MODEL)
    for artifact, kind in ((p1_loaded, "P1"), (p4_loaded, "P4")):
        if artifact["ordered_feature_columns"] != columns or artifact["effective_hgb_params"] != hp or artifact["training_fold_ids"] != [1, 2, 3, 4] or artifact["K"] != 77 or not isinstance(artifact["model"], HistGradientBoostingClassifier):
            raise RuntimeError(f"{kind} reload/config sanity check failed")
    if p4_loaded["target"] != "BENEFICIAL_vs_NEUTRAL_HARMFUL" or p4_loaded["class_weighting"] != "NONE" or p4_loaded["resampling"] != "NONE":
        raise RuntimeError("P4 target/weighting sanity check failed")
    common = {"status": "PASS", "operation": "PRODUCTION_FINAL_FIT", "workflow": "A", "training_scope": ["F1", "F2", "F3", "F4"], "fold0_used": False, "public_labels_used": False, "K": 77, "feature_count": 36, "feature_names": columns, "feature_schema_sha256": schema_hash(columns), "estimator": "sklearn.ensemble.HistGradientBoostingClassifier", "hyperparameters": hp, "random_seed": hp["random_state"], "training_query_count": len(query_ids), "training_action_count": len(rows), "training_data_provenance": ["artifacts/task1/evaluation/strict_cv_v2/folds.json", "data/raw/btc/LegalIR/train.json", "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl", "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"], "canonical_contract_sources": ["reports/task1/step2p1_model_contract.json", "reports/task1/step2_k77_model_contract_v2.json", "reports/task1/step2p4_discriminability_report.json", "scripts/analysis/step2p1_controlled_pairwise_policy.py", "scripts/analysis/step2p4_proxy_discriminability_diagnostic.py"]}
    p1_config = {**common, "model": "P1", "pair_construction": p1_contract["pair_generation"], "pair_weighting": "uniform_1.0", "training_pair_count_before_mirroring": 1666645, "training_pair_count_after_mirroring": 3333290, "model_sha256": p1_hash}
    p4_config = {**common, "model": "P4", "target": "BENEFICIAL_vs_NEUTRAL_HARMFUL", "negative_classes": ["NEUTRAL", "HARMFUL"], "class_weighting": "NONE", "resampling": "NONE", "class_counts": class_counts, "model_sha256": p4_hash}
    P1_CONFIG.write_text(json.dumps(p1_config, indent=2) + "\n", encoding="utf-8")
    P4_CONFIG.write_text(json.dumps(p4_config, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PASS", "P1_status": "PASS", "P4_status": "PASS", "P1_artifact_path": str(P1_MODEL.relative_to(ROOT)), "P4_artifact_path": str(P4_MODEL.relative_to(ROOT)), "P1_sha256": p1_hash, "P4_sha256": p4_hash, "P1_config_sha256": sha256(P1_CONFIG), "P4_config_sha256": sha256(P4_CONFIG), "F1_F4_used": True, "Fold0_used": False, "public_labels_used": False, "K": 77, "feature_count": 36, "P1_canonical_contract_match": True, "P4_canonical_contract_match": True, "P4_target": "BENEFICIAL_vs_NEUTRAL_HARMFUL", "P4_harmful_included_as_negative": True, "P4_class_weighting": "none", "P4_resampling": "none", "hyperparameter_sweep": False, "feature_sweep": False, "model_family_change": False, "beta_change": False, "threshold_selection_executed": False, "public_inference_executed": False, "submission_generated": False, "reload_sanity_checks": {"P1": "PASS", "P4": "PASS", "feature_schema_exact": True, "no_Fold0_training_rows": True, "no_public_training_rows_or_labels": True}}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "P1_sha256": p1_hash, "P4_sha256": p4_hash, "queries": len(query_ids), "actions": len(rows), "pairs": 3333290, "p4_class_counts": class_counts}, indent=2))


if __name__ == "__main__":
    main()
