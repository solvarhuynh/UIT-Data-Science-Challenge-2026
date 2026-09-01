import json
import runpy
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/task1"
CONTRACT = R / "step2p1_model_contract.json"
MODELS = R / "step2p4_proxy_models"
OUTER_PRED = R / "step2p4_proxy_predictions.jsonl"
INNER_PRED = R / "step2p4_proxy_inner_oof_predictions.jsonl"
REPORT = R / "step2p4_discriminability_report.json"


def matrix(rows, columns):
    return np.asarray([[row["features"][col] for col in columns] for row in rows], dtype=np.float32)


def target(rows):
    return np.asarray([int(row["truth_delta"] > 1e-12) for row in rows], dtype=np.uint8)


def counts(y):
    positive = int(y.sum())
    total = len(y)
    return {"total_actions": total, "beneficial_actions": positive,
            "non_beneficial_actions": total - positive,
            "beneficial_prevalence": positive / total}


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(a), "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def identity(row):
    return {"incoming_doc_id": row["incoming_doc_id"], "incoming_union_rank": row["incoming_union_rank"],
            "drop_rank": row["drop_rank"], "dropped_doc_id": row["dropped_doc_id"]}


def save_model(path, model, columns, hp, train_folds, held_out_fold, train_y, scope):
    dump({"model": model, "artifact_type": "STEP2_P4_DIAGNOSTIC_ONLY", "prediction_scope": scope,
          "ordered_feature_columns": columns, "effective_hgb_params": hp, "training_fold_ids": train_folds,
          "held_out_fold": held_out_fold, "training_class_counts": counts(train_y)}, path)


def main():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    columns, hp = contract["feature_columns"], contract["hyperparameters"]
    if not (contract["status"] == "FROZEN_PRE_FIT" and len(columns) == len(set(columns)) == 36 and contract["fixed_K"] == 77):
        raise RuntimeError("CONTRACT_ERROR frozen P1 feature contract")
    ns = runpy.run_path(str(ROOT / "scripts/analysis/step2_k77_oof_policy_realizability.py"), run_name="p4_parent")
    folds = ns["target_folds"]()
    records, gold = ns["load_gold"](set(folds))
    baseline, _ = ns["load_baseline"](set(folds), folds)
    candidates, _ = ns["load_candidates"](set(folds), folds)
    actions, incoming = ns["make_actions"](folds, records, gold, baseline, candidates, columns)
    if incoming != 403322 or sum(len(actions[f]) for f in range(1, 5)) != 806644:
        raise RuntimeError("CONTRACT_ERROR K77 action population")
    if MODELS.exists():
        shutil.rmtree(MODELS)
    MODELS.mkdir(parents=True)
    outer_rows = []
    inner_rows = []
    outer_auc, class_counts, outer_models, inner_models = {}, {}, 0, 0
    for held in range(1, 5):
        train_folds = [f for f in range(1, 5) if f != held]
        train_rows = [row for f in train_folds for row in actions[f]]
        x_train, y_train = matrix(train_rows, columns), target(train_rows)
        model = HistGradientBoostingClassifier(**hp)
        model.fit(x_train, y_train)
        path = MODELS / f"outer_fold{held}.joblib"
        save_model(path, model, columns, hp, train_folds, held, y_train, "OUTER_HELD_OUT")
        outer_models += 1
        test_rows, y_test = actions[held], target(actions[held])
        if len(np.unique(y_test)) != 2:
            raise RuntimeError(f"CONTRACT_ERROR held-out fold {held} lacks a target class")
        p = model.predict_proba(matrix(test_rows, columns))[:, 1]
        outer_auc[f"F{held}"] = float(roc_auc_score(y_test, p))
        class_counts[str(held)] = counts(y_test)
        for row, score, label in zip(test_rows, p, y_test):
            outer_rows.append({"prediction_scope": "OUTER_HELD_OUT", "query_id": row["query_id"], "fold": held,
                               "action_identity": identity(row), "p_hat_beneficial": float(score),
                               "POST_PREDICTION_EVALUATION": {"y_B": int(label), "true_action_class": row["label"],
                                                               "truth_recall_delta": row["truth_delta"]}})
        del model, x_train, y_train, test_rows, y_test, p, train_rows
        for valid in train_folds:
            inner_train_folds = [f for f in train_folds if f != valid]
            inner_train_rows = [row for f in inner_train_folds for row in actions[f]]
            x_inner, y_inner = matrix(inner_train_rows, columns), target(inner_train_rows)
            inner_model = HistGradientBoostingClassifier(**hp)
            inner_model.fit(x_inner, y_inner)
            ipath = MODELS / f"outer_fold{held}_inner_valid_fold{valid}.joblib"
            save_model(ipath, inner_model, columns, hp, inner_train_folds, valid, y_inner, "INNER_HELD_OUT")
            inner_models += 1
            valid_rows = actions[valid]
            scores = inner_model.predict_proba(matrix(valid_rows, columns))[:, 1]
            for row, score in zip(valid_rows, scores):
                inner_rows.append({"prediction_scope": "INNER_HELD_OUT", "outer_held_fold": held,
                                   "inner_valid_fold": valid, "query_id": row["query_id"], "fold": valid,
                                   "action_identity": identity(row), "p_hat_beneficial": float(score)})
            del inner_model, inner_train_rows, x_inner, y_inner, valid_rows, scores
    with OUTER_PRED.open("w", encoding="utf-8", newline="\n") as handle:
        for row in outer_rows:
            handle.write(json.dumps(row) + "\n")
    with INNER_PRED.open("w", encoding="utf-8", newline="\n") as handle:
        for row in inner_rows:
            handle.write(json.dumps(row) + "\n")
    all_y = np.asarray([row["POST_PREDICTION_EVALUATION"]["y_B"] for row in outer_rows], dtype=np.uint8)
    all_p = np.asarray([row["p_hat_beneficial"] for row in outer_rows], dtype=np.float64)
    p1 = json.loads((R / "step2p1f_failure_attribution_report.json").read_text(encoding="utf-8"))
    if not (p1["status"] == "PASS" and p1["oracle_positive_queries"] == 426 and
            [p1["trs"][x]["query_count"] for x in ("S_SUCCESS", "T_THRESHOLD_LOSS", "R_RANKING_DISCRIMINATION_LOSS")] == [6, 63, 357]):
        raise RuntimeError("CONTRACT_ERROR canonical P1 success/failure reference")
    # Reconstruct only the canonical P1 query populations; no P4 score influences this membership.
    score_rows = [json.loads(x) for x in (R / "step2p1f_full_action_scores.jsonl").read_text(encoding="utf-8").splitlines() if x]
    qtop = {}
    oracle = {}
    for row in score_rows:
        q = row["query_id"]
        oracle[q] = max(oracle.get(q, 0.0), row["truth_recall_delta"])
        if row["is_query_top_scored"]:
            qtop[q] = row
    success = {q for q, row in qtop.items() if oracle[q] > 0 and row["true_action_class"] == "BENEFICIAL" and row["would_execute"]}
    failure = {q for q, row in qtop.items() if oracle[q] > 0 and row["true_action_class"] != "BENEFICIAL"}
    if len(success) != 6 or len(failure) != 357:
        raise RuntimeError("CONTRACT_ERROR P1 success/failure membership")
    distributions = {"SUCCESS": summary([row["p_hat_beneficial"] for row in outer_rows if row["query_id"] in success]),
                     "FAILURE": summary([row["p_hat_beneficial"] for row in outer_rows if row["query_id"] in failure])}
    report = {
        "status": "PASS", "experiment": "STEP 2-P4 — Label-Free Beneficial-Action Proxy Discriminability Diagnostic",
        "experiment_type": "DIAGNOSTIC_ONLY_NO_BRANCH_GATE", "training_executed": True, "policy_modified": False,
        "new_policy_scoring_executed": False, "fold0_or_public_used": False, "policy_oof_strict": True,
        "end_to_end_selection_oof": False, "k77_selected_exploratorily_on_folds_1_4": True, "K": 77, "features": 36,
        "target": {"positive": "BENEFICIAL_truth_recall_delta_gt_1e-12", "negative": "NEUTRAL_OR_HARMFUL", "positive_label": 1},
        "model": {"class": "sklearn.ensemble.HistGradientBoostingClassifier", "loss": "log_loss", "hyperparameters": hp,
                  "uniform_sample_influence": True, "class_weight": None, "sample_weight": None,
                  "outer_models_expected": 4, "outer_models_trained": outer_models,
                  "inner_models_expected": 12, "inner_models_trained": inner_models},
        "class_counts": {"per_fold": class_counts, "pooled": counts(all_y)},
        "auc": {**outer_auc, "pooled_outer_oof": float(roc_auc_score(all_y, all_p)), "chance_reference": 0.5,
                "orientation": "HIGHER_SCORE_MEANS_MORE_BENEFICIAL", "directional_auc_reversal_used": False},
        "feature_importance": {"status": "NOT_AVAILABLE_WITHOUT_NEW_UNPREREGISTERED_METHOD", "method": None, "ranking": [],
                               "limitation": "No canonical HGB feature-importance method was found, and HistGradientBoostingClassifier exposes no direct feature_importances_ attribute."},
        "p1_success_failure_proxy_distribution": distributions,
        "prediction_outputs": {"outer_held_out": str(OUTER_PRED.relative_to(ROOT)), "outer_rows": len(outer_rows),
                               "inner_held_out": str(INNER_PRED.relative_to(ROOT)), "inner_rows": len(inner_rows)},
        "model_output_directory": str(MODELS.relative_to(ROOT)),
        "standing_policy_comparator": {"R": 357, "D77_exact": -0.0007142857142857143, "gain_sum": -4.0, "S": 6, "T": 63},
        "branch_gate": "NONE", "preserved_status": {"relative_ordering_hypothesis": "SUPPORTED", "P3_sub_branch": "FORMALLY_CLOSED",
          "deployable_BN_identification": "UNRESOLVED_PROXY_LEARNABILITY", "pair_family": "STRONGLY_WEAKENED", "pair_reweighting": "NOT_AUTHORIZED",
          "B_vs_N_only": "NOT_AUTHORIZED", "rank_feature_intervention": "NOT_JUSTIFIED", "raw_pairwise_dispersion": "DEFERRED",
          "HGB_predictive_capacity": "UNRESOLVED", "HGB_API_objective_flexibility": "BLOCKED_BY_IMPLEMENTATION_FOR_PAIRWISE_MARGIN",
          "feature_expansion": "NOT_JUSTIFIED", "K77": "NOT_REJECTED", "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"},
        "notes": ["Gold-derived labels were used only to construct training targets and after held-out probabilities were frozen for AUC/descriptive reporting.",
                  "Pointwise diagnostic scores were not thresholded and did not modify P1 ranking, aggregation, thresholds, or policy metrics."],
        "next_state": "STEP2_P4_COMPLETE_PENDING_PROFESSOR_REVIEW"}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "outer_models": outer_models, "inner_models": inner_models,
                      "auc": report["auc"], "success_median": distributions["SUCCESS"]["median"],
                      "failure_median": distributions["FAILURE"]["median"]}, indent=2))


if __name__ == "__main__":
    main()
