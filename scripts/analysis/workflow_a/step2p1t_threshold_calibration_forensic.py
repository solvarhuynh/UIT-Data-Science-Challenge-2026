"""Read-only forensic for frozen STEP 2-P1 pairwise scores; never fits models."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "task1"
OUT = R / "step2p1t_threshold_calibration_report.json"
THRESHOLDS = {1: 0.8871766063648394, 2: 0.9158352964999534, 3: 0.8885338990548798, 4: 0.9072451320464932}
EXPECTED_D = {1: 0.0009523809523809524, 2: 0.0, 3: -0.002380952380952381, 4: -0.0014285714285714286}
BINS = [(0.0, 0.005, "[0, 0.005]"), (0.005, 0.010, "(0.005, 0.010]"), (0.010, 0.020, "(0.010, 0.020]"), (0.020, 0.050, "(0.020, 0.050]"), (0.050, None, "> 0.050")]


def read_json(name: str):
    return json.loads((R / name).read_text(encoding="utf-8"))


def dist(values):
    if not values:
        return {"count": 0, "min": None, "p10": None, "p25": None, "median": None, "mean": None, "p75": None, "p90": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"count": int(len(a)), "min": float(a.min()), "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)), "mean": float(a.mean()), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def bin_summary(rows, value_key, headroom_key):
    out = {}
    for lo, hi, label in BINS:
        selected = [x for x in rows if x[value_key] >= lo and (hi is None or x[value_key] <= hi)]
        out[label] = {"count": len(selected), "fraction": len(selected) / len(rows) if rows else None, "Recall_headroom_sum": sum(x[headroom_key] for x in selected)}
    return out


def main():
    gate = read_json("step2p1f_r2_reproduction_gate_report.json")
    attr = read_json("step2p1f_failure_attribution_report.json")
    corr = read_json("step2p1f_pairfamily_correlation_report.json")
    metric = read_json("step2p1f_metric_contract_resolution_report.json")
    contract = read_json("step2p1_model_contract.json")
    realizability = read_json("step2p1_realizability_report.json")
    decisions = [json.loads(x) for x in (R / "step2p1_oof_policy_decisions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [json.loads(x) for x in (R / "step2p1f_full_action_scores.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    models = [R / "step2p1f_models" / f"outer_fold{i}.joblib" for i in range(1, 5)]
    top = [x for x in rows if x["is_query_top_scored"]]
    executed = [x for x in top if x["would_execute"]]
    top_classes = Counter(x["true_action_class"] for x in top)
    exec_classes = Counter(x["true_action_class"] for x in executed)
    trs = attr["trs"]
    preconditions = {
        "r2_gate_pass": gate.get("status") == "PASS" and gate.get("reproduction_gate_pass") is True,
        "models_persisted_4": len(gate.get("models_persisted", [])) == 4 and all(x.exists() for x in models),
        "full_action_rows": len(rows) == 806644,
        "K77": contract.get("fixed_K") == 77,
        "oracle_positive_426": attr.get("oracle_positive_queries") == 426,
        "trs_6_63_357": [trs[k]["query_count"] for k in ("S_SUCCESS", "T_THRESHOLD_LOSS", "R_RANKING_DISCRIMINATION_LOSS")] == [6, 63, 357],
        "canonical_gain_D": gate.get("canonical_gain_sum") == -4.0 and gate.get("canonical_D77") == -0.0007142857142857143,
        "top_class_counts": top_classes == Counter({"BENEFICIAL": 69, "NEUTRAL": 5405, "HARMFUL": 126}),
        "executed_class_counts": exec_classes == Counter({"BENEFICIAL": 6, "NEUTRAL": 392, "HARMFUL": 12}),
    }
    if not all(preconditions.values()):
        OUT.write_text(json.dumps({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1T — Pairwise-Specific Threshold/Calibration Forensic", "training_executed": False, "preconditions": preconditions}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "CONTRACT_ERROR", "preconditions": preconditions}, indent=2))
        return
    for x in top:
        x["threshold_margin"] = x["pairwise_policy_score"] - x["outer_threshold"]
    btop = [x for x in top if x["true_action_class"] == "BENEFICIAL"]
    s = [x for x in btop if x["would_execute"]]
    t = [x for x in btop if not x["would_execute"]]
    if not (len(top) == 5600 and len(btop) == 69 and len(s) == 6 and len(t) == 63 and all(x["threshold_margin"] >= 0 for x in s) and all(x["threshold_margin"] < 0 for x in t)):
        OUT.write_text(json.dumps({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1T — Pairwise-Specific Threshold/Calibration Forensic", "training_executed": False, "reason": "B-top S/T decomposition or margins mismatched"}, indent=2) + "\n", encoding="utf-8")
        print('{"status": "CONTRACT_ERROR"}')
        return
    for x in t:
        x["distance_below_threshold"] = -x["threshold_margin"]
    for x in executed:
        x["positive_threshold_margin"] = x["threshold_margin"]
    b_per_query = [{"query_id": x["query_id"], "fold": x["fold"], "top_pairwise_score": x["pairwise_policy_score"], "fold_threshold": x["outer_threshold"], "threshold_margin": x["threshold_margin"], "current_outcome": "S_SUCCESS" if x["would_execute"] else "T_THRESHOLD_LOSS", "truth_recall_delta": x["truth_recall_delta"], "incoming_union_rank": x["incoming_union_rank"], "drop_rank": x["drop_rank"]} for x in btop]
    selectivity = {}
    for cls in ("BENEFICIAL", "NEUTRAL", "HARMFUL"):
        z = [x for x in top if x["true_action_class"] == cls]
        crossings = sum(x["would_execute"] for x in z)
        selectivity[cls] = {"count": len(z), "top_score_distribution": dist([x["pairwise_policy_score"] for x in z]), "threshold_margin_distribution": dist([x["threshold_margin"] for x in z]), "threshold_crossing_count": crossings, "threshold_crossing_rate": crossings / len(z)}
    executed_subsets = {}
    for cls in ("BENEFICIAL", "NEUTRAL", "HARMFUL"):
        z = [x for x in executed if x["true_action_class"] == cls]
        executed_subsets[cls] = {"count": len(z), "top_score_distribution": dist([x["pairwise_policy_score"] for x in z]), "threshold_margin_distribution": dist([x["threshold_margin"] for x in z]), "Recall_delta_sum": sum(x["truth_recall_delta"] for x in z)}
    folds = {}
    for f in range(1, 5):
        z = [x for x in top if x["fold"] == f]
        bz, nz, hz = ([x for x in z if x["true_action_class"] == c] for c in ("BENEFICIAL", "NEUTRAL", "HARMFUL"))
        sz, tz = [x for x in bz if x["would_execute"]], [x for x in bz if not x["would_execute"]]
        eh = [x for x in hz if x["would_execute"]]
        folds[str(f)] = {"threshold": THRESHOLDS[f], "total_queries": len(z), "B_top_count": len(bz), "S_count": len(sz), "T_count": len(tz), "S_conversion_rate": len(sz) / len(bz) if bz else None, "N_top_count": len(nz), "N_executed_count": sum(x["would_execute"] for x in nz), "N_crossing_rate": sum(x["would_execute"] for x in nz) / len(nz) if nz else None, "H_top_count": len(hz), "H_executed_count": len(eh), "H_crossing_rate": len(eh) / len(hz) if hz else None, "median_B_top_score": dist([x["pairwise_policy_score"] for x in bz])["median"], "median_B_top_margin": dist([x["threshold_margin"] for x in bz])["median"], "median_T_margin": dist([x["threshold_margin"] for x in tz])["median"], "median_executed_H_margin": dist([x["threshold_margin"] for x in eh])["median"], "canonical_D_fold": EXPECTED_D[f]}
    inner = {"status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS", "inner_oof_threshold_curve_status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS", "inner_oof_refit_permitted": False, "folds": {str(f): {"selected_threshold": THRESHOLDS[f], "selected_inner_objective": None, "selected_threshold_unique": None, "exact_tie_count": None, "next_lower_threshold": None, "next_lower_objective": None, "next_higher_threshold": None, "next_higher_objective": None} for f in range(1, 5)}, "evidence": "P1 source serializes only outer selected thresholds; inspected persisted P1 artifacts contain no inner candidate/objective curve."}
    report = {"status": "PASS", "experiment": "STEP 2-P1T — Pairwise-Specific Threshold/Calibration Forensic", "scientific_status": "DIAGNOSTIC_ONLY_NO_BRANCH_GATE", "training_executed": False, "policy_oof_strict": True, "end_to_end_selection_oof": False, "k77_selected_exploratorily_on_folds_1_4": True, "interpretation_scope": "PAIRWISE_THRESHOLD_CALIBRATION_FORENSIC_CONDITIONAL_ON_EXPLORATORY_FIXED_K77", "K": 77, "fixed_thresholds": {str(k): v for k, v in THRESHOLDS.items()}, "precondition_checks": preconditions, "canonical_counts": {"queries": 5600, "oracle_positive": 426, "B_top": 69, "N_top": 5405, "H_top": 126, "S": 6, "T": 63, "R": 357, "executed_B": 6, "executed_N": 392, "executed_H": 12}, "beneficial_top_69": {"per_query": b_per_query, "S_score_distribution": dist([x["pairwise_policy_score"] for x in s]), "T_score_distribution": dist([x["pairwise_policy_score"] for x in t]), "S_margin_distribution": dist([x["threshold_margin"] for x in s]), "T_margin_distribution": dist([x["threshold_margin"] for x in t]), "per_fold": {str(f): {"S_count": sum(x["fold"] == f for x in s), "T_count": sum(x["fold"] == f for x in t), "median_S_margin": dist([x["threshold_margin"] for x in s if x["fold"] == f])["median"], "mean_S_margin": dist([x["threshold_margin"] for x in s if x["fold"] == f])["mean"], "median_T_margin": dist([x["threshold_margin"] for x in t if x["fold"] == f])["median"], "mean_T_margin": dist([x["threshold_margin"] for x in t if x["fold"] == f])["mean"]} for f in range(1, 5)}, "T_distance_bins": bin_summary(t, "distance_below_threshold", "truth_recall_delta")}, "top_class_selectivity": selectivity, "executed_subsets": executed_subsets, "harmful_margin_bins": bin_summary([x for x in executed if x["true_action_class"] == "HARMFUL"], "positive_threshold_margin", "truth_recall_delta"), "S_vs_executed_H": {"S_score_distribution": executed_subsets["BENEFICIAL"]["top_score_distribution"], "executed_H_score_distribution": executed_subsets["HARMFUL"]["top_score_distribution"], "S_margin_distribution": executed_subsets["BENEFICIAL"]["threshold_margin_distribution"], "executed_H_margin_distribution": executed_subsets["HARMFUL"]["threshold_margin_distribution"], "median_score_S": executed_subsets["BENEFICIAL"]["top_score_distribution"]["median"], "median_score_H_exec": executed_subsets["HARMFUL"]["top_score_distribution"]["median"], "median_margin_S": executed_subsets["BENEFICIAL"]["threshold_margin_distribution"]["median"], "median_margin_H_exec": executed_subsets["HARMFUL"]["threshold_margin_distribution"]["median"]}, "per_fold": folds, "inner_oof_threshold_audit": inner, "preserved_status": {"remaining_ranking_loss": "PRIMARY_REMAINING_BOTTLENECK", "threshold_calibration": "NEWLY_SIGNIFICANT_SECONDARY_BOTTLENECK", "pair_family_causal_driver": "UNRESOLVED_NOT_ESTABLISHED", "mean_pairwise_win_probability_calibration": "SUPPORTED_HYPOTHESIS", "pair_reweighting": "NOT_AUTHORIZED", "B_vs_N_only_training": "NOT_AUTHORIZED", "HGB": "UNRESOLVED", "feature_expansion": "NOT_JUSTIFIED", "K77": "NOT_REJECTED", "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"}, "notes": ["Read-only analysis of the serialized frozen action scores; no model.fit, rescore, threshold optimization, or calibration fit was performed.", "Outer labels were used only as diagnostic truth.", "Pair-family correlations preserved: Spearman %.16g; Pearson %.16g." % (corr["query_local_analysis"]["spearman"], corr["query_local_analysis"]["pearson"])]}
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "B_top": len(btop), "S": len(s), "T": len(t), "S_median_score": report["beneficial_top_69"]["S_score_distribution"]["median"], "T_median_score": report["beneficial_top_69"]["T_score_distribution"]["median"], "T_bins": report["beneficial_top_69"]["T_distance_bins"], "crossing_rates": {k: v["threshold_crossing_rate"] for k, v in selectivity.items()}}, indent=2))


if __name__ == "__main__":
    main()
