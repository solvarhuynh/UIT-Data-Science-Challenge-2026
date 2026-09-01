"""Read-only P1A-R robustness forensic for the fixed MEAN--MEDIAN gap."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "task1"
OUT = R / "step2p1ar_disagreement_robustness_report.json"
MEAN_T = {1: 0.8871766063648394, 2: 0.9158352964999534, 3: 0.8885338990548798, 4: 0.9072451320464932}
MED_T = {1: float("inf"), 2: float("inf"), 3: 0.9837219720799008, 4: float("inf")}


def read(name): return json.loads((R / name).read_text(encoding="utf-8"))
def key(x): return (x["query_id"], x["incoming_doc_id"], x["drop_rank"])
def dump(data): OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def top(rows, score):
    return sorted(rows, key=lambda x: (-x[score], -x["drop_rank"], x["incoming_union_rank"], x["incoming_doc_id"]))[0]


def dist(values):
    if not values:
        return {"count": 0, **{k: None for k in ("min", "p10", "p25", "median", "mean", "p75", "p90", "max")}}
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)), "mean": float(a.mean()),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def compare(brows, hrows, gap, reference=None):
    b, h = [x[gap] for x in brows], [x[gap] for x in hrows]
    result = {"B": dist(b), "H": dist(h), "B_count": len(b), "H_count": len(h)}
    if not b or not h:
        result.update({"raw_AUC": None, "directional_AUC": None, "raw_orientation": None,
                       "orientation_matches_original_n18": None, "P_gap_B_gt_H": None,
                       "ties": None, "tie_probability": None, "AUC_status": "INSUFFICIENT_CLASS_SUPPORT"})
        return result
    wins, ties, total = sum(x > y for x in b for y in h), sum(x == y for x in b for y in h), len(b) * len(h)
    raw = (wins + .5 * ties) / total
    orientation = "HIGHER_GAP_ASSOCIATED_WITH_B" if raw >= .5 else "LOWER_GAP_ASSOCIATED_WITH_B"
    result.update({"raw_AUC": raw, "directional_AUC": max(raw, 1 - raw), "raw_orientation": orientation,
                   "orientation_matches_original_n18": None if reference is None else orientation == reference,
                   "P_gap_B_gt_H": wins / total, "ties": ties, "tie_probability": ties / total, "AUC_status": "AVAILABLE"})
    return result


def group_distributions(rows):
    return {c: {"signed_gap": dist([x["signed_gap"] for x in rows if x["class"] == c]),
                "absolute_gap": dist([x["absolute_gap"] for x in rows if x["class"] == c])}
            for c in ("BENEFICIAL", "NEUTRAL", "HARMFUL")}


def orientation_summary(items):
    supported = [x for x in items if x["AUC_status"] == "AVAILABLE"]
    matches = sum(x["orientation_matches_original_n18"] for x in supported)
    reverses = len(supported) - matches
    if not supported: status = "INSUFFICIENT_FOLD_SUPPORT"
    elif matches == len(supported): status = "CONSISTENT_ALL_SUPPORTED_FOLDS"
    elif reverses == len(supported): status = "SYSTEMATIC_REVERSAL"
    elif matches / len(supported) >= .75: status = "MOSTLY_CONSISTENT"
    else: status = "MIXED_ORIENTATION"
    return {"folds_with_sufficient_support": len(supported), "folds_matching_original_orientation": matches,
            "folds_reversing_original_orientation": reverses, "orientation_robustness": status}


def robustness(all_result, fold_summary):
    if all_result["AUC_status"] != "AVAILABLE": return "ROBUSTNESS_NOT_SUPPORTED"
    if all_result["orientation_matches_original_n18"] is False or fold_summary["orientation_robustness"] == "SYSTEMATIC_REVERSAL":
        return "ORIENTATION_UNSTABLE"
    if all_result["directional_AUC"] >= .70 and fold_summary["orientation_robustness"] != "SYSTEMATIC_REVERSAL": return "ROBUSTNESS_VISIBLE"
    if all_result["directional_AUC"] >= .60 and fold_summary["orientation_robustness"] in ("CONSISTENT_ALL_SUPPORTED_FOLDS", "MOSTLY_CONSISTENT"):
        return "ROBUSTNESS_WEAK"
    return "ROBUSTNESS_NOT_SUPPORTED"


def main():
    p1ac, p1af, p1a, attr, p1c, contract = (read("step2p1ac_confidence_signal_report.json"), read("step2p1af_topaction_change_report.json"),
        read("step2p1a_realizability_report.json"), read("step2p1f_failure_attribution_report.json"),
        read("step2p1c_realizability_report.json"), read("step2p1_model_contract.json"))
    pre = {"p1ac_pass": p1ac["status"] == "PASS", "p1af_pass": p1af["status"] == "PASS", "p1a_pass": p1a["status"] == "PASS",
           "p1c_pass": p1c["status"] == "PASS", "K77": contract["fixed_K"] == 77,
           "original_BH": p1ac["canonical_populations"]["mean_executed_B"] == 6 and p1ac["canonical_populations"]["mean_executed_H"] == 12,
           "original_strongest_auc": p1ac["strongest_available_signal"]["ROC_AUC_directional"] == .875}
    med = {}
    with (R / "step2p1a_median_scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line); med[key(x)] = x
    groups = defaultdict(list); nmean = 0
    with (R / "step2p1f_full_action_scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line); nmean += 1
            if key(x) in med: groups[x["query_id"]].append(x)
    pre.update({"mean_rows_806644": nmean == 806644, "median_rows_806644": len(med) == 806644,
                "exact_action_join": nmean == len(med), "queries_5600": len(groups) == 5600})
    if not all(pre.values()):
        dump({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic", "preconditions": pre}); print(json.dumps(pre, indent=2)); return
    states = []
    for q, rows in groups.items():
        mt = top(rows, "pairwise_policy_score"); dt = top([med[key(x)] for x in rows], "median_pairwise_policy_score")
        same = key(mt) == key(dt); dmt = med[key(mt)]
        states.append({"query_id": q, "fold": mt["fold"], "class": mt["true_action_class"], "delta": mt["truth_recall_delta"],
                       "mean_execute": mt["pairwise_policy_score"] >= MEAN_T[mt["fold"]],
                       "median_execute": dt["median_pairwise_policy_score"] >= MED_T[mt["fold"]], "top_same": same,
                       "signed_gap": mt["pairwise_policy_score"] - dmt["median_pairwise_policy_score"],
                       "absolute_gap": abs(mt["pairwise_policy_score"] - dmt["median_pairwise_policy_score"])})
    top_counts = Counter(x["class"] for x in states)
    executed = [x for x in states if x["mean_execute"]]
    ec = Counter(x["class"] for x in executed)
    # Oracle headroom is exactly derivable from already-persisted action truth deltas, not from a model.
    oracle = {q: max([r["truth_recall_delta"] for r in rows] + [0.0]) for q, rows in groups.items()}
    op = [x for x in states if oracle[x["query_id"]] > 0]
    same_noop = [x for x in states if x["top_same"] and x["mean_execute"] and not x["median_execute"]]
    changed_noop = [x for x in states if not x["top_same"] and x["mean_execute"] and not x["median_execute"]]
    pre.update({"top_classes_69_5405_126": top_counts == Counter({"BENEFICIAL": 69, "NEUTRAL": 5405, "HARMFUL": 126}),
                "executed_BNH_6_392_12": ec == Counter({"BENEFICIAL": 6, "NEUTRAL": 392, "HARMFUL": 12}),
                "oracle_positive_426": len(op) == 426, "same_top_noop_337": len(same_noop) == 337, "changed_top_noop_56": len(changed_noop) == 56})
    if not all(pre.values()):
        dump({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic", "preconditions": pre,
              "observed": {"top_counts": dict(top_counts), "execution_counts": dict(ec), "oracle_positive": len(op), "same_noop": len(same_noop), "changed_noop": len(changed_noop)}}); print(json.dumps(pre, indent=2)); return
    def bh(rows): return ([x for x in rows if x["class"] == "BENEFICIAL"], [x for x in rows if x["class"] == "HARMFUL"])
    original_b, original_h = bh(executed)
    original = {gap: compare(original_b, original_h, gap) for gap in ("signed_gap", "absolute_gap")}
    refs = {gap: original[gap]["raw_orientation"] for gap in original}
    for gap in original: original[gap]["orientation_matches_original_n18"] = True
    if original["signed_gap"]["directional_AUC"] != .875 and original["absolute_gap"]["directional_AUC"] != .875:
        dump({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic", "preconditions": pre, "original_n18": original}); return
    def paired(rows):
        b, h = bh(rows)
        return {gap: compare(b, h, gap, refs[gap]) for gap in ("signed_gap", "absolute_gap")}
    all_top = paired(states); oracle_top = paired(op); same_abst = paired(same_noop); changed_abst = paired(changed_noop)
    fold = {}
    per_gap = {"signed_gap": [], "absolute_gap": []}
    for f in range(1, 5):
        results = paired([x for x in states if x["fold"] == f])
        fold[str(f)] = results
        for gap in per_gap: per_gap[gap].append(results[gap])
    directions = {gap: orientation_summary(per_gap[gap]) for gap in per_gap}
    labels = [("ORIGINAL_EXECUTED_B_VS_H", original), ("ALL_BTOP_VS_HTOP", all_top),
              ("ORACLE_POSITIVE_BTOP_VS_HTOP", oracle_top), ("SAME_TOP_ABSTENTION_B_VS_H", same_abst),
              ("TOP_CHANGED_ABSTENTION_B_VS_H", changed_abst)]
    stability = {}
    for gap in ("signed_gap", "absolute_gap"):
        stability[gap] = [{"population": label, "B_count": obj[gap]["B_count"], "H_count": obj[gap]["H_count"],
                           "raw_AUC": obj[gap]["raw_AUC"], "directional_AUC": obj[gap]["directional_AUC"],
                           "orientation_match": obj[gap]["orientation_matches_original_n18"],
                           "delta_directional_AUC_vs_original": None if obj[gap]["directional_AUC"] is None else obj[gap]["directional_AUC"] - original[gap]["directional_AUC"]}
                          for label, obj in labels]
    report = {"status": "PASS", "experiment": "STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic",
      "scientific_status": "DIAGNOSTIC_ONLY_NO_BRANCH_GATE", "training_executed": False, "new_scoring_executed": False,
      "new_formula_tested": False, "threshold_optimization_performed": False, "policy_counterfactual_performed": False,
      "policy_oof_strict": True, "end_to_end_selection_oof": False, "k77_selected_exploratorily_on_folds_1_4": True,
      "decision_confidence_signal_selected_exploratorily_on_folds_1_4": True,
      "interpretation_scope": "POST_SELECTION_DISAGREEMENT_SIGNAL_ROBUSTNESS_FORENSIC_CONDITIONAL_ON_EXPLORATORY_FIXED_K77", "K": 77,
      "preconditions": pre, "original_n18": {"B": 6, "H": 12, "signed_gap": original["signed_gap"], "absolute_gap": original["absolute_gap"],
          "reference_variant_for_auc_0_875": "signed_gap", "reference_orientation": refs},
      "population_all_mean_top": {"B": 69, "H": 126, **all_top},
      "population_oracle_positive": {"B": sum(x["class"] == "BENEFICIAL" for x in op), "H": sum(x["class"] == "HARMFUL" for x in op),
          "oracle_headroom_sum_B_top": sum(oracle[x["query_id"]] for x in op if x["class"] == "BENEFICIAL"),
          "oracle_headroom_sum_H_top": sum(oracle[x["query_id"]] for x in op if x["class"] == "HARMFUL"), **oracle_top},
      "population_same_top_abstention": {"total": 337, "class_counts": dict(Counter(x["class"] for x in same_noop)),
          "class_distributions": group_distributions(same_noop), **same_abst,
          "neutral_vs_B_and_H_descriptive": {"NEUTRAL": group_distributions(same_noop)["NEUTRAL"]}},
      "population_mean_executed_baseline": {"total": 410, "B": 6, "N": 392, "H": 12, "label": "NOT_INDEPENDENT_BASELINE_POPULATION",
          "class_distributions": group_distributions(executed), **original},
      "population_top_changed_abstention": {"total": 56, "class_counts": dict(Counter(x["class"] for x in changed_noop)),
          "class_distributions": group_distributions(changed_noop), **changed_abst},
      "per_fold_all_Btop_vs_Htop": fold, "direction_consistency": directions, "effect_size_stability": stability,
      "robustness_status": {gap: robustness(all_top[gap], directions[gap]) for gap in ("signed_gap", "absolute_gap")},
      "preserved_status": {"ranking_bottleneck": "PRIMARY", "decision_confidence_representation": "SUPPORTED", "top1_top2_margin": "WEAKENED",
          "underlying_pairwise_dispersion": "SUPPORTED_HYPOTHESIS_ONLY", "decision_confidence_intervention": "NOT_YET_AUTHORIZED",
          "raw_pairwise_rescoring": "NOT_AUTHORIZED", "third_reducer": "NOT_AUTHORIZED", "pair_family": "UNRESOLVED_NOT_ESTABLISHED",
          "pair_reweighting": "NOT_AUTHORIZED", "HGB": "UNRESOLVED", "feature_expansion": "NOT_JUSTIFIED", "K77": "NOT_REJECTED",
          "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"},
      "notes": ["Only the preselected signed and absolute same-action MEAN-MEDIAN gaps were evaluated.",
          "Raw AUC orientation, not directional AUC alone, is used for every consistency statement.",
          "Post-selection descriptive forensic only; no intervention, confidence direction, threshold, or policy is authorized."]}
    dump(report)
    print(json.dumps({"status": "PASS", "original": {g: original[g]["directional_AUC"] for g in original},
      "all_top": {g: {"raw": all_top[g]["raw_AUC"], "directional": all_top[g]["directional_AUC"], "match": all_top[g]["orientation_matches_original_n18"]} for g in all_top},
      "robustness": report["robustness_status"], "directions": directions}, indent=2))


if __name__ == "__main__": main()
