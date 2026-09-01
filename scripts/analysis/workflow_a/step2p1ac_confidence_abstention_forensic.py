"""STEP 2-P1A-C: read-only confidence / abstention forensic.

This script only reads persisted action-score artifacts.  It never loads a
model, fits, scores, selects a threshold, or constructs a counterfactual
policy.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "task1"
OUT = R / "step2p1ac_confidence_signal_report.json"
MEAN_T = {1: 0.8871766063648394, 2: 0.9158352964999534,
          3: 0.8885338990548798, 4: 0.9072451320464932}


def read(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def ident(row):
    return (row["query_id"], row["incoming_doc_id"], row["drop_rank"])


def top_two(rows, score):
    ranked = sorted(rows, key=lambda x: (-x[score], -x["drop_rank"],
                                         x["incoming_union_rank"], x["incoming_doc_id"]))
    return ranked[0], ranked[1]


def summary(values):
    if not values:
        return {k: None for k in ("count", "min", "p25", "median", "mean", "p75", "max")} | {"count": 0}
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "min": float(a.min()),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def auc_details(beneficial, harmful):
    wins = sum(b > h for b in beneficial for h in harmful)
    ties = sum(b == h for b in beneficial for h in harmful)
    total = len(beneficial) * len(harmful)
    raw = (wins + 0.5 * ties) / total
    directional = max(raw, 1.0 - raw)
    direction = "HIGHER_ASSOCIATED_WITH_BENEFICIAL" if raw >= 0.5 else "LOWER_ASSOCIATED_WITH_BENEFICIAL"
    strength = ("WEAK" if directional < 0.60 else "MODEST" if directional < 0.70 else
                "VISIBLE" if directional < 0.80 else "STRONG_DESCRIPTIVE_SEPARATION")
    return {"ROC_AUC_raw": raw, "ROC_AUC_directional": directional, "direction": direction,
            "P_signal_B_gt_signal_H": wins / total, "ties": ties, "tie_probability": ties / total,
            "strength_label": strength, "small_n_warning": True}


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for z in order[i:j]:
            result[z] = rank
        i = j
    return result


def pearson(x, y):
    if len(x) < 2:
        return None
    a, b = np.asarray(x, float), np.asarray(y, float)
    da, db = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float(np.dot(da, da) * np.dot(db, db)))
    return None if denominator == 0 else float(np.dot(da, db) / denominator)


def location(abstained, b, h):
    m, bm, hm = np.median(abstained), np.median(b), np.median(h)
    db, dh = abs(m - bm), abs(m - hm)
    if min(bm, hm) <= m <= max(bm, hm):
        label = "BETWEEN_B_AND_H"
    elif db < dh:
        label = "CLOSER_TO_B"
    elif dh < db:
        label = "CLOSER_TO_H"
    else:
        label = "OUTSIDE_BOTH"
    return {"median": float(m), "relative_to_B_H": label,
            "absolute_distance_to_B_median": float(db), "absolute_distance_to_H_median": float(dh)}


def main():
    p1a, p1af, contract = (read("step2p1a_realizability_report.json"),
                           read("step2p1af_topaction_change_report.json"), read("step2p1_model_contract.json"))
    preconditions = {
        "p1a_pass": p1a["status"] == "PASS", "p1af_pass": p1af["status"] == "PASS",
        "K77": contract["fixed_K"] == 77,
    }
    med = {}
    with (R / "step2p1a_median_scores.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            med[ident(x)] = x
    groups = defaultdict(list)
    mean_rows = 0
    with (R / "step2p1f_full_action_scores.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            k = ident(m)
            if k not in med:
                preconditions["exact_action_key_join"] = False
                continue
            groups[m["query_id"]].append((m, med[k]))
            mean_rows += 1
    preconditions.update({"mean_rows_806644": mean_rows == 806644,
                          "median_rows_806644": len(med) == 806644,
                          "exact_action_key_join": preconditions.get("exact_action_key_join", True) and len(med) == mean_rows,
                          "queries_5600": len(groups) == 5600})
    if not all(preconditions.values()):
        OUT.write_text(json.dumps({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1A-C — Decision-Confidence / Abstention Failure Forensic", "preconditions": preconditions}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "CONTRACT_ERROR", "preconditions": preconditions}, indent=2)); return

    states = []
    for query_id, pairs in groups.items():
        mrows, drows = [x[0] for x in pairs], [x[1] for x in pairs]
        mt, mt2 = top_two(mrows, "pairwise_policy_score")
        dt, dt2 = top_two(drows, "median_pairwise_policy_score")
        if not mt["is_query_top_scored"] or not dt["is_median_top_action"]:
            preconditions["persisted_top_markers_match_recomputed"] = False
        dm = med[ident(mt)]
        mexec = mt["pairwise_policy_score"] >= MEAN_T[mt["fold"]]
        states.append({"query_id": query_id, "fold": mt["fold"], "mean": mt, "median": dt,
                       "mean_execute": mexec, "median_execute": False if math.isinf(float("inf")) else False,
                       "top_changed": ident(mt) != ident(dt),
                       "mean_top1_minus_top2": mt["pairwise_policy_score"] - mt2["pairwise_policy_score"],
                       "median_top1_minus_top2": dt["median_pairwise_policy_score"] - dt2["median_pairwise_policy_score"],
                       "same_action_mean_median_gap": mt["pairwise_policy_score"] - dm["median_pairwise_policy_score"],
                       "absolute_same_action_mean_median_gap": abs(mt["pairwise_policy_score"] - dm["median_pairwise_policy_score"]),
                       "incoming_union_rank": mt["incoming_union_rank"], "drop_rank": mt["drop_rank"],
                       "truth_recall_delta": mt["truth_recall_delta"]})
    # All MEDIAN thresholds are intentionally fixed historical values; only their resulting action is read.
    median_thresholds = {1: float("inf"), 2: float("inf"), 3: 0.9837219720799008, 4: float("inf")}
    for s in states:
        s["median_execute"] = s["median"]["median_pairwise_policy_score"] >= median_thresholds[s["fold"]]
    preconditions["persisted_top_markers_match_recomputed"] = preconditions.get("persisted_top_markers_match_recomputed", True)

    executed = [s for s in states if s["mean_execute"]]
    counts = Counter(s["mean"]["true_action_class"] for s in executed)
    same_noop = [s for s in states if not s["top_changed"] and s["mean_execute"] and not s["median_execute"]]
    changed_noop = [s for s in states if s["top_changed"] and s["mean_execute"] and not s["median_execute"]]
    population_ok = (counts == Counter({"BENEFICIAL": 6, "NEUTRAL": 392, "HARMFUL": 12}) and
                     len(executed) == 410 and len(same_noop) == 337 and len(changed_noop) == 56)
    if not population_ok:
        OUT.write_text(json.dumps({"status": "CONTRACT_ERROR", "experiment": "STEP 2-P1A-C — Decision-Confidence / Abstention Failure Forensic", "preconditions": preconditions, "observed_counts": {"executed": len(executed), "BNH": dict(counts), "same_top_execute_to_noop": len(same_noop), "top_changed_execute_to_noop": len(changed_noop)}}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "CONTRACT_ERROR", "counts": dict(counts)}, indent=2)); return

    availability = {
        "pairwise_probability_std": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "pairwise_probability_variance": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "pairwise_probability_IQR": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "pairwise_probability_MAD": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "fraction_pairwise_probability_gt_0_5": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "pairwise_vote_entropy": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "inner_model_agreement": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
        "mean_top1_minus_top2": {"source_artifact": "step2p1f_full_action_scores.jsonl", "available": True, "requires_new_model_scoring": False},
        "median_top1_minus_top2": {"source_artifact": "step2p1a_median_scores.jsonl", "available": True, "requires_new_model_scoring": False},
        "mean_minus_median_same_action": {"source_artifact": "step2p1f_full_action_scores.jsonl + step2p1a_median_scores.jsonl", "available": True, "requires_new_model_scoring": False},
        "absolute_mean_minus_median_same_action": {"source_artifact": "step2p1f_full_action_scores.jsonl + step2p1a_median_scores.jsonl", "available": True, "requires_new_model_scoring": False},
        "incoming_union_rank": {"source_artifact": "step2p1f_full_action_scores.jsonl", "available": True, "requires_new_model_scoring": False, "scope": "persisted structural frozen feature"},
        "drop_rank": {"source_artifact": "step2p1f_full_action_scores.jsonl", "available": True, "requires_new_model_scoring": False, "scope": "persisted structural frozen feature"},
        "other_frozen_36_feature_values": {"source_artifact": None, "available": False, "requires_new_model_scoring": True, "status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"},
    }
    signal_specs = [("mean_top1_minus_top2", "SCORE_MARGIN"), ("median_top1_minus_top2", "SCORE_MARGIN"),
                    ("same_action_mean_median_gap", "MEAN_MEDIAN_DISAGREEMENT"),
                    ("absolute_same_action_mean_median_gap", "MEAN_MEDIAN_DISAGREEMENT"),
                    ("incoming_union_rank", "FROZEN_FEATURE"), ("drop_rank", "FROZEN_FEATURE")]
    b = [s for s in executed if s["mean"]["true_action_class"] == "BENEFICIAL"]
    h = [s for s in executed if s["mean"]["true_action_class"] == "HARMFUL"]
    n = [s for s in executed if s["mean"]["true_action_class"] == "NEUTRAL"]
    signal_results, table, abstention, changed_context, correlations = {}, [], {}, {}, {}
    primary = {"BENEFICIAL_EXECUTED": b, "HARMFUL_EXECUTED": h, "NEUTRAL_EXECUTED": n,
               "SAME_TOP_EXECUTE_TO_NOOP": same_noop, "TOP_CHANGED_EXECUTE_TO_NOOP": changed_noop}
    for name, family in signal_specs:
        groups_summary = {key: summary([s[name] for s in rows]) for key, rows in primary.items()}
        auc = auc_details([s[name] for s in b], [s[name] for s in h])
        signal_results[name] = {"signal_family": family, "available": True, "group_distributions": groups_summary,
                                "B_vs_H": auc}
        table.append({"signal_name": name, "signal_family": family, "available": True,
                      "B_count": 6, "H_count": 12, "B_median": groups_summary["BENEFICIAL_EXECUTED"]["median"],
                      "H_median": groups_summary["HARMFUL_EXECUTED"]["median"], "B_mean": groups_summary["BENEFICIAL_EXECUTED"]["mean"],
                      "H_mean": groups_summary["HARMFUL_EXECUTED"]["mean"], **auc})
        abstention[name] = {"same_top_execute_to_noop": groups_summary["SAME_TOP_EXECUTE_TO_NOOP"],
                            "median_location_relative_to_B_H": location([s[name] for s in same_noop], [s[name] for s in b], [s[name] for s in h])}
        changed_context[name] = {"top_changed_execute_to_noop": groups_summary["TOP_CHANGED_EXECUTE_TO_NOOP"],
                                 "same_top_execute_to_noop": groups_summary["SAME_TOP_EXECUTE_TO_NOOP"]}
        x, y = [s[name] for s in b + h], [s["truth_recall_delta"] for s in b + h]
        correlations[name] = {"n": 18, "Pearson_r": pearson(x, y), "Spearman_rho": pearson(average_ranks(x), average_ranks(y)), "small_n_warning": True,
                              "interpretation": "DESCRIPTIVE_ONLY_NO_CAUSAL_CLAIM"}
    table.sort(key=lambda x: x["ROC_AUC_directional"], reverse=True)
    per_fold = {}
    for fold in range(1, 5):
        fs = [s for s in states if s["fold"] == fold]
        fe = [s for s in fs if s["mean_execute"]]
        fb = [s for s in fe if s["mean"]["true_action_class"] == "BENEFICIAL"]
        fh = [s for s in fe if s["mean"]["true_action_class"] == "HARMFUL"]
        fn = [s for s in fe if s["mean"]["true_action_class"] == "NEUTRAL"]
        fsa = [s for s in fs if not s["top_changed"] and s["mean_execute"] and not s["median_execute"]]
        values = {}
        for name, _ in signal_specs:
            values[name] = {"B_median": summary([s[name] for s in fb])["median"], "H_median": summary([s[name] for s in fh])["median"],
                            "N_median": summary([s[name] for s in fn])["median"], "same_top_abstention_median": summary([s[name] for s in fsa])["median"],
                            "AUC_status": "INSUFFICIENT_CLASS_SUPPORT" if not fb or not fh else "AVAILABLE"}
            if fb and fh: values[name]["ROC_AUC_directional"] = auc_details([s[name] for s in fb], [s[name] for s in fh])["ROC_AUC_directional"]
        per_fold[str(fold)] = {"mean_executed_B": len(fb), "mean_executed_H": len(fh), "mean_executed_N": len(fn), "primary_signals": values}
    best = table[0]
    max_auc = best["ROC_AUC_directional"]
    headline = ("STRONG_BUT_SMALL_N_CONFIDENCE_SEPARATION" if max_auc >= .80 else
                "VISIBLE_CONFIDENCE_SEPARATION" if max_auc >= .70 else
                "WEAK_CONFIDENCE_SEPARATION" if max_auc >= .60 else "NO_VISIBLE_CONFIDENCE_SEPARATION")
    report = {"status": "PASS", "experiment": "STEP 2-P1A-C — Decision-Confidence / Abstention Failure Forensic",
              "scientific_status": "DIAGNOSTIC_ONLY_NO_BRANCH_GATE", "training_executed": False, "new_scoring_executed": False,
              "new_formula_tested": False, "threshold_optimization_performed": False, "policy_counterfactual_performed": False,
              "policy_oof_strict": True, "end_to_end_selection_oof": False, "k77_selected_exploratorily_on_folds_1_4": True,
              "interpretation_scope": "DECISION_CONFIDENCE_ABSTENTION_FORENSIC_CONDITIONAL_ON_EXPLORATORY_FIXED_K77", "K": 77,
              "preconditions": preconditions, "artifact_availability": availability,
              "inner_model_agreement_status": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS",
              "canonical_populations": {"mean_executed_B": 6, "mean_executed_N": 392, "mean_executed_H": 12,
                                        "mean_executions": 410, "same_top_execute_to_noop": len(same_noop), "top_changed_execute_to_noop": len(changed_noop)},
              "signal_results": signal_results, "primary_B_vs_H_table": table, "abstention_population": abstention,
              "top_changed_abstention_population": changed_context, "per_fold": per_fold,
              "realized_delta_correlations": correlations, "headline_status": headline, "strongest_available_signal": best,
              "preserved_status": {"remaining_ranking_bottleneck": "PRIMARY", "decision_confidence_hypothesis": "STRONGLY_SUPPORTED",
                                   "aggregation_as_ranking_fix": "WEAKENED", "third_reducer": "NOT_AUTHORIZED", "pair_family": "UNRESOLVED_NOT_ESTABLISHED",
                                   "pair_reweighting": "NOT_AUTHORIZED", "HGB": "UNRESOLVED", "feature_expansion": "NOT_JUSTIFIED",
                                   "K77": "NOT_REJECTED", "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"},
              "notes": ["Pure post-hoc analysis from persisted serialized score/action rows; zero model fit and zero new model scoring.",
                        "Raw pairwise probability vectors, inner-model agreement traces, and exact values for frozen features other than persisted structural metadata were not serialized and were not reconstructed.",
                        "All AUCs are descriptive only with B=6 and H=12; no confidence rule, threshold, selection, or intervention is authorized."]}
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "canonical": report["canonical_populations"], "strongest": {"name": best["signal_name"], "auc": best["ROC_AUC_directional"], "direction": best["direction"]}, "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
