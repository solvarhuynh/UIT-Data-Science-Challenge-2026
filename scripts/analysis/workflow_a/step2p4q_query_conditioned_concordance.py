import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/task1"
P4_REPORT = R / "step2p4_discriminability_report.json"
PRED = R / "step2p4_proxy_predictions.jsonl"
P1_ATTR = R / "step2p1f_failure_attribution_report.json"
P1_SCORES = R / "step2p1f_full_action_scores.jsonl"
OUT = R / "step2p4q_concordance_report.json"


def empty():
    return {"oracle_positive_queries": 0, "included_queries": 0, "excluded_queries": 0,
            "excluded_NO_B": 0, "excluded_NO_N": 0, "excluded_NO_B_AND_NO_N": 0,
            "B_actions": 0, "N_actions": 0, "BN_pairs": 0, "B_above_N_pairs": 0,
            "tie_pairs": 0, "B_below_N_pairs": 0, "majority_concordant_queries": 0,
            "exact_50_50_queries": 0, "non_majority_queries": 0}


def finalize(x):
    if x["BN_pairs"] != x["B_above_N_pairs"] + x["tie_pairs"] + x["B_below_N_pairs"]:
        raise RuntimeError("CONTRACT_ERROR pair count identity")
    if x["included_queries"] + x["excluded_queries"] != x["oracle_positive_queries"]:
        raise RuntimeError("CONTRACT_ERROR query count identity")
    x["pair_pooled_concordance"] = (x["B_above_N_pairs"] + 0.5 * x["tie_pairs"]) / x["BN_pairs"] if x["BN_pairs"] else None
    x["query_majority_concordance_rate"] = x["majority_concordant_queries"] / x["included_queries"] if x["included_queries"] else None
    return x


def main():
    p4 = json.loads(P4_REPORT.read_text(encoding="utf-8"))
    checks = {"p4_status": p4.get("status") == "PASS", "experiment_type": p4.get("experiment_type") == "DIAGNOSTIC_ONLY_NO_BRANCH_GATE",
              "K77": p4.get("K") == 77, "features36": p4.get("features") == 36,
              "policy_unchanged": p4.get("policy_modified") is False and p4.get("new_policy_scoring_executed") is False,
              "fold0_public_absent": p4.get("fold0_or_public_used") is False,
              "auc_direction_not_reversed": p4.get("auc", {}).get("directional_auc_reversal_used") is False}
    predictions = [json.loads(line) for line in PRED.read_text(encoding="utf-8").splitlines() if line]
    keys = [(x["query_id"], x["action_identity"]["incoming_doc_id"], x["action_identity"]["drop_rank"]) for x in predictions]
    checks.update({"outer_rows_806644": len(predictions) == 806644, "outer_scope_only": all(x.get("prediction_scope") == "OUTER_HELD_OUT" for x in predictions),
                   "folds_1_to_4": sorted(set(x["fold"] for x in predictions)) == [1, 2, 3, 4],
                   "unique_action_predictions": len(keys) == len(set(keys)), "one_finite_score_each": all(isinstance(x.get("p_hat_beneficial"), float) for x in predictions)})
    if not all(checks.values()):
        raise RuntimeError(f"CONTRACT_ERROR P4 frozen prediction contract: {checks}")
    attr = json.loads(P1_ATTR.read_text(encoding="utf-8"))
    if not (attr.get("status") == "PASS" and attr.get("oracle_positive_queries") == 426 and
            [attr["trs"][name]["query_count"] for name in ("S_SUCCESS", "T_THRESHOLD_LOSS", "R_RANKING_DISCRIMINATION_LOSS")] == [6, 63, 357]):
        raise RuntimeError("CONTRACT_ERROR canonical P1 oracle-positive reference")
    canonical = [json.loads(line) for line in P1_SCORES.read_text(encoding="utf-8").splitlines() if line]
    oracle, canonical_fold = {}, {}
    for row in canonical:
        q = row["query_id"]
        oracle[q] = max(oracle.get(q, 0.0), row["truth_recall_delta"])
        canonical_fold[q] = row["fold"]
    oracle_positive = {q for q, delta in oracle.items() if delta > 1e-12}
    if len(oracle_positive) != 426:
        raise RuntimeError("CONTRACT_ERROR oracle-positive population")
    by_query = defaultdict(list)
    for row in predictions:
        by_query[row["query_id"]].append(row)
    if set(by_query) != set(oracle):
        raise RuntimeError("CONTRACT_ERROR prediction/canonical query coverage")
    pooled = empty()
    per = {f"F{fold}": empty() for fold in range(1, 5)}
    audits = []
    for q in sorted(oracle_positive, key=lambda z: int(z) if z.isdigit() else z):
        fold = canonical_fold[q]
        targets = (pooled, per[f"F{fold}"])
        for x in targets:
            x["oracle_positive_queries"] += 1
        rows = by_query[q]
        bs = [row for row in rows if row["POST_PREDICTION_EVALUATION"]["true_action_class"] == "BENEFICIAL"]
        ns = [row for row in rows if row["POST_PREDICTION_EVALUATION"]["true_action_class"] == "NEUTRAL"]
        for x in targets:
            x["B_actions"] += len(bs); x["N_actions"] += len(ns)
        if not bs or not ns:
            reason = "NO_B_AND_NO_N" if not bs and not ns else "NO_B" if not bs else "NO_N"
            for x in targets:
                x["excluded_queries"] += 1; x[f"excluded_{reason}"] += 1
            continue
        above = ties = below = 0
        for b in bs:
            for n in ns:
                if b["p_hat_beneficial"] > n["p_hat_beneficial"]: above += 1
                elif b["p_hat_beneficial"] == n["p_hat_beneficial"]: ties += 1
                else: below += 1
        pair_count = above + ties + below
        concordance = (above + 0.5 * ties) / pair_count
        audits.append({"query_id": q, "fold": fold, "B_actions": len(bs), "N_actions": len(ns), "BN_pairs": pair_count,
                       "B_above_N_pairs": above, "tie_pairs": ties, "B_below_N_pairs": below, "query_concordance": concordance})
        for x in targets:
            x["included_queries"] += 1; x["BN_pairs"] += pair_count; x["B_above_N_pairs"] += above
            x["tie_pairs"] += ties; x["B_below_N_pairs"] += below
            if concordance > 0.5: x["majority_concordant_queries"] += 1
            elif concordance == 0.5: x["exact_50_50_queries"] += 1
            else: x["non_majority_queries"] += 1
    pooled = finalize(pooled)
    per = {name: finalize(value) for name, value in per.items()}
    report = {"status": "PASS", "experiment": "STEP 2-P4-Q — Query-Conditioned B-vs-N Proxy Concordance Diagnostic",
              "experiment_type": "DIAGNOSTIC_ONLY_NO_BRANCH_GATE", "training_executed": False, "new_inference_executed": False,
              "policy_modified": False, "fold0_or_public_used": False, "policy_oof_strict": True, "end_to_end_selection_oof": False,
              "k77_selected_exploratorily_on_folds_1_4": True, "K": 77,
              "frozen_prediction_source": "reports/task1/step2p4_proxy_predictions.jsonl", "preflight": checks,
              "population": {"canonical_oracle_positive": 426, "included_queries": pooled["included_queries"], "excluded_queries": pooled["excluded_queries"],
                             "excluded_reasons": {"NO_B": pooled["excluded_NO_B"], "NO_N": pooled["excluded_NO_N"], "NO_B_AND_NO_N": pooled["excluded_NO_B_AND_NO_N"]}},
              "pooled": {key: pooled[key] for key in ("B_actions", "N_actions", "BN_pairs", "B_above_N_pairs", "tie_pairs", "B_below_N_pairs", "pair_pooled_concordance", "majority_concordant_queries", "exact_50_50_queries", "query_majority_concordance_rate")},
              "per_fold": per, "query_level_concordance_audit": audits, "tie_handling": "EXACT_EQUALITY_COUNTS_AS_0.5", "chance_reference": 0.5,
              "auc_direction_reversal": False, "standing_policy_comparator": {"R": 357, "D77_exact": -0.0007142857142857143, "gain_sum": -4.0, "S": 6, "T": 63},
              "branch_gate": "NONE", "preserved_status": {"P4_global_signal": "STRONG_DESCRIPTIVE_OOF_SIGNAL", "label_free_signal_existence": "SUPPORTED",
                "within_query_discriminability": "UNDER_DIAGNOSTIC", "proxy_gate": "NOT_AUTHORIZED", "proxy_reranking": "NOT_AUTHORIZED",
                "feature_signal": "FEATURE_SIGNAL_PRESENT", "feature_expansion": "NOT_JUSTIFIED", "HGB_beneficiality_recognition": "SUPPORTED_CAPACITY_FOR_BENEFICIALITY_RECOGNITION",
                "HGB_final_ranking_capacity": "UNRESOLVED", "HGB_API_objective_flexibility": "BLOCKED_BY_IMPLEMENTATION_FOR_PAIRWISE_MARGIN", "P3_sub_branch": "CLOSED",
                "pair_reweighting": "NOT_AUTHORIZED", "B_vs_N_only": "NOT_AUTHORIZED", "rank_feature_intervention": "NOT_JUSTIFIED", "raw_pairwise_dispersion": "DEFERRED",
                "K77": "NOT_REJECTED", "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"},
              "notes": ["Read-only diagnostic: frozen outer-held-out P4 predictions were reused exactly; no models were loaded for inference.",
                        "Truth labels were joined only after frozen scores to select the preregistered B/N diagnostic population and compute concordance.",
                        "No proxy threshold, gate, reranking, P1 fusion, policy modification, or D77 policy computation occurred."],
              "next_state": "STEP2_P4Q_COMPLETE_PENDING_PROFESSOR_REVIEW"}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "population": report["population"], "pooled": report["pooled"],
                      "per_fold_concordance": {name: value["pair_pooled_concordance"] for name, value in per.items()}}, indent=2))


if __name__ == "__main__":
    main()
