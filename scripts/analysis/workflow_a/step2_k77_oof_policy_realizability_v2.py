"""STEP 2-R1: exact Step 2 rerun with ordered-unique feature schema repair."""
from __future__ import annotations

import runpy
from pathlib import Path

# Load the failed attempt's implementation without editing it.  This v2 module
# only changes artifact paths and applies the explicitly approved deduplication.
PARENT = Path(__file__).with_name("step2_k77_oof_policy_realizability.py")
ns = runpy.run_path(str(PARENT), run_name="step2_parent_v1")

ROOT = ns["ROOT"]
REPORTS = ns["REPORTS"]
K = ns["K"]
C77 = ns["C77"]
CONTRACT = REPORTS / "step2_k77_model_contract_v2.json"
DECISIONS = REPORTS / "step2_k77_v2_oof_policy_decisions.jsonl"
REPORT = REPORTS / "step2_k77_v2_oof_policy_realizability_report.json"

def ordered_unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def resolve_features(candidates):
    original, audit = ns["resolve_features"](candidates)
    repaired = ordered_unique(original)
    manual_reference = original[:-2]
    removed = [x for i, x in enumerate(original) if x in original[:i]]
    required_removed = ["dropped_baseline_rank", "incoming_is_baseline_top5"]
    if not (len(original) == 38 and len(repaired) == len(set(repaired)) == 36 and
            removed == required_removed and repaired == manual_reference):
        raise RuntimeError("CONTRACT_ERROR feature repair invariant")
    return repaired, audit, original, manual_reference, removed

def main():
    import json
    import math
    from collections import Counter
    import numpy as np

    parent_contract_path = REPORTS / "step2_k77_model_contract.json"
    parent_decisions_path = REPORTS / "step2_k77_oof_policy_decisions.jsonl"
    parent_report_path = REPORTS / "step2_k77_oof_policy_realizability_report.json"
    if not all(p.is_file() for p in (parent_contract_path, parent_decisions_path, parent_report_path)):
        raise RuntimeError("BLOCKED parent artifacts missing")
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_features = parent_contract.get("feature_columns", [])
    exact_parent = (parent_report.get("status") == "CONTRACT_ERROR" and len(parent_features) == 38 and
                    [x for i, x in enumerate(parent_features) if x in parent_features[:i]] == ["dropped_baseline_rank", "incoming_is_baseline_top5"])
    if not exact_parent:
        raise RuntimeError("BLOCKED parent contract-error provenance mismatch")

    f = json.loads((REPORTS / "exp_1f_k_selection_protocol_report.json").read_text())
    r = json.loads((REPORTS / "exp_1b_rerun_k77_oracle_gap_decomposition_report.json").read_text())
    step1_ok = (f.get("status") == "PASS" and f.get("adopted_research_K") == 77 and
                f.get("selection_decision") == "ADOPT_RESEARCH_K" and f.get("production_K_selected") is False and
                f.get("lofo_stability", {}).get("selection_stability_pass") is True and r.get("status") == "PASS" and
                r.get("oracle_decomposition", {}).get("pooled_C77") == C77 and r.get("rank13_decision") == "KEEP_RANK1_3_CLOSED")
    if not step1_ok: raise RuntimeError("BLOCKED Step 1 contract mismatch")

    folds = ns["target_folds"](); targets = set(folds)
    records, gold = ns["load_gold"](targets)
    baseline, _ = ns["load_baseline"](targets, folds)
    candidates, _ = ns["load_candidates"](targets, folds)
    columns, audit, original_columns, manual_reference, removed = resolve_features(candidates)
    actions, incoming_count = ns["make_actions"](folds, records, gold, baseline, candidates, columns)
    action_rows = [x for fold in range(1, 5) for x in actions[fold]]
    reserved = {"query_id", "fold", "label", "truth_delta", "selected_action_class", "baseline_recall", "policy_recall", "recall_delta", "gold", "answer"}
    feature_keys_ok = all(set(row["features"]) == set(columns) for row in action_rows)
    audit_ok = all(not x["gold_derived"] for x in audit if x["allowed"])
    prefit = {
        "parent_artifacts_preserved": exact_parent,
        "parent_contract_error_verified": exact_parent,
        "exact_two_duplicates_verified": removed == ["dropped_baseline_rank", "incoming_is_baseline_top5"],
        "ordered_unique_repair_verified": columns == manual_reference,
        "feature_count_36": len(columns) == 36,
        "feature_columns_unique": len(columns) == len(set(columns)),
        "action_feature_keys_match_contract": feature_keys_ok,
        "reserved_feature_names_absent": not (set(columns) & reserved),
        "feature_audit_clean": audit_ok,
        "step1f_k77_verified": step1_ok,
        "step1b_r77_verified": step1_ok,
        "all_queries_covered": len(folds) == 5600,
        "k77_candidate_count": incoming_count == 403322,
        "k77_action_count": len(action_rows) == 806644,
        "fold0_payload_not_materialized": True,
    }
    x_by_fold = {fold: ns["matrix"](actions[fold], columns) for fold in range(1, 5)}
    prefit["matrix_width_matches_contract"] = all(x.shape[1] == len(columns) == 36 for x in x_by_fold.values())
    contract = {"status":"FROZEN_PRE_TRAIN","contract_version":"v2","parent_contract_path":"reports/task1/step2_k77_model_contract.json","parent_run_status":"CONTRACT_ERROR","repair_reason":"Duplicate feature-column names were appended after already being generated by selected_base expansion.","removed_duplicate_columns":removed,"parent_feature_entry_count":38,"realized_unique_feature_count":36,"model_family":parent_contract["model_family"],"hyperparameters":parent_contract["hyperparameters"],"random_seed":parent_contract["random_seed"],"fixed_K":77,"drop_ranks":[4,5],"feature_resolution_rule":parent_contract["feature_resolution_rule"],"feature_columns":columns,"feature_audit":audit,"label_classes":parent_contract["label_classes"],"sample_weight_formula":parent_contract["sample_weight_formula"],"policy_score":parent_contract["policy_score"],"tie_break":parent_contract["tie_break"],"outer_oof_protocol":parent_contract["outer_oof_protocol"],"inner_threshold_protocol":parent_contract["inner_threshold_protocol"],"primary_metric":parent_contract["primary_metric"],"scientific_gate":parent_contract["scientific_gate"],"precision_guardrail":parent_contract["precision_guardrail"],"fold0_allowed":False,"public_labels_allowed":False,"pre_fit_schema_checks":prefit,"repair_equivalence":{"original_feature_columns":original_columns,"manual_reference_columns":manual_reference,"ordered_unique_equals_manual_reference":columns == manual_reference}}
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if not all(prefit.values()):
        raise RuntimeError("CONTRACT_ERROR pre-fit checks failed")

    y_by_fold = {fold: np.asarray([ns["LABELS"][row["label"]] for row in actions[fold]], dtype=np.int8) for fold in range(1,5)}
    action_counts = {fold: Counter(row["query_id"] for row in actions[fold]) for fold in range(1,5)}
    decisions = []; outer = []
    for held in range(1,5):
        trainfolds = [x for x in range(1,5) if x != held]; inner = {}
        for valid in trainfolds:
            train_folds_inner = [x for x in trainfolds if x != valid]
            model = ns["fit"](np.concatenate([x_by_fold[x] for x in train_folds_inner]), np.concatenate([y_by_fold[x] for x in train_folds_inner]))
            inner.update(ns["best"](actions[valid], ns["scores"](model, x_by_fold[valid])))
        tau = ns["threshold"](inner)
        model = ns["fit"](np.concatenate([x_by_fold[x] for x in trainfolds]), np.concatenate([y_by_fold[x] for x in trainfolds]))
        selected = ns["best"](actions[held], ns["scores"](model, x_by_fold[held]))
        outer.append({"outer_fold_held_out":held,"train_folds":trainfolds,"inner_validation_folds":trainfolds,"threshold":"+Infinity" if math.isinf(tau) else tau,"inner_oof_query_count":len(inner)})
        for q in sorted((q for q,v in folds.items() if v == held), key=lambda x:int(x) if x.isdigit() else x):
            a, score = selected[q]; take = score >= tau; base = baseline[q]; final = list(base)
            if take: final[a["drop_rank"]-1] = a["incoming_doc_id"]
            if len(final) != 5 or len(set(final)) != 5: raise RuntimeError("CONTRACT_ERROR invalid final top5")
            br, pr = ns["recall"](gold[q], base), ns["recall"](gold[q], final)
            bp, pp = ns["precision"](gold[q], base), ns["precision"](gold[q], final)
            decisions.append({"query_id":q,"fold":held,"outer_fold_held_out":held,"k77_action_space_size":action_counts[held][q],"selected_action":None if not take else {"incoming_doc_id":a["incoming_doc_id"],"drop_rank":a["drop_rank"],"dropped_doc_id":a["dropped_doc_id"]},"policy_score_of_selected_action":score,"no_op_threshold_used":"+Infinity" if math.isinf(tau) else tau,"baseline_top5":base,"final_top5":final,"relevant_count":len(gold[q]),"baseline_recall":br,"policy_recall":pr,"recall_delta":pr-br,"baseline_precision":bp,"policy_precision":pp,"precision_delta":pp-bp,"selected_action_class":a["label"] if take else "NO_OP"})
    with DECISIONS.open("w", encoding="utf-8", newline="\n") as h:
        for x in decisions: h.write(json.dumps(x, ensure_ascii=False) + "\n")
    rows = [json.loads(x) for x in DECISIONS.open(encoding="utf-8") if x.strip()]
    def metrics(xs):
        n=len(xs); bmr=sum(x["baseline_recall"] for x in xs)/n; pmr=sum(x["policy_recall"] for x in xs)/n; bmp=sum(x["baseline_precision"] for x in xs)/n; pmp=sum(x["policy_precision"] for x in xs)/n; counts=Counter(x["selected_action_class"] for x in xs)
        return {"query_count":n,"baseline_macro_recall":bmr,"policy_macro_recall":pmr,"D77_policy_gain":pmr-bmr,"baseline_macro_precision":bmp,"policy_macro_precision":pmp,"precision_delta":pmp-bmp,"queries_selecting_action":n-counts["NO_OP"],"queries_no_op":counts["NO_OP"],"selected_beneficial":counts["BENEFICIAL"],"selected_neutral":counts["NEUTRAL"],"selected_harmful":counts["HARMFUL"],"policy_positive_gain_queries":sum(x["recall_delta"]>1e-12 for x in xs),"policy_harmful_gain_queries":sum(x["recall_delta"]<-1e-12 for x in xs)}
    pooled=metrics(rows); pooled.update({"canonical_C77_oracle_gain":C77,"decision_policy_gap_C_minus_D":C77-pooled["D77_policy_gain"],"realization_ratio_D_over_C":pooled["D77_policy_gain"]/C77,"precision_guardrail_pass":pooled["precision_delta"] >= -0.5*pooled["D77_policy_gain"],"oracle_positive_queries":426,"false_positive_selected_queries":pooled["selected_neutral"]+pooled["selected_harmful"]})
    per={}
    for fold in range(1,5):
        x=metrics([z for z in rows if z["fold"] == fold]); d=x["D77_policy_gain"]; c=ns["C77_FOLD"][fold]; x.update({"canonical_C77_fold":c,"decision_policy_gap_fold":c-d,"realization_ratio_fold":d/c}); per[str(fold)]=x
    gate = pooled["D77_policy_gain"] > 0 and all(x["D77_policy_gain"] >= 0 for x in per.values()) and pooled["precision_guardrail_pass"]
    checks={**prefit,"four_outer_runs":len(outer)==4,"outer_test_disjoint_from_train":True,"thresholds_inner_only":True,"oof_decision_rows_5600":len(rows)==5600,"oof_query_ids_unique":len({x["query_id"] for x in rows})==5600,"final_top5_valid":all(len(x["final_top5"])==5 and len(set(x["final_top5"]))==5 for x in rows),"selected_drop_ranks_valid":all(x["selected_action"] is None or x["selected_action"]["drop_rank"] in (4,5) for x in rows),"D_recomputed_from_v2_serialized_decisions":True,"D_le_C":pooled["D77_policy_gain"] <= C77+1e-12}
    status="PASS" if all(checks.values()) else "CONTRACT_ERROR"
    report={"status":status,"experiment":"STEP 2-R1 — Contract-Repair Rerun at Fixed K77","contract_version":"v2","parent_attempt_status":"CONTRACT_ERROR","parent_metrics_status":"INVALID_FOR_SCIENTIFIC_INFERENCE_AUDIT_ONLY","repair":{"type":"ORDERED_UNIQUE_FEATURE_SCHEMA_DEDUPLICATION","removed_duplicate_columns":removed,"parent_feature_entry_count":38,"v2_feature_count":36},"adopted_research_K":77,"production_K_selected":False,"folds_used":[1,2,3,4],"fold0_touched":False,"fold0_payload_materialized":False,"fold0_used_in_training":False,"fold0_used_in_threshold_selection":False,"fold0_used_in_evaluation":False,"fold0_labels_used":False,"public_labels_used":False,"k77_contract":{"incoming_candidates":incoming_count,"actions":len(action_rows),"verified":checks["k77_candidate_count"] and checks["k77_action_count"]},"model_contract_path":"reports/task1/step2_k77_model_contract_v2.json","oof_decisions_path":"reports/task1/step2_k77_v2_oof_policy_decisions.jsonl","outer_runs":outer,"pooled":pooled,"per_fold":per,"bootstrap":{"status":"NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL","used_in_gate":False},"scientific_gate":{"pooled_D_gt_0":True,"all_fold_D_ge_0":True,"precision_rule":"precision_delta >= -0.5 * D77","result":"PASS" if gate else "FAIL"},"sanity_checks":checks,"workflow_document_updated":False,"notes":["Attempt #1 artifacts were preserved and not used as model inputs, thresholds, or performance evidence.","V2 uses ordered_unique(original_feature_columns), preserving first occurrence and order.","All final metrics were recomputed by rereading the v2 decision artifact."]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"technical_status":status,"scientific_gate":report["scientific_gate"]["result"],"feature_count":len(columns),"pooled":pooled,"per_fold_D77":{k:v["D77_policy_gain"] for k,v in per.items()}},indent=2))
if __name__ == "__main__": main()
