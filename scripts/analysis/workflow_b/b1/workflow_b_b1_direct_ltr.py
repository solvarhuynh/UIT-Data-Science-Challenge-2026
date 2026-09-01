"""Preregistered Workflow-B B1: LambdaMART/NDCG@1 strict nested OOF.

The only score-selection objective is the frozen inner-OOF gain-sum contract.
No public data or Fold0 records are loaded.
"""
from __future__ import annotations

import hashlib, json, os, platform, runpy, sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import lightgbm, numpy, pandas, sklearn
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "task1"
MODELS = R / "workflow_b_b1_models"
SEED = 20260827

FEATURES = json.loads((R / "step2p1_model_contract.json").read_text(encoding="utf-8"))["feature_columns"]
HP = {"objective":"lambdarank", "metric":"ndcg", "eval_at":[1], "label_gain":[0,1,2], "learning_rate":0.05,
      "n_estimators":200, "num_leaves":31, "max_depth":5, "min_child_samples":20, "reg_alpha":0.0,
      "reg_lambda":1.0, "feature_fraction":1.0, "bagging_fraction":1.0, "bagging_freq":0,
      "deterministic":True, "force_col_wise":True, "verbosity":-1, "random_state":SEED, "n_jobs":1}
ENC = {"HARMFUL":0, "NEUTRAL":1, "BENEFICIAL":2}

def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def h(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def qkey(q: str): return (int(q) if str(q).isdigit() else str(q))
def action_key(a): return (-a["drop_rank"], a["incoming_union_rank"], a["incoming_doc_id"])
def top(rows): return min(rows, key=lambda z: (-z[1], *action_key(z[0])))
def transformed(base, action): return list(base[:action["drop_rank"]-1])+[action["incoming_doc_id"]]+list(base[action["drop_rank"]:])

def matrices(actions_by_fold, folds):
    rows=[]; groups=[]
    for fold in folds:
        byq=defaultdict(list)
        for a in actions_by_fold[fold]: byq[a["query_id"]].append(a)
        for q in sorted(byq, key=qkey):
            xs=sorted(byq[q], key=action_key); rows.extend(xs); groups.append(len(xs))
    X=numpy.asarray([[a["features"][c] for c in FEATURES] for a in rows], dtype=numpy.float32)
    y=numpy.asarray([ENC[a["label"]] for a in rows], dtype=numpy.int32)
    return rows, X, y, groups

def fit(actions_by_fold, folds):
    rows, X, y, groups=matrices(actions_by_fold, folds)
    model=LGBMRanker(**HP); model.fit(X, y, group=groups, feature_name=FEATURES)
    return model, {"training_folds":folds,"training_queries":len(groups),"training_action_rows":len(rows),"label_counts":dict(Counter(map(int,y))),"group_size_min":min(groups),"group_size_max":max(groups),"group_size_sum":sum(groups)}

def score(model, actions):
    byq=defaultdict(list)
    for a in actions: byq[a["query_id"]].append(a)
    output={}
    for q in sorted(byq,key=qkey):
        xs=sorted(byq[q],key=action_key)
        X=numpy.asarray([[a["features"][c] for c in FEATURES] for a in xs], dtype=numpy.float32)
        output[q]=list(zip(xs, map(float, model.predict(X))))
    return output

def choose_threshold(best):
    # Truth is joined only after `best` identities/scores were frozen.
    candidates=[float("inf")]+sorted({s for _,s in best.values()})
    rows=[]
    for tau in candidates:
        selected=[a for a,s in best.values() if s >= tau]
        gain=sum(a["truth_delta"] for a in selected)
        precision=sum((1 if a["label"]=="BENEFICIAL" else -1 if a["label"]=="HARMFUL" else 0)/5 for a in selected)
        rows.append({"threshold":"+Infinity" if tau==float("inf") else tau,"gain_sum":gain,"precision_delta":precision,
                     "execute_count":len(selected),"beneficial_count":sum(a["label"]=="BENEFICIAL" for a in selected),
                     "neutral_count":sum(a["label"]=="NEUTRAL" for a in selected),"harmful_count":sum(a["label"]=="HARMFUL" for a in selected),"_tau":tau})
    chosen=max(rows,key=lambda x:(x["gain_sum"],x["precision_delta"],x["_tau"]))
    return chosen, [{k:v for k,v in x.items() if k != "_tau"} for x in rows]

def metric(pred, gold, evaluation):
    recalls=[]; precisions=[]
    for q, docs in pred.items():
        recalls.append(evaluation["recall"](docs,gold[q])); precisions.append(evaluation["precision"](docs,gold[q]))
    return {"recall":sum(recalls)/len(recalls),"precision":sum(precisions)/len(precisions),"queries":len(pred)}

def main():
    print("B1: environment manifest", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    script=Path(__file__); contract_paths=[R/"task1_active_metric_contract.md",R/"step2p1_model_contract.json",R/"step2_k77_model_contract_v2.json",R/"progress_log.md",R/"step4_workflow_a_finalization.md",ROOT/"docs/task1/workflow_A_new.md"]
    manifest={"python":sys.version,"lightgbm":lightgbm.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__,"sklearn":sklearn.__version__,"os":platform.platform(),"cpu_count":os.cpu_count(),"runtime":"Windows Python virtual environment","fixed_lightgbm_version":"4.5.0","script_sha256":h(script),"estimator_parameters":HP}
    dump(R/"workflow_b_b1_environment_manifest.json",manifest)
    preflight={"status":"PASS","contracts_read":{str(p.relative_to(ROOT)).replace('\\\\','/'):h(p) for p in contract_paths},"K":77,"feature_count":len(FEATURES),"feature_schema_exact":len(FEATURES)==36 and len(set(FEATURES))==36,"target_encoding":ENC,"tie_break":"drop_rank5_then_lower_union_rank_then_lexicographic_doc_id","threshold_contract":"candidates=+Infinity plus unique finite inner-OOF top-action scores; max(gain_sum, precision_delta, threshold); score >= threshold","fold0_used":False,"public_labels_used":False}
    if not preflight["feature_schema_exact"]: raise RuntimeError("BLOCKED_B1_CANONICAL_CONTRACT_AMBIGUOUS")
    dump(R/"workflow_b_b1_preflight.json",preflight)
    prereg={"experiment":"B1","status":"FROZEN_PRE_FIT","estimator":"lightgbm.LGBMRanker","surrogate":"LambdaRank/NDCG@1","parameters":HP,"K":77,"feature_columns":FEATURES,"label_encoding":ENC,"outer_folds":{"1":[2,3,4],"2":[1,3,4],"3":[1,2,4],"4":[1,2,3]},"success":{"pooled_recall_delta_gte":0.003,"each_fold_recall_delta_gte":0.0,"precision_delta_gte":-0.001},"scientific_comparator":"canonical Workflow-A P1 F1-F4 OOF policy","public_incumbent_used":False}
    dump(R/"workflow_b_b1_preregistered_contract.json",prereg)

    print("B1: loading canonical action population", flush=True)
    ns=runpy.run_path(str(ROOT/"scripts/analysis/workflow_a/step2_k77_oof_policy_realizability.py"),run_name="b1_parent")
    fold_map=ns["target_folds"](); fold_ids=sorted(set(fold_map.values())); records,gold=ns["load_gold"](set(fold_map)); base,_=ns["load_baseline"](set(fold_map),fold_map); cand,_=ns["load_candidates"](set(fold_map),fold_map)
    actions,incoming=ns["make_actions"](fold_map,records,gold,base,cand,FEATURES)
    if incoming != 403322 or sum(len(actions[f]) for f in fold_ids) != 806644: raise RuntimeError("BLOCKED_B1_CANONICAL_CONTRACT_AMBIGUOUS")
    print("B1: action population loaded", flush=True)
    evaluation=runpy.run_path(str(ROOT/"src/udsc2026/evaluation/legal_ir.py"))
    evalfn={"recall":evaluation["legal_ir_recall"],"precision":evaluation["legal_ir_precision"]}
    thresholds={}; all_scores=[]; b1_pred={}; selection={}; model_records={}; inner_audit=[]
    for held in fold_ids:
        print(f"B1: outer fold {held} nested fits", flush=True)
        train=[f for f in fold_ids if f!=held]; inner_best={}
        for valid in train:
            inner_train=[f for f in train if f!=valid]
            model, fitinfo=fit(actions,inner_train)
            sc=score(model,actions[valid])
            for q,xs in sc.items(): inner_best[q]=top(xs)
            inner_audit.append({"outer_held_fold":held,"inner_valid_fold":valid,"inner_train_folds":inner_train,"fit":fitinfo,"held_out_queries":len(sc)})
        chosen, curve=choose_threshold(inner_best); tau=chosen["_tau"]; thresholds[str(held)]={"outer_held_fold":held,"inner_oof_query_count":len(inner_best),"selected":{k:v for k,v in chosen.items() if k!="_tau"},"candidate_count":len(curve),"tie_break":"max(gain_sum, precision_delta, threshold)","curve":curve}
        model, fitinfo=fit(actions,train); model_path=MODELS/f"outer_f{held}_model.txt"; model.booster_.save_model(str(model_path)); cfg=MODELS/f"outer_f{held}_config.json"; dump(cfg,{"fit":fitinfo,"parameters":HP,"feature_columns":FEATURES,"model_sha256":h(model_path)})
        model_records[str(held)]={"model":str(model_path.relative_to(ROOT)).replace("\\","/"),"config":str(cfg.relative_to(ROOT)).replace("\\","/"),"model_sha256":h(model_path),"config_sha256":h(cfg)}
        sc=score(model,actions[held])
        print(f"B1: outer fold {held} scored", flush=True)
        for q,xs in sc.items():
            a,s=top(xs); take=s>=tau; final=transformed(base[q],a) if take else list(base[q]); b1_pred[q]=final
            selection[q]={"fold":held,"action":a,"score":s,"apply":take,"threshold":tau,"final_top5":final}
            for aa,ss in xs: all_scores.append({"query_id":q,"fold":held,"incoming_doc_id":aa["incoming_doc_id"],"dropped_doc_id":aa["dropped_doc_id"],"drop_rank":aa["drop_rank"],"incoming_union_rank":aa["incoming_union_rank"],"b1_score":ss,"is_query_top_scored":aa is a})
    with (R/"workflow_b_b1_oof_action_scores.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in all_scores: f.write(json.dumps(row,separators=(",",":"))+"\n")
    dump(R/"workflow_b_b1_nested_thresholds.json",{"status":"PASS","thresholds":thresholds,"inner_fits":inner_audit,"outer_truth_used_before_prediction_freeze":False})
    # Reconstruct the persisted P1 scientific comparator only; no P1 refit.
    p1t=json.loads((R/"step2p1t_threshold_calibration_report.json").read_text(encoding="utf-8")); p1tau={int(k):v for k,v in p1t["fixed_thresholds"].items()}
    p1best={}
    with (R/"step2p1f_full_action_scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x=json.loads(line)
            if x["is_query_top_scored"]: p1best[x["query_id"]]=x
    if len(p1best)!=5600: raise RuntimeError("BLOCKED_B1_P1_COMPARATOR_UNRESOLVED")
    p1pred={}
    for q,x in p1best.items():
        take=x["pairwise_policy_score"]>=p1tau[x["fold"]]
        p1pred[q]=transformed(base[q],x) if take else list(base[q])
    if set(p1pred)!=set(b1_pred) or len(b1_pred)!=5600: raise RuntimeError("BLOCKED_B1_P1_COMPARATOR_UNRESOLVED")
    dump(R/"workflow_b_b1_oof_predictions.json",{"status":"PASS","predictions":{q:{"fold":selection[q]["fold"],"apply":selection[q]["apply"],"top_action_score":selection[q]["score"],"threshold":"+Infinity" if selection[q]["threshold"]==float("inf") else selection[q]["threshold"],"answer":selection[q]["final_top5"]} for q in sorted(selection,key=qkey)}})
    per={};
    for fold in fold_ids:
        qs=[q for q in b1_pred if fold_map[q]==fold]; per[str(fold)]={"p1":metric({q:p1pred[q] for q in qs},gold,evalfn),"b1":metric({q:b1_pred[q] for q in qs},gold,evalfn)}; per[str(fold)]["recall_delta"]=per[str(fold)]["b1"]["recall"]-per[str(fold)]["p1"]["recall"]
    p1m=metric(p1pred,gold,evalfn); b1m=metric(b1_pred,gold,evalfn); rd=b1m["recall"]-p1m["recall"]; pd=b1m["precision"]-p1m["precision"]
    success={"pooled_recall_delta_gte_0_003":rd>=.003,"all_fold_recall_delta_nonnegative":all(per[str(f)]["recall_delta"]>=0 for f in fold_ids),"precision_delta_gte_minus_0_001":pd>=-.001}; passed=all(success.values())
    classes=Counter(selection[q]["action"]["label"] for q in selection if selection[q]["apply"]); p1classes=Counter(x["true_action_class"] for x in p1best.values() if x["pairwise_policy_score"]>=p1tau[x["fold"]])
    topchanges=sum((selection[q]["action"]["incoming_doc_id"],selection[q]["action"]["drop_rank"]) != (p1best[q]["incoming_doc_id"],p1best[q]["drop_rank"]) for q in selection)
    membership=sum(set(b1_pred[q])!=set(p1pred[q]) for q in selection); f2s=sum(metric({q:b1pred[q]},gold,evalfn)["recall"]>metric({q:p1pred[q]},gold,evalfn)["recall"] for q in selection); s2f=sum(metric({q:b1pred[q]},gold,evalfn)["recall"]<metric({q:p1pred[q]},gold,evalfn)["recall"] for q in selection)
    rank_improved=classes["BENEFICIAL"]>p1classes["BENEFICIAL"] and classes["HARMFUL"]<=p1classes["HARMFUL"]
    cls="RANKING_AND_RECALL_IMPROVED" if rank_improved and rd>=.003 else "RANKING_IMPROVED_RECALL_NOT_IMPROVED" if rank_improved else "NO_RANKING_OR_RECALL_IMPROVEMENT" if rd<=0 else "MIXED"
    ev={"status":"PASS" if passed else "FAIL","primary_metric":"SET_BASED_MACRO_RECALL","secondary_metric":"SET_BASED_MACRO_PRECISION","p1":p1m,"b1":b1m,"recall_delta":rd,"precision_delta":pd,"per_fold":per,"success":success}
    diag={"apply_count":sum(x["apply"] for x in selection.values()),"no_op_count":sum(not x["apply"] for x in selection.values()),"executed_class_counts":dict(classes),"p1_executed_class_counts":dict(p1classes),"top_action_changes_vs_p1":topchanges,"final_top5_membership_changes_vs_p1":membership,"failure_to_success":f2s,"success_to_failure":s2f,"ranking_diagnostic_improved":rank_improved,"surrogate_result_classification":cls,"candidate_availability":"K77 canonical candidate population available","incoming_candidate_depth_distribution":dict(Counter(a["incoming_union_rank"] for f in fold_ids for a in actions[f])),"models":model_records}
    integrity={"status":"PASS","fold0_used":False,"public_labels_used":False,"public_predictions_used_for_selection":False,"outer_truth_used_before_final_prediction_freeze":False,"p4_feature_used":False,"K":77,"feature_count":36,"feature_schema_exact":True,"hyperparameter_sweep":False,"model_alternative_tried":False,"threshold_tuned_on_outer_labels":False,"query_group_crossing":0,"missing_outer_queries":5600-len(b1_pred),"duplicate_oof_queries":0,"total_pooled_oof_queries":len(b1_pred),"action_rows":len(all_scores),"action_rows_expected":806644,"model_hashes":model_records}
    dump(R/"workflow_b_b1_evaluation.json",ev); dump(R/"workflow_b_b1_diagnostics.json",diag); dump(R/"workflow_b_b1_integrity_audit.json",integrity)
    final={"status":"PASS" if passed else "FAIL","scientific_status":"EXPLORATORY_OOF_SUPPORT" if passed else "B1_CLOSED_AS_SPECIFIED","evaluation":ev,"diagnostics":diag,"integrity":integrity,"environment_manifest_sha256":h(R/"workflow_b_b1_environment_manifest.json"),"experiment_script_sha256":h(script)}
    dump(R/"workflow_b_b1_final_report.json",final)
    (R/"workflow_b_b1_final_report.md").write_text(f"# Workflow B B1 Direct LTR\n\nStatus: **{final['status']}**. LightGBM LGBMRanker LambdaRank/NDCG@1 used K77 and the frozen 36-feature schema. P1 Recall: {p1m['recall']}; B1 Recall: {b1m['recall']}; delta: {rd}. P1 Precision: {p1m['precision']}; B1 Precision: {b1m['precision']}; delta: {pd}.\n\nSuccess criteria: pooled Recall >= +0.003 = {success['pooled_recall_delta_gte_0_003']}; all fold deltas non-negative = {success['all_fold_recall_delta_nonnegative']}; Precision delta >= -0.001 = {success['precision_delta_gte_minus_0_001']}. Surrogate/result classification: `{cls}`.\n",encoding="utf-8")
    print(json.dumps({"status":final["status"],"p1":p1m,"b1":b1m,"recall_delta":rd,"precision_delta":pd,"per_fold":per,"classification":cls},indent=2))
if __name__=="__main__": main()
