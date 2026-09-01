"""STEP 2-P2: read-only localization of frozen pairwise ranking failures."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "task1"
OUT = R / "step2p2_ranking_failure_localization_report.json"
EPS = 1e-12


def read(name): return json.loads((R / name).read_text(encoding="utf-8"))
def dump(data): OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def action_key(x): return (x["query_id"], x["incoming_doc_id"], x["drop_rank"])
def rank_key(x): return (-x["pairwise_policy_score"], -x["drop_rank"], x["incoming_union_rank"], x["incoming_doc_id"])
def dist(v):
    if not v: return {"count": 0, **{k: None for k in ("min","p10","p25","median","mean","p75","p90","max")}}
    a=np.asarray(v,float); return {"count":int(a.size),"min":float(a.min()),"p10":float(np.percentile(a,10)),"p25":float(np.percentile(a,25)),"median":float(np.median(a)),"mean":float(a.mean()),"p75":float(np.percentile(a,75)),"p90":float(np.percentile(a,90)),"max":float(a.max())}
def auc(success, failure):
    if not success or not failure: return {"ROC_AUC_raw":None,"ROC_AUC_directional":None,"AUC_status":"INSUFFICIENT_CLASS_SUPPORT"}
    wins=sum(s>f for s in success for f in failure); ties=sum(s==f for s in success for f in failure); total=len(success)*len(failure); raw=(wins+.5*ties)/total
    return {"ROC_AUC_raw":raw,"ROC_AUC_directional":max(raw,1-raw),"P_success_gt_failure":wins/total,"ties":ties,"AUC_status":"AVAILABLE"}
def group_dist(rows, field): return {"success":dist([x[field] for x in rows if x["outcome"]=="SUCCESS"]),"failure":dist([x[field] for x in rows if x["outcome"]=="FAILURE"])}
def depth_bucket(rank): return "1-20" if rank<=20 else "21-29" if rank<=29 else "30-45" if rank<=45 else "46-77"
def gap_bucket(g):
    if abs(g)<=EPS:return "TIE"
    if g<-.20:return "DECISIVELY_WRONG"
    if g<-.05:return "MODERATELY_WRONG"
    if g<0:return "NEAR_TIE_WRONG"
    if g<=.05:return "NARROW_SUCCESS"
    if g<=.20:return "MODERATE_SUCCESS"
    return "DECISIVE_SUCCESS"
def fractions(x):
    t=x["BN_pairs"]+x["BH_pairs"]+x["NH_pairs"]
    return {k:(x[k+"_pairs"]/t if t else None) for k in ("BN","BH","NH")}
def quartiles(rows, field):
    ordered=sorted(rows,key=lambda x:(x[field],int(x["query_id"]))); n=len(ordered); result={}
    for i,x in enumerate(ordered): x[field+"_quartile"]="Q"+str(min(4,(4*i)//n+1))
    for q in ("Q1","Q2","Q3","Q4"):
        z=[x for x in rows if x[field+"_quartile"]==q]; s=sum(x["outcome"]=="SUCCESS" for x in z); f=len(z)-s
        result[q]={"query_count":len(z),"success_count":s,"failure_count":f,"failure_rate":f/len(z),"median_B_minus_N":dist([x["B_minus_N"] for x in z])["median"],"oracle_headroom_sum":sum(x["oracle"] for x in z)}
    return result


def main():
    attr, gate, contract, p1a, p1af, p1ar = (read("step2p1f_failure_attribution_report.json"), read("step2p1f_r2_reproduction_gate_report.json"), read("step2p1_model_contract.json"), read("step2p1a_realizability_report.json"), read("step2p1af_topaction_change_report.json"), read("step2p1ar_disagreement_robustness_report.json"))
    pre={"K77":contract["fixed_K"]==77,"p1a_pass":p1a["status"]=="PASS","p1af_pass":p1af["status"]=="PASS","p1ar_pass":p1ar["status"]=="PASS","r2_pass":gate["status"]=="PASS"}
    groups=defaultdict(list); count=0
    with (R/"step2p1f_full_action_scores.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            x=json.loads(line); groups[x["query_id"]].append(x); count+=1
    pre.update({"action_rows_806644":count==806644,"queries_5600":len(groups)==5600})
    states=[]
    for q, rows in groups.items():
        oracle=max([x["truth_recall_delta"] for x in rows]+[0.0])
        if oracle<=0: continue
        by={c:[x for x in rows if x["true_action_class"]==c] for c in ("BENEFICIAL","NEUTRAL","HARMFUL")}
        b=max(by["BENEFICIAL"],key=lambda x:(x["pairwise_policy_score"],-x["drop_rank"],-x["incoming_union_rank"],x["incoming_doc_id"]))
        n=max(by["NEUTRAL"],key=lambda x:(x["pairwise_policy_score"],-x["drop_rank"],-x["incoming_union_rank"],x["incoming_doc_id"]))
        h=max(by["HARMFUL"],key=lambda x:(x["pairwise_policy_score"],-x["drop_rank"],-x["incoming_union_rank"],x["incoming_doc_id"])) if by["HARMFUL"] else None
        gap=b["pairwise_policy_score"]-n["pairwise_policy_score"]
        outcome="SUCCESS" if gap>EPS else "FAILURE" if gap<-EPS else "TIE"
        tops=sorted(rows,key=rank_key); c=Counter(x["true_action_class"] for x in rows); x={"query_id":q,"fold":b["fold"],"oracle":oracle,"outcome":outcome,"class_top":tops[0]["true_action_class"],"B_count":c["BENEFICIAL"],"N_count":c["NEUTRAL"],"H_count":c["HARMFUL"],"total_actions":len(rows),"unique_incoming_doc_ids":len({a["incoming_doc_id"] for a in rows}),"drop_rank_alternatives":len({a["drop_rank"] for a in rows}),"best_B_score":b["pairwise_policy_score"],"best_N_score":n["pairwise_policy_score"],"best_H_score":None if h is None else h["pairwise_policy_score"],"top_score":tops[0]["pairwise_policy_score"],"second_best_score":tops[1]["pairwise_policy_score"],"top1_minus_top2_action_margin":tops[0]["pairwise_policy_score"]-tops[1]["pairwise_policy_score"],"B_minus_N":gap,"depth":b["incoming_union_rank"],"depth_bucket":depth_bucket(b["incoming_union_rank"]),"incoming_union_rank_best_B":b["incoming_union_rank"],"incoming_union_rank_best_N":n["incoming_union_rank"],"drop_rank_best_B":b["drop_rank"],"drop_rank_best_N":n["drop_rank"]}
        x.update({"BN_pairs":c["BENEFICIAL"]*c["NEUTRAL"],"BH_pairs":c["BENEFICIAL"]*c["HARMFUL"],"NH_pairs":c["NEUTRAL"]*c["HARMFUL"]}); x.update({k+"_fraction":v for k,v in fractions(x).items()}); states.append(x)
    s=[x for x in states if x["outcome"]=="SUCCESS"]; f=[x for x in states if x["outcome"]=="FAILURE"]; ties=[x for x in states if x["outcome"]=="TIE"]
    pre.update({"oracle_positive_426":len(states)==426,"success_69":len(s)==69,"failure_357":len(f)==357,"ties_0":len(ties)==0,"headroom_329_2":sum(x["oracle"] for x in states)==329.2,"median_gap":abs(dist([x["B_minus_N"] for x in states])["median"]-(-.0944695955058123))<1e-12})
    if not all(pre.values()): dump({"status":"CONTRACT_ERROR","experiment":"STEP 2-P2 — Ranking-Failure Localization Diagnostic","preconditions":pre,"observed":{"states":len(states),"success":len(s),"failure":len(f),"ties":len(ties)}}); print(json.dumps(pre,indent=2)); return
    for x in states: x["gap_bucket"]=gap_bucket(x["B_minus_N"])
    depth={}
    for d in ("1-20","21-29","30-45","46-77"):
        z=[x for x in states if x["depth_bucket"]==d]; zs=[x for x in z if x["outcome"]=="SUCCESS"]; zf=[x for x in z if x["outcome"]=="FAILURE"]
        depth[d]={"success_count":len(zs),"failure_count":len(zf),"total_count":len(z),"failure_rate":len(zf)/len(z),"oracle_headroom_sum_success":sum(x["oracle"] for x in zs),"oracle_headroom_sum_failure":sum(x["oracle"] for x in zf),"median_B_minus_N_success":dist([x["B_minus_N"] for x in zs])["median"],"median_B_minus_N_failure":dist([x["B_minus_N"] for x in zf])["median"],"top_class_counts":{"success":dict(Counter(x["class_top"] for x in zs)),"failure":dict(Counter(x["class_top"] for x in zf))}}
    shallow=[x for x in states if x["depth"]<=20]; deep=[x for x in states if x["depth"]>20]
    comp={k:group_dist(states,k) for k in ("B_count","N_count","H_count")}
    for x in states:
        x["B_fraction"]=x["B_count"]/x["total_actions"]; x["N_fraction"]=x["N_count"]/x["total_actions"]; x["H_fraction"]=x["H_count"]/x["total_actions"]
    comp.update({k:group_dist(states,k) for k in ("B_fraction","N_fraction","H_fraction","total_actions")})
    pf={k:group_dist(states,k) for k in ("BN_pairs","BH_pairs","NH_pairs","BN_fraction","BH_fraction","NH_fraction")}
    patterns={}
    for x in states:
        present=(x["BN_pairs"]>0,x["BH_pairs"]>0,x["NH_pairs"]>0); x["presence_pattern"]={(True,False,False):"BN_ONLY",(True,True,False):"BN_BH",(True,False,True):"BN_NH",(True,True,True):"BN_BH_NH"}.get(present,"OTHER")
    for pat in ("BN_ONLY","BN_BH","BN_NH","BN_BH_NH","OTHER"):
        z=[x for x in states if x["presence_pattern"]==pat]; patterns[pat]={"success_count":sum(x["outcome"]=="SUCCESS" for x in z),"failure_count":sum(x["outcome"]=="FAILURE" for x in z),"failure_rate":None if not z else sum(x["outcome"]=="FAILURE" for x in z)/len(z),"oracle_headroom_sum":sum(x["oracle"] for x in z),"logical_note":"Oracle-positive guarantees B; categories without BN can occur only when N_count=0, which does not occur here."}
    gaps={}
    for g in ("DECISIVELY_WRONG","MODERATELY_WRONG","NEAR_TIE_WRONG","TIE","NARROW_SUCCESS","MODERATE_SUCCESS","DECISIVE_SUCCESS"):
        z=[x for x in states if x["gap_bucket"]==g]; gaps[g]={"query_count":len(z),"share_of_426":len(z)/426,"oracle_headroom_sum":sum(x["oracle"] for x in z),"depth_distribution":dict(Counter(x["depth_bucket"] for x in z)),"fold_distribution":dict(Counter(str(x["fold"]) for x in z)),"median_B_count":dist([x["B_count"] for x in z])["median"],"median_N_count":dist([x["N_count"] for x in z])["median"],"median_H_count":dist([x["H_count"] for x in z])["median"]}
    score={k:group_dist(states,k) for k in ("best_B_score","best_N_score","top_score","second_best_score","top1_minus_top2_action_margin")}; score["best_H_score"]={"success":dist([x["best_H_score"] for x in s if x["best_H_score"] is not None]),"failure":dist([x["best_H_score"] for x in f if x["best_H_score"] is not None])}
    counts={k:group_dist(states,k) for k in ("total_actions","unique_incoming_doc_ids","drop_rank_alternatives")}
    incoming={"incoming_union_rank_best_B":group_dist(states,"incoming_union_rank_best_B"),"incoming_union_rank_best_N":group_dist(states,"incoming_union_rank_best_N"),"drop_rank_best_B":group_dist(states,"drop_rank_best_B"),"drop_rank_best_N":group_dist(states,"drop_rank_best_N"),"incoming_min_source_rank":{"status":"NOT_PERSISTED"},"incoming_source_support":{"status":"NOT_PERSISTED"}}
    qs={k:quartiles(states,k) for k in ("BN_fraction","NH_fraction")}
    folds={}
    for fold in range(1,5):
        z=[x for x in states if x["fold"]==fold]; zs=[x for x in z if x["outcome"]=="SUCCESS"]; zf=[x for x in z if x["outcome"]=="FAILURE"]
        folds[str(fold)]={"oracle_positive_count":len(z),"ranking_success_count":len(zs),"ranking_failure_count":len(zf),"failure_rate":len(zf)/len(z),"median_B_minus_N":dist([x["B_minus_N"] for x in z])["median"],"depth_bucket_failure_counts":dict(Counter(x["depth_bucket"] for x in zf)),"gap_bucket_counts":dict(Counter(x["gap_bucket"] for x in z)),"median_B_count":dist([x["B_count"] for x in z])["median"],"median_N_count":dist([x["N_count"] for x in z])["median"],"median_H_count":dist([x["H_count"] for x in z])["median"],"median_BN_fraction":dist([x["BN_fraction"] for x in z])["median"],"median_NH_fraction":dist([x["NH_fraction"] for x in z])["median"],"top_class_counts":dict(Counter(x["class_top"] for x in z))}
    aucs={k:auc([x[k] for x in s],[x[k] for x in f]) for k in ("depth","B_count","N_count","H_count","BN_fraction","NH_fraction","best_B_score","best_N_score","incoming_union_rank_best_B","incoming_union_rank_best_N")}
    bn_rates=[v["failure_rate"] for v in qs["BN_fraction"].values()]; nh_rates=[v["failure_rate"] for v in qs["NH_fraction"].values()]
    conditions={"concentrated_shallow":len([x for x in f if x["depth"]<=20])/len(f)>=.75 and sum(x["outcome"]=="FAILURE" for x in shallow)/len(shallow)>=sum(x["outcome"]=="FAILURE" for x in deep)/len(deep)+.10,"concentrated_deep":len([x for x in f if x["depth"]>20])/len(f)>=.50 and sum(x["outcome"]=="FAILURE" for x in deep)/len(deep)>=sum(x["outcome"]=="FAILURE" for x in shallow)/len(shallow)+.10,"concentrated_pair_composition":max(bn_rates)-min(bn_rates)>=.20 or max(nh_rates)-min(nh_rates)>=.20,"concentrated_score_margin":max(gaps[g]["query_count"] for g in ("DECISIVELY_WRONG","MODERATELY_WRONG","NEAR_TIE_WRONG"))/len(f)>=.70}
    active=sum(conditions.values()); status="MULTI_FACTOR_CONCENTRATION" if active>=2 else "CONCENTRATED_SHALLOW" if conditions["concentrated_shallow"] else "CONCENTRATED_DEEP" if conditions["concentrated_deep"] else "CONCENTRATED_PAIR_COMPOSITION" if conditions["concentrated_pair_composition"] else "CONCENTRATED_SCORE_MARGIN" if conditions["concentrated_score_margin"] else "DIFFUSE_FAILURE_PATTERN"
    report={"status":"PASS","experiment":"STEP 2-P2 — Ranking-Failure Localization Diagnostic","scientific_status":"DIAGNOSTIC_ONLY_NO_BRANCH_GATE","training_executed":False,"new_scoring_executed":False,"pair_weighting_changed":False,"objective_changed":False,"aggregation_changed":False,"K_changed":False,"policy_oof_strict":True,"end_to_end_selection_oof":False,"k77_selected_exploratorily_on_folds_1_4":True,"interpretation_scope":"PAIRWISE_RANKING_FAILURE_LOCALIZATION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77","K":77,"preconditions":pre,"canonical_reproduction":{"oracle_positive":426,"success":69,"failure":357,"ties":0,"best_B_gt_best_N":69,"best_B_lt_best_N":357,"best_B_minus_best_N_median":dist([x["B_minus_N"] for x in states])["median"],"oracle_headroom_sum":329.2},"depth_analysis":{"buckets":depth,"shallow_vs_deep":{"shallow":{"success":sum(x["outcome"]=="SUCCESS" for x in shallow),"failure":sum(x["outcome"]=="FAILURE" for x in shallow),"failure_rate":sum(x["outcome"]=="FAILURE" for x in shallow)/len(shallow),"oracle_headroom":sum(x["oracle"] for x in shallow)},"deep":{"success":sum(x["outcome"]=="SUCCESS" for x in deep),"failure":sum(x["outcome"]=="FAILURE" for x in deep),"failure_rate":sum(x["outcome"]=="FAILURE" for x in deep)/len(deep),"oracle_headroom":sum(x["oracle"] for x in deep)}}},"action_class_composition":comp,"pair_family_composition":pf,"pair_family_presence_patterns":patterns,"gap_analysis":{"distributions":{"success":dist([x["B_minus_N"] for x in s]),"failure":dist([x["B_minus_N"] for x in f])},"buckets":gaps,"tautology_warning":"B_minus_N sign defines success/failure; buckets characterize severity, not mechanism."},"score_analysis":score,"candidate_action_count_analysis":counts,"incoming_rank_profile":incoming,"pair_composition_quartiles":qs,"per_fold":folds,"oracle_headroom":{"success":{"query_count":69,"oracle_headroom_sum":sum(x["oracle"] for x in s),"share_of_329_2":sum(x["oracle"] for x in s)/329.2},"failure":{"query_count":357,"oracle_headroom_sum":sum(x["oracle"] for x in f),"share_of_329_2":sum(x["oracle"] for x in f)/329.2}},"success_failure_descriptive_auc":aucs,"localization_summary":{"status":status,"conditions":conditions,"pair_quartile_failure_rate_range":{"BN_fraction":max(bn_rates)-min(bn_rates),"NH_fraction":max(nh_rates)-min(nh_rates)}},"preserved_status":{"ranking_bottleneck":"PRIMARY","pair_family":"UNRESOLVED_NOT_ESTABLISHED","pair_reweighting":"NOT_AUTHORIZED","B_vs_N_only":"NOT_AUTHORIZED","raw_pairwise_dispersion":"DEFERRED","HGB":"UNRESOLVED","feature_expansion":"NOT_JUSTIFIED","K77":"NOT_REJECTED","smaller_K":"DEFERRED","workflow_B":"NOT_ACTIVE"},"notes":["Read-only descriptive localization using persisted frozen MEAN action scores.","Local BN/BH/NH counts are action-space composition, not literal training exposure.","No causal pair-family claim, intervention recommendation, or branch authorization is made."]}
    dump(report); print(json.dumps({"status":"PASS","summary":status,"headroom":report["oracle_headroom"],"depth":report["depth_analysis"]["shallow_vs_deep"],"quartile_ranges":report["localization_summary"]["pair_quartile_failure_rate_range"]},indent=2))

if __name__=="__main__": main()
