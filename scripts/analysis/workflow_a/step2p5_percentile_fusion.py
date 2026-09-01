import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/task1"
OUTER_P1 = R / "step2p1f_full_action_scores.jsonl"
OUTER_P4 = R / "step2p4_proxy_predictions.jsonl"
INNER_P1 = R / "step2p5_p1_inner_full_action_scores.jsonl"
INNER_P4 = R / "step2p4_proxy_inner_oof_predictions.jsonl"
P5A = R / "step2p5a_reproducibility_report.json"
INNER_PREFLIGHT = R / "step2p5_inner_join_preflight.json"
FUSED = R / "step2p5_fused_scores.jsonl"
THRESH = R / "step2p5_threshold_report.json"
REPORT = R / "step2p5_realizability_report.json"
BETA = 0.05
BASE_R = 357
BASE_D = -0.0007142857142857143
BASE_GAIN = -4.0
BASE_FOLD_R = {1: 81, 2: 79, 3: 96, 4: 101}


def read_jsonl(path):
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def key_outer(x):
    a = x.get("action_identity", x)
    return (int(x["fold"]), str(x["query_id"]), str(a["incoming_doc_id"]), int(a["incoming_union_rank"]), int(a["drop_rank"]), str(a["dropped_doc_id"]))


def key_inner(x):
    a = x.get("action_id", x.get("action_identity", x))
    outer = x.get("outer_fold", x.get("outer_held_fold"))
    inner = x.get("inner_fold", x.get("inner_valid_fold"))
    return (int(outer), int(inner), str(x["query_id"]), str(a["incoming_doc_id"]), int(a["incoming_union_rank"]), int(a["drop_rank"]), str(a["dropped_doc_id"]))


def tie_key(row, score):
    return (-score, -row["drop_rank"], row["incoming_union_rank"], row["incoming_doc_id"])


def percentile(rows):
    n = len(rows)
    if n < 2:
        raise RuntimeError("CONTRACT_ERROR single-action query")
    values = sorted(row["proxy"] for row in rows)
    positions = {}
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and values[end] == values[pos]:
            end += 1
        positions[values[pos]] = (pos + 0.5 * (end - pos - 1), end - pos)
        pos = end
    for row in rows:
        mid, _ = positions[row["proxy"]]
        row["midrank0"] = mid
        row["percentile"] = mid / (n - 1)
        row["fused"] = row["p1"] + BETA * row["percentile"]
        if not 0.0 <= row["percentile"] <= 1.0:
            raise RuntimeError("CONTRACT_ERROR percentile range")
    ordered_p1 = sorted(rows, key=lambda x: tie_key(x, x["p1"]))
    ordered_fused = sorted(rows, key=lambda x: tie_key(x, x["fused"]))
    for i, row in enumerate(ordered_p1, 1): row["p1_rank"] = i
    for i, row in enumerate(ordered_fused, 1): row["p5_rank"] = i
    return ordered_p1[0], ordered_fused[0]


def join_preflight(p1rows, p4rows, keyfn, scope, expected):
    c1, c4 = Counter(keyfn(x) for x in p1rows), Counter(keyfn(x) for x in p4rows)
    s1, s4 = set(c1), set(c4)
    examples = []
    for key in list(s1 - s4)[:10]: examples.append({"type": "P1_UNMATCHED", "key": key})
    for key in list(s4 - s1)[:10]: examples.append({"type": "P4_UNMATCHED", "key": key})
    for key, count in list((c1 - Counter({k: 1 for k in c1})).items())[:10]: examples.append({"type": "P1_DUPLICATE", "key": key, "count": count + 1})
    for key, count in list((c4 - Counter({k: 1 for k in c4})).items())[:10]: examples.append({"type": "P4_DUPLICATE", "key": key, "count": count + 1})
    p1dup, p4dup = sum(v - 1 for v in c1.values() if v > 1), sum(v - 1 for v in c4.values() if v > 1)
    p1un, p4un = len(s1 - s4), len(s4 - s1)
    folds_ok = all(x.get("prediction_scope") == scope for x in p4rows) and all(all(v in (1,2,3,4) for v in keyfn(x)[:2] if isinstance(v, int)) for x in p1rows + p4rows)
    passed = len(p1rows) == len(p4rows) == expected and not p1dup and not p4dup and not p1un and not p4un and folds_ok
    return {"status": "PASS" if passed else "BLOCKED_INNER_JOIN_CONFLICT", "P1_rows": len(p1rows), "P4_rows": len(p4rows), "joined_rows": len(s1 & s4), "P1_duplicate_keys": p1dup, "P4_duplicate_keys": p4dup, "P1_unmatched_rows": p1un, "P4_unmatched_rows": p4un, "fold_mismatches": 0 if folds_ok else 1, "one_to_one": passed, "fold0_or_public_used": False, "pass": passed, "mismatch_examples": examples}


def choose_threshold(tops):
    candidates = [float("inf")] + sorted({row["fused"] for row in tops})
    options = []
    for tau in candidates:
        taken = [row for row in tops if row["fused"] >= tau]
        gain = sum(row["truth_delta"] for row in taken)
        precision_delta = sum((1 if row["label"] == "BENEFICIAL" else -1 if row["label"] == "HARMFUL" else 0) / 5 for row in taken)
        options.append((gain, precision_delta, tau, taken))
    return max(options, key=lambda x: (x[0], x[1], x[2])), len(candidates)


def stats(values):
    values = sorted(values)
    if not values: return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    def q(p):
        i = (len(values) - 1) * p; lo, hi = int(i), math.ceil(i)
        return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (i - lo)
    return {"count": len(values), "min": values[0], "p25": q(.25), "median": q(.5), "p75": q(.75), "max": values[-1]}


def main():
    p5a = json.loads(P5A.read_text(encoding="utf-8"))
    valid_p5a = p5a.get("status") == "PASS" and p5a["full_action_artifact"].get("artifact_trusted") is True and p5a["full_action_artifact"].get("rows") == 2419932 and p5a["full_action_artifact"].get("queries") == 16800 and p5a["reproduction_gate"].get("identity_matches") == 16800 and p5a["reproduction_gate"].get("score_matches") == 16800 and p5a["reproduction_gate"].get("max_abs_score_error") == 0.0
    if not valid_p5a: raise RuntimeError("BLOCKED_ARTIFACT_PROVENANCE_CONFLICT P5-A gate")
    inner_p1, inner_p4 = list(read_jsonl(INNER_P1)), list(read_jsonl(INNER_P4))
    inner = join_preflight(inner_p1, inner_p4, key_inner, "INNER_HELD_OUT", 2419932)
    inner.update({"P1_inner_source": str(INNER_P1.relative_to(ROOT)).replace('\\','/'), "P4_inner_source": str(INNER_P4.relative_to(ROOT)).replace('\\','/'), "join_keys": ["outer_fold", "inner_fold", "query_id", "incoming_doc_id", "incoming_union_rank", "drop_rank", "dropped_doc_id"]})
    INNER_PREFLIGHT.write_text(json.dumps(inner, indent=2) + "\n", encoding="utf-8")
    if not inner["pass"]: raise RuntimeError("BLOCKED_INNER_JOIN_CONFLICT")
    outer_p1, outer_p4 = list(read_jsonl(OUTER_P1)), list(read_jsonl(OUTER_P4))
    outer = join_preflight(outer_p1, outer_p4, key_outer, "OUTER_HELD_OUT", 806644)
    outer["status"] = "PASS" if outer["pass"] else "BLOCKED_OUTER_JOIN_CONFLICT"
    if not outer["pass"]: raise RuntimeError("BLOCKED_OUTER_JOIN_CONFLICT")
    # Inner truth joins only after scores/ranks are frozen; canonical outer full-score artifact supplies immutable action truth by identity.
    truth = {key_outer(x): {"label": x["true_action_class"], "truth_delta": x["truth_recall_delta"]} for x in outer_p1}
    p4i = {key_inner(x): x for x in inner_p4}; by_inner = defaultdict(list)
    for x in inner_p1:
        key = key_inner(x); y = p4i[key]; outer_fold, inner_fold, q, doc, rank, drop, dropped = key
        by_inner[(outer_fold, inner_fold, q)].append({"fold": inner_fold, "query_id": q, "incoming_doc_id": doc, "incoming_union_rank": rank, "drop_rank": drop, "dropped_doc_id": dropped, "p1": x["P1_score"], "proxy": y["p_hat_beneficial"]})
    inner_tops = defaultdict(list); inner_action_count = defaultdict(int)
    for (outer_fold, inner_fold, q), rows in by_inner.items():
        _, p5top = percentile(rows)
        t = truth[(inner_fold, q, p5top["incoming_doc_id"], p5top["incoming_union_rank"], p5top["drop_rank"], p5top["dropped_doc_id"])]
        inner_tops[outer_fold].append({**p5top, **t, "outer_fold": outer_fold, "inner_fold": inner_fold})
        inner_action_count[outer_fold] += len(rows)
    thresholds = {}; threshold_rows = {}
    for f in range(1,5):
        chosen, n = choose_threshold(inner_tops[f]); thresholds[f] = chosen[2]
        threshold_rows[str(f)] = {"outer_fold": f, "inner_folds_used": [x for x in range(1,5) if x != f], "number_inner_queries": len(inner_tops[f]), "number_inner_actions": inner_action_count[f], "P1_inner_source": str(INNER_P1.relative_to(ROOT)).replace('\\','/'), "P4_inner_source": str(INNER_P4.relative_to(ROOT)).replace('\\','/'), "beta": BETA, "normalization": "WITHIN_QUERY_ZERO_BASED_AVERAGE_PERCENTILE", "selected_threshold": "+Infinity" if math.isinf(chosen[2]) else chosen[2], "gain_sum_at_selected_threshold": chosen[0], "threshold_candidate_count": n, "threshold_tie_rule": "max(gain_sum, precision_delta, threshold)", "outer_truth_used": False}
    THRESH.write_text(json.dumps({"status": "PASS", "experiment": "STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention", "all_thresholds_selected_from_inner_oof_only": True, "folds": threshold_rows}, indent=2) + "\n", encoding="utf-8")
    p4o = {key_outer(x): x for x in outer_p4}; by_outer = defaultdict(list)
    for x in outer_p1:
        key = key_outer(x); y = p4o[key]; fold, q, doc, rank, drop, dropped = key
        by_outer[(fold,q)].append({"fold": fold, "query_id": q, "incoming_doc_id": doc, "incoming_union_rank": rank, "drop_rank": drop, "dropped_doc_id": dropped, "p1": x["pairwise_policy_score"], "proxy": y["p_hat_beneficial"], "label": x["true_action_class"], "truth_delta": x["truth_recall_delta"]})
    alltops=[]; score_rows=[]; percentile_flipped=[]; percentile_unflipped=[]; flips=Counter(); flipfold={f:Counter() for f in range(1,5)}
    for (fold,q), rows in by_outer.items():
        p1top, p5top = percentile(rows); changed = action_key(p1top) != action_key(p5top)
        for row in rows:
            score_rows.append({"prediction_scope":"OUTER_HELD_OUT","query_id":q,"fold":fold,"action_identity":action_id(row),"P1_score":row["p1"],"p_hat_beneficial":row["proxy"],"query_action_count":len(rows),"proxy_midrank0":row["midrank0"],"proxy_percentile":row["percentile"],"beta":BETA,"fused_score":row["fused"],"P1_rank":row["p1_rank"],"P5_rank":row["p5_rank"],"is_P1_top":row is p1top,"is_P5_top":row is p5top})
        top = {**p5top, "p1top":p1top, "changed":changed}; alltops.append(top)
        (percentile_flipped if changed else percentile_unflipped).append(p5top["percentile"])
    with FUSED.open("w", encoding="utf-8", newline="\n") as h:
        for row in score_rows: h.write(json.dumps(row)+"\n")
    oracle = {}
    for (_, q), rows in by_outer.items():
        oracle[q] = max(row["truth_delta"] for row in rows)
    per={}; total_gain=0; executed=[]
    for f in range(1,5):
        tops=[x for x in alltops if x["fold"]==f]; taken=[x for x in tops if x["fused"]>=thresholds[f]]; executed += taken
        op=[x for x in tops if oracle[x["query_id"]]>0]; r=sum(x["label"]!="BENEFICIAL" for x in op); gain=sum(x["truth_delta"] for x in taken); total_gain+=gain
        per[str(f)]={"oracle_positive_query_count":len(op),"baseline_R":BASE_FOLD_R[f],"P5_R":r,"delta_R":r-BASE_FOLD_R[f],"D77_exact":gain/1400,"delta_D77_exact":gain/1400-[-0.0,0.0009523809523809524,0.0,-0.002380952380952381,-0.0014285714285714286][f],"gain_sum":gain}
    op=[x for x in alltops if oracle[x["query_id"]]>0]; R=sum(x["label"]!="BENEFICIAL" for x in op); S=sum(x["label"]=="BENEFICIAL" and x["fused"]>=thresholds[x["fold"]] for x in op); T=sum(x["label"]=="BENEFICIAL" and x["fused"]<thresholds[x["fold"]] for x in op)
    for x in alltops:
        flips["all_queries_total"]+=1; flips["top_action_changed" if x["changed"] else "top_action_unchanged"]+=1
        if oracle[x["query_id"]]>0:
            a=x["p1top"]["label"]=="BENEFICIAL"; b=x["label"]=="BENEFICIAL"; name=("success_to_success" if a and b else "success_to_failure" if a else "failure_to_success" if b else "failure_to_failure")
            flips[name]+=1; flipfold[x["fold"]][name]+=1
    nonw=sum(per[str(f)]["P5_R"]<=BASE_FOLD_R[f] for f in range(1,5)); d=total_gain/5600
    criteria={"pooled_R_improves":R<BASE_R,"D77_non_worsens":d>=BASE_D,"folds_non_worsened":nonw,"fold_robustness_met":nonw>=3,"overall_preregistered_validation":R<BASE_R and d>=BASE_D and nonw>=3}
    report={"status":"PASS","experiment":"STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention","experiment_type":"CONTROLLED_SINGLE_VARIABLE_INTERVENTION","training_executed":False,"new_model_inference_executed":False,"policy_modified":True,"fold0_or_public_used":False,"policy_oof_strict":True,"end_to_end_selection_oof":False,"k77_selected_exploratorily_on_folds_1_4":True,"K":77,"beta":BETA,"artifacts":{"outer_P1":str(OUTER_P1.relative_to(ROOT)),"outer_P4":str(OUTER_P4.relative_to(ROOT)),"inner_P1":str(INNER_P1.relative_to(ROOT)),"inner_P4":str(INNER_P4.relative_to(ROOT)),"P5A_reproduction":str(P5A.relative_to(ROOT))},"join_preflight":{"inner":"PASS","outer":"PASS"},"fusion_contract":{"P1_transform":"IDENTITY","proxy_transform":"WITHIN_QUERY_ZERO_BASED_AVERAGE_PERCENTILE","percentile_formula":"midrank0/(n-1)","beta":BETA,"formula":"P1_score + 0.05*proxy_percentile","proxy_ties":"AVERAGED","fused_score_ties":"CANONICAL_P1_TIE_BREAK"},"baseline":{"R":357,"D77_exact":BASE_D,"gain_sum":-4.0,"S":6,"T":63},"P5":{"R":R,"D77_exact":d,"gain_sum":total_gain,"S":S,"T":T},"delta":{"R":R-357,"D77_exact":d-BASE_D,"gain_sum":total_gain+4.0},"per_fold":per,"ranking_flips":{"all_queries":dict(flips),"oracle_positive":{k:flips[k] for k in ("failure_to_success","success_to_failure","failure_to_failure","success_to_success")},"per_fold":{str(f):dict(flipfold[f]) for f in range(1,5)}},"proxy_percentile_distribution":{"FLIPPED":stats(percentile_flipped),"UNFLIPPED":stats(percentile_unflipped),"population_definition":"P5-selected top percentile for flipped queries; shared selected top percentile for unflipped queries"},"preregistered_criteria":criteria,"branch_gate":"NONE","preserved_status":{"P4_global_signal":"STRONG_DESCRIPTIVE_OOF_SIGNAL","P4Q_within_query_signal":"SUPPORTED_DESCRIPTIVELY","absolute_proxy_gate":"NOT_AUTHORIZED","proxy_only_ranking":"NOT_AUTHORIZED","P3":"CLOSED","pair_reweighting":"NOT_AUTHORIZED","B_vs_N_only":"NOT_AUTHORIZED","rank_feature_intervention":"NOT_JUSTIFIED","raw_pairwise_dispersion":"DEFERRED","feature_expansion":"NOT_JUSTIFIED","K77":"NOT_REJECTED","smaller_K":"DEFERRED","workflow_B":"NOT_ACTIVE"},"notes":["All fusion inputs were frozen artifacts; no model was loaded for prediction.","Inner truth was joined only after inner fused scores/ranks were frozen for canonical gain-sum threshold selection.","Outer truth was joined only after outer fusion/ranking were frozen."],"next_state":"STEP2_P5_COMPLETE_PENDING_PROFESSOR_REVIEW"}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","P5":report["P5"],"criteria":criteria,"thresholds":{f:threshold_rows[str(f)]["selected_threshold"] for f in range(1,5)}},indent=2))


def action_id(row):
    return {"incoming_doc_id":row["incoming_doc_id"],"incoming_union_rank":row["incoming_union_rank"],"drop_rank":row["drop_rank"],"dropped_doc_id":row["dropped_doc_id"]}


def action_key(row):
    return (row["incoming_doc_id"],row["incoming_union_rank"],row["drop_rank"],row["dropped_doc_id"])


if __name__ == "__main__": main()
