"""CPU-only final local-regime selector over frozen Task1 expert artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.phase1_step4_recovery import load_fold_map, load_gold, metric, qkey
from private_task1.scripts.analysis.guarded_vs_direct_top5_meta_selector import (
    load_guarded, load_zip_submission, rows, sha256, summary_vs_a,
    validate_submission, write_json, write_jsonl,
)
from private_task1.scripts.analysis.final_multi_expert_stack import (
    bge_order, build_private_c, load_candidates, load_flat_worklist, load_scores,
    rrf_ranking,
)

OUT = ROOT / "private_task1/experiments/09205_final_last_slot_local_regime_selector"
TEAM = ROOT / "private_task1/submissions/team_09205"
MULTI_OUT = ROOT / "private_task1/experiments/09205_final_multi_expert_stack"
INC_ZIP = TEAM / "submission_private_constrained_dual_anchor_rrf_09205.zip"
SAFE_ZIP = TEAM / "submission_private_dual_anchor_guarded_09198.zip"
G_ZIP = ROOT / "private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip"
D_ZIP = TEAM / "submission_private_09205_final_multi_expert_stack.zip"
FINAL_ZIP = TEAM / "submission_private_09205_FINAL_LAST_SLOT.zip"
PRIVATE_OFFICIAL = ROOT / "private_task1/input/private-official.json"
DIRECT_OOF = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl"
C_OOF = ROOT / "private_task1/experiments/direct_k20_listwise_top5/oof_predictions.jsonl"
PRIVATE_B = ROOT / "private_task1/experiments/09205_last_chance/private_expert_b_top5.json"
VAL_K20 = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
VAL_BGE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
PRIV_K20 = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIV_BGE = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
R_VAL_WORK = ROOT / "private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl"
R_VAL_SCORE = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl"
EXPECTED_INC = "aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae"
KS = (15, 30, 60, 120)
EXPERTS = ("A", "B", "C", "R", "D")
SEED = 20260923


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def question(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or row.get("text") or "")


def per_query_recall(pred: list[str], row: dict[str, Any]) -> float:
    gold = set(map(str, row.get("answer", [])))
    return len(gold & set(pred)) / max(1, len(gold))


def rrf_full(q, docs, scores):
    return rrf_ranking(q, docs, scores)


def expert_artifacts(folds, gold, vc, vs, rvc, rvs):
    direct = {str(x["query_id"]): x for x in rows(DIRECT_OOF)}
    A = load_guarded(direct)
    B = {q: list(map(str, direct[q]["direct_top5"])) for q in folds}
    C_rows = {str(x["query_id"]): x for x in rows(C_OOF)}
    C = {q: list(map(str, C_rows[q]["listwise_top5"])) for q in folds}
    D_rows = {str(x["query_id"]): x for x in rows(MULTI_OUT / "stack_oof_predictions.jsonl")}
    D = {q: list(map(str, D_rows[q]["overlay_top5"])) for q in folds}
    R, R10, rval = {}, {}, {}
    for q in folds:
        full, score = rrf_full(q, rvc[q], rvs)
        R[q], R10[q], rval[q] = full[:5], full[:10], score
    out = {"A": A, "B": B, "C": C, "R": R, "D": D}
    if any(set(v) != set(folds) for v in out.values()):
        raise RuntimeError("validation expert population mismatch")
    return out, R10, rval, direct, D_rows


def overlap(a, b):
    return len(set(a) & set(b)) / 5.0


def cue_features(text: str):
    low = text.lower()
    citation = int(bool(re.search(r"\b(điều|khoản|điểm|nghị định|thông tư|luật|quyết định)\b", low)))
    number = int(bool(re.search(r"\b\d+(?:/\d+)?(?:/[a-zđ-]+)?\b|\b(thứ nhất|thứ hai|bao nhiêu)\b", low)))
    return len(text.split()), citation, number


def source_overlap_stats(q, docs, scores):
    source_top = {}
    for s in ("dense", "bm25", "knn_word"):
        source_top[s] = [d for d, _ in sorted(((d, m["source_ranks"].get(s, 10**9)) for d, m in docs.items() if s in m["source_ranks"]), key=lambda x: (x[1], x[0]))][:5]
    source_top["bge"] = bge_order(q, docs, scores)[:5]
    pairs = (("dense", "bm25"), ("dense", "bge"), ("dense", "knn_word"), ("bm25", "bge"), ("bm25", "knn_word"), ("bge", "knn_word"))
    return [overlap(source_top[a], source_top[b]) for a, b in pairs]


def base_query_features(q, ex, r10, docs, scores, rscore, text, dconf):
    pairs = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"),
             ("A", "R"), ("B", "R"), ("C", "R"), ("D", "R"))
    vals = [overlap(ex[a], ex[b]) for a, b in pairs]
    union_size = len(set().union(*(set(ex[e]) for e in EXPERTS)))
    rank1 = Counter(ex[e][0] for e in EXPERTS)
    a45_votes = [sum(d in ex[e] for e in EXPERTS) for d in ex["A"][3:5]]
    vals += [float(union_size), float(max(rank1.values())), *map(float, a45_votes), float(len(rank1))]
    vals += [float(dconf.get("stack_boundary_margin", 0.0)), float(dconf.get("net_vote_gain", 0.0)),
             float(dconf.get("number_of_membership_changes", len(set(ex["D"])-set(ex["A"]))))]
    full = sorted(rscore, key=lambda d: (-rscore[d], d))
    margins = [rscore[full[0]]-rrscore for rrscore in [rscore[full[1]]]]
    margins += [rscore[full[3]]-rscore[full[4]], rscore[full[4]]-rscore[full[5]]]
    supports = [docs[d]["source_support"] for d in r10[:5]]
    vals += list(map(float, margins)) + [statistics.mean(supports), float(min(supports))]
    vals += source_overlap_stats(q, docs, scores)
    br = {d: i for i, d in enumerate(bge_order(q, docs, scores), 1)}
    low = int(margins[2] <= .00119048)
    high = int(union_size >= 9 or statistics.mean(supports) <= 1.6)
    strong = int(max(abs(br[d]-docs[d]["candidate_rank"]) for d in r10[:5]) >= 17)
    single = int(sum(s <= 1 for s in supports) >= 2)
    vals += [float(low), float(high), float(strong), float(single), *map(float, cue_features(text))]
    return vals


def domain_prob(vbase, pbase):
    Xv, Xp = np.asarray(vbase, float), np.asarray(pbase, float)
    X = np.vstack([Xv, Xp]); y = np.r_[np.zeros(len(Xv)), np.ones(len(Xp))]
    pred = np.zeros(len(y)); idx = np.arange(len(y))
    for k in range(5):
        test, train = idx[idx % 5 == k], idx[idx % 5 != k]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=500, random_state=SEED))
        m.fit(X[train], y[train]); pred[test] = m.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, pred))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=500, random_state=SEED))
    model.fit(X, y)
    return auc, model.predict_proba(Xv)[:, 1], model.predict_proba(Xp)[:, 1]


def utilities(experts, gold):
    out = {}
    for q in gold:
        a = per_query_recall(experts["A"][q], gold[q])
        out[q] = {e: per_query_recall(experts[e][q], gold[q])-a for e in EXPERTS}
    return out


def local_stats(neighbors, distances, utility):
    weights = 1/(distances + .05); result = {}
    for e in EXPERTS:
        u = np.asarray([utility[q][e] for q in neighbors], float)
        mean = float(np.average(u, weights=weights)); harm = float(np.average(u < 0, weights=weights)); win = float(np.average(u > 0, weights=weights))
        negatives = np.abs(u[u < 0]); mean_neg = float(np.average(negatives, weights=weights[u < 0])) if len(negatives) else 0.
        shrunk = mean*len(neighbors)/(len(neighbors)+30)
        result[e] = {"local_mean": mean, "local_harm": harm, "local_win": win, "shrunk": shrunk,
                     "local_score": shrunk-.75*harm*mean_neg}
    result["A"] = {**result["A"], "local_score": 0.0}
    return result


def selector_prediction(ex, stats):
    selected = max(EXPERTS, key=lambda e: (stats[e]["local_score"], -EXPERTS.index(e)))
    if selected != "A" and stats[selected]["local_score"] > 0 and stats[selected]["local_win"] > stats[selected]["local_harm"]:
        return list(ex[selected]), selected
    return list(ex["A"]), "A"


def fusion_prediction(ex, r10, stats):
    rel = {e: max(0., stats[e]["local_score"]) for e in EXPERTS}
    rel["A"] = .25 + max(0., stats["A"]["local_score"])
    total = sum(rel.values()); rel = {e: v/total for e, v in rel.items()}
    ranks = {e: {d: i for i, d in enumerate(ex[e], 1)} for e in EXPERTS}
    ranks["R"] = {d: i for i, d in enumerate(r10, 1)}
    docs = set().union(*(set(v) for v in ex.values()), set(r10))
    score = {d: sum(rel[e]/(30+ranks[e][d]) for e in EXPERTS if d in ranks[e]) for d in docs}
    order = sorted(docs, key=lambda d: (-score[d], d)); chosen = order[:5]
    for anchor in reversed(ex["A"][:3]):
        if anchor not in chosen:
            drop = min((d for d in chosen if d not in ex["A"][:3]), key=lambda d: (score[d], d))
            chosen[chosen.index(drop)] = anchor
    chosen = sorted(set(chosen), key=lambda d: (-score[d], d))
    # Exact A Top1-3 lock; remaining documents retain local-RRF order.
    tail = [d for d in chosen if d not in ex["A"][:3]]
    return list(ex["A"][:3]) + tail[:2], score, rel


def neighbor_map(train, test, matrix, qindex, kmax=120):
    train_sorted = sorted(train, key=qkey); test_sorted = sorted(test, key=qkey)
    model = NearestNeighbors(n_neighbors=min(kmax, len(train_sorted)), metric="euclidean", algorithm="brute", n_jobs=-1)
    model.fit(np.asarray([matrix[qindex[q]] for q in train_sorted]))
    dist, idx = model.kneighbors(np.asarray([matrix[qindex[q]] for q in test_sorted]))
    return {q: ([train_sorted[i] for i in idx[j]], dist[j]) for j, q in enumerate(test_sorted)}


def predict_local(test, neighbor_data, k, experts, r10, utility, policy):
    pred, audit = {}, {}
    for q in test:
        ns, ds = neighbor_data[q]; ns, ds = ns[:k], ds[:k]
        stats = local_stats(ns, ds, utility); ex = {e: experts[e][q] for e in EXPERTS}
        if policy == "selector":
            answer, selected = selector_prediction(ex, stats); score = stats[selected]["local_score"]
            audit[q] = {"selected_expert": selected, "confidence": score, "stats": stats}
        else:
            answer, scores, rel = fusion_prediction(ex, r10[q], stats)
            rejected = [d for d in scores if d not in answer]
            confidence = min(scores[d] for d in answer)-max(scores[d] for d in rejected)
            audit[q] = {"selected_expert": "LOCAL_FUSION", "confidence": float(confidence), "stats": stats, "reliability": rel, "document_scores": scores}
        pred[q] = answer
    return pred, audit


def choose_k(train, matrix, qindex, experts, r10, utility, gold, folds, policy):
    by_k = {k: {} for k in KS}; audits = {k: {} for k in KS}
    for inner in sorted(set(folds[q] for q in train)):
        pool = [q for q in train if folds[q] != inner]; test = [q for q in train if folds[q] == inner]
        nm = neighbor_map(pool, test, matrix, qindex)
        for k in KS:
            pred, audit = predict_local(test, nm, k, experts, r10, utility, policy); by_k[k].update(pred); audits[k].update(audit)
    candidates = []
    for k in KS:
        base = {q: experts["A"][q] for q in train}
        report = summary_vs_a(by_k[k], base, {q: gold[q] for q in train}, {q: folds[q] for q in train})
        report["net_gain"] = report["improved"]-report["harmed"]
        eligible = report["improved"] > report["harmed"] and report["precision_delta"] >= -.0015
        candidates.append({"k": k, "eligible": eligible, **report})
    valid = [x for x in candidates if x["eligible"]]
    if not valid:
        return None, candidates
    selected = sorted(valid, key=lambda x: (-x["recall"], -x["net_gain"], x["k"]))[0]
    return int(selected["k"]), candidates


def outer_oof(matrix, qindex, experts, r10, utility, gold, folds, policy):
    pred, audit, choices = {}, {}, {}
    for f in sorted(set(folds.values())):
        train = [q for q in folds if folds[q] != f]; test = [q for q in folds if folds[q] == f]
        k, candidates = choose_k(train, matrix, qindex, experts, r10, utility, gold, folds, policy)
        choices[f"F{f}"] = {"selected_k": k, "candidates": candidates}
        if k is None:
            pred.update({q: list(experts["A"][q]) for q in test}); audit.update({q: {"selected_expert": "A", "confidence": 0., "stats": {}} for q in test})
        else:
            nm = neighbor_map(train, test, matrix, qindex); p, a = predict_local(test, nm, k, experts, r10, utility, policy); pred.update(p); audit.update(a)
    report = summary_vs_a(pred, experts["A"], gold, folds); report["net_gain"] = report["improved"]-report["harmed"]
    report["improved_harmed_ratio"] = None if report["harmed"] == 0 else report["improved"]/report["harmed"]
    report["selected_counts"] = dict(Counter(audit[q]["selected_expert"] for q in audit))
    return pred, audit, choices, report


def final_k(matrix, qindex, experts, r10, utility, gold, folds, policy):
    return choose_k(list(folds), matrix, qindex, experts, r10, utility, gold, folds, policy)


def accepted(report):
    return report["recall_delta"] > 0 and report["net_gain"] > 0 and report["improved"] > report["harmed"] and report["precision_delta"] >= -.002


def write_zip(payload, expected):
    jpath = FINAL_ZIP.with_suffix(".json")
    jpath.write_text(json.dumps({q: {"answer": payload[q]} for q in sorted(payload, key=qkey)}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jpath, arcname="submission.json")
    return {"path": str(FINAL_ZIP.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(FINAL_ZIP), "validator": validate_submission(payload, expected)}


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    required = ["final_report.json", "final_report.md", "stack_oof_report.json", "multi_expert_oracle.json", "domain_adaptation.json", "private_stack_actions.jsonl", "stack_oof_predictions.jsonl"]
    if any(not (MULTI_OUT/x).is_file() for x in required):
        raise RuntimeError("multi-expert audit artifact missing")
    mf, mo, md = read_json(MULTI_OUT/"final_report.json"), read_json(MULTI_OUT/"multi_expert_oracle.json"), read_json(MULTI_OUT/"domain_adaptation.json")
    dsha = sha256(D_ZIP); dval = validate_submission(load_zip_submission(D_ZIP), set(map(str, read_json(PRIVATE_OFFICIAL))))
    gate = str(mf["oof_gate"]); violation = gate == "FAIL" and D_ZIP.is_file()
    audit0 = {"multi_expert_oracle": mo["multi_union_oracle"], "oracle_delta_vs_A": mo["oracle_delta_multi_vs_A"], "domain_auc": md["domain_auc"],
              "raw_stack_oof": mf["raw_stack"], "gated_stack_oof": mf["gated_stack"], "oof_gate": gate,
              "gate_violation": violation, "private_proposed_changes": mf["private_proposed_changes"], "private_final_changes": mf["private_final_changes"],
              "zip_sha256": dsha, "validator": dval}
    write_json(OUT/"stage0_multi_expert_audit.json", audit0)

    private_raw = read_json(PRIVATE_OFFICIAL); private_ids = set(map(str, private_raw))
    I, S, G, Dp = load_zip_submission(INC_ZIP), load_zip_submission(SAFE_ZIP), load_zip_submission(G_ZIP), load_zip_submission(D_ZIP)
    if sha256(INC_ZIP) != EXPECTED_INC or not validate_submission(I, private_ids)["pass"]:
        raise RuntimeError("incumbent freeze failed")
    folds_all = load_fold_map(); folds = {q: f for q, f in folds_all.items() if f in {1,2,3,4}}
    gold_all = load_gold(); gold = {q: gold_all[q] for q in folds}
    vc, vs = load_candidates(VAL_K20), load_scores(VAL_BGE); rvc, rvs = load_flat_worklist(R_VAL_WORK), load_scores(R_VAL_SCORE)
    experts, vr10, vrscore, direct, drows = expert_artifacts(folds, gold, vc, vs, rvc, rvs)
    pc, ps = load_candidates(PRIV_K20), load_scores(PRIV_BGE)
    Bp = {str(q): list(map(str, v)) for q, v in read_json(PRIVATE_B)["top5"].items()}
    Cp = build_private_c(gold, folds, experts["A"], vc, vs, pc, ps, G)
    write_json(OUT/"private_expert_c_reconstructed.json", {"queries": len(Cp), "top5": Cp, "reason": "Private C prediction artifact was genuinely absent; frozen historical C configuration fitted once on F1-F4."})
    Rp, pr10, prscore = {}, {}, {}
    for q in pc:
        full, score = rrf_full(q, pc[q], ps); Rp[q], pr10[q], prscore[q] = full[:5], full[:10], score
    pexperts = {"A": G, "B": Bp, "C": Cp, "R": Rp, "D": Dp}

    pactions = {str(x["query_id"]): x for x in rows(MULTI_OUT/"private_stack_actions.jsonl")}
    vbase, pbase = {}, {}
    for q in folds:
        conf = drows[q]["confidence"] if set(experts["D"][q]) != set(experts["A"][q]) else {}
        vbase[q] = base_query_features(q, {e: experts[e][q] for e in EXPERTS}, vr10[q], rvc[q], rvs, vrscore[q], question(gold[q]), conf)
    for q in pc:
        action = pactions.get(q, {})
        conf = {"stack_boundary_margin": action.get("stack_score_margin", 0.), "net_vote_gain": 0., "number_of_membership_changes": len(set(Dp[q])-set(G[q]))}
        pbase[q] = base_query_features(q, {e: pexperts[e][q] for e in EXPERTS}, pr10[q], pc[q], ps, prscore[q], question(private_raw[q]), conf)
    vorder, porder = sorted(folds, key=qkey), sorted(pc, key=qkey)
    domain_auc, vp, pp = domain_prob([vbase[q] for q in vorder], [pbase[q] for q in porder])
    vfeat = {q: vbase[q]+[float(x)] for q, x in zip(vorder, vp)}; pfeat = {q: pbase[q]+[float(x)] for q, x in zip(porder, pp)}
    scaler = RobustScaler(); scaler.fit(np.asarray([vfeat[q] for q in vorder]+[pfeat[q] for q in porder], float))
    vmatrix = scaler.transform(np.asarray([vfeat[q] for q in vorder], float)); pmatrix = scaler.transform(np.asarray([pfeat[q] for q in porder], float))
    qindex = {q: i for i, q in enumerate(vorder)}; utility = utilities(experts, gold)
    # Cosine is diagnostic only and never enters selection.
    cosine_diag = float(np.mean(np.min(pairwise_distances(pmatrix[:min(200,len(pmatrix))], vmatrix, metric="cosine"), axis=1)))

    p1, p1audit, p1choices, p1report = outer_oof(vmatrix, qindex, experts, vr10, utility, gold, folds, "selector")
    p2, p2audit, p2choices, p2report = outer_oof(vmatrix, qindex, experts, vr10, utility, gold, folds, "fusion")
    p0 = {q: experts["D"][q] for q in folds}; p0report = summary_vs_a(p0, experts["A"], gold, folds); p0report["net_gain"] = p0report["improved"]-p0report["harmed"]
    policies = {"P0_MULTI_STACK": (p0, p0report), "P1_LOCAL_SELECTOR": (p1, p1report), "P2_LOCAL_FUSION": (p2, p2report)}
    eligible = {k: v for k, v in policies.items() if (k != "P0_MULTI_STACK" or not violation) and accepted(v[1])}
    risk = "NORMAL_LAST_SLOT"
    if eligible:
        selected = max(eligible, key=lambda k: (eligible[k][1]["recall"], eligible[k][1]["net_gain"], -eligible[k][1]["changed"]))
    else:
        extreme = {k: v for k, v in policies.items() if (k != "P0_MULTI_STACK" or not violation) and -.0005 <= v[1]["recall_delta"] <= 0 and v[1]["improved"] >= v[1]["harmed"] and v[1]["net_gain"] >= 0 and all(x["recall_delta"] >= -.002 for x in v[1]["folds"].values())}
        if not extreme:
            selected = None
        else:
            selected = max(extreme, key=lambda k: extreme[k][1]["recall"]); risk = "EXTREME"
    oof_report = {"stage0": audit0, "domain_auc_recomputed": domain_auc, "cosine_distance_diagnostic_mean_private_to_nearest_validation": cosine_diag,
                  "P0_MULTI_STACK": p0report, "P1_LOCAL_SELECTOR": p1report, "P2_LOCAL_FUSION": p2report,
                  "p1_nested_k": p1choices, "p2_nested_k": p2choices, "selected_policy": selected, "risk_mode": risk,
                  "private_labels_used": False, "fold0_used": False}
    write_json(OUT/"local_oof_report.json", oof_report)
    write_jsonl(OUT/"local_oof_predictions.jsonl", ({"query_id": q, "fold": folds[q], "A": experts["A"][q], "P0": p0[q], "P1": p1[q], "P2": p2[q], "P1_audit": p1audit[q], "P2_audit": {k:v for k,v in p2audit[q].items() if k != "document_scores"}} for q in vorder))
    if selected is None:
        report = {"status": "KEEP_09205_INCUMBENT", **oof_report, "elapsed_seconds": time.monotonic()-started, "gpu_runs": 0, "modal_runs": 0, "auto_submit": False}
        write_json(OUT/"final_report.json", report); (OUT/"final_report.md").write_text("# Final local regime selector\n\nNo policy passed the aggressive last-slot gate. Incumbent retained.\n", encoding="utf-8"); print(json.dumps(report, indent=2)); return 0

    selected_report = policies[selected][1]
    if selected == "P0_MULTI_STACK":
        shutil.copyfile(D_ZIP, FINAL_ZIP); payload = Dp
        private_audit = [pactions[q] for q in sorted(pactions, key=qkey) if Dp[q] != I[q]]
    else:
        policy_name = "selector" if selected == "P1_LOCAL_SELECTOR" else "fusion"
        k, k_candidates = final_k(vmatrix, qindex, experts, vr10, utility, gold, folds, policy_name)
        if k is None:
            raise RuntimeError("selected local policy has no full-bank K")
        all_nm = neighbor_map(vorder, porder, np.vstack([vmatrix, pmatrix]), {**qindex, **{q: len(vorder)+i for i,q in enumerate(porder)}})
        ppred, paudit = predict_local(porder, all_nm, k, pexperts, pr10, utility, policy_name)
        mutable = {q for q in private_ids if set(I[q]) == set(G[q]) == set(S[q])}; protected = private_ids-mutable
        proposals = []
        for q in porder:
            candidate = list(I[q][:3]) + [d for d in ppred[q] if d not in I[q][:3]][:2]
            entered, dropped = set(candidate)-set(I[q]), set(I[q])-set(candidate)
            if not entered or len(entered) > 2 or len(candidate) != 5 or len(set(candidate)) != 5 or I[q][:3] != G[q][:3]:
                continue
            exceptional = False
            if q in protected:
                if selected != "P2_LOCAL_FUSION" or len(entered) != 1:
                    continue
                inc, drop = next(iter(entered)), next(iter(dropped))
                votes_in = sum(inc in pexperts[e][q] for e in EXPERTS); votes_drop = sum(drop in pexperts[e][q] for e in EXPERTS)
                support_stats = [paudit[q]["stats"][e] for e in EXPERTS if inc in pexperts[e][q]]
                exceptional = votes_in >= 4 and votes_drop <= 1 and any(s["local_win"] > .8 and s["local_harm"] < .1 for s in support_stats)
                if not exceptional:
                    continue
            proposals.append({"query_id": q, "old_top5": I[q], "new_top5": candidate, "confidence": float(paudit[q]["confidence"]),
                              "membership_changes": len(entered), "entered": sorted(entered), "dropped": sorted(dropped),
                              "protected_exception": exceptional, "selected_expert": paudit[q]["selected_expert"],
                              "local_stats": paudit[q]["stats"]})
        expected_changes = selected_report["changed"]/5600*2080; cap = min(120, max(20, round(expected_changes*1.5)))
        proposals.sort(key=lambda x: (-x["confidence"], qkey(x["query_id"]))); private_audit = proposals[:cap]
        payload = {q: list(I[q]) for q in I}
        for x in private_audit: payload[x["query_id"]] = x["new_top5"]
        write_zip(payload, private_ids)
    final_val = validate_submission(payload, private_ids); changed = [q for q in I if payload[q] != I[q]]
    top13 = sum(payload[q][:3] != I[q][:3] for q in I); one = sum(len(set(payload[q])-set(I[q])) == 1 for q in changed); two = sum(len(set(payload[q])-set(I[q])) == 2 for q in changed); order_only = sum(set(payload[q]) == set(I[q]) for q in changed)
    protected_set = private_ids-{q for q in private_ids if set(I[q]) == set(G[q]) == set(S[q])}; protected_changes = sum(q in protected_set for q in changed)
    net_per = selected_report["net_gain"]/max(1, selected_report["changed"]); expected_gain = len(changed)*net_per; proxy = expected_gain/2080
    submission = {"path": str(FINAL_ZIP.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(FINAL_ZIP), "validator": final_val}
    if not final_val["pass"] or top13:
        raise RuntimeError("final local submission firewall failed")
    write_jsonl(OUT/"private_actions.jsonl", private_audit)
    report = {"status": "FINAL_LAST_SLOT_READY", **oof_report, "selected_final_policy": selected, "risk_mode": risk,
              "private_changes": len(changed), "one_doc_changes": one, "two_doc_changes": two, "order_only_changes": order_only, "protected_189_overrides": protected_changes,
              "expected_private_net_gain_proxy": expected_gain, "expected_private_recall_delta_proxy": proxy,
              "submission": submission, "top1_3_changed": top13, "elapsed_seconds": time.monotonic()-started,
              "private_labels_used": False, "gpu_runs": 0, "modal_runs": 0, "auto_submit": False}
    write_json(OUT/"final_report.json", report)
    (OUT/"final_report.md").write_text(f"# Final last-slot local regime selector\n\nSelected `{selected}` in `{risk}` mode. OOF Recall `{selected_report['recall']}` (delta `{selected_report['recall_delta']}`), changed/improved/harmed `{selected_report['changed']}/{selected_report['improved']}/{selected_report['harmed']}`. Private changes `{len(changed)}`; validator PASS; SHA `{submission['sha256']}`. No automatic submission.\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
