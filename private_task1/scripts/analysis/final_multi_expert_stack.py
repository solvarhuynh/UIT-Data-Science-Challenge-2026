"""Final CPU-only adaptive multi-expert stack over the frozen 0.92054 incumbent."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.phase1_step4_recovery import load_fold_map, load_gold, metric, qkey
from private_task1.scripts.analysis.guarded_vs_direct_top5_meta_selector import (
    load_guarded, load_zip_submission, rows, sha256, summary_vs_a,
    validate_submission, write_json, write_jsonl,
)
from private_task1.scripts.analysis.direct_k20_listwise_top5 import (
    HP as C_HP, build_features as c_build_features, load_bge as c_load_bge,
    load_candidate_file as c_load_candidates, make_universe as c_make_universe,
    matrices as c_matrices, rank_predictions as c_rank_predictions,
)

OUT = ROOT / "private_task1/experiments/09205_final_multi_expert_stack"
TEAM = ROOT / "private_task1/submissions/team_09205"
INC_ZIP = TEAM / "submission_private_constrained_dual_anchor_rrf_09205.zip"
SAFE_ZIP = TEAM / "submission_private_dual_anchor_guarded_09198.zip"
G_ZIP = ROOT / "private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip"
PRIVATE_OFFICIAL = ROOT / "private_task1/input/private-official.json"
DIRECT_OOF = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl"
C_OOF = ROOT / "private_task1/experiments/direct_k20_listwise_top5/oof_predictions.jsonl"
VAL_K20 = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
VAL_BGE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
PRIV_K20 = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIV_BGE = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
R_VAL_WORK = ROOT / "private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl"
R_VAL_SCORE = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl"
PRIVATE_B = ROOT / "private_task1/experiments/09205_last_chance/private_expert_b_top5.json"
FINAL_ZIP = TEAM / "submission_private_09205_final_multi_expert_stack.zip"
EXPECTED_SHA = "aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae"
EXPECTED = {"A": .9300505952380952, "B": .9293809523809524, "C": .9305863095238095, "R": .924514880952381}
SEED = 20260923
TARGET_FOLDS = {1, 2, 3, 4}
QUANTILES = (.70, .80, .90, .95)
STACK_HP = dict(objective="lambdarank", metric="ndcg", eval_at=[5], label_gain=[0, 1],
                n_estimators=180, learning_rate=.03, num_leaves=7, max_depth=3,
                min_child_samples=30, subsample=.9, colsample_bytree=.9,
                reg_lambda=2., random_state=SEED, n_jobs=-1, verbosity=-1,
                deterministic=True, force_col_wise=True)
RRF_WEIGHTS = {"dense": .2, "bge": .3, "knn_word": .2, "bm25": .3}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def doc_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def question(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or row.get("text") or "")


def load_candidates(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for row in rows(path):
        q = str(row["query_id"])
        if q in out:
            raise RuntimeError(f"duplicate candidate query: {q}")
        docs = {}
        for item in row["candidates"]:
            d = str(item["document_id"])
            sr = {str(k): int(v) for k, v in (item.get("source_ranks") or {}).items()}
            docs[d] = {"candidate_rank": int(item["candidate_rank"]), "source_ranks": sr,
                       "rrf_score": float(item.get("rrf_score", sum(1/(60+r) for r in sr.values()))),
                       "source_support": len(sr)}
        if len(docs) != 20:
            raise RuntimeError(f"not K20: {q}:{len(docs)}")
        out[q] = docs
    return out


def load_scores(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in rows(path):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in out:
            raise RuntimeError(f"duplicate score: {key}")
        if not all(math.isfinite(float(row[x])) for x in ("bge_ft_score", "bge_base_score")):
            raise RuntimeError(f"nonfinite BGE: {key}")
        out[key] = row
    return out


def load_flat_worklist(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows(path):
        q, d = str(row["query_id"]), str(row["document_id"])
        sr = {str(k): int(v) for k, v in (row.get("source_ranks") or {}).items()}
        if d in out.setdefault(q, {}):
            raise RuntimeError(f"duplicate flat worklist identity: {(q, d)}")
        out[q][d] = {"candidate_rank": int(row["candidate_rank"]), "source_ranks": sr,
                     "rrf_score": float(row.get("rrf_score", sum(1/(60+r) for r in sr.values()))),
                     "source_support": len(sr)}
    if any(len(v) != 20 for v in out.values()):
        raise RuntimeError("canonical R worklist is not K20")
    return out


def bge_order(q: str, docs: dict[str, dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    return sorted(docs, key=lambda d: (-float(scores[(q, d)]["bge_ft_score"]),
                                      -float(scores[(q, d)]["bge_base_score"]),
                                      docs[d]["candidate_rank"], doc_key(d)))


def rrf_ranking(q: str, docs: dict[str, dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    br = {d: i for i, d in enumerate(bge_order(q, docs, scores), 1)}
    values, ranks = {}, {}
    for d, meta in docs.items():
        ranks[d] = dict(meta["source_ranks"])
        ranks[d]["bge"] = br[d]
        values[d] = sum(w / (2 + ranks[d][name]) for name, w in RRF_WEIGHTS.items() if name in ranks[d])
    order = sorted(docs, key=lambda d: (-values[d], *(ranks[d].get(s, 10**9) for s in RRF_WEIGHTS), d))
    return order, values


def load_experts(folds: dict[str, int], candidates, scores, r_candidates, r_scores):
    saved = {str(x["query_id"]): x for x in rows(DIRECT_OOF)}
    if set(saved) != set(folds):
        raise RuntimeError("A/B OOF population mismatch")
    A = load_guarded(saved)
    B = {q: [str(d) for d in saved[q]["direct_top5"]] for q in folds}
    crows = {str(x["query_id"]): x for x in rows(C_OOF)}
    if set(crows) != set(folds):
        raise RuntimeError("C OOF population mismatch")
    C = {q: [str(d) for d in crows[q]["listwise_top5"]] for q in folds}
    R, rscore = {}, {}
    for q in folds:
        rank, vals = rrf_ranking(q, r_candidates[q], r_scores)
        R[q], rscore[q] = rank[:5], vals
    return {"A": A, "B": B, "C": C, "R": R}, rscore


def minmax(values: dict[str, float]) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    span = max(1e-12, hi-lo)
    return {k: (v-lo)/span for k, v in values.items()}


FEATURE_NAMES = (
    "in_A", "rank_A", "rr_A", "in_B", "rank_B", "rr_B", "in_C", "rank_C", "rr_C",
    "in_R", "rank_R", "rr_R", "expert_vote_count", "expert_top3_vote_count", "best_expert_rank",
    "mean_reciprocal_expert_rank", "expert_rank_std", "A_B_agree", "A_C_agree", "B_C_agree",
    "bge_ft_raw", "bge_ft_minmax", "bge_base_raw", "bge_base_minmax", "union_rrf",
    "candidate_rank_norm", "rr_dense", "rr_bm25", "rr_knn_word", "source_support",
    "source_support_count", "missing_dense", "missing_bm25", "missing_knn_word",
    "A_B_top5_overlap", "A_C_top5_overlap", "B_C_top5_overlap", "all_expert_union_size",
    "LOW_MARGIN", "HIGH_DISAGREEMENT", "STRONG_BGE_REORDER", "SINGLE_SOURCE_DOMINANCE", "query_length",
)


def build_query_features(q: str, expert: dict[str, list[str]], docs, scores, rscore, qtext: str):
    ranks = {e: {d: i for i, d in enumerate(expert[e], 1)} for e in expert}
    ft = {d: float(scores[(q, d)]["bge_ft_score"]) for d in docs}
    base = {d: float(scores[(q, d)]["bge_base_score"]) for d in docs}
    nft, nbase = minmax(ft), minmax(base)
    bo = bge_order(q, docs, scores); br = {d: i for i, d in enumerate(bo, 1)}
    overlaps = {"AB": len(set(expert["A"]) & set(expert["B"])), "AC": len(set(expert["A"]) & set(expert["C"])), "BC": len(set(expert["B"]) & set(expert["C"]))}
    union_size = len(set().union(*(set(v) for v in expert.values())))
    rorder = sorted(docs, key=lambda d: (-rscore[d], doc_key(d)))
    margin = rscore[rorder[4]] - rscore[rorder[5]]
    top_support = [docs[d]["source_support"] for d in expert["A"]]
    low = int(margin <= .00119048)
    high = int(sum(top_support)/5 <= 1.6 or sum(x <= 1 for x in top_support) >= 2 or union_size >= 9)
    strong = int(max(abs(br[d] - docs[d]["candidate_rank"]) for d in expert["A"]) >= 17)
    single = int(sum(x <= 1 for x in top_support) >= 2)
    qlen = len(qtext.split())
    feats = {}
    for d, meta in docs.items():
        vals = []
        eranks = []
        for e in ("A", "B", "C", "R"):
            rank = ranks[e].get(d)
            vals += [float(rank is not None), float(rank or 6), 0. if rank is None else 1/(60+rank)]
            if rank is not None:
                eranks.append(rank)
        membership = {e: d in ranks[e] for e in ranks}
        vals += [float(len(eranks)), float(sum(d in expert[e][:3] for e in expert)), float(min(eranks) if eranks else 6),
                 float(statistics.mean(1/(60+r) for r in eranks) if eranks else 0),
                 float(statistics.pstdev(eranks) if len(eranks)>1 else 0),
                 float(membership["A"] and membership["B"]), float(membership["A"] and membership["C"]),
                 float(membership["B"] and membership["C"]), ft[d], nft[d], base[d], nbase[d],
                 float(meta["rrf_score"]), (21-meta["candidate_rank"])/20]
        for source in ("dense", "bm25", "knn_word"):
            rank = meta["source_ranks"].get(source)
            vals.append(0. if rank is None else 1/(60+rank))
        vals += [float(meta["source_support"]), float(meta["source_support"]),
                 float("dense" not in meta["source_ranks"]), float("bm25" not in meta["source_ranks"]),
                 float("knn_word" not in meta["source_ranks"]), overlaps["AB"]/5, overlaps["AC"]/5,
                 overlaps["BC"]/5, float(union_size), float(low), float(high), float(strong), float(single), float(qlen)]
        if len(vals) != len(FEATURE_NAMES):
            raise RuntimeError("feature width mismatch")
        feats[d] = vals
    qdomain = [float(qlen), float(union_size), overlaps["AB"]/5, overlaps["AC"]/5, overlaps["BC"]/5,
               float(statistics.mean(ft.values())), float(statistics.pstdev(ft.values())),
               float(statistics.mean(base.values())), float(statistics.pstdev(base.values())),
               float(statistics.mean(m["source_support"] for m in docs.values())), float(margin),
               float(low), float(high), float(strong), float(single)]
    return feats, qdomain


def candidate_set(expert: dict[str, list[str]], rfull: list[str]) -> list[str]:
    wanted = set().union(*(set(expert[e]) for e in ("A", "B", "C")), set(rfull[:10]))
    return sorted(wanted, key=doc_key)


def fit_ranker(qids: list[str], features, gold, weights):
    X, y, groups, sw = [], [], [], []
    for q in sorted(qids, key=qkey):
        docs = sorted(features[q], key=doc_key)
        wanted = set(map(str, gold[q].get("answer", [])))
        X.extend(features[q][d] for d in docs); y.extend(int(d in wanted) for d in docs)
        groups.append(len(docs)); sw.extend([weights[q]]*len(docs))
    model = LGBMRanker(**STACK_HP)
    model.fit(np.asarray(X, np.float32), np.asarray(y, np.int32), group=groups, sample_weight=np.asarray(sw, np.float32), feature_name=list(FEATURE_NAMES))
    return model


def predict(model, qids: Iterable[str], features):
    out, raw = {}, {}
    for q in qids:
        docs = sorted(features[q], key=doc_key)
        values = model.predict(np.asarray([features[q][d] for d in docs], np.float32))
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite stack score {q}")
        raw[q] = dict(zip(docs, map(float, values)))
        out[q] = sorted(docs, key=lambda d: (-raw[q][d], doc_key(d)))[:5]
    return out, raw


def confidence(q: str, stack: list[str], A: list[str], scores: dict[str, float], expert: dict[str, list[str]]):
    entered, dropped = list(set(stack)-set(A)), list(set(A)-set(stack))
    rejected = [d for d in scores if d not in stack]
    margin = min(scores[d] for d in stack) - max(scores[d] for d in rejected)
    votes = {d: sum(d in expert[e] for e in expert) for d in scores}
    erank = {d: min([expert[e].index(d)+1 for e in expert if d in expert[e]] or [999]) for d in scores}
    return {"entered": entered, "dropped": dropped, "stack_boundary_margin": float(margin),
            "entered_vote_count_min": min([votes[d] for d in entered] or [0]),
            "entered_best_expert_rank": min([erank[d] for d in entered] or [999]),
            "dropped_vote_count_max": max([votes[d] for d in dropped] or [0]),
            "net_vote_gain": float(statistics.mean([votes[d] for d in entered] or [0])-statistics.mean([votes[d] for d in dropped] or [0])),
            "A_stack_overlap": len(set(A)&set(stack))/5, "number_of_membership_changes": len(entered),
            "top1_3_lock": all(d in stack for d in A[:3]), "expert_votes": votes,
            "expert_ranks": {d: {e: (expert[e].index(d)+1 if d in expert[e] else None) for e in expert} for d in set(entered+dropped)}}


def apply_gate(base, stack, conf, threshold):
    out = {q: list(v) for q, v in base.items()}
    chosen = set()
    if threshold is None:
        return out, chosen
    for q in out:
        c = conf[q]
        if (set(stack[q]) != set(base[q]) and c["stack_boundary_margin"] >= threshold and c["net_vote_gain"] >= 0
                and c["number_of_membership_changes"] <= 2 and c["top1_3_lock"]):
            # Preserve A ordering for shared docs and stack ordering for entrants, while locking A Top1-3.
            tail = [d for d in stack[q] if d not in base[q][:3]]
            out[q] = list(base[q][:3]) + tail[:2]
            if len(set(out[q])) != 5:
                continue
            chosen.add(q)
    return out, chosen


def threshold_choice(train, base, stack, conf, gold, folds):
    margins = [conf[q]["stack_boundary_margin"] for q in train if set(stack[q]) != set(base[q]) and conf[q]["stack_boundary_margin"] > 0]
    candidates = []
    if not margins:
        return None, candidates
    for quant in QUANTILES:
        t = float(np.quantile(np.asarray(margins), quant))
        pred, chosen = apply_gate({q: base[q] for q in train}, {q: stack[q] for q in train}, conf, t)
        report = summary_vs_a(pred, {q: base[q] for q in train}, {q: gold[q] for q in train}, {q: folds[q] for q in train})
        stable = report["improved"] > report["harmed"] and all(x["recall_delta"] >= 0 for x in report["folds"].values())
        candidates.append({"quantile": quant, "threshold": t, "eligible": stable, "chosen": len(chosen), **report})
    valid = [x for x in candidates if x["eligible"]]
    if not valid:
        return None, candidates
    return sorted(valid, key=lambda x: (-x["recall"], x["changed"], -x["threshold"]))[0], candidates


def union_oracle(experts, gold):
    out = {}
    for q in gold:
        union = list(dict.fromkeys(d for e in ("A", "B", "C", "R") for d in experts[e][q]))
        wanted = set(map(str, gold[q].get("answer", [])))
        good = [d for d in union if d in wanted][:5]
        out[q] = good + [d for d in union if d not in good][:5-len(good)]
    return out


def build_private_c(gold, folds, A_val, candidates_val, scores_val, candidates_priv, scores_priv, A_priv):
    val_bge = c_load_bge(VAL_BGE); priv_bge = c_load_bge(PRIV_BGE)
    vu = c_make_universe(c_load_candidates(VAL_K20), A_val); pu = c_make_universe(c_load_candidates(PRIV_K20), A_priv)
    vf = c_build_features(vu, A_val, val_bge); pf = c_build_features(pu, A_priv, priv_bge)
    X, y, groups = c_matrices(list(folds), vf, gold)
    hp = dict(C_HP); hp["n_jobs"] = -1
    model = LGBMRanker(**hp); model.fit(X, y, group=groups, feature_name=list(next(iter(vf.values())).values()) and None)
    return c_rank_predictions(list(candidates_priv), pf, pu, A_priv, model)


def write_submission(payload, expected_ids):
    json_path = FINAL_ZIP.with_suffix(".json")
    obj = {q: {"answer": payload[q]} for q in sorted(payload, key=qkey)}
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="submission.json")
    return {"json": str(json_path.relative_to(ROOT)).replace("\\", "/"), "zip": str(FINAL_ZIP.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(FINAL_ZIP), "validator": validate_submission(payload, expected_ids)}


def main() -> int:
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    private_raw = read_json(PRIVATE_OFFICIAL); private_ids = set(map(str, private_raw))
    I, S, G = load_zip_submission(INC_ZIP), load_zip_submission(SAFE_ZIP), load_zip_submission(G_ZIP)
    freeze = {"sha256": sha256(INC_ZIP), "expected_sha256": EXPECTED_SHA, "validator": validate_submission(I, private_ids)}
    if freeze["sha256"] != EXPECTED_SHA or not freeze["validator"]["pass"]:
        raise RuntimeError("incumbent freeze gate failed")
    folds_all = load_fold_map(); folds = {q: f for q, f in folds_all.items() if f in TARGET_FOLDS}
    gold_all = load_gold(); gold = {q: gold_all[q] for q in folds}
    vc, vs = load_candidates(VAL_K20), load_scores(VAL_BGE)
    if set(vc) != set(folds) or set(vs) != {(q, d) for q in vc for d in vc[q]}:
        raise RuntimeError("validation identity mismatch")
    rvc, rvs = load_flat_worklist(R_VAL_WORK), load_scores(R_VAL_SCORE)
    if set(rvc) != set(folds) or set(rvs) != {(q, d) for q in rvc for d in rvc[q]}:
        raise RuntimeError("canonical R identity mismatch")
    experts, rscore = load_experts(folds, vc, vs, rvc, rvs)
    metrics = {e: metric(experts[e], gold) for e in experts}
    for e, expected in EXPECTED.items():
        if abs(metrics[e]["recall"] - expected) > 1e-14:
            raise RuntimeError(f"expert {e} reproduction failed: {metrics[e]['recall']} != {expected}")
    # Add canonical R Top10 rows to the PV1 expert universe without replacing PV1 metadata.
    for q in folds:
        rfull, _ = rrf_ranking(q, rvc[q], rvs)
        for d in rfull[:10]:
            if d not in vc[q]:
                vc[q][d] = dict(rvc[q][d]); vs[(q, d)] = rvs[(q, d)]
    oracle = union_oracle(experts, gold); om = metric(oracle, gold)
    sizes = [len(set().union(*(set(experts[e][q]) for e in experts))) for q in folds]
    def absent_gain(e):
        return sum(bool((set(experts[e][q])-set(experts["A"][q])) & set(map(str, gold[q].get("answer", [])))) for q in folds)
    oracle_report = {"A_recall": metrics["A"]["recall"], "AB_oracle": .9337708333333333,
                     "multi_union_oracle": om["recall"], "oracle_delta_multi_vs_A": om["recall"]-metrics["A"]["recall"],
                     "headroom": "STRONG_HEADROOM" if om["recall"]-metrics["A"]["recall"] >= .005 else "LOW_HEADROOM",
                     "union_size": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)},
                     "relevant_absent_A": {e: absent_gain(e) for e in ("B", "C", "R")}}
    write_json(OUT/"expert_reproduction.json", {"metrics": metrics, "expected": EXPECTED, "fold0_used": False})
    write_json(OUT/"multi_expert_oracle.json", oracle_report)

    pc, ps = load_candidates(PRIV_K20), load_scores(PRIV_BGE)
    # Domain adaptation is built from label-free K20/BGE/query distribution summaries.
    val_domain, priv_domain = {}, {}
    for q in folds:
        rfull, rv = rrf_ranking(q, rvc[q], rvs); ex = {e: experts[e][q] for e in experts}
        _, val_domain[q] = build_query_features(q, ex, vc[q], vs, {d: rv.get(d, 0.0) for d in vc[q]}, question(gold[q]))
    # Private expert-free proxy uses R for all expert slots; expert fields are omitted from domain features below.
    for q in pc:
        rfull, rv = rrf_ranking(q, pc[q], ps); ex = {e: rfull[:5] for e in experts}
        _, priv_domain[q] = build_query_features(q, ex, pc[q], ps, rv, question(private_raw[q]))
    domain_cols = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Xv = np.asarray([[val_domain[q][i] for i in domain_cols] for q in sorted(folds, key=qkey)], float)
    Xp = np.asarray([[priv_domain[q][i] for i in domain_cols] for q in sorted(pc, key=qkey)], float)
    X = np.vstack([Xv, Xp]); y = np.r_[np.zeros(len(Xv)), np.ones(len(Xp))]
    domain = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=500, random_state=SEED))
    # Deterministic five-way out-of-sample AUC, then refit all rows.
    pred = np.zeros(len(y)); idx = np.arange(len(y))
    for k in range(5):
        test = idx[idx % 5 == k]; train = idx[idx % 5 != k]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=500, random_state=SEED))
        m.fit(X[train], y[train]); pred[test] = m.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, pred)); domain.fit(X, y)
    pv = domain.predict_proba(Xv)[:, 1]; w = np.clip(pv/np.maximum(1e-6, 1-pv), .25, 4.); w /= w.mean()
    qweights = dict(zip(sorted(folds, key=qkey), map(float, w)))
    write_json(OUT/"domain_adaptation.json", {"domain_auc": auc, "classifier": "StandardScaler+LogisticRegression", "validation_queries": len(Xv), "private_queries": len(Xp), "weight_mean": float(w.mean()), "weight_min": float(w.min()), "weight_max": float(w.max()), "private_labels_used": False})

    features, rfulls, candidate_stats = {}, {}, []
    for q in folds:
        rfull, rv = rrf_ranking(q, rvc[q], rvs); rfulls[q] = rfull
        # Scores are defined on the canonical-R universe; extend them deterministically
        # to the merged PV1 universe only for feature construction.
        merged_rv = {d: rv.get(d, 0.0) for d in vc[q]}
        ex = {e: experts[e][q] for e in experts}; full, _ = build_query_features(q, ex, vc[q], vs, merged_rv, question(gold[q]))
        keep = candidate_set(ex, rfull); features[q] = {d: full[d] for d in keep}; candidate_stats.append(len(keep))
    raw, raw_scores, overlay, oof_rows, choices = {}, {}, {q: list(experts["A"][q]) for q in folds}, [], {}
    for f in sorted(TARGET_FOLDS):
        train = [q for q in folds if folds[q] != f]; test = [q for q in folds if folds[q] == f]
        model = fit_ranker(train, features, gold, qweights)
        train_pred, train_scores = predict(model, train, features); test_pred, test_scores = predict(model, test, features)
        conf_train = {q: confidence(q, train_pred[q], experts["A"][q], train_scores[q], {e: experts[e][q] for e in experts}) for q in train}
        conf_test = {q: confidence(q, test_pred[q], experts["A"][q], test_scores[q], {e: experts[e][q] for e in experts}) for q in test}
        choice, candidates = threshold_choice(train, experts["A"], train_pred, conf_train, gold, folds)
        threshold = None if choice is None else choice["threshold"]
        held, chosen = apply_gate({q: experts["A"][q] for q in test}, test_pred, conf_test, threshold)
        raw.update(test_pred); raw_scores.update(test_scores); overlay.update(held)
        choices[f"F{f}"] = {"selected": choice, "candidates": candidates}
        for q in test:
            oof_rows.append({"query_id": q, "fold": f, "A_top5": experts["A"][q], "stack_top5": test_pred[q], "overlay_top5": held[q], "selected": q in chosen, "confidence": conf_test[q]})
    raw_report = summary_vs_a(raw, experts["A"], gold, folds)
    gated = summary_vs_a(overlay, experts["A"], gold, folds); gated["net_relevant_document_gain"] = gated["improved"]-gated["harmed"]
    gated["improved_harmed_ratio"] = None if gated["harmed"] == 0 else gated["improved"]/gated["harmed"]
    primary = gated["recall_delta"] >= .001 and gated["improved"] > gated["harmed"] and gated["net_relevant_document_gain"] >= 6 and all(x["recall_delta"] >= -.0005 for x in gated["folds"].values()) and gated["precision_delta"] >= -.0005 and gated["top1_3_changed"] == 0
    secondary = gated["recall_delta"] > 0 and gated["improved"] >= 1.5*gated["harmed"] and gated["net_relevant_document_gain"] >= 4 and all(x["recall_delta"] >= -.001 for x in gated["folds"].values())
    gate = "STRONG" if primary else "LAST_SLOT" if secondary else "FAIL"
    oof_report = {"candidate_size": {"min": min(candidate_stats), "median": statistics.median(candidate_stats), "max": max(candidate_stats)},
                  "raw": raw_report, "gated": gated, "selected_thresholds": choices, "gate": gate,
                  "fold0_used": False, "private_labels_used": False, "elapsed_seconds": time.monotonic()-started}
    write_jsonl(OUT/"stack_oof_predictions.jsonl", sorted(oof_rows, key=lambda x: qkey(x["query_id"])))
    write_json(OUT/"stack_oof_report.json", oof_report)

    mutable = {q for q in private_ids if set(I[q]) == set(G[q]) == set(S[q])}; immutable = private_ids-mutable
    final_report = {"current_incumbent": .920544597, "incumbent_freeze": freeze, "experts": metrics, "oracle": oracle_report,
                    "domain_auc": auc, "raw_stack": raw_report, "gated_stack": gated, "oof_gate": gate,
                    "private_mutable": len(mutable), "private_immutable": len(immutable), "private_proposed_changes": 0,
                    "private_final_changes": 0, "submission": None, "gpu_runs": 0, "modal_runs": 0,
                    "private_labels_used": False, "fold0_used": False, "auto_submit": False}
    if gate == "FAIL" or time.monotonic()-started >= 90*60:
        final_report["status"] = "KEEP_09205_INCUMBENT"
        write_json(OUT/"private_stack_actions.jsonl", [])
        write_json(OUT/"final_report.json", final_report)
        (OUT/"final_report.md").write_text(f"# Final multi-expert stack\n\nOOF gate: **{gate}**. Gated Recall `{gated['recall']}`, delta `{gated['recall_delta']}`; changed/improved/harmed `{gated['changed']}/{gated['improved']}/{gated['harmed']}`. Incumbent retained byte-for-byte.\n", encoding="utf-8")
        print(json.dumps(final_report, ensure_ascii=False, indent=2)); return 0

    # Promotion-only path: recover/final-fit B and C, then fit the frozen stacker once.
    A_priv = G; B_priv = {str(q): list(map(str, v)) for q, v in read_json(PRIVATE_B)["top5"].items()}
    C_priv = build_private_c(gold, folds, experts["A"], vc, vs, pc, ps, A_priv)
    R_priv, rpriv_score = {}, {}
    for q in pc:
        R_priv[q], rpriv_score[q] = rrf_ranking(q, pc[q], ps)
    pexperts = {"A": A_priv, "B": B_priv, "C": C_priv, "R": {q: R_priv[q][:5] for q in pc}}
    pfeatures = {}
    for q in pc:
        ex = {e: pexperts[e][q] for e in pexperts}; full, _ = build_query_features(q, ex, pc[q], ps, rpriv_score[q], question(private_raw[q]))
        pfeatures[q] = {d: full[d] for d in candidate_set(ex, R_priv[q])}
    final_model = fit_ranker(list(folds), features, gold, qweights); pstack, pscores = predict(final_model, pc, pfeatures)
    pconf = {q: confidence(q, pstack[q], A_priv[q], pscores[q], {e: pexperts[e][q] for e in pexperts}) for q in pc}
    selected_quantiles = [v["selected"]["quantile"] for v in choices.values() if v["selected"] is not None]
    # Reuse one of the four frozen quantiles exactly; mode first, then the
    # higher quantile as deterministic tie-break. Never invent an in-between threshold.
    counts = Counter(selected_quantiles)
    final_quantile = float(max(counts, key=lambda x: (counts[x], x)))
    train_margins = [next(x["confidence"]["stack_boundary_margin"] for x in oof_rows if x["query_id"] == q) for q in folds if set(raw[q]) != set(experts["A"][q]) and next(x["confidence"]["stack_boundary_margin"] for x in oof_rows if x["query_id"] == q) > 0]
    final_threshold = float(np.quantile(np.asarray(train_margins), final_quantile))
    _proposed_a, selected = apply_gate(A_priv, pstack, pconf, final_threshold); selected &= mutable
    def incumbent_locked_candidate(q: str) -> list[str]:
        tail = [d for d in pstack[q] if d not in I[q][:3]]
        return list(I[q][:3]) + tail[:2]
    selected = {q for q in selected if incumbent_locked_candidate(q) != I[q]}
    proposed_count = len(selected)
    rate = gated["changed"]/5600; expected_changes = round(rate*len(mutable)); cap = min(40, max(10, round(expected_changes*1.25)))
    ranked = sorted(selected, key=lambda q: (-pconf[q]["stack_boundary_margin"], -pconf[q]["net_vote_gain"], qkey(q)))[:cap]
    selected = set(ranked); payload = {q: list(I[q]) for q in I}
    for q in ranked:
        # The incumbent and Guarded sets are equal in the mutable region, but
        # their order can differ. Preserve the incumbent Top1-3 byte-for-byte.
        candidate = incumbent_locked_candidate(q)
        if len(candidate) != 5 or len(set(candidate)) != 5 or len(set(candidate)-set(I[q])) > 2:
            raise RuntimeError(f"invalid automatic private action: {q}")
        payload[q] = candidate
    actions = [{"query_id": q, "old_top5": I[q], "new_top5": payload[q], "stack_score_margin": pconf[q]["stack_boundary_margin"],
                "expert_votes": pconf[q]["expert_votes"], "expert_ranks": pconf[q]["expert_ranks"], "confidence_threshold": final_threshold,
                "selection_reason": "automatic_stacked_ranker_confidence_gate"} for q in ranked]
    immutable_mutations = sum(payload[q] != I[q] for q in immutable); top13 = sum(payload[q][:3] != I[q][:3] for q in I)
    if immutable_mutations or top13:
        raise RuntimeError("private firewall failed")
    submission = write_submission(payload, private_ids)
    if not submission["validator"]["pass"]:
        raise RuntimeError("final validator failed")
    write_jsonl(OUT/"private_stack_actions.jsonl", actions)
    final_report.update({"status": "FINAL_AUTOMATED_CHALLENGER_READY", "private_proposed_changes": proposed_count, "private_final_changes": len(ranked),
                         "private_max_changes": cap, "final_quantile": final_quantile, "final_threshold": final_threshold,
                         "immutable_mutations": immutable_mutations, "top1_3_changed": top13, "submission": submission})
    write_json(OUT/"final_report.json", final_report)
    (OUT/"final_report.md").write_text(f"# Final multi-expert stack\n\nOOF gate: **{gate}**. Final automatic Private changes: `{len(ranked)}`. Validator: **PASS**. No automatic submission.\n", encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
