"""Step 2: frozen K=77 rank-4/5 residual policy under strict nested OOF.

Only folds 1--4 are materialized.  Aggregate JSONL inputs are structurally
screened by query_id before a target payload is decoded.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/task1"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
CONTRACT = REPORTS / "step2_k77_model_contract.json"
DECISIONS = REPORTS / "step2_k77_oof_policy_decisions.jsonl"
REPORT = REPORTS / "step2_k77_oof_policy_realizability_report.json"
K = 77
TARGET_FOLDS = {1, 2, 3, 4}
LABELS = {"BENEFICIAL": 0, "NEUTRAL": 1, "HARMFUL": 2}
HP = {"loss": "log_loss", "learning_rate": 0.05, "max_iter": 200, "max_depth": 5, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False, "random_state": 20260827}
C77 = 0.058785714285714274
C77_FOLD = {1: 0.04922619047619048, 2: 0.060416666666666674, 3: 0.06107142857142857, 4: 0.06442857142857143}
V3_BASE = ("neural_max", "neural_second", "neural_mean", "neural_min", "neural_std", "neural_max_minus_mean", "neural_top2_mean", "bm25_max", "bm25_mean", "union_rank", "reciprocal_union_rank", "source_support", "min_source_rank", "has_bge_support", "bm25_rank", "adaptive_k500_rank", "knn_char_rank", "knn_word_rank", "bm25_missing", "adaptive_k500_missing", "knn_char_missing", "knn_word_missing", "question_token_length", "question_char_length", "baseline_rank", "is_baseline_top5")
V3_DIFF = ("neural_max", "neural_mean", "neural_second", "union_rank", "source_support", "bm25_max")

def ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace(): i += 1
    return i

def skip_value(s: str, i: int) -> int:
    i = ws(s, i)
    if s[i] == '"': return json.JSONDecoder().raw_decode(s, i)[1]
    if s[i] not in "[{":
        while i < len(s) and s[i] not in ",]}": i += 1
        return i
    stack = ['}' if s[i] == '{' else ']']; i += 1; quoted = escaped = False
    while i < len(s) and stack:
        c = s[i]
        if quoted:
            if escaped: escaped = False
            elif c == '\\': escaped = True
            elif c == '"': quoted = False
        elif c == '"': quoted = True
        elif c == '{': stack.append('}')
        elif c == '[': stack.append(']')
        elif c == stack[-1]: stack.pop()
        i += 1
    return i

def top_string(s: str, wanted: str) -> str:
    d = json.JSONDecoder(); i = ws(s, 0)
    if i >= len(s) or s[i] != '{': raise ValueError("JSONL row must be an object")
    i = ws(s, i + 1)
    while i < len(s) and s[i] != '}':
        key, i = d.raw_decode(s, i); i = ws(s, i)
        if s[i] != ':': raise ValueError("invalid JSON")
        i = ws(s, i + 1)
        if key == wanted:
            value, _ = d.raw_decode(s, i)
            if not isinstance(value, str): raise ValueError("query_id must be string")
            return value
        i = ws(s, skip_value(s, i))
        if i < len(s) and s[i] == ',': i = ws(s, i + 1)
    raise ValueError(f"missing {wanted}")

def target_folds() -> dict[str, int]:
    # Reuse the approved structural fold reader; it never materializes Fold0 IDs.
    import importlib.util
    p = REPORTS / "step0_metric_contract_audit.py"; spec = importlib.util.spec_from_file_location("step0", p)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    result = module.read_target_fold_map(FOLDS)
    if len(result) != 5600 or set(result.values()) != TARGET_FOLDS: raise ValueError("invalid target fold map")
    return result

def target_rows(path: Path, targets: set[str]):
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            qid = top_string(line, "query_id")
            if qid not in targets:
                skipped += 1; continue
            yield json.loads(line)
    return skipped

def load_gold(targets: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    raw = TRAIN.read_text(encoding="utf-8-sig"); d = json.JSONDecoder(); i = ws(raw, 0) + 1; records = {}
    while ws(raw, i) < len(raw) and raw[ws(raw, i)] != '}':
        i = ws(raw, i); qid, i = d.raw_decode(raw, i); i = ws(raw, i)
        if raw[i] != ':': raise ValueError("invalid train JSON")
        i = ws(raw, i + 1)
        if qid in targets: record, i = d.raw_decode(raw, i); records[str(qid)] = record
        else: i = skip_value(raw, i)
        i = ws(raw, i)
        if i < len(raw) and raw[i] == ',': i += 1
    gold = {q: {str(x) for x in r["answer"]} for q, r in records.items()}
    if set(gold) != targets or any(not x for x in gold.values()): raise ValueError("target gold coverage invalid")
    return records, gold

def load_baseline(targets: set[str], folds: dict[str, int]) -> tuple[dict[str, list[str]], int]:
    out = {}; skipped = 0
    with BASELINE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = top_string(line, "query_id")
            if q not in targets: skipped += 1; continue
            row = json.loads(line); top = [str(x) for x in row["top5"]]
            if int(row["fold"]) != folds[q] or len(top) != 5 or len(set(top)) != 5: raise ValueError(f"baseline invalid: {q}")
            out[q] = top
    if set(out) != targets: raise ValueError("baseline coverage invalid")
    return out, skipped

def load_candidates(targets: set[str], folds: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    out = {}; skipped = 0
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = top_string(line, "query_id")
            if q not in targets: skipped += 1; continue
            row = json.loads(line)
            if int(row["fold"]) != folds[q]: raise ValueError(f"candidate fold invalid: {q}")
            raw = row.get("candidates") if isinstance(row.get("candidates"), list) else [row]
            unique = {}
            for pos, x in enumerate(raw, 1):
                doc = str(x["doc_id"]); y = dict(x); y["doc_id"] = doc; y["union_rank"] = int(y.get("union_rank", pos)); unique.setdefault(doc, y)
            out[q] = sorted(unique.values(), key=lambda x: (int(x["union_rank"]), x["doc_id"]))
    if set(out) != targets: raise ValueError("candidate coverage invalid")
    return out, skipped

def rank_value(doc: dict[str, Any], name: str) -> float:
    ranks = doc.get("source_ranks") or {}; value = ranks.get(name, doc.get(name))
    return float(value) if value not in (None, "") else 0.0

def doc_value(doc: dict[str, Any], name: str, qtokens: float, qchars: float, base_rank: int | None) -> float:
    if name == "union_rank": return float(doc.get("union_rank", 1e9))
    if name == "reciprocal_union_rank": return 1.0 / max(1.0, float(doc.get("union_rank", 1e9)))
    if name == "source_support": return float(doc.get("source_support", len(doc.get("source_ranks") or {})))
    if name == "min_source_rank":
        vals = [rank_value(doc, x) for x in (doc.get("source_ranks") or {}) if rank_value(doc, x) > 0]
        return min(vals) if vals else 0.0
    if name == "has_bge_support": return float(any("bge" in str(x).lower() for x in (doc.get("source_ranks") or {})))
    if name in ("bm25_rank", "adaptive_k500_rank", "knn_char_rank", "knn_word_rank"): return rank_value(doc, name)
    if name.endswith("_missing"):
        return float(rank_value(doc, name.removesuffix("_missing")) == 0.0)
    if name == "question_token_length": return qtokens
    if name == "question_char_length": return qchars
    if name == "baseline_rank": return float(base_rank or 0)
    if name == "is_baseline_top5": return float(base_rank is not None)
    raise KeyError(name)

def resolve_features(candidates: dict[str, list[dict[str, Any]]]) -> tuple[list[str], list[dict[str, Any]]]:
    available = {"union_rank", "reciprocal_union_rank", "source_support", "min_source_rank", "has_bge_support", "bm25_rank", "adaptive_k500_rank", "knn_char_rank", "knn_word_rank", "bm25_missing", "adaptive_k500_missing", "knn_char_missing", "knn_word_missing", "question_token_length", "question_char_length", "baseline_rank", "is_baseline_top5"}
    audit = []
    for n in V3_BASE:
        ok = n in available; audit.append({"feature_name": n, "source": "candidate_refs_full/source_ranks" if n not in {"question_token_length", "question_char_length", "baseline_rank", "is_baseline_top5"} else "target train question/baseline metadata", "dtype": "float32", "inference_available": ok, "gold_derived": False, "allowed": ok})
    selected_base = [n for n in V3_BASE if n in available]
    columns = [f"incoming_{n}" for n in selected_base] + [f"dropped_{n}" for n in selected_base] + [f"diff_{n}" for n in V3_DIFF if n in available] + ["dropped_baseline_rank", "incoming_is_baseline_top5"]
    return columns, audit

def recall(g: set[str], top: list[str]) -> float: return len(g & set(top)) / len(g)
def precision(g: set[str], top: list[str]) -> float: return len(g & set(top)) / 5.0

def make_actions(folds, records, gold, baseline, candidates, columns):
    by_fold = defaultdict(list); total_in = 0
    for q in sorted(folds, key=lambda x: int(x) if x.isdigit() else x):
        base = baseline[q]; bydoc = {x["doc_id"]: x for x in candidates[q]}; qtext = str(records[q].get("question", "")); qt, qc = float(len(qtext.split())), float(len(qtext))
        incoming = [x for x in candidates[q] if x["union_rank"] <= K and x["doc_id"] not in set(base)]; total_in += len(incoming)
        for inc in incoming:
            for drop in (4, 5):
                dropped_id = base[drop - 1]; dropped = bydoc.get(dropped_id, {"doc_id": dropped_id, "union_rank": 1e9, "source_ranks": {}, "source_support": 0})
                feat = {}
                for n in V3_BASE:
                    if f"incoming_{n}" in columns:
                        feat[f"incoming_{n}"] = doc_value(inc, n, qt, qc, None)
                        feat[f"dropped_{n}"] = doc_value(dropped, n, qt, qc, drop)
                for n in V3_DIFF:
                    if f"diff_{n}" in columns: feat[f"diff_{n}"] = feat[f"incoming_{n}"] - feat[f"dropped_{n}"]
                feat["dropped_baseline_rank"] = float(drop); feat["incoming_is_baseline_top5"] = 0.0
                final = list(base); final[drop - 1] = inc["doc_id"]; delta = recall(gold[q], final) - recall(gold[q], base)
                label = "BENEFICIAL" if delta > 1e-12 else "HARMFUL" if delta < -1e-12 else "NEUTRAL"
                by_fold[folds[q]].append({"query_id": q, "fold": folds[q], "incoming_doc_id": inc["doc_id"], "incoming_union_rank": int(inc["union_rank"]), "drop_rank": drop, "dropped_doc_id": dropped_id, "baseline_top5": base, "features": feat, "label": label, "truth_delta": delta})
    if total_in != 403322 or sum(len(x) for x in by_fold.values()) != 806644: raise ValueError(f"K77 action contract mismatch: {total_in}")
    return by_fold, total_in

def matrix(rows, columns): return np.asarray([[r["features"][c] for c in columns] for r in rows], dtype=np.float32)
def weights(y):
    c = Counter(int(value) for value in y); n = len(y)
    if set(c) != set(LABELS.values()): raise RuntimeError(f"BLOCKED absent class: {dict(c)}")
    return np.asarray([n / (3 * c[int(value)]) for value in y], dtype=np.float64)
def fit(x, y):
    m = HistGradientBoostingClassifier(**HP); m.fit(x, y, sample_weight=weights(y)); return m
def scores(model, x):
    p = model.predict_proba(x); idx = {int(x): i for i, x in enumerate(model.classes_)}
    return [float(x) for x in p[:, idx[LABELS["BENEFICIAL"]]] - p[:, idx[LABELS["HARMFUL"]]]]
def best(rows, values):
    grouped = defaultdict(list)
    for r, s in zip(rows, values): grouped[r["query_id"]].append((r, s))
    return {q: sorted(x, key=lambda z: (-z[1], -z[0]["drop_rank"], z[0]["incoming_union_rank"], z[0]["incoming_doc_id"]))[0] for q, x in grouped.items()}
def threshold(bestmap):
    pairs = list(bestmap.values()); candidates = [math.inf] + sorted({s for _, s in pairs})
    choices = []
    for t in candidates:
        chosen = [r for r, s in pairs if s >= t]; gain = sum(r["truth_delta"] for r in chosen) / len(pairs)
        # precision is a secondary tie-break over policy predictions (no-op keeps baseline).
        # Per-query baseline precision is computed later; its constant part need not be retained here.
        pdelta = sum((1.0 if r["label"] == "BENEFICIAL" else -1.0 if r["label"] == "HARMFUL" else 0.0) / 5.0 for r in chosen) / len(pairs)
        choices.append((gain, pdelta, t))
    return max(choices, key=lambda x: (x[0], x[1], x[2] if math.isfinite(x[2]) else math.inf))[2]

def main():
    # Validate preconditions before any target payload data is read for model work.
    f = json.loads((REPORTS / "exp_1f_k_selection_protocol_report.json").read_text()); r = json.loads((REPORTS / "exp_1b_rerun_k77_oracle_gap_decomposition_report.json").read_text())
    ok = f.get("status") == "PASS" and f.get("adopted_research_K") == 77 and f.get("selection_decision") == "ADOPT_RESEARCH_K" and f.get("production_K_selected") is False and f.get("lofo_stability", {}).get("selection_stability_pass") is True and r.get("status") == "PASS" and r.get("oracle_decomposition", {}).get("pooled_C77") == C77 and r.get("rank13_decision") == "KEEP_RANK1_3_CLOSED"
    if not ok: raise RuntimeError("BLOCKED Step 1 contract mismatch")
    folds = target_folds(); targets = set(folds); records, gold = load_gold(targets); baseline, bskip = load_baseline(targets, folds); candidates, cskip = load_candidates(targets, folds)
    columns, audit = resolve_features(candidates)
    if not columns or any(x["allowed"] and x["gold_derived"] for x in audit): raise RuntimeError("CONTRACT_ERROR feature audit")
    contract = {"status":"FROZEN_PRE_TRAIN","model_family":"sklearn.ensemble.HistGradientBoostingClassifier","hyperparameters":HP,"random_seed":20260827,"fixed_K":77,"drop_ranks":[4,5],"feature_resolution_rule":"Ordered intersection of historical V3A action features, K77 precomputed candidate/baseline/query metadata availability, and non-gold inference-safe features; unavailable neural/reranker features are excluded.","feature_columns":columns,"feature_audit":audit,"label_classes":["BENEFICIAL","NEUTRAL","HARMFUL"],"sample_weight_formula":"N/(3*n_class)","policy_score":"P(BENEFICIAL)-P(HARMFUL)","tie_break":"drop_rank5_then_lower_union_rank_then_lexicographic_doc_id","outer_oof_protocol":{"held_out_folds":[1,2,3,4],"train_is_complement":True},"inner_threshold_protocol":{"rotations":3,"candidates":"+infinity plus all unique finite inner-OOF best-action policy scores","objective":"macro_recall_gain","tie_break":["larger_macro_precision","larger_tau","deterministic_numeric_order"]},"primary_metric":"macro_recall","scientific_gate":{"pooled_D_gt_0":True,"each_fold_D_ge_0":True},"precision_guardrail":"precision_delta >= -0.5 * D77","fold0_allowed":False,"public_labels_allowed":False}
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")  # must precede first fit
    actions, incoming_count = make_actions(folds, records, gold, baseline, candidates, columns)
    # Materialize the frozen feature matrix once per fold.  This is a pure CPU
    # execution optimization; values and the frozen feature schema are unchanged.
    x_by_fold = {fold: matrix(actions[fold], columns) for fold in range(1, 5)}
    y_by_fold = {fold: np.asarray([LABELS[row["label"]] for row in actions[fold]], dtype=np.int8) for fold in range(1, 5)}
    query_action_counts = {fold: Counter(row["query_id"] for row in actions[fold]) for fold in range(1, 5)}
    decisions = []; outer = []
    for held in range(1, 5):
        trainfolds = [x for x in range(1,5) if x != held]; inner = {}
        for valid in trainfolds:
            train_folds_inner = [ff for ff in trainfolds if ff != valid]
            train_x = np.concatenate([x_by_fold[ff] for ff in train_folds_inner])
            train_y = np.concatenate([y_by_fold[ff] for ff in train_folds_inner])
            validrows = actions[valid]
            inner.update(best(validrows, scores(fit(train_x, train_y), x_by_fold[valid])))
        tau = threshold(inner)
        final_x = np.concatenate([x_by_fold[ff] for ff in trainfolds])
        final_y = np.concatenate([y_by_fold[ff] for ff in trainfolds])
        selected = best(actions[held], scores(fit(final_x, final_y), x_by_fold[held]))
        outer.append({"outer_fold_held_out":held,"train_folds":trainfolds,"inner_validation_folds":trainfolds,"threshold":tau,"inner_oof_query_count":len(inner)})
        for q in sorted((q for q,v in folds.items() if v == held), key=lambda x:int(x) if x.isdigit() else x):
            a, score = selected.get(q, (None, None)); take = a is not None and score >= tau; base = baseline[q]; final = list(base)
            if take: final[a["drop_rank"]-1] = a["incoming_doc_id"]
            if len(final) != 5 or len(set(final)) != 5: raise RuntimeError("CONTRACT_ERROR invalid final top5")
            br, pr, bp, pp = recall(gold[q], base), recall(gold[q], final), precision(gold[q], base), precision(gold[q], final)
            decisions.append({"query_id":q,"fold":held,"outer_fold_held_out":held,"k77_action_space_size":query_action_counts[held][q],"selected_action":None if not take else {"incoming_doc_id":a["incoming_doc_id"],"drop_rank":a["drop_rank"],"dropped_doc_id":a["dropped_doc_id"]},"policy_score_of_selected_action":score,"no_op_threshold_used":tau,"baseline_top5":base,"final_top5":final,"relevant_count":len(gold[q]),"baseline_recall":br,"policy_recall":pr,"recall_delta":pr-br,"baseline_precision":bp,"policy_precision":pp,"precision_delta":pp-bp,"selected_action_class":a["label"] if take else "NO_OP"})
    with DECISIONS.open("w", encoding="utf-8", newline="\n") as h:
        for x in decisions: h.write(json.dumps(x, ensure_ascii=False) + "\n")
    # Re-read serialized OOF decisions: all reported metrics below are sourced from this file.
    rows = [json.loads(x) for x in DECISIONS.open(encoding="utf-8") if x.strip()]
    if len(rows) != 5600 or len({x["query_id"] for x in rows}) != 5600: raise RuntimeError("CONTRACT_ERROR OOF serialization")
    def metrics(xs):
        n=len(xs); bmr=sum(x["baseline_recall"] for x in xs)/n; pmr=sum(x["policy_recall"] for x in xs)/n; bmp=sum(x["baseline_precision"] for x in xs)/n; pmp=sum(x["policy_precision"] for x in xs)/n
        counts=Counter(x["selected_action_class"] for x in xs); return {"query_count":n,"baseline_macro_recall":bmr,"policy_macro_recall":pmr,"D77_policy_gain":pmr-bmr,"baseline_macro_precision":bmp,"policy_macro_precision":pmp,"precision_delta":pmp-bmp,"queries_selecting_action":n-counts["NO_OP"],"queries_no_op":counts["NO_OP"],"selected_beneficial":counts["BENEFICIAL"],"selected_neutral":counts["NEUTRAL"],"selected_harmful":counts["HARMFUL"],"policy_positive_gain_queries":sum(x["recall_delta"]>1e-12 for x in xs),"policy_harmful_gain_queries":sum(x["recall_delta"]<-1e-12 for x in xs)}
    pooled=metrics(rows); pooled.update({"canonical_C77_oracle_gain":C77,"decision_policy_gap_C_minus_D":C77-pooled["D77_policy_gain"],"realization_ratio_D_over_C":pooled["D77_policy_gain"]/C77,"precision_guardrail_pass":pooled["precision_delta"] >= -0.5*pooled["D77_policy_gain"],"oracle_positive_queries":426,"false_positive_selected_queries":pooled["selected_neutral"]+pooled["selected_harmful"]})
    per={}
    for fold in range(1,5):
        x=metrics([z for z in rows if z["fold"]==fold]); d=x["D77_policy_gain"]; x.update({"canonical_C77_fold":C77_FOLD[fold],"decision_policy_gap_fold":C77_FOLD[fold]-d,"realization_ratio_fold":d/C77_FOLD[fold]}); per[str(fold)]=x
    gate = pooled["D77_policy_gain"] > 0 and all(x["D77_policy_gain"] >= 0 for x in per.values()) and pooled["precision_guardrail_pass"]
    checks={"step1f_k77_verified":True,"step1b_r77_verified":True,"all_queries_covered":True,"four_outer_runs":len(outer)==4,"outer_test_disjoint_from_train":True,"thresholds_inner_only":True,"feature_audit_clean":all(not x["gold_derived"] for x in audit if x["allowed"]),"k77_candidate_count":incoming_count==403322,"k77_action_count":sum(len(x) for x in actions.values())==806644,"oof_decision_rows_5600":len(rows)==5600,"oof_query_ids_unique":len({x["query_id"] for x in rows})==5600,"final_top5_valid":all(len(x["final_top5"])==5 and len(set(x["final_top5"]))==5 for x in rows),"selected_drop_ranks_valid":all(x["selected_action"] is None or x["selected_action"]["drop_rank"] in (4,5) for x in rows),"D_recomputed_from_serialized_decisions":True,"D_le_C":pooled["D77_policy_gain"] <= C77+1e-12,"fold0_payload_not_materialized":True}
    status="PASS" if all(checks.values()) else "CONTRACT_ERROR"
    report={"status":status,"experiment":"STEP 2 — Strict-OOF Policy Realizability Test at Fixed Research K=77","adopted_research_K":77,"production_K_selected":False,"official_primary_metric":"macro_recall","official_secondary_metric":"macro_precision","folds_used":[1,2,3,4],"fold0_touched":False,"fold0_payload_materialized":False,"fold0_used_in_training":False,"fold0_used_in_threshold_selection":False,"fold0_used_in_evaluation":False,"fold0_labels_used":False,"public_labels_used":False,"total_queries":5600,"k77_contract":{"incoming_candidates":incoming_count,"actions":sum(len(x) for x in actions.values()),"verified":checks["k77_candidate_count"] and checks["k77_action_count"]},"model_contract_path":"reports/task1/step2_k77_model_contract.json","policy_family":"HistGradientBoostingClassifier","outer_runs":outer,"pooled":pooled,"per_fold":per,"bootstrap":{"status":"NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL","protocol_source":None,"result":None,"used_in_gate":False},"scientific_gate":{"pass_requires_pooled_D_gt_0":True,"pass_requires_all_fold_D_ge_0":True,"precision_rule":"precision_delta >= -0.5 * D77","result":"PASS" if gate else "FAIL"},"sanity_checks":checks,"workflow_document_updated":False,"notes":["No compatible Task1 query-level bootstrap protocol was found in scripts/tests/docs; none was invented.","Contract was written before the first model.fit.","All reported metrics were recomputed by rereading the serialized OOF decisions."]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"technical_status":status,"scientific_gate":report["scientific_gate"]["result"],"feature_count":len(columns),"pooled":pooled,"per_fold_D77":{k:v["D77_policy_gain"] for k,v in per.items()},"outputs":[str(CONTRACT.relative_to(ROOT)),str(DECISIONS.relative_to(ROOT)),str(REPORT.relative_to(ROOT))]}, indent=2))
if __name__ == "__main__": main()
