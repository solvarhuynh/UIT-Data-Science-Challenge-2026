"""Last-day CPU-only direct document listwise Top5 experiment.

The contract is written before the first model fit.  The candidate universe is
current PV1 K20 union the frozen incumbent Top5; labels are used only as
F1--F4 targets/evaluation, never to construct membership or features.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analysis.phase1_step4_recovery import load_fold_map, load_gold, qkey

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

OUT = ROOT / "private_task1/experiments/direct_k20_listwise_top5"
K20_VAL = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
K20_VAL_WORK = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
BGE_VAL = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
K20_PRIV = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
BGE_PRIV = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
INC_OOF = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl"
INC_SWAPS = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_swaps.jsonl"
INC_NESTED = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_result.json"
INC_PRIVATE = ROOT / "private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.json"

VAL_UNIVERSE = OUT / "validation_candidate_universe.jsonl"
CONTRACT = OUT / "contract.json"
FEATURE_MANIFEST = OUT / "feature_manifest.json"
RUNTIME_REPORT = OUT / "runtime_report.json"
OOF = OUT / "oof_predictions.jsonl"
PHASE_REPORT = OUT / "phase_report.json"
PHASE_MD = OUT / "phase_report.md"
PRIVATE_PRED = OUT / "private_predictions.jsonl"
FINAL_REPORT = OUT / "final_report.json"
FINAL_MD = OUT / "final_report.md"

EXPECTED_K20_SHA = "5bb1f804b629a65611ee0f9e6b11c4b95026a42c3303b395b3dbc0d92ab8a2e2"
EXPECTED_INC_RECALL = 0.9300505952380952
SEED = 20260827
FEATURES = (
    "current_rank", "current_rank_missing", "reciprocal_current_rank",
    "dense_rank", "dense_missing", "bm25_rank", "bm25_missing",
    "knn_word_rank", "knn_word_missing", "source_support_count", "rrf_score",
    "bge_ft_score", "bge_ft_missing", "bge_ft_rank", "bge_base_score",
    "bge_base_missing", "incumbent_membership", "incumbent_rank",
    "incumbent_rank_missing",
)
HP = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [5],
    "label_gain": [0, 1], "learning_rate": 0.05, "n_estimators": 200,
    "num_leaves": 31, "max_depth": 5, "min_child_samples": 20,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "feature_fraction": 1.0,
    "bagging_fraction": 1.0, "bagging_freq": 0, "deterministic": True,
    "force_col_wise": True, "verbosity": -1, "random_state": SEED,
    "n_jobs": 1,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object JSONL row: {path}:{no}")
                yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def doc_key(value: str):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def load_candidate_file(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows(path):
        qid = str(row["query_id"])
        if qid in result:
            raise RuntimeError(f"duplicate query candidate row: {path}:{qid}")
        docs = {}
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 20:
            raise RuntimeError(f"not K20: {path}:{qid}")
        for item in candidates:
            did = str(item["document_id"])
            if did in docs:
                raise RuntimeError(f"duplicate candidate identity: {(qid, did)}")
            source = {str(k): int(v) for k, v in (item.get("source_ranks") or {}).items()}
            docs[did] = {
                "document_id": did, "current_rank": int(item["candidate_rank"]),
                "source_ranks": source, "rrf_score": float(item.get("rrf_score", sum(1.0 / (60 + v) for v in source.values()))),
            }
        result[qid] = docs
    if any(len(docs) != 20 for docs in result.values()):
        raise RuntimeError(f"K20 candidate cardinality failure: {path}")
    return result


def load_bge(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    result = {}
    for row in rows(path):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in result:
            raise RuntimeError(f"duplicate BGE score: {path}:{key}")
        ft, base = float(row["bge_ft_score"]), float(row["bge_base_score"])
        if not (math.isfinite(ft) and math.isfinite(base)):
            raise RuntimeError(f"nonfinite BGE score: {key}")
        result[key] = {"bge_ft_score": ft, "bge_base_score": base}
    return result


def load_incumbent_val(folds: dict[str, int]) -> dict[str, list[str]]:
    result = {}
    for row in rows(INC_OOF):
        qid = str(row["query_id"])
        if qid in result or qid not in folds or int(row["fold"]) != folds[qid]:
            raise RuntimeError(f"invalid incumbent OOF identity: {qid}")
        result[qid] = [str(v) for v in row["baseline_top5"]]
    for row in rows(INC_SWAPS):
        qid, pos = str(row["query_id"]), int(row["pos"])
        if qid not in result or pos not in (4, 5):
            raise RuntimeError(f"invalid incumbent nested swap: {qid}")
        if result[qid][pos - 1] != str(row["dropped"]):
            raise RuntimeError(f"incumbent dropped identity mismatch: {qid}")
        result[qid][pos - 1] = str(row["added"])
    if set(result) != set(folds) or any(len(v) != 5 or len(set(v)) != 5 for v in result.values()):
        raise RuntimeError("incumbent validation Top5 is incomplete or duplicated")
    return result


def load_incumbent_private() -> dict[str, list[str]]:
    raw = json.loads(INC_PRIVATE.read_text(encoding="utf-8"))
    result = {str(q): [str(v) for v in item["answer"]] for q, item in raw.items()}
    if len(result) != 2080 or any(len(v) != 5 or len(set(v)) != 5 for v in result.values()):
        raise RuntimeError("Private incumbent is not 2080 unique Top5")
    return result


def make_universe(base: dict[str, dict[str, dict[str, Any]]], incumbent: dict[str, list[str]]) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    added = 0
    for qid, docs in base.items():
        merged = {did: dict(meta) for did, meta in docs.items()}
        for did in incumbent[qid]:
            if did not in merged:
                merged[did] = {"document_id": did, "current_rank": 21, "source_ranks": {}, "rrf_score": 0.0, "anchor_only": True}
                added += 1
        result[qid] = merged
    return result


def bge_ranks(qid: str, docs: dict[str, dict[str, Any]], bge: dict[tuple[str, str], dict[str, float]]) -> dict[str, int]:
    scored = [did for did in docs if (qid, did) in bge]
    ordered = sorted(scored, key=lambda did: (-bge[(qid, did)]["bge_ft_score"], -bge[(qid, did)]["bge_base_score"], docs[did]["current_rank"], doc_key(did)))
    return {did: i for i, did in enumerate(ordered, 1)}


def build_features(universe: dict[str, dict[str, dict[str, Any]]], incumbent: dict[str, list[str]], bge: dict[tuple[str, str], dict[str, float]]) -> dict[str, dict[str, list[float]]]:
    result = {}
    for qid, docs in universe.items():
        ranks = bge_ranks(qid, docs, bge)
        incumbent_rank = {did: i for i, did in enumerate(incumbent[qid], 1)}
        qrows = {}
        for did, meta in docs.items():
            source = meta["source_ranks"]
            current = int(meta["current_rank"])
            current_missing = int(current > 20)
            def source_value(name: str):
                value = int(source.get(name, 1000))
                return value, int(name not in source)
            dense, dense_missing = source_value("dense")
            bm25, bm25_missing = source_value("bm25")
            knn, knn_missing = source_value("knn_word")
            if (qid, did) in bge:
                ft = bge[(qid, did)]["bge_ft_score"]
                base = bge[(qid, did)]["bge_base_score"]
                brank, bmissing = ranks[did], 0
            else:
                ft, base, brank, bmissing = 0.0, 0.0, 1000, 1
            irank = incumbent_rank.get(did, 0)
            qrows[did] = [
                float(current), float(current_missing), 1.0 / (1.0 + current),
                float(dense), float(dense_missing), float(bm25), float(bm25_missing),
                float(knn), float(knn_missing), float(len(source)), float(meta["rrf_score"]),
                ft, float(bmissing), float(brank), base, float(bmissing),
                float(did in incumbent_rank), float(irank or 0), float(not irank),
            ]
        result[qid] = qrows
    return result


def rank_predictions(qids, features, universe, incumbent, model):
    output = {}
    for qid in qids:
        docs = sorted(features[qid], key=doc_key)
        values = model.predict(np.asarray([features[qid][did] for did in docs], dtype=np.float32))
        scored = dict(zip(docs, map(float, values)))
        output[qid] = sorted(docs, key=lambda did: (-scored[did], -int(did in incumbent[qid]), incumbent[qid].index(did) if did in incumbent[qid] else 999, universe[qid][did]["current_rank"], doc_key(did)))
        if not np.isfinite(values).all() or len(output[qid][:5]) != 5 or len(set(output[qid][:5])) != 5:
            raise RuntimeError(f"invalid listwise prediction: {qid}")
    return {qid: value[:5] for qid, value in output.items()}


def matrices(qids, features, gold):
    x, y, groups = [], [], []
    for qid in sorted(qids, key=qkey):
        docs = sorted(features[qid], key=doc_key)
        wanted = {str(v) for v in gold[qid].get("answer", [])}
        x.extend(features[qid][did] for did in docs)
        y.extend(int(did in wanted) for did in docs)
        groups.append(len(docs))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), groups


def metrics(pred, gold):
    recalls, precisions = [], []
    for qid, answer in pred.items():
        record = gold[qid]
        wanted = set(str(v) for v in (record.get("answer", []) if isinstance(record, dict) else record))
        hits = len(wanted & set(answer[:5]))
        recalls.append(hits / len(wanted))
        precisions.append(hits / 5.0)
    return {"queries": len(pred), "recall": statistics.mean(recalls), "precision": statistics.mean(precisions)}


def mutation_stats(base, proposed, gold):
    stats = Counter()
    depths = Counter()
    for qid in base:
        if base[qid] == proposed[qid]:
            stats["unchanged"] += 1
            continue
        stats["changed"] += 1
        n = sum(a != b for a, b in zip(base[qid], proposed[qid]))
        stats[f"changed_{n}_docs"] += 1
        wanted = {str(v) for v in gold[qid].get("answer", [])}
        old_hits, new_hits = len(wanted & set(base[qid])), len(wanted & set(proposed[qid]))
        if new_hits > old_hits: stats["improved"] += 1
        elif new_hits < old_hits: stats["harmed"] += 1
        else: stats["neutral_changed"] += 1
        for did in set(proposed[qid]) - set(base[qid]):
            depths[str(did)] += 1
    return {"changed": stats["changed"], "unchanged": stats["unchanged"], "improved": stats["improved"], "harmed": stats["harmed"], "neutral_changed": stats["neutral_changed"], "net_relevant_document_change": stats["improved"] - stats["harmed"], "mutation_counts": {str(i): stats[f"changed_{i}_docs"] for i in range(1, 6)}}


def entered_rank_distribution(base, proposed, universe):
    values = Counter()
    for qid in base:
        for did in set(proposed[qid]) - set(base[qid]):
            rank = int(universe[qid][did]["current_rank"])
            values["6-20" if rank <= 20 else "incumbent_anchor_outside_k20"] += 1
    return dict(values)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes = {"validation_worklist": sha(K20_VAL_WORK), "validation_candidates": sha(K20_VAL), "private_candidates": sha(K20_PRIV), "validation_bge": sha(BGE_VAL), "private_bge": sha(BGE_PRIV), "incumbent_oof": sha(INC_OOF), "incumbent_swaps": sha(INC_SWAPS), "incumbent_nested": sha(INC_NESTED), "private_incumbent": sha(INC_PRIVATE)}
    if input_hashes["validation_worklist"] != EXPECTED_K20_SHA:
        raise RuntimeError("current validation worklist SHA mismatch")
    folds_all = load_fold_map()
    folds = {qid: fold for qid, fold in folds_all.items() if fold in (1, 2, 3, 4)}
    if Counter(folds.values()) != Counter({1: 1400, 2: 1400, 3: 1400, 4: 1400}):
        raise RuntimeError("F1-F4 fold population mismatch")
    gold_all = load_gold(); gold = {qid: gold_all[qid] for qid in folds}
    val_base = load_candidate_file(K20_VAL); priv_base = load_candidate_file(K20_PRIV)
    val_bge = load_bge(BGE_VAL); priv_bge = load_bge(BGE_PRIV)
    val_inc = load_incumbent_val(folds); priv_inc = load_incumbent_private()
    val_universe = make_universe(val_base, val_inc)
    priv_universe = make_universe(priv_base, priv_inc)
    anchors_added = sum(len(val_universe[q]) - 20 for q in val_universe)
    write_jsonl(VAL_UNIVERSE, ({"query_id": qid, "candidates": [{"document_id": did, **meta} for did, meta in sorted(docs.items(), key=lambda x: (x[1]["current_rank"], doc_key(x[0])))]} for qid, docs in sorted(val_universe.items(), key=lambda x: qkey(x[0]))))
    val_features = build_features(val_universe, val_inc, val_bge)
    priv_features = build_features(priv_universe, priv_inc, priv_bge)
    feature_manifest = {"features": list(FEATURES), "count": len(FEATURES), "feature_semantics": "fixed ranks/scores/missing flags from current artifacts; no Qwen/gold-derived features", "validation_candidate_rows": sum(len(v) for v in val_universe.values()), "private_candidate_rows": sum(len(v) for v in priv_universe.values()), "input_hashes": input_hashes, "fold0_used": False, "private_labels_used": False}
    write_json(FEATURE_MANIFEST, feature_manifest)
    contract = {"status": "FROZEN_PRE_FIT", "scientific_comparator": "GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED", "incumbent_recall": EXPECTED_INC_RECALL, "folds": {"F1": 1400, "F2": 1400, "F3": 1400, "F4": 1400, "Fold0": 0}, "candidate_universe": "CURRENT_PV1_K20 UNION incumbent V2_NESTED Top5; membership label-free", "feature_schema": list(FEATURES), "target": "document_id in F1-F4 gold answer", "estimator": "lightgbm.LGBMRanker LambdaRank/NDCG@5", "parameters": HP, "random_seed": SEED, "tie_break": ["higher score", "incumbent membership", "lower incumbent rank", "lower current rank", "stable document_id"], "evaluation_gate": {"all_fold_delta_gte": 0, "positive_folds": 3, "pooled_recall_delta": 0.0015, "pooled_precision_delta": -0.0002, "improved_gt_harmed": True}, "private_rule": "same feature schema and candidate construction; no labels; 5 direct documents", "input_hashes": input_hashes, "fold0_used": False, "private_labels_used": False}
    write_json(CONTRACT, contract)

    # Technical smoke, before any scientific fold fit.
    smoke_ids = sorted(folds, key=qkey)[:50]
    sx, sy, sg = matrices(smoke_ids, val_features, gold)
    smoke_model = LGBMRanker(**HP)
    smoke_start = time.perf_counter(); smoke_model.fit(sx, sy, group=sg, feature_name=list(FEATURES)); smoke_pred = smoke_model.predict(sx); smoke_seconds = time.perf_counter() - smoke_start
    if not np.isfinite(smoke_pred).all():
        raise RuntimeError("technical smoke produced nonfinite scores")
    write_json(RUNTIME_REPORT, {"technical_smoke": {"status": "PASS", "queries": len(smoke_ids), "rows": len(sx), "groups": sg, "fit_predict_seconds": smoke_seconds, "finite_scores": True}, "estimator": "lightgbm.LGBMRanker", "lightgbm": lightgbm.__version__, "n_jobs": HP["n_jobs"], "outer_folds_sequential": True})

    baseline = val_inc
    base_metric = metrics(baseline, gold)
    stored = json.loads(INC_NESTED.read_text(encoding="utf-8"))
    replay_pass = abs(base_metric["recall"] - EXPECTED_INC_RECALL) <= 1e-12 and abs(base_metric["recall"] - float(stored["proposed"]["recall"])) <= 1e-12
    if not replay_pass:
        raise RuntimeError(f"incumbent replay failed: {base_metric}")

    oof = {}; fold_reports = {}
    for held in (1, 2, 3, 4):
        train_ids = [qid for qid in folds if folds[qid] != held]; test_ids = [qid for qid in folds if folds[qid] == held]
        x, y, groups = matrices(train_ids, val_features, gold)
        model = LGBMRanker(**HP); model.fit(x, y, group=groups, feature_name=list(FEATURES))
        pred = rank_predictions(test_ids, val_features, val_universe, val_inc, model); oof.update(pred)
        fold_reports[f"F{held}"] = {"train_queries": len(train_ids), "train_rows": len(x), "test_queries": len(test_ids), "scores_finite": True}
    if set(oof) != set(baseline):
        raise RuntimeError("OOF query coverage mismatch")
    proposed_metric = metrics(oof, gold)
    per_fold = {}
    for fold in (1, 2, 3, 4):
        ids = [qid for qid in oof if folds[qid] == fold]
        fold_gold = {qid: gold[qid] for qid in ids}
        bm, pm = metrics({qid: baseline[qid] for qid in ids}, fold_gold), metrics({qid: oof[qid] for qid in ids}, fold_gold)
        per_fold[f"F{fold}"] = {"incumbent": bm, "listwise": pm, "recall_delta": pm["recall"] - bm["recall"], "precision_delta": pm["precision"] - bm["precision"]}
    stat = mutation_stats(baseline, oof, gold)
    stat["entered_document_current_rank_source"] = entered_rank_distribution(baseline, oof, val_universe)
    rd = proposed_metric["recall"] - base_metric["recall"]; pd = proposed_metric["precision"] - base_metric["precision"]
    oracle_values = []
    for qid, docs in val_universe.items():
        oracle_values.append(min(5, len(set(docs) & {str(v) for v in gold[qid].get("answer", [])})) / len(gold[qid].get("answer", [])))
    gate = replay_pass and all(v["recall_delta"] >= 0 for v in per_fold.values()) and sum(v["recall_delta"] > 0 for v in per_fold.values()) >= 3 and rd >= 0.0015 and pd >= -0.0002 and stat["improved"] > stat["harmed"]
    report = {"status": "DIRECT_K20_LISTWISE_TOP5_COMPLETE", "contract": rel(CONTRACT), "input_hashes": input_hashes, "candidate_universe": {"queries": len(val_universe), "rows": sum(len(v) for v in val_universe.values()), "min": min(map(len, val_universe.values())), "median": statistics.median(map(len, val_universe.values())), "max": max(map(len, val_universe.values())), "incumbent_anchors_added_outside_k20": anchors_added}, "technical_preflight": "PASS", "incumbent_replay_gate": "PASS", "incumbent": base_metric, "listwise": proposed_metric, "recall_delta": rd, "precision_delta": pd, "per_fold": per_fold, "mutations": stat, "candidate_oracle_recall": statistics.mean(oracle_values), "oof_predictions_sha256": None, "scientific_gate": "STRONG_PASS" if gate else "FAIL", "gpu_runs": 0, "modal_gpu_runs": 0}
    write_jsonl(OOF, ({"query_id": qid, "fold": folds[qid], "incumbent_top5": baseline[qid], "listwise_top5": oof[qid]} for qid in sorted(oof, key=qkey)))
    report["oof_predictions_sha256"] = sha(OOF)
    if not gate:
        report["private_final_fit"] = "NOT_RUN"; report["private_challenger"] = "NOT_CREATED"; report["direct_listwise_decision"] = "CLOSE"
        report["last_day_branch_audit"] = {"LEGAL_STRUCTURE_CITATION_RETRIEVAL": "highest remaining non-Qwen headroom candidate, but no execution authorized here", "UNBOUNDED_EXISTING_UNION": "artifact-ready but larger runtime/risk", "FULL_DOCUMENT_STRUCTURAL_RERANKING": "larger candidate/feature preparation risk", "SAME_BGE_HARD_NEGATIVE_FT_V2": "operationally heavy and not executable safely in last-day CPU-only window", "recommendation": "KEEP_GUARDED_DIRECT_V2_NESTED_INCUMBENT_AND_CLOSE_NEW_EXPERIMENTS"}
        write_json(PHASE_REPORT, report)
        PHASE_MD.write_text(f"# Direct listwise Top5\n\nGate: **FAIL**. Recall `{proposed_metric['recall']}`; delta `{rd}`; fold deltas `{ {k: v['recall_delta'] for k, v in per_fold.items()} }`; changed/improved/harmed `{stat['changed']}/{stat['improved']}/{stat['harmed']}`. Private fit was not run.\n", encoding="utf-8")
        print_summary(report, proposed_metric, per_fold, stat, oracle_values)
        return 0

    # Promotion path is reached only if all frozen gates pass.
    x, y, groups = matrices(list(folds), val_features, gold); final_model = LGBMRanker(**HP); final_model.fit(x, y, group=groups, feature_name=list(FEATURES))
    final_dir = OUT / "final_model"; final_dir.mkdir(parents=True, exist_ok=True); model_path = final_dir / "model.txt"; final_model.booster_.save_model(str(model_path)); write_json(final_dir / "manifest.json", {"model": rel(model_path), "model_sha256": sha(model_path), "parameters": HP, "features": list(FEATURES), "input_hashes": input_hashes, "fold0_used": False, "private_labels_used": False})
    private_pred = rank_predictions(list(priv_universe), priv_features, priv_universe, priv_inc, final_model); write_jsonl(PRIVATE_PRED, ({"query_id": qid, "answer": private_pred[qid], "incumbent_answer": priv_inc[qid]} for qid in sorted(private_pred, key=qkey)))
    private_stat = {"queries": len(private_pred), "changed": sum(private_pred[q] != priv_inc[q] for q in private_pred), "mutation_counts": dict(Counter(sum(a != b for a, b in zip(private_pred[q], priv_inc[q])) for q in private_pred))}
    report.update({"private_final_fit": "PASS", "private": private_stat, "private_challenger": "NOT_CREATED_UNTIL_VALIDATOR_IMPLEMENTATION"})
    write_json(FINAL_REPORT, report); FINAL_MD.write_text("# Direct listwise Top5\n\nScientific gate PASS; Private structural prediction prepared.\n", encoding="utf-8")
    print_summary(report, proposed_metric, per_fold, stat, oracle_values); return 0


def print_summary(report, proposed, per_fold, stat, oracle):
    print("BRANCH: DIRECT_K20_PLUS_ANCHORS_DOCUMENT_LISTWISE_TOP5")
    print("GPU_RUNS: 0\nMODAL_GPU_RUNS: 0")
    print(f"TECHNICAL_PREFLIGHT: {report['technical_preflight']}")
    print(f"VALIDATION_QUERIES: {report['candidate_universe']['queries']}/5600")
    print(f"CANDIDATE_ROWS: {report['candidate_universe']['rows']}")
    print(f"FEATURE_COUNT: {len(FEATURES)}")
    print("ESTIMATOR: lightgbm.LGBMRanker(objective=lambdarank, metric=ndcg, eval_at=5, n_jobs=1)")
    print(f"INCUMBENT_REPLAY_GATE: {report['incumbent_replay_gate']}")
    print(f"INCUMBENT_RECALL: {report['incumbent']['recall']}")
    print(f"LISTWISE_OOF_RECALL: {proposed['recall']}")
    print(f"LISTWISE_OOF_RECALL_DELTA: {report['recall_delta']}")
    print(f"LISTWISE_OOF_PRECISION: {proposed['precision']}")
    print(f"LISTWISE_OOF_PRECISION_DELTA: {report['precision_delta']}")
    for fold, value in per_fold.items(): print(f"{fold}_DELTA: {value['recall_delta']}")
    print(f"OOF_CHANGED: {stat['changed']}\nOOF_IMPROVED: {stat['improved']}\nOOF_HARMED: {stat['harmed']}")
    print(f"CANDIDATE_ORACLE_RECALL: {statistics.mean(oracle)}")
    print(f"SCIENTIFIC_GATE: {report['scientific_gate']}")
    if report['scientific_gate'] == 'FAIL':
        print("PRIVATE_FINAL_FIT: NOT_RUN\nPRIVATE_CHALLENGER: NOT_CREATED\nLAST_DAY_NEXT_BRANCH: KEEP_GUARDED_DIRECT_V2_NESTED_INCUMBENT_AND_CLOSE_NEW_EXPERIMENTS\nFINAL_STATUS: CLOSED_DIRECT_LISTWISE\nNEXT_ACTION: KEEP_EXISTING_GUARDED_DIRECT_SUBMISSION")
    else:
        print("FINAL_STATUS: READY_FOR_MANUAL_SUBMISSION_REVIEW\nNEXT_ACTION: MANUAL_REVIEW_BEFORE_SUBMISSION")


if __name__ == "__main__":
    raise SystemExit(main())
