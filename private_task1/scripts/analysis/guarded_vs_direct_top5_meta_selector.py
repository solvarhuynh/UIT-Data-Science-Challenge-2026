"""CPU-only, strict nested meta-selector: Guarded V2 versus Direct Top5.

The incumbent 09205 submission is read-only.  Private overlay construction is
intentionally unreachable unless the predeclared F1--F4 promotion gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.guarded_direct_k20_residual import (  # noqa: E402
    TARGET_FOLDS, build_direct_inputs, build_private_inputs, candidate_metadata,
    direct_outer_oof, fit_direct_model, load_and_merge_scores, load_fold_map,
    load_gold, load_pv1_metadata, load_pv1_worklist, metric, qkey,
    score_query_ids,
)

OUT = ROOT / "private_task1/experiments/guarded_vs_direct_top5_meta_selector"
TEAM = ROOT / "private_task1/submissions/team_09205"
INCUMBENT = TEAM / "submission_private_constrained_dual_anchor_rrf_09205.zip"
GUARDED_PRIVATE = ROOT / "private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.json"
DIRECT_OOF = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl"
GUARDED_SWAPS = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_swaps.jsonl"
GUARDED_RESULT = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_result.json"
DIRECT_REPLAY = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_replay_summary.json"
PRIVATE_OFFICIAL = ROOT / "private_task1/input/private-official.json"

EXPECTED_SHA = "aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae"
EXPECTED_A_RECALL = 0.9300505952380952
EXPECTED_B_RECALL = 0.9293809523809524
FROZEN_QUANTILES = (0.90, 0.95, 0.975)
SEED = 20260923


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object JSONL row: {path}:{number}")
                yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def load_zip_submission(path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if members != ["submission.json"]:
            raise RuntimeError(f"unexpected ZIP members: {members}")
        raw = json.loads(archive.read("submission.json"))
    return {str(qid): [str(x) for x in row["answer"]] for qid, row in raw.items()}


def validate_submission(payload: dict[str, list[str]], expected_ids: set[str]) -> dict[str, Any]:
    nulls = sum(any(not str(doc).strip() for doc in docs) for docs in payload.values())
    duplicates = sum(len(docs) != 5 or len(set(docs)) != 5 for docs in payload.values())
    return {
        "queries": len(payload), "expected_queries": len(expected_ids),
        "missing": len(expected_ids - set(payload)), "extra": len(set(payload) - expected_ids),
        "duplicates": duplicates, "null": nulls,
        "pass": len(payload) == len(expected_ids) and not (expected_ids - set(payload))
        and not (set(payload) - expected_ids) and duplicates == 0 and nulls == 0,
    }


def load_guarded(a_rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result = {qid: list(row["baseline_top5"]) for qid, row in a_rows.items()}
    for row in rows(GUARDED_SWAPS):
        qid, pos = str(row["query_id"]), int(row["pos"])
        if result[qid][pos - 1] != str(row["dropped"]):
            raise RuntimeError(f"guarded swap lineage mismatch: {qid}")
        result[qid][pos - 1] = str(row["added"])
    if any(len(v) != 5 or len(set(v)) != 5 for v in result.values()):
        raise RuntimeError("guarded OOF Top5 invalid")
    return result


def per_query_hits(pred: list[str], answer: set[str]) -> int:
    return len(set(pred) & answer)


def class_of(a: list[str], b: list[str], answer: set[str]) -> str:
    delta = per_query_hits(b, answer) - per_query_hits(a, answer)
    return "B_BETTER" if delta > 0 else "A_BETTER" if delta < 0 else "SAME"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


FEATURE_NAMES = (
    "top5_set_overlap", "membership_changed_count", "a_top4_top5_direct_margin",
    "a_top5_best_outside_direct_margin", "b_rank5_score", "b_rank6_score",
    "b_rank5_rank6_margin", "b_top5_min_score", "b_top5_mean_score",
    "entered_candidate_rank_minus_dropped", "entered_union_rrf_minus_dropped",
    "entered_source_support_minus_dropped", "entered_rr_bm25_minus_dropped",
    "entered_rr_dense_minus_dropped", "entered_rr_knn_word_minus_dropped",
    "entered_bge_ft_minus_dropped", "entered_bge_base_minus_dropped",
    "low_margin", "high_disagreement", "strong_bge_reorder", "single_source_dominance",
)


def query_features(qid: str, a: list[str], b: list[str], scores: dict[str, float], meta: dict[str, dict[str, Any]], score_rows: dict[tuple[str, str], dict[str, Any]]) -> list[float]:
    order = sorted(scores, key=lambda did: (-float(scores[did]), did))
    rank = {did: i for i, did in enumerate(order, 1)}
    entered, dropped = list(set(b) - set(a)), list(set(a) - set(b))
    def direct(doc: str) -> float: return float(scores[doc])
    def support(doc: str) -> float: return float(meta[doc]["source_support"])
    def src(doc: str, name: str) -> float: return -float(meta[doc]["source_ranks"].get(name, 21))
    def bge(doc: str, name: str) -> float: return float(score_rows[(qid, doc)][name])
    def rrf(doc: str) -> float: return float(score_rows[(qid, doc)].get("rrf_score", 0.0))
    def diff(fn) -> float: return mean([fn(doc) for doc in entered]) - mean([fn(doc) for doc in dropped])
    b5, b6 = direct(order[4]), direct(order[5])
    a4, a5 = direct(a[3]), direct(a[4])
    best_outside = max((direct(doc) for doc in order if doc not in set(a)), default=direct(a[4]))
    mean_support = mean([support(doc) for doc in b])
    single_source_count = sum(support(doc) <= 1 for doc in b)
    max_movement = max(abs(rank[doc] - int(meta[doc]["candidate_rank"])) for doc in b)
    # Thresholds are the pre-existing F1--F4 diagnostic cutoffs, not labels.
    flags = (int(b5 - b6 <= 0.00119048), int(mean_support <= 1.6 or single_source_count >= 2), int(max_movement >= 17), int(single_source_count >= 2))
    return [
        len(set(a) & set(b)) / 5.0, float(len(set(a) ^ set(b)) // 2), a4 - a5,
        direct(a[4]) - best_outside, b5, b6, b5 - b6, min(direct(doc) for doc in b), mean([direct(doc) for doc in b]),
        diff(lambda doc: float(meta[doc]["candidate_rank"])), diff(rrf), diff(support),
        diff(lambda doc: src(doc, "bm25")), diff(lambda doc: src(doc, "dense")), diff(lambda doc: src(doc, "knn_word")),
        diff(lambda doc: bge(doc, "bge_ft_score")), diff(lambda doc: bge(doc, "bge_base_score")), *flags,
    ]


def eligible(a: list[str], b: list[str]) -> bool:
    return a[:3] == b[:3] and len(set(a) ^ set(b)) // 2 <= 2


def make_selector() -> Any:
    # Current scikit-learn selects multinomial automatically for lbfgs.  Its
    # explicit ``multi_class`` argument was removed in the installed release.
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", random_state=SEED,
        solver="lbfgs", max_iter=500,
    ))


def utility(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    columns = {str(label): index for index, label in enumerate(model.classes_)}
    return probabilities[:, columns.get("B_BETTER", -1)] - 2.0 * probabilities[:, columns.get("A_BETTER", -1)]


def apply_actions(ids: Iterable[str], a: dict[str, list[str]], b: dict[str, list[str]], utilities: dict[str, float], threshold: float | None) -> dict[str, list[str]]:
    result = {}
    for qid in ids:
        take = threshold is not None and utilities.get(qid, -math.inf) >= threshold and eligible(a[qid], b[qid])
        result[qid] = list(b[qid] if take else a[qid])
    return result


def summary_vs_a(pred: dict[str, list[str]], a: dict[str, list[str]], gold: dict[str, dict[str, Any]], folds: dict[str, int]) -> dict[str, Any]:
    overall_a, overall_p = metric(a, gold), metric(pred, gold)
    result: dict[str, Any] = {"recall": float(overall_p["recall"]), "precision": float(overall_p["precision"]), "recall_delta": float(overall_p["recall"] - overall_a["recall"]), "precision_delta": float(overall_p["precision"] - overall_a["precision"]), "folds": {}}
    for fold in sorted(set(folds.values())):
        ids = [qid for qid in pred if folds[qid] == fold]
        am, pm = metric({q: a[q] for q in ids}, {q: gold[q] for q in ids}), metric({q: pred[q] for q in ids}, {q: gold[q] for q in ids})
        result["folds"][f"F{fold}"] = {"recall_delta": float(pm["recall"] - am["recall"]), "precision_delta": float(pm["precision"] - am["precision"])}
    changed = [q for q in pred if pred[q] != a[q]]
    result.update({"changed": len(changed), "improved": sum(per_query_hits(pred[q], set(map(str, gold[q].get("answer", [])))) > per_query_hits(a[q], set(map(str, gold[q].get("answer", [])))) for q in changed), "harmed": sum(per_query_hits(pred[q], set(map(str, gold[q].get("answer", [])))) < per_query_hits(a[q], set(map(str, gold[q].get("answer", [])))) for q in changed), "neutral_changed": 0, "top1_3_changed": sum(pred[q][:3] != a[q][:3] for q in pred)})
    result["neutral_changed"] = result["changed"] - result["improved"] - result["harmed"]
    return result


def select_threshold(train_ids: list[str], inner_u: dict[str, float], a: dict[str, list[str]], b: dict[str, list[str]], gold: dict[str, dict[str, Any]], folds: dict[str, int]) -> dict[str, Any]:
    values = np.asarray([inner_u[q] for q in train_ids if eligible(a[q], b[q])], dtype=float)
    if not len(values): return {"status": "NO_ELIGIBLE"}
    candidates = []
    for q in FROZEN_QUANTILES:
        threshold = float(np.quantile(values, q))
        pred = apply_actions(train_ids, a, b, inner_u, threshold)
        report = summary_vs_a(pred, {x: a[x] for x in train_ids}, {x: gold[x] for x in train_ids}, {x: folds[x] for x in train_ids})
        stable = all(v["recall_delta"] >= 0 for v in report["folds"].values())
        candidates.append({"quantile": q, "threshold": threshold, "stable": stable, **report})
    valid = [x for x in candidates if x["stable"]]
    if not valid: return {"status": "NO_STABLE_THRESHOLD", "candidates": candidates}
    best = sorted(valid, key=lambda x: (-x["recall_delta"], x["harmed"], -x["threshold"]))[0]
    return {"status": "SELECTED", "selected_quantile": best["quantile"], "selected_threshold": best["threshold"], "candidates": candidates}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    official = json.loads(PRIVATE_OFFICIAL.read_text(encoding="utf-8"))
    incumbent = load_zip_submission(INCUMBENT)
    stage0 = {"sha256": sha256(INCUMBENT), "expected_sha256": EXPECTED_SHA, "validator": validate_submission(incumbent, set(map(str, official)))}
    if stage0["sha256"] != EXPECTED_SHA or not stage0["validator"]["pass"]: raise RuntimeError("incumbent freeze gate failed")

    all_folds = load_fold_map(); folds = {q: f for q, f in all_folds.items() if f in TARGET_FOLDS}
    gold_all = load_gold(); gold = {q: gold_all[q] for q in folds}
    _cr, _md, docs = load_pv1_metadata(); _wr, work = load_pv1_worklist(); reconstructed, _merge = load_and_merge_scores(work)
    score_rows = {(str(r["query_id"]), str(r["document_id"])): r for r in reconstructed}
    metadata = candidate_metadata(docs, score_rows); _src, _ext, _names, cache = build_direct_inputs(docs, score_rows)
    direct_scores, direct_rankings, _reports = direct_outer_oof(folds, gold, cache)
    saved = {str(r["query_id"]): r for r in rows(DIRECT_OOF)}
    if set(saved) != set(folds) or any(saved[q]["direct_top5"] != direct_rankings[q][:5] for q in folds): raise RuntimeError("Expert B exact reproduction failed")
    a = load_guarded(saved); b = {q: list(saved[q]["direct_top5"]) for q in saved}
    a_metric, b_metric = metric(a, gold), metric(b, gold)
    if abs(float(a_metric["recall"]) - EXPECTED_A_RECALL) > 1e-14 or abs(float(b_metric["recall"]) - EXPECTED_B_RECALL) > 1e-14: raise RuntimeError("Expert A/B metric replay failed")

    classes = {q: class_of(a[q], b[q], set(map(str, gold[q].get("answer", [])))) for q in folds}
    oracle = {q: b[q] if classes[q] == "B_BETTER" else a[q] for q in folds}
    oracle_report = summary_vs_a(oracle, a, gold, folds)
    anatomical = {"B_BETTER": sum(v == "B_BETTER" for v in classes.values()), "A_BETTER": sum(v == "A_BETTER" for v in classes.values()), "SAME": sum(v == "SAME" for v in classes.values()), "membership_changed": sum(set(a[q]) != set(b[q]) for q in folds), "oracle": oracle_report}
    write_jsonl(OUT / "expert_a_oof.jsonl", ({"query_id": q, "fold": folds[q], "top5": a[q]} for q in sorted(folds, key=qkey)))
    write_jsonl(OUT / "expert_b_oof.jsonl", ({"query_id": q, "fold": folds[q], "top5": b[q], "direct_scores": direct_scores[q]} for q in sorted(folds, key=qkey)))
    if oracle_report["recall_delta"] < 0.002:
        report = {"status": "CLOSE_META_SELECTOR", "stage0": stage0, "expert_a_recall": a_metric["recall"], "expert_b_recall": b_metric["recall"], "anatomy": anatomical, "reason": "oracle delta below predeclared +0.002", "gpu_runs": 0, "modal_gpu_runs": 0}
        write_json(OUT / "meta_selector_report.json", report); return 0

    data_ids = [q for q in sorted(folds, key=qkey) if set(a[q]) != set(b[q])]
    x = {q: query_features(q, a[q], b[q], direct_scores[q], metadata[q], score_rows) for q in data_ids}
    y = {q: classes[q] for q in data_ids}
    outer_pred: dict[str, list[str]] = {q: list(a[q]) for q in folds}; meta_rows = []; selections = {}
    for outer in sorted(set(folds.values())):
        train_ids = [q for q in data_ids if folds[q] != outer]; test_ids = [q for q in data_ids if folds[q] == outer]
        inner_u: dict[str, float] = {}
        for inner in sorted(set(folds.values()) - {outer}):
            hold = [q for q in train_ids if folds[q] == inner]; fit = [q for q in train_ids if folds[q] != inner]
            model = make_selector(); model.fit(np.asarray([x[q] for q in fit]), [y[q] for q in fit]); values = utility(model, np.asarray([x[q] for q in hold])); inner_u.update(dict(zip(hold, map(float, values))))
        selected = select_threshold(train_ids, inner_u, a, b, gold, folds); selections[f"F{outer}"] = selected
        if selected["status"] == "SELECTED":
            model = make_selector(); model.fit(np.asarray([x[q] for q in train_ids]), [y[q] for q in train_ids]); values = utility(model, np.asarray([x[q] for q in test_ids])); test_u = dict(zip(test_ids, map(float, values))); chosen = apply_actions(test_ids, a, b, test_u, selected["selected_threshold"]); outer_pred.update(chosen)
            for q in test_ids: meta_rows.append({"query_id": q, "fold": outer, "target": y[q], "utility": test_u[q], "threshold": selected["selected_threshold"], "take_b": chosen[q] == b[q], "eligible": eligible(a[q], b[q])})
        else:
            for q in test_ids: meta_rows.append({"query_id": q, "fold": outer, "target": y[q], "utility": None, "threshold": None, "take_b": False, "eligible": eligible(a[q], b[q])})
    meta_report = summary_vs_a(outer_pred, a, gold, folds)
    gate = meta_report["recall_delta"] >= 0.0015 and all(v["recall_delta"] >= 0 for v in meta_report["folds"].values()) and sum(v["recall_delta"] > 0 for v in meta_report["folds"].values()) >= 3 and meta_report["precision_delta"] >= -0.0002 and meta_report["improved"] > meta_report["harmed"] and meta_report["top1_3_changed"] == 0
    write_jsonl(OUT / "meta_oof_predictions.jsonl", meta_rows)
    report = {"status": "META_GATE_PASS" if gate else "CLOSE_META_SELECTOR", "stage0": stage0, "expert_a_recall": a_metric["recall"], "expert_b_recall": b_metric["recall"], "anatomy": anatomical, "thresholds_by_outer_fold": selections, "meta_oof": meta_report, "meta_gate": "PASS" if gate else "FAIL", "private_application": "NOT_RUN" if not gate else "PENDING_FINAL_FIT", "fold0_used": False, "public_labels_used": False, "private_labels_used": False, "gpu_runs": 0, "modal_gpu_runs": 0}
    write_json(OUT / "meta_selector_report.json", report)
    write_json(OUT / "meta_model_manifest.json", {"family": "StandardScaler+multinomial LogisticRegression", "random_state": SEED, "utility": "P_BETTER - 2.0*P_A_BETTER", "quantiles": list(FROZEN_QUANTILES), "features": list(FEATURE_NAMES), "final_fit": "NOT_RUN_UNLESS_META_GATE_PASS" if not gate else "PENDING"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
