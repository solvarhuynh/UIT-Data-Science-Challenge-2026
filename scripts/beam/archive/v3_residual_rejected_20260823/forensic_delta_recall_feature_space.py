"""CPU-only forensic over the exact Delta-Recall V1 residual feature space."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from .common import jsonl, sort_query_ids, write_jsonl
    from .forensic_stage1_hard_errors import fixed_stage1_config
    from .train_delta_recall_residual import FOLDS, TOP_K, build_residual_records, fixed_stage1_ranked, residual_feature_names, validate_actions
    from .train_residual_policy import feature_names
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from forensic_stage1_hard_errors import fixed_stage1_config
    from train_delta_recall_residual import FOLDS, TOP_K, build_residual_records, fixed_stage1_ranked, residual_feature_names, validate_actions
    from train_residual_policy import feature_names


EPS = 1e-12
WINDOWS: tuple[int | None, ...] = (1, 2, 3, 5, 10, 20, 50, None)
FORBIDDEN_OUTCOME_TOKENS = ("label", "gain", "gold", "delta_recall", "resulting_recall")


def action_threshold(v3b_report: Path) -> float | None:
    report = json.loads(Path(v3b_report).read_text(encoding="utf-8"))
    return report["outer_cv"]["best_v3b_raw"].get("action_threshold")


def delta_class(value: float) -> str:
    if value > EPS:
        return "POSITIVE"
    if value < -EPS:
        return "NEGATIVE"
    return "ZERO"


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype="float64")
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None, "sum": 0.0}
    return {"count": int(len(finite)), "mean": float(finite.mean()), "median": float(np.median(finite)), "p25": float(np.quantile(finite, .25)), "p75": float(np.quantile(finite, .75)), "min": float(finite.min()), "max": float(finite.max()), "sum": float(finite.sum())}


def roc_auc(positive: list[float], negative: list[float]) -> float | None:
    """Tie-aware AUC for finite diagnostic scores."""
    if not positive or not negative:
        return None
    pos = np.asarray(positive, dtype="float64"); neg = np.asarray(negative, dtype="float64")
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return None
    y_true = np.concatenate((np.ones(len(pos), dtype="int8"), np.zeros(len(neg), dtype="int8")))
    y_score = np.concatenate((pos, neg))
    return float(roc_auc_score(y_true, y_score))


def effect_size(positive: list[float], negative: list[float]) -> float | None:
    a = np.asarray(positive, dtype="float64"); b = np.asarray(negative, dtype="float64")
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def feature_families(name: str) -> list[str]:
    lower = name.lower(); tags = []
    if "neural" in lower: tags.append("NEURAL")
    if any(token in lower for token in ("bm25", "union", "source", "knn", "bge", "retrieval", "rank", "support")): tags.append("RETRIEVAL")
    if any(token in lower for token in ("stage1_score", "p_benefit", "p_harm", "stage1_rank")): tags.append("STAGE1")
    if any(token in lower for token in ("top1", "top2", "margin", "mean_stage1", "std_stage1", "median_stage1", "score_z", "percentile")): tags.append("QUERY_CONTEXT")
    if "candidate_minus_incumbent_" in lower: tags.append("RELATIONAL")
    return tags


def challenger_rows(records: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if any(any(token in name.lower() for token in FORBIDDEN_OUTCOME_TOKENS) for name in names):
        raise AssertionError("outcome field reached forensic feature schema")
    rows = []
    for record in records:
        if record["is_keep_stage1"]:
            continue
        if len(record["x"]) != len(names):
            raise AssertionError("residual vector does not match feature schema")
        rows.append({"query_id": str(record["query_id"]), "fold": int(record["fold"]), "stage1_rank": int(record["stage1_rank"]), "delta_recall": float(record["delta_recall"]), "delta_class": delta_class(float(record["delta_recall"])), "challenger_label": str(record["action"]["label"]), "incumbent_label": str(record["incumbent"]["label"]), "feature_dict": dict(zip(names, map(float, record["x"]))), "incoming_doc_id": str(record["action"]["incoming_doc_id"]), "drop_rank": int(record["action"]["drop_rank"])})
    return rows


def _comparison(rows: list[dict[str, Any]], feature: str, negative_classes: set[str]) -> dict[str, Any]:
    positive = [float(row["feature_dict"][feature]) for row in rows if row["delta_class"] == "POSITIVE"]
    negative = [float(row["feature_dict"][feature]) for row in rows if row["delta_class"] in negative_classes]
    auc = roc_auc(positive, negative)
    return {"positive": summary(positive), "nonpositive": summary(negative), "mean_difference": (float(np.mean(positive) - np.mean(negative)) if positive and negative else None), "standardized_effect_size": effect_size(positive, negative), "raw_auc": auc, "auc_separation": (max(auc, 1.0 - auc) if auc is not None else None)}


def _sentinel(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = np.asarray([float(row["feature_dict"][feature]) for row in rows], dtype="float64")
    finite = values[np.isfinite(values)]
    extreme = int(np.sum(np.abs(finite) >= 1e5))
    q25, q75 = (float(np.quantile(finite, .25)), float(np.quantile(finite, .75))) if len(finite) else (None, None)
    return {"finite_count": int(len(finite)), "missing_nonfinite_count": int(len(values) - len(finite)), "extreme_value_fraction": float(extreme / len(values)) if len(values) else 0.0, "robust_median": float(np.median(finite)) if len(finite) else None, "robust_iqr": (q75 - q25) if q25 is not None and q75 is not None else None, "sentinel_sensitive": bool(extreme or len(finite) != len(values))}


def _fold_stability(rows: list[dict[str, Any]], feature: str, negative_classes: set[str]) -> dict[str, Any]:
    per_fold = {}; signs = []; separations = []
    for fold in FOLDS:
        stats = _comparison([row for row in rows if row["fold"] == fold], feature, negative_classes)
        per_fold[str(fold)] = {"mean_difference": stats["mean_difference"], "raw_auc": stats["raw_auc"], "auc_separation": stats["auc_separation"]}
        if stats["mean_difference"] is not None and stats["mean_difference"] != 0:
            signs.append(1 if stats["mean_difference"] > 0 else -1)
        if stats["auc_separation"] is not None:
            separations.append(float(stats["auc_separation"]))
    return {"per_fold": per_fold, "sign_consistent_across_4_folds": len(signs) == 4 and len(set(signs)) == 1, "positive_sign_fraction": float(sum(sign > 0 for sign in signs) / len(signs)) if signs else None, "min_fold_auc_separation": min(separations) if separations else None, "mean_fold_auc_separation": float(np.mean(separations)) if separations else None}


def query_pairwise(rows: list[dict[str, Any]], feature: str, negative_classes: set[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["query_id"]].append(row)
    per_query = []; per_fold: dict[int, list[tuple[float, float]]] = defaultdict(list); greater_wins = lower_wins = pairs = 0
    for query_id, values in grouped.items():
        positive = [float(row["feature_dict"][feature]) for row in values if row["delta_class"] == "POSITIVE"]
        comparison = [float(row["feature_dict"][feature]) for row in values if row["delta_class"] in negative_classes]
        if not positive or not comparison:
            continue
        pair_values = [(a, b) for a in positive for b in comparison if np.isfinite(a) and np.isfinite(b)]
        if not pair_values:
            continue
        greater = sum(float(a > b) + .5 * float(a == b) for a, b in pair_values) / len(pair_values)
        lower = sum(float(a < b) + .5 * float(a == b) for a, b in pair_values) / len(pair_values)
        fold = int(values[0]["fold"]); per_query.append((greater, lower)); per_fold[fold].append((greater, lower)); greater_wins += greater * len(pair_values); lower_wins += lower * len(pair_values); pairs += len(pair_values)
    def pack(values: list[tuple[float, float]]) -> dict[str, float | int | None]:
        greater = float(np.mean([value[0] for value in values])) if values else None; lower = float(np.mean([value[1] for value in values])) if values else None
        return {"positive_feature_greater_pair_win_rate": greater, "positive_feature_lower_pair_win_rate": lower, "query_pairwise_separation": max(greater, lower) if greater is not None and lower is not None else None, "eligible_query_count": len(values)}
    macro = pack(per_query); micro_greater = greater_wins / pairs if pairs else None; micro_lower = lower_wins / pairs if pairs else None
    return {"micro": {"positive_feature_greater_pair_win_rate": micro_greater, "positive_feature_lower_pair_win_rate": micro_lower, "query_pairwise_separation": max(micro_greater, micro_lower) if micro_greater is not None and micro_lower is not None else None, "pair_count": pairs}, "macro": macro, "per_fold": {str(fold): pack(values) for fold, values in sorted(per_fold.items())}}


def cross_tabs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_label = Counter((row["delta_class"], row["challenger_label"]) for row in rows)
    triple = Counter((row["incumbent_label"], row["challenger_label"], row["delta_class"]) for row in rows)
    labels = ("BENEFIT", "NEUTRAL", "HARM")
    return {"delta_class_x_challenger_label": {f"{delta}|{label}": int(delta_label[(delta, label)]) for delta in ("POSITIVE", "ZERO", "NEGATIVE") for label in labels}, "incumbent_label_x_challenger_label_x_delta_class": {f"{incumbent}|{challenger}|{delta}": int(triple[(incumbent, challenger, delta)]) for incumbent in labels for challenger in labels for delta in ("POSITIVE", "ZERO", "NEGATIVE")}}


def delta_query_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["query_id"]].append(row)
    return {"queries_with_any_positive": sum(any(row["delta_class"] == "POSITIVE" for row in values) for values in grouped.values()), "queries_with_only_zero": sum(all(row["delta_class"] == "ZERO" for row in values) for values in grouped.values()), "queries_with_any_negative": sum(any(row["delta_class"] == "NEGATIVE" for row in values) for values in grouped.values())}


def per_fold_delta_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for fold in FOLDS:
        values = [row for row in rows if row["fold"] == fold]
        counts = Counter(row["delta_class"] for row in values)
        result[str(fold)] = {"delta_class_counts": {name: int(counts[name]) for name in ("POSITIVE", "ZERO", "NEGATIVE")}, "delta_query_counts": delta_query_counts(values), "delta_magnitude_summary": {"positive": summary(row["delta_recall"] for row in values if row["delta_class"] == "POSITIVE"), "negative": summary(row["delta_recall"] for row in values if row["delta_class"] == "NEGATIVE")}}
    return result


def _rank_bin(rank: int) -> str:
    if rank <= 5: return str(rank)
    if rank <= 10: return "6-10"
    if rank <= 20: return "11-20"
    if rank <= 50: return "21-50"
    return ">50"


def positive_rank_distribution(ranked: dict[str, dict[str, Any]], anchor_threshold: float | None) -> dict[str, Any]:
    first = Counter(); best = Counter(); examples = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]; incumbent = item["best"]; keep = float(incumbent["gain"]) if anchor_threshold is None or float(incumbent["stage1_score"]) >= float(anchor_threshold) else 0.0
        positive = [(rank, action, float(action["gain"]) - keep) for rank, action in enumerate(item["all"][1:], start=1) if float(action["gain"]) - keep > EPS]
        if not positive: continue
        first_rank = positive[0][0]; best_rank, best_action, best_delta = max(positive, key=lambda value: (value[2], -value[0]))
        first[_rank_bin(first_rank)] += 1; best[_rank_bin(best_rank)] += 1
        examples.append({"query_id": query_id, "fold": int(incumbent["fold"]), "first_positive_rank": first_rank, "best_delta_rank": best_rank, "best_delta": best_delta, "best_positive_doc_id": str(best_action["incoming_doc_id"])})
    return {"query_count": len(examples), "first_positive_challenger_rank": dict(first), "best_delta_challenger_rank": dict(best), "examples": examples}


def oracle_curve(ranked: dict[str, dict[str, Any]], anchor_threshold: float | None) -> dict[str, Any]:
    by_window = {}; previous = -float("inf")
    for window in WINDOWS:
        recalls = []; positive_queries = positive_actions = benefit_queries = 0
        for item in ranked.values():
            incumbent = item["best"]; anchor = anchor_threshold is None or float(incumbent["stage1_score"]) >= float(anchor_threshold); baseline = float(incumbent["baseline_recall"]); keep_gain = float(incumbent["gain"]) if anchor else 0.0
            candidates = item["all"][1:] if window is None else item["all"][1:window + 1]
            deltas = [float(action["gain"]) - keep_gain for action in candidates]
            recalls.append(baseline + max([keep_gain, *[float(action["gain"]) for action in candidates]]))
            positive_queries += int(any(delta > EPS for delta in deltas)); positive_actions += sum(delta > EPS for delta in deltas); benefit_queries += int(any(str(action["label"]) == "BENEFIT" for action in candidates))
        macro = float(np.mean(recalls)) if recalls else 0.0
        if macro + 1e-12 < previous: raise AssertionError("oracle curve is not monotonic")
        previous = macro; key = "ALL" if window is None else str(window)
        by_window[key] = {"oracle_macro_recall": macro, "queries_with_positive_opportunity": int(positive_queries), "positive_action_count": int(positive_actions), "queries_with_benefit_reachable": int(benefit_queries)}
    keep = float(np.mean([float(item["best"]["baseline_recall"]) + (float(item["best"]["gain"]) if anchor_threshold is None or float(item["best"]["stage1_score"]) >= float(anchor_threshold) else 0.0) for item in ranked.values()])) if ranked else 0.0
    for value in by_window.values(): value["delta_vs_KEEP_STAGE1"] = value["oracle_macro_recall"] - keep
    return {"fixed_v3b_keep_macro_recall": keep, "by_k": by_window}


def analyze(ranked: dict[str, dict[str, Any]], action_names: list[str], anchor_threshold: float | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records, names = build_residual_records(ranked, action_names, anchor_threshold, labeled=True)
    if names != residual_feature_names(action_names): raise AssertionError("forensic did not reuse exact residual schema")
    rows = challenger_rows(records, names)
    delta_counts = Counter(row["delta_class"] for row in rows); query_counts = delta_query_counts(rows)
    benefit_queries = {row["query_id"] for row in rows if row["challenger_label"] == "BENEFIT"}; positive_queries = {row["query_id"] for row in rows if row["delta_class"] == "POSITIVE"}
    comparisons = {"positive_vs_zero": {"ZERO"}, "positive_vs_negative": {"NEGATIVE"}, "positive_vs_nonpositive": {"ZERO", "NEGATIVE"}}
    rankings: dict[str, list[dict[str, Any]]] = {name: [] for name in comparisons}
    for feature in names:
        for comparison_name, negative_classes in comparisons.items():
            sentinel = _sentinel(rows, feature)
            stats = _comparison(rows, feature, negative_classes); stability = _fold_stability(rows, feature, negative_classes)
            pair = query_pairwise(rows, feature, negative_classes)
            rankings[comparison_name].append({"feature": feature, "families": feature_families(feature), "comparison": comparison_name, "sentinel": sentinel, "pooled": stats, "fold_stability": stability, "query_conditioned": pair})
    def ranking_key(row: dict[str, Any]) -> tuple[int, int, float, float, float, float]:
        stable = int(bool(row["fold_stability"]["sign_consistent_across_4_folds"]))
        min_fold = row["fold_stability"]["min_fold_auc_separation"] or -1.0
        macro_pair = row["query_conditioned"]["macro"]["query_pairwise_separation"] or -1.0
        pooled = row["pooled"]["auc_separation"] or -1.0
        effect = abs(row["pooled"]["standardized_effect_size"] or 0.0)
        not_sentinel = int(not row["sentinel"]["sentinel_sensitive"])
        return stable, min_fold, macro_pair, pooled, effect, not_sentinel
    for values in rankings.values(): values.sort(key=ranking_key, reverse=True)
    family_summary = {}
    for family in ("NEURAL", "RETRIEVAL", "STAGE1", "QUERY_CONTEXT", "RELATIONAL"):
        values = [row for row in rankings["positive_vs_zero"] if family in row["families"] and not row["sentinel"]["sentinel_sensitive"]]
        family_summary[family] = {"feature_count": len(values), "sign_consistent_feature_count": sum(bool(row["fold_stability"]["sign_consistent_across_4_folds"]) for row in values), "best_pooled_positive_vs_zero_separation": max((row["pooled"]["auc_separation"] or 0.0 for row in values), default=None), "best_min_fold_separation": max((row["fold_stability"]["min_fold_auc_separation"] or 0.0 for row in values), default=None), "best_query_conditioned_separation": max((row["query_conditioned"]["macro"]["query_pairwise_separation"] or 0.0 for row in values), default=None)}
    curve = oracle_curve(ranked, anchor_threshold); rank_distribution = positive_rank_distribution(ranked, anchor_threshold)
    tabs = cross_tabs(rows)
    report = {"status": "DELTA_RECALL_FEATURE_FORENSIC_V2_COMPLETE", "folds": list(FOLDS), "query_count": len(ranked), "anchor_threshold": anchor_threshold, "anchor_semantics": "execute Stage1 incumbent iff stage1_score >= fixed V3B action threshold", "fixed_v3b_keep_macro_recall": curve["fixed_v3b_keep_macro_recall"], "delta_class_counts": {name: int(delta_counts[name]) for name in ("POSITIVE", "ZERO", "NEGATIVE")}, "delta_class_counts_by_fold": per_fold_delta_report(rows), "delta_query_counts": query_counts, "delta_magnitude_summary": {"positive": summary(row["delta_recall"] for row in rows if row["delta_class"] == "POSITIVE"), "negative": summary(row["delta_recall"] for row in rows if row["delta_class"] == "NEGATIVE")}, "benefit_delta_cross_tab": tabs["delta_class_x_challenger_label"], "incumbent_challenger_delta_cross_tab": tabs["incumbent_label_x_challenger_label_x_delta_class"], "benefit_query_overlap": {"queries_with_any_BENEFIT": len(benefit_queries), "queries_with_any_positive_delta": len(positive_queries), "queries_with_both": len(benefit_queries & positive_queries), "BENEFIT_without_positive_delta": len(benefit_queries - positive_queries), "positive_delta_without_BENEFIT": len(positive_queries - benefit_queries)}, "oracle_curve_by_k": curve["by_k"], "benefit_reachability_by_k": {key: value["queries_with_benefit_reachable"] for key, value in curve["by_k"].items()}, "positive_rank_distribution": {key: value for key, value in rank_distribution.items() if key != "examples"}, "feature_schema": names, "feature_rankings": {name: values[:50] for name, values in rankings.items()}, "feature_family_summary": family_summary, "query_conditioned_feature_rankings": rankings["positive_vs_zero"][:50], "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True, "descriptive_only": True}
    return report, [*rankings["positive_vs_zero"], *rankings["positive_vs_negative"], *rankings["positive_vs_nonpositive"]], rank_distribution["examples"]


def self_test() -> dict[str, bool]:
    def action(query: str, fold: int, doc: str, label: str, gain: float, score: float, feature: float) -> dict[str, Any]:
        return {"query_id": query, "fold": fold, "incoming_doc_id": doc, "drop_rank": 5, "baseline_recall": .5, "gain": gain, "label": label, "stage1_score": score, "stage1_p_benefit": max(score, 0.0), "stage1_p_harm": 0.0, "features": {"diff_neural_mean": feature}}
    incumbent = action("q", 1, "inc", "BENEFIT", .5, .9, 2.0); positive = action("q", 1, "pos", "BENEFIT", 1.0, .8, 3.0); zero = action("q", 1, "zero", "NEUTRAL", .5, .7, 2.0); negative = action("q", 1, "neg", "HARM", 0.0, .6, 1.0)
    ranked = {"q": {"best": incumbent, "second": positive, "all": [incumbent, positive, zero, negative]}}
    records, names = build_residual_records(ranked, ["diff_neural_mean"], .9, labeled=True); rows = challenger_rows(records, names)
    fold0_rejected = False
    try: validate_actions([{"query_id": "bad", "fold": 0}])
    except ValueError: fold0_rejected = True
    exact_schema = all(len(record["x"]) == len(names) for record in records) and len(names) == len(residual_feature_names(["diff_neural_mean"]))
    no_outcomes = not any(any(token in name.lower() for token in FORBIDDEN_OUTCOME_TOKENS) for name in names)
    classes = [delta_class(.5), delta_class(0.0), delta_class(-.5)] == ["POSITIVE", "ZERO", "NEGATIVE"]
    cross_total = sum(cross_tabs(rows)["delta_class_x_challenger_label"].values()) == len(rows)
    pair_rows = [{"query_id": "p", "fold": 1, "delta_class": "POSITIVE", "feature_dict": {"x": 3.0}}, {"query_id": "p", "fold": 1, "delta_class": "ZERO", "feature_dict": {"x": 2.0}}, {"query_id": "p", "fold": 1, "delta_class": "NEGATIVE", "feature_dict": {"x": 100.0}}]
    reverse_rows = [{"query_id": "r", "fold": 1, "delta_class": "POSITIVE", "feature_dict": {"x": 1.0}}, {"query_id": "r", "fold": 1, "delta_class": "ZERO", "feature_dict": {"x": 2.0}}, {"query_id": "r", "fold": 1, "delta_class": "NEGATIVE", "feature_dict": {"x": 3.0}}]
    scoped_zero = query_pairwise(pair_rows, "x", {"ZERO"})
    scoped_negative = query_pairwise(pair_rows, "x", {"NEGATIVE"})
    scoped_nonpositive = query_pairwise(pair_rows, "x", {"ZERO", "NEGATIVE"})
    pair_greater = scoped_zero["macro"]["positive_feature_greater_pair_win_rate"] == 1.0 and scoped_zero["macro"]["query_pairwise_separation"] == 1.0
    pair_lower = query_pairwise(reverse_rows, "x", {"ZERO", "NEGATIVE"})["macro"]["positive_feature_lower_pair_win_rate"] == 1.0
    pairwise_scoped = scoped_negative["macro"]["positive_feature_lower_pair_win_rate"] == 1.0 and scoped_negative["macro"]["query_pairwise_separation"] == 1.0 and scoped_nonpositive["macro"]["query_pairwise_separation"] != scoped_zero["macro"]["query_pairwise_separation"] and scoped_nonpositive["macro"]["query_pairwise_separation"] != scoped_negative["macro"]["query_pairwise_separation"]
    auc_cases = roc_auc([2, 3], [0, 1]) == 1.0 and roc_auc([0, 1], [2, 3]) == 0.0 and abs(float(roc_auc([1], [1])) - .5) < 1e-12
    auc_nonfinite = abs(float(roc_auc([2, float("nan"), float("inf")], [0, 1, float("-inf")])) - 1.0) < 1e-12
    curve = oracle_curve(ranked, .9); values = [row["oracle_macro_recall"] for row in curve["by_k"].values()]
    oracle_monotonic = all(right + 1e-12 >= left for left, right in zip(values, values[1:])); keep_correct = curve["fixed_v3b_keep_macro_recall"] == 1.0; increasing_k_safe = values[-1] + 1e-12 >= values[0]
    sentinel = _sentinel([{**rows[0], "feature_dict": {**rows[0]["feature_dict"], names[0]: 1e6}}], names[0])["sentinel_sensitive"]
    family_tags = feature_families("candidate_minus_incumbent_diff_neural_mean")
    benefit_is_slice = rows[1]["challenger_label"] == "NEUTRAL" and rows[1]["delta_class"] == "ZERO"
    toy_report, toy_rankings, toy_examples = analyze(ranked, ["diff_neural_mean"], .9)
    analyze_toy = toy_report["delta_class_counts"] == {"POSITIVE": 1, "ZERO": 1, "NEGATIVE": 1} and len(toy_rankings) == 3 * len(toy_report["feature_schema"]) and bool(toy_examples)
    return {"fold0_rejected": fold0_rejected, "exact_residual_feature_vector": exact_schema, "outcome_fields_absent": no_outcomes, "delta_classes": classes, "benefit_cross_tab_total": cross_total, "query_pairwise_greater": pair_greater, "query_pairwise_lower": pair_lower, "query_pairwise_scoped": pairwise_scoped, "auc_exact_cases": auc_cases, "auc_nonfinite_filtered": auc_nonfinite, "oracle_curve_monotonic": oracle_monotonic, "oracle_keep_baseline": keep_correct, "increasing_k_cannot_lower_oracle": increasing_k_safe, "sentinel_sensitive_detected": sentinel, "family_relational_neural": "RELATIONAL" in family_tags and "NEURAL" in family_tags, "benefit_is_diagnostic_slice": benefit_is_slice, "analyze_exact_feature_space_toy": analyze_toy}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path); parser.add_argument("--v3b-report", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "delta_training_enabled": False, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if any(value is None for value in (args.actions, args.v3b_report, args.output_dir)): parser.error("--actions, --v3b-report, and --output-dir are required")
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    validate_actions(actions)
    if any(any(key not in row for key in ("label", "gain", "baseline_recall")) for row in actions): raise ValueError("forensic requires labeled folds1--4 actions")
    names = feature_names(actions); threshold = action_threshold(args.v3b_report); ranked = fixed_stage1_ranked(actions, names, fixed_stage1_config(args.v3b_report))
    report, rankings, examples = analyze(ranked, names, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "delta_feature_forensic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "delta_feature_rankings.jsonl", rankings)
    write_jsonl(args.output_dir / "delta_positive_query_examples.jsonl", examples)


if __name__ == "__main__": main()
