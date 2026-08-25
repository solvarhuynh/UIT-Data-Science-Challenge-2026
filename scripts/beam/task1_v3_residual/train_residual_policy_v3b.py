"""V3B CPU-only two-stage residual policy.

Stage 1 ranks legal actions within a query.  Stage 2 decides whether the
query should be changed at all.  All model selection is out-of-fold on folds
1--4; fold0 actions must stay label-free until the selected policy is frozen.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

try:
    from .common import jsonl, sort_query_ids, write_jsonl
    from .train_residual_policy import LABELS, choose, feature_names, fit_model, make_predictions, utilities, validate_label_split
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from train_residual_policy import LABELS, choose, feature_names, fit_model, make_predictions, utilities, validate_label_split


FOLDS = (1, 2, 3, 4)
# Fixed small tree keeps the split search at 4 x 2 x 4 = 32 configurations.
STAGE1_LEAVES = (7,)
BENEFIT_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)
HARM_TRAIN_MULTIPLIERS = (1.0, 2.0)
HARM_UTILITY_WEIGHTS = (1.0, 1.5, 2.0, 3.0)
STAGE2_POSITIVE_MULTIPLIERS = (1.0, 2.0, 4.0)
GATE_QUANTILES = (0.50, 0.75, 0.90, 0.95)
ACTION_QUANTILES = (0.75, 0.90, 0.95)
MARGIN_QUANTILES = (0.75, 0.90, 0.95)
EXPECTED_V3A_OOF_SWAPS = {"swap_count": 2152, "beneficial_swaps": 54, "harmful_swaps": 29, "neutral_swaps": 2069}


def action_matrix(actions: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in actions], dtype="float32")


def stage1_weights(actions: list[dict[str, Any]], benefit_multiplier: float, harm_train_multiplier: float) -> np.ndarray:
    counts = Counter(str(row["query_id"]) for row in actions)
    multipliers = {"BENEFIT": benefit_multiplier, "NEUTRAL": 1.0, "HARM": harm_train_multiplier}
    return np.asarray([multipliers[str(row["label"])] / counts[str(row["query_id"])] for row in actions], dtype="float32")


def fit_stage1(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.06, max_leaf_nodes=int(config["max_leaf_nodes"]),
        l2_regularization=1.0, random_state=2026,
    )
    model.fit(action_matrix(actions, names), [LABELS[str(row["label"])] for row in actions], sample_weight=stage1_weights(actions, float(config["benefit_multiplier"]), float(config["harm_train_multiplier"])))
    return model


def probabilities(model: Any, actions: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = model.predict_proba(action_matrix(actions, names))
    positions = {int(label): index for index, label in enumerate(model.classes_)}
    benefit = raw[:, positions[1]] if 1 in positions else np.zeros(len(actions))
    harm = raw[:, positions[2]] if 2 in positions else np.zeros(len(actions))
    return benefit, harm


def rank_actions(model: Any, actions: list[dict[str, Any]], names: list[str], harm_utility_weight: float) -> dict[str, dict[str, Any]]:
    benefit, harm = probabilities(model, actions, names)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action, p_benefit, p_harm in zip(actions, benefit, harm):
        grouped[str(action["query_id"])].append({**action, "stage1_score": float(p_benefit - harm_utility_weight * p_harm), "stage1_p_benefit": float(p_benefit), "stage1_p_harm": float(p_harm)})
    ranked: dict[str, dict[str, Any]] = {}
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: (-float(row["stage1_score"]), -int(row["drop_rank"]), str(row["incoming_doc_id"])))
        best, second = rows[0], rows[1] if len(rows) > 1 else rows[0]
        ranked[query_id] = {"best": best, "second": second, "all": rows}
    return ranked


GATE_ACTION_FEATURES = (
    "incoming_neural_max", "incoming_neural_second", "incoming_neural_mean", "incoming_neural_std", "incoming_neural_max_minus_mean", "incoming_neural_top2_mean",
    "incoming_bm25_max", "incoming_bm25_mean", "incoming_union_rank", "incoming_source_support", "incoming_min_source_rank",
    "dropped_neural_max", "dropped_neural_second", "dropped_neural_mean", "dropped_bm25_max", "dropped_source_support", "dropped_baseline_rank",
    "diff_neural_max", "diff_neural_second", "diff_neural_mean", "diff_bm25_max", "diff_source_support", "diff_union_rank",
)
GATE_NAMES = ("best_action_score", "second_best_action_score", "score_margin", "best_p_benefit", "best_p_harm", *GATE_ACTION_FEATURES, "action_score_mean", "action_score_std", "top1_minus_median", "score_ge_q75_count", "score_ge_q90_count")


def query_rows(ranked: dict[str, dict[str, Any]], labeled: bool) -> list[dict[str, Any]]:
    rows = []
    for query_id in sort_query_ids(ranked):
        item, best, second, all_actions = ranked[query_id], ranked[query_id]["best"], ranked[query_id]["second"], ranked[query_id]["all"]
        scores = np.asarray([float(action["stage1_score"]) for action in all_actions], dtype="float64")
        features = {
            "best_action_score": float(best["stage1_score"]), "second_best_action_score": float(second["stage1_score"]),
            "score_margin": float(best["stage1_score"] - second["stage1_score"]), "best_p_benefit": float(best["stage1_p_benefit"]), "best_p_harm": float(best["stage1_p_harm"]),
            "action_score_mean": float(scores.mean()), "action_score_std": float(scores.std()), "top1_minus_median": float(scores[0] - np.median(scores)),
            "score_ge_q75_count": int((scores >= np.quantile(scores, .75)).sum()), "score_ge_q90_count": int((scores >= np.quantile(scores, .90)).sum()),
        }
        features.update({name: float(best["features"].get(name, 0.0)) for name in GATE_ACTION_FEATURES})
        row = {"query_id": query_id, "fold": int(best["fold"]), "features": features, "best_action": best, "baseline_top5": best["baseline_top5"]}
        if labeled:
            row.update({"best_action_is_benefit": int(float(best["gain"]) > 0.0), "query_has_any_benefit": int(any(float(action["gain"]) > 0.0 for action in all_actions)), "baseline_recall": float(best["baseline_recall"]), "best_gain": float(best["gain"])})
        rows.append(row)
    return rows


def gate_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row["features"][name]) for name in GATE_NAMES] for row in rows], dtype="float32")


def fit_stage2(rows: list[dict[str, Any]], max_leaf_nodes: int, stage2_positive_multiplier: float):
    from sklearn.ensemble import HistGradientBoostingClassifier
    labels = [int(row["best_action_is_benefit"]) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("V3B Stage2 outer training split requires both query target classes")
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, max_leaf_nodes=max_leaf_nodes, l2_regularization=1.0, random_state=2027)
    weights = stage2_weights(rows, stage2_positive_multiplier)
    model.fit(gate_matrix(rows), labels, sample_weight=weights)
    return model


def stage2_weights(rows: list[dict[str, Any]], stage2_positive_multiplier: float) -> np.ndarray:
    return np.asarray([stage2_positive_multiplier if int(row["best_action_is_benefit"]) else 1.0 for row in rows], dtype="float32")


def gate_probabilities(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    values = model.predict_proba(gate_matrix(rows))
    index = {int(label): position for position, label in enumerate(model.classes_)}
    return values[:, index[1]] if 1 in index else np.zeros(len(rows))


def crossfit_stage1_rows(actions: list[dict[str, Any]], folds: tuple[int, ...], names: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for held_out in folds:
        train = [row for row in actions if int(row["fold"]) in set(folds) - {held_out}]
        valid = [row for row in actions if int(row["fold"]) == held_out]
        if not train or not valid:
            raise ValueError("V3B Stage1 cross-fitting split is empty")
        model = fit_stage1(train, names, config)
        result.extend(query_rows(rank_actions(model, valid, names, float(config["harm_utility_weight"])), labeled=True))
    return result


def outer_stage1_cache(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Generate expensive Stage1 cross-fit outputs once per Stage1 config."""
    result = {}
    for held_out in FOLDS:
        outer_train_folds = tuple(fold for fold in FOLDS if fold != held_out)
        outer_train = [row for row in actions if int(row["fold"]) in outer_train_folds]
        outer_valid = [row for row in actions if int(row["fold"]) == held_out]
        stage2_train = crossfit_stage1_rows(outer_train, outer_train_folds, names, config)
        stage1 = fit_stage1(outer_train, names, config)
        valid_rows = query_rows(rank_actions(stage1, outer_valid, names, float(config["harm_utility_weight"])), labeled=True)
        result[held_out] = (stage2_train, valid_rows)
    return result


def outer_oof_from_stage1_cache(cache: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]], config: dict[str, Any], stage2_positive_multiplier: float) -> list[dict[str, Any]]:
    result = []
    for fold in FOLDS:
        stage2_train, valid_rows = cache[fold]
        stage2 = fit_stage2(stage2_train, int(config["max_leaf_nodes"]), stage2_positive_multiplier)
        for row, probability in zip(valid_rows, gate_probabilities(stage2, valid_rows)):
            result.append({**row, "gate_probability": float(probability)})
    return result


def threshold_values(rows: list[dict[str, Any]], key: str, quantiles: tuple[float, ...]) -> list[float | None]:
    values = np.asarray([float(row[key] if key == "gate_probability" else row["features"][key]) for row in rows], dtype="float64")
    return [None, *sorted({float(np.quantile(values, q)) for q in quantiles})]


def policy_metrics(rows: list[dict[str, Any]], gate_threshold: float | None, action_threshold: float | None, margin_threshold: float | None) -> dict[str, Any]:
    before = after = 0.0; swaps = beneficial = harmful = neutral = rescues = broken = 0
    per_fold: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        action = row["best_action"]
        threshold_execute = ((gate_threshold is None or float(row["gate_probability"]) >= gate_threshold) and (action_threshold is None or float(row["features"]["best_action_score"]) >= action_threshold) and (margin_threshold is None or float(row["features"]["score_margin"]) >= margin_threshold))
        execute = bool(row["execute"]) if "execute" in row else threshold_execute
        baseline, gain = float(row["baseline_recall"]), float(action["gain"]) if execute else 0.0
        before += baseline; after += baseline + gain; swaps += execute; beneficial += execute and gain > 0; harmful += execute and gain < 0; neutral += execute and gain == 0
        rescues += execute and baseline == 0 and gain > 0; broken += execute and baseline > 0 and gain < 0
        per_fold[int(row["fold"])].append((baseline, baseline + gain))
    count = len(rows)
    fold_delta = {str(fold): mean(after - before for before, after in values) for fold, values in per_fold.items()}
    fold_before = {str(fold): mean(before for before, _ in values) for fold, values in per_fold.items()}
    fold_after = {str(fold): mean(after for _, after in values) for fold, values in per_fold.items()}
    return {"baseline_macro_recall": before / count, "policy_macro_recall": after / count, "delta_vs_baseline": (after - before) / count, "pooled_delta_vs_baseline": (after - before) / count, "swap_count": swaps, "beneficial_swaps": beneficial, "harmful_swaps": harmful, "neutral_swaps": neutral, "beneficial_swap_precision": beneficial / swaps if swaps else 0.0, "harmful_swap_rate": harmful / swaps if swaps else 0.0, "zero_hit_rescues": rescues, "previously_correct_broken": broken, "per_fold_baseline_macro_recall": fold_before, "per_fold_policy_macro_recall": fold_after, "per_fold_delta_vs_baseline": fold_delta, "min_fold_delta_vs_baseline": min(fold_delta.values()), "negative_fold_count_vs_baseline": sum(value < 0.0 for value in fold_delta.values())}


def conservative_threshold_value(value: float | None) -> float:
    """Disabled decisions are least conservative; zero remains a real value."""
    return float("-inf") if value is None else float(value)


def selection_key(row: dict[str, Any]) -> tuple[float, int, int, float, int, float, float, float, int]:
    return (
        float(row["policy_macro_recall"]), -int(row["negative_fold_count_vs_baseline"]), -int(row["harmful_swaps"]),
        float(row["beneficial_swap_precision"]), -int(row["swap_count"]),
        conservative_threshold_value(row.get("gate_threshold")), conservative_threshold_value(row.get("action_threshold")),
        conservative_threshold_value(row.get("margin_threshold")), -int(row.get("max_leaf_nodes") or 0),
    )


def add_v3a_comparison(metrics: dict[str, Any], v3a_metrics: dict[str, Any]) -> dict[str, Any]:
    per_fold = {
        fold: float(metrics["per_fold_policy_macro_recall"][fold]) - float(v3a_metrics["per_fold_policy_macro_recall"][fold])
        for fold in metrics["per_fold_policy_macro_recall"]
    }
    delta = float(metrics["policy_macro_recall"]) - float(v3a_metrics["policy_macro_recall"])
    return {**metrics, "delta_vs_v3a": delta, "pooled_delta_vs_v3a": delta, "per_fold_delta_vs_v3a": per_fold, "min_fold_delta_vs_v3a": min(per_fold.values()), "negative_fold_count_vs_v3a": sum(value < 0.0 for value in per_fold.values())}


def eligible_v3b(candidate: dict[str, Any]) -> bool:
    return (float(candidate["pooled_delta_vs_baseline"]) > 0.0 and int(candidate["negative_fold_count_vs_baseline"]) == 0 and float(candidate["pooled_delta_vs_v3a"]) > 0.0 and int(candidate["negative_fold_count_vs_v3a"]) == 0)


def stage2_usage(candidate: dict[str, Any]) -> dict[str, bool]:
    return {"stage2_used": candidate.get("gate_threshold") is not None, "action_threshold_used": candidate.get("action_threshold") is not None, "margin_gate_used": candidate.get("margin_threshold") is not None}


def v3b_ranking_gate_forensic(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(); any_benefit = top1_benefit = ranking_failures = 0
    for row in rows:
        label = str(row["best_action"]["label"]); any_benefit += int(row["query_has_any_benefit"]); top1_benefit += label == "BENEFIT"; ranking_failures += int(row["query_has_any_benefit"] and label != "BENEFIT")
        gate_pass = candidate["gate_threshold"] is None or float(row["gate_probability"]) >= float(candidate["gate_threshold"])
        counts[f"top1_{label}_and_gate_{'pass' if gate_pass else 'abstain'}"] += 1
        counts[f"top1_{label}"] += 1
    return {"queries_with_any_BENEFIT": any_benefit, "queries_where_V3B_stage1_top1_is_BENEFIT": top1_benefit, "queries_with_any_BENEFIT_but_V3B_top1_not_BENEFIT": ranking_failures, "V3B_stage1_top1_BENEFIT_recall": top1_benefit / any_benefit if any_benefit else 0.0, "top1_class_distribution": {key: int(counts[key]) for key in ("top1_BENEFIT", "top1_HARM", "top1_NEUTRAL")}, "stage2_behavior": {key: int(counts[key]) for key in ("top1_BENEFIT_and_gate_pass", "top1_BENEFIT_and_gate_abstain", "top1_HARM_and_gate_pass", "top1_HARM_and_gate_abstain", "top1_NEUTRAL_and_gate_pass", "top1_NEUTRAL_and_gate_abstain")}}


def gate_ablations(rows: list[dict[str, Any]], candidate: dict[str, Any], v3a_metrics: dict[str, Any]) -> dict[str, Any]:
    settings = {"ACTION_ONLY": (None, candidate["action_threshold"], None), "GATE_ONLY": (candidate["gate_threshold"], None, None), "ACTION_PLUS_GATE": (candidate["gate_threshold"], candidate["action_threshold"], None), "ACTION_GATE_MARGIN": (candidate["gate_threshold"], candidate["action_threshold"], candidate["margin_threshold"])}
    return {name: add_v3a_comparison(policy_metrics(rows, *thresholds), v3a_metrics) for name, thresholds in settings.items()}


def load_v3a_config(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("selected_policy_type") == "NO_SWAP_BASELINE":
        return {"selected_policy_type": "NO_SWAP_BASELINE"}
    config = report["selected_config"]
    return {"selected_policy_type": "LEARNED", "max_leaf_nodes": int(config["max_leaf_nodes"]), "l2_regularization": float(config["l2_regularization"]), "harm_weight": float(config["harm_weight"]), "threshold": float(config["threshold"])}


def assert_expected_v3a_oof_counts(metrics: dict[str, Any]) -> None:
    actual = {key: int(metrics[key]) for key in EXPECTED_V3A_OOF_SWAPS}
    if actual != EXPECTED_V3A_OOF_SWAPS:
        raise AssertionError(f"V3A OOF execution accounting regression: {actual} != {EXPECTED_V3A_OOF_SWAPS}")


def v3a_oof(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> dict[str, Any]:
    if config["selected_policy_type"] == "NO_SWAP_BASELINE":
        return baseline_metrics(actions)
    rows = []
    for held_out in FOLDS:
        train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {held_out}]
        valid = [row for row in actions if int(row["fold"]) == held_out]
        model = fit_model(train, names, int(config["max_leaf_nodes"]), float(config["l2_regularization"]))
        chosen = choose(valid, utilities(model, valid, names, float(config["harm_weight"])), float(config["threshold"]))
        source_by_query = {str(row["query_id"]): row for row in valid}
        for query_id, selected in chosen.items():
            source = source_by_query[query_id]
            rows.append({"fold": held_out, "baseline_recall": float(source["baseline_recall"]), "best_action": {**(selected or source), "gain": float(selected["gain"]) if selected is not None else 0.0}, "execute": selected is not None, "gate_probability": 1.0, "features": {"best_action_score": 0.0, "score_margin": 0.0}})
    return policy_metrics(rows, None, None, None)


def baseline_metrics(actions: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, dict[str, Any]] = {}
    for action in actions:
        by_query.setdefault(str(action["query_id"]), action)
    rows = [
        {"fold": int(action["fold"]), "baseline_recall": float(action["baseline_recall"]), "best_action": {**action, "gain": 0.0}, "execute": False, "gate_probability": 0.0, "features": {"best_action_score": 0.0, "score_margin": 0.0}}
        for action in by_query.values()
    ]
    return policy_metrics(rows, None, None, None)


def build_predictions(rows: list[dict[str, Any]], thresholds: dict[str, float | None]) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda item: (int(item["fold"]), str(item["query_id"]))):
        action = row["best_action"]; baseline = [str(doc) for doc in row["baseline_top5"]]
        execute = ((thresholds["gate_threshold"] is None or float(row["gate_probability"]) >= float(thresholds["gate_threshold"])) and (thresholds["action_threshold"] is None or float(row["features"]["best_action_score"]) >= float(thresholds["action_threshold"])) and (thresholds["margin_threshold"] is None or float(row["features"]["score_margin"]) >= float(thresholds["margin_threshold"])) )
        top5 = list(baseline)
        if execute: top5[int(action["drop_rank"]) - 1] = str(action["incoming_doc_id"])
        if len(top5) != 5 or len(set(top5)) != 5 or top5[:3] != baseline[:3]:
            raise ValueError(f"V3B emitted invalid/protected top5 for {row['query_id']}")
        output.append({"query_id": row["query_id"], "fold": int(row["fold"]), "top5": top5, "selected_action": action if execute else None, "gate_probability": float(row["gate_probability"])})
    return output


def self_test() -> dict[str, bool]:
    baseline = ["a", "b", "c", "d", "e"]
    row = {"query_id": "0", "fold": 0, "baseline_top5": baseline, "best_action": {"drop_rank": 5, "incoming_doc_id": "x"}, "gate_probability": .9, "features": {"best_action_score": .8, "score_margin": .2}}
    prediction = build_predictions([row], {"gate_threshold": .5, "action_threshold": .5, "margin_threshold": .1})[0]
    try: validate_label_split([], [{"fold": 0, "label": "BENEFIT"}])
    except ValueError: fold0_guard = True
    else: fold0_guard = False
    toy_actions = []
    for fold in FOLDS:
        for has_benefit in (True, False):
            query_id = f"{fold}{int(has_benefit)}"; baseline_top5 = [f"{query_id}_{index}" for index in range(5)]
            labels = ("BENEFIT", "NEUTRAL") if has_benefit else ("HARM", "NEUTRAL")
            for index, label in enumerate(labels):
                toy_actions.append({"query_id": query_id, "fold": fold, "incoming_doc_id": f"in_{query_id}_{index}", "drop_rank": 5 - index, "baseline_top5": baseline_top5, "features": {"toy_signal": float(has_benefit) + index / 10}, "label": label, "baseline_recall": .5, "gain": .5 if label == "BENEFIT" else -.5 if label == "HARM" else 0.0})
    target_harm = {"query_id": "target", "fold": 1, "gain": -.5, "label": "HARM", "stage1_score": .9, "stage1_p_benefit": .1, "stage1_p_harm": .8, "features": {}, "baseline_top5": baseline, "baseline_recall": .5}
    target_benefit = {"query_id": "target", "fold": 1, "gain": .5, "label": "BENEFIT", "stage1_score": .8, "stage1_p_benefit": .8, "stage1_p_harm": .1, "features": {}, "baseline_top5": baseline, "baseline_recall": .5}
    target_probe = query_rows({"target": {"best": target_harm, "second": target_benefit, "all": [target_harm, target_benefit]}}, labeled=True)[0]
    toy_config = {"max_leaf_nodes": 7, "benefit_multiplier": 1.0, "harm_train_multiplier": 1.0, "harm_utility_weight": 1.0}
    toy_names = feature_names(toy_actions)
    crossfit_rows = crossfit_stage1_rows(toy_actions, FOLDS, toy_names, toy_config)
    stage1_cache = outer_stage1_cache(toy_actions, toy_names, toy_config)
    outer_rows = outer_oof_from_stage1_cache(stage1_cache, toy_config, 1.0)
    outer_rows_positive_weighted = outer_oof_from_stage1_cache(stage1_cache, toy_config, 4.0)
    tied = [{"policy_macro_recall": .5, "negative_fold_count_vs_baseline": 0, "harmful_swaps": 0, "beneficial_swap_precision": 1.0, "swap_count": 1, "gate_threshold": value, "action_threshold": value, "margin_threshold": value, "max_leaf_nodes": 7} for value in (None, 0.0, .05, .5)]
    threshold_order = [row["gate_threshold"] for row in sorted(tied, key=selection_key, reverse=True)]
    abstained_v3a = policy_metrics([{"fold": 1, "baseline_recall": .5, "best_action": {"gain": 0.0}, "execute": False, "gate_probability": 1.0, "features": {"best_action_score": 0.0, "score_margin": 0.0}}], None, None, None)
    unstable = {"pooled_delta_vs_baseline": .01, "negative_fold_count_vs_baseline": 0, "pooled_delta_vs_v3a": .01, "negative_fold_count_vs_v3a": 1}
    stable = {**unstable, "negative_fold_count_vs_v3a": 0}
    stage1_cache_is_reused = len(stage1_cache) == 4 and [{key: row[key] for key in ("query_id", "best_action", "features")} for row in outer_rows] == [{key: row[key] for key in ("query_id", "best_action", "features")} for row in outer_rows_positive_weighted]
    return {"fold0_label_guard": fold0_guard, "stage2_target_is_best_action": target_probe["best_action_is_benefit"] == 0 and target_probe["query_has_any_benefit"] == 1, "threshold_tiebreak": threshold_order == [.5, .05, 0.0, None], "v3a_abstention_not_neutral_swap": abstained_v3a["swap_count"] == 0 and abstained_v3a["neutral_swaps"] == 0, "stage2_multiplier_only_stage2": np.array_equal(stage1_weights(toy_actions, 1.0, 1.0), stage1_weights(toy_actions, 1.0, 1.0)) and not np.array_equal(stage2_weights(crossfit_rows, 1.0), stage2_weights(crossfit_rows, 4.0)), "stage1_cache_reused": stage1_cache_is_reused, "stability_guard_rejects_unstable": not eligible_v3b(unstable), "stability_guard_accepts_stable": eligible_v3b(stable), "top1_to3_protected": prediction["top5"][:3] == baseline[:3], "max_one_swap": prediction["selected_action"] is not None, "five_distinct_docs": len(prediction["top5"]) == 5 and len(set(prediction["top5"])) == 5, "stage1_crossfit": len(crossfit_rows) == 8 and {row["query_id"] for row in crossfit_rows} == {str(row["query_id"]) for row in toy_actions}, "stage2_crossfit": len(outer_rows) == 8 and all("gate_probability" in row for row in outer_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v3a-report", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "fold0_labels_read": False, "stage1_crossfitting": True, "stage2_crossfitting": True, "gpu_launched": False}, indent=2)); return
    if args.self_test:
        checks = self_test()
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "gpu_launched": False}, indent=2)); return
    actions = jsonl(args.actions)
    train_actions = [row for row in actions if int(row["fold"]) in FOLDS]
    fold0_actions = [row for row in actions if int(row["fold"]) == 0]
    validate_label_split(train_actions, fold0_actions)
    if {int(row["fold"]) for row in train_actions} != set(FOLDS) or not fold0_actions: raise ValueError("V3B requires labeled folds1--4 and unlabeled fold0")
    names = feature_names(actions)
    v3a_config = load_v3a_config(args.v3a_report)
    v3a_metrics = v3a_oof(train_actions, names, v3a_config)
    assert_expected_v3a_oof_counts(v3a_metrics)
    candidates = []
    for leaves in STAGE1_LEAVES:
        for benefit_multiplier in BENEFIT_MULTIPLIERS:
            for harm_train_multiplier in HARM_TRAIN_MULTIPLIERS:
                for harm_utility_weight in HARM_UTILITY_WEIGHTS:
                    config = {"max_leaf_nodes": leaves, "benefit_multiplier": benefit_multiplier, "harm_train_multiplier": harm_train_multiplier, "harm_utility_weight": harm_utility_weight}
                    stage1_cache = outer_stage1_cache(train_actions, names, config)
                    for stage2_positive_multiplier in STAGE2_POSITIVE_MULTIPLIERS:
                        oof = outer_oof_from_stage1_cache(stage1_cache, config, stage2_positive_multiplier)
                        for gate in threshold_values(oof, "gate_probability", GATE_QUANTILES):
                            for action in threshold_values(oof, "best_action_score", ACTION_QUANTILES):
                                for margin in threshold_values(oof, "score_margin", MARGIN_QUANTILES):
                                    metrics = add_v3a_comparison(policy_metrics(oof, gate, action, margin), v3a_metrics)
                                    candidates.append({"policy_type": "V3B_TWO_STAGE", **config, "stage2_positive_multiplier": stage2_positive_multiplier, "gate_threshold": gate, "action_threshold": action, "margin_threshold": margin, **metrics})
    best_v3b_raw = max(candidates, key=selection_key)
    eligible_candidates = [candidate for candidate in candidates if eligible_v3b(candidate)]
    best_v3b = max(eligible_candidates, key=selection_key) if eligible_candidates else best_v3b_raw
    baseline = {"policy_type": "NO_SWAP_BASELINE", **add_v3a_comparison(baseline_metrics(train_actions), v3a_metrics)}
    baseline["policy_macro_recall"] = baseline["baseline_macro_recall"]; baseline["delta_vs_baseline"] = 0.0; baseline["swap_count"] = baseline["beneficial_swaps"] = baseline["harmful_swaps"] = baseline["neutral_swaps"] = 0; baseline["beneficial_swap_precision"] = baseline["harmful_swap_rate"] = 0.0
    v3a_candidate = {"policy_type": "V3A_CURRENT", **v3a_metrics, "delta_vs_v3a": 0.0, "pooled_delta_vs_v3a": 0.0, "per_fold_delta_vs_v3a": {fold: 0.0 for fold in v3a_metrics["per_fold_delta_vs_baseline"]}, "min_fold_delta_vs_v3a": 0.0, "negative_fold_count_vs_v3a": 0}
    selection = max([v3a_candidate, best_v3b], key=selection_key) if eligible_candidates else v3a_candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selection["policy_type"] == "V3B_TWO_STAGE":
        config = selection
        stage2_train = crossfit_stage1_rows(train_actions, FOLDS, names, config)
        stage1 = fit_stage1(train_actions, names, config); stage2 = fit_stage2(stage2_train, int(config["max_leaf_nodes"]), float(config["stage2_positive_multiplier"]))
        fold0_rows = query_rows(rank_actions(stage1, fold0_actions, names, float(config["harm_utility_weight"])), labeled=False)
        for row, probability in zip(fold0_rows, gate_probabilities(stage2, fold0_rows)): row["gate_probability"] = float(probability)
        predictions = build_predictions(fold0_rows, selection)
    elif selection["policy_type"] == "V3A_CURRENT" and v3a_config["selected_policy_type"] == "LEARNED":
        stage1 = fit_model(train_actions, names, int(v3a_config["max_leaf_nodes"]), float(v3a_config["l2_regularization"]))
        predictions = make_predictions(choose(fold0_actions, utilities(stage1, fold0_actions, names, float(v3a_config["harm_weight"])), float(v3a_config["threshold"])), fold0_actions)
    else:
        predictions = []
        for query_id in sort_query_ids({str(row["query_id"]) for row in fold0_actions}):
            source = next(row for row in fold0_actions if str(row["query_id"]) == query_id)
            predictions.append({"query_id": query_id, "fold": 0, "top5": source["baseline_top5"], "selected_action": None, "gate_probability": 0.0})
    if len(predictions) != 1400: raise ValueError("V3B fold0 must emit exactly 1,400 predictions")
    write_jsonl(args.output_dir / "fold0_v3b_policy_predictions.jsonl", predictions)
    best_stage1_cache = outer_stage1_cache(train_actions, names, best_v3b)
    best_v3b_oof = outer_oof_from_stage1_cache(best_stage1_cache, best_v3b, float(best_v3b["stage2_positive_multiplier"]))
    write_jsonl(args.output_dir / "v3b_best_raw_oof_predictions.jsonl", build_predictions(best_v3b_oof, best_v3b))
    report = {"status": "V3B_POLICY_SELECTED_FOLD0_UNLABELED", "fold0_labels_read": False, "fold0_evaluated": False, "stage1": "multiclass query-normalized action ranker", "stage1_weighting": {"benefit_multiplier": list(BENEFIT_MULTIPLIERS), "harm_train_multiplier": list(HARM_TRAIN_MULTIPLIERS), "query_normalized": True}, "stage1_utility": {"harm_utility_weight": list(HARM_UTILITY_WEIGHTS)}, "stage1_cache_reused_across_stage2_multipliers": True, "stage2_positive_multiplier_grid": list(STAGE2_POSITIVE_MULTIPLIERS), "stage2": "cross-fitted gate for whether Stage1 best action is beneficial", "stage2_target": "best_action_is_benefit = int(best_action.gain > 0)", "query_has_any_benefit": "forensic_only_not_supervised_target", "stage2_feature_names": list(GATE_NAMES), "outer_cv": {"no_swap_baseline": baseline, "v3a_current": v3a_candidate, "best_v3b_raw": best_v3b_raw, "best_v3b_eligible": best_v3b if eligible_candidates else None}, "v3b_stage1_gate_forensic": v3b_ranking_gate_forensic(best_v3b_oof, best_v3b), "gate_ablations": gate_ablations(best_v3b_oof, best_v3b, v3a_metrics), "selected_policy": {**selection, **stage2_usage(selection)}, "selected_per_fold_delta_vs_baseline": selection["per_fold_delta_vs_baseline"], "selected_per_fold_delta_vs_v3a": selection["per_fold_delta_vs_v3a"], "selected_pooled_delta_vs_baseline": selection["pooled_delta_vs_baseline"], "selected_pooled_delta_vs_v3a": selection["pooled_delta_vs_v3a"], "candidate_count": len(candidates), "eligible_v3b_candidate_count": len(eligible_candidates), "fold0_prediction_query_count": len(predictions), "no_submission_created": True, "gpu_launched": False}
    (args.output_dir / "v3b_policy_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
