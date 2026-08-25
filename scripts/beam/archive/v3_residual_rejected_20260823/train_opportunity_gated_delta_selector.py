"""CPU-only opportunity-gated Delta-Recall selector, folds 1--4 only.

The fixed V3B Stage1 model supplies the ranking.  This experiment learns two
small, metric-aligned components on top of that ranking:

* a query-level opportunity gate; and
* a challenger-only selector over the next TOP_K actions.

All model selection is nested OOF on folds 1--4.  Fold0 is intentionally not
accepted by this module.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import jsonl, sort_query_ids, write_jsonl
    from .forensic_stage1_hard_errors import fixed_stage1_config
    from .train_residual_policy import feature_names
    from .train_residual_policy_v3b import FOLDS, fit_stage1, load_v3a_config, rank_actions, v3a_oof
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from forensic_stage1_hard_errors import fixed_stage1_config
    from train_residual_policy import feature_names
    from train_residual_policy_v3b import FOLDS, fit_stage1, load_v3a_config, rank_actions, v3a_oof


TOP_K = 20
THRESHOLD_QUANTILES = (.75, .85, .90, .925, .95, .975, .99)
ORACLE_K10_REFERENCE = 0.9610327381
ORACLE_K20_REFERENCE = 0.9670446429
OUTCOME_KEYS = {"label", "gain", "gold", "delta_recall", "resulting_recall", "baseline_recall"}

GATE_CONFIG = {
    "objective": "binary:logistic", "tree_method": "hist", "n_estimators": 150,
    "max_depth": 3, "learning_rate": .05, "min_child_weight": 5,
    "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
    "random_state": 2041, "n_jobs": 16,
}
SELECTOR_CONFIG = {**GATE_CONFIG, "random_state": 2042}

GATE_NAMES = (
    "incumbent_stage1_score", "incumbent_stage1_p_benefit", "incumbent_stage1_p_harm",
    "top1_stage1_score", "top2_stage1_score", "top1_top2_margin", "mean_stage1_score",
    "std_stage1_score", "median_stage1_score", "top1_minus_median",
    "max_candidate_stage1_p_benefit", "mean_candidate_stage1_p_benefit",
    "top2_candidate_stage1_p_benefit", "max_p_benefit_difference",
    "max_candidate_incoming_neural_max", "max_candidate_incoming_neural_mean",
    "max_candidate_minus_incumbent_incoming_neural_max",
    "max_candidate_minus_incumbent_incoming_neural_mean",
)

SELECTOR_NAMES = (
    "candidate_stage1_p_benefit", "candidate_stage1_p_harm", "candidate_stage1_score",
    "p_benefit_difference", "p_harm_difference", "candidate_minus_incumbent_stage1_score",
    "stage1_rank", "candidate_score_z", "candidate_score_percentile",
    "candidate_incoming_neural_max", "candidate_incoming_neural_mean",
    "candidate_incoming_neural_top2_mean", "candidate_incoming_neural_second",
    "candidate_diff_neural_max", "candidate_diff_neural_mean", "candidate_diff_neural_second",
    "candidate_minus_incumbent_incoming_neural_max",
    "candidate_minus_incumbent_incoming_neural_mean",
    "candidate_minus_incumbent_incoming_neural_top2_mean",
    "candidate_minus_incumbent_incoming_neural_second",
    "candidate_minus_incumbent_diff_neural_max",
    "candidate_minus_incumbent_diff_neural_mean",
    "candidate_minus_incumbent_diff_neural_second",
)


def validate_actions(actions: list[dict[str, Any]]) -> None:
    folds_by_query: dict[str, set[int]] = defaultdict(set)
    for row in actions:
        fold = int(row["fold"])
        if fold not in FOLDS:
            raise ValueError(f"opportunity-gated selector accepts only folds1--4: {row['query_id']} fold={fold}")
        folds_by_query[str(row["query_id"])].add(fold)
    crossed = [query for query, folds in folds_by_query.items() if len(folds) != 1]
    if crossed:
        raise ValueError(f"query belongs to more than one fold: {crossed[:5]}")


def action_matrix(actions: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in actions], dtype="float32")


def strip_outcomes(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in actions:
        value = {key: item for key, item in row.items() if key not in OUTCOME_KEYS}
        if any(key in value for key in OUTCOME_KEYS):
            raise AssertionError("outcome field leaked into Stage1 inference action")
        clean.append(value)
    return clean


def anchor_execute(incumbent: dict[str, Any], threshold: float | None) -> bool:
    return threshold is None or float(incumbent["stage1_score"]) >= float(threshold)


def keep_gain(incumbent: dict[str, Any], threshold: float | None) -> float:
    return float(incumbent.get("gain", 0.0)) if anchor_execute(incumbent, threshold) else 0.0


def keep_recall(incumbent: dict[str, Any], threshold: float | None) -> float:
    return float(incumbent.get("baseline_recall", 0.0)) + keep_gain(incumbent, threshold)


def feature_value(action: dict[str, Any], name: str) -> float:
    return float(action.get("features", {}).get(name, 0.0))


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def _max_feature(actions: list[dict[str, Any]], name: str) -> float:
    values = _finite([feature_value(action, name) for action in actions])
    return float(values.max()) if len(values) else 0.0


def _mean_feature(actions: list[dict[str, Any]], name: str) -> float:
    values = _finite([feature_value(action, name) for action in actions])
    return float(values.mean()) if len(values) else 0.0


def gate_features(item: dict[str, Any]) -> dict[str, float]:
    incumbent = item["best"]
    challengers = item["all"][1:TOP_K + 1]
    scores = _finite([float(action["stage1_score"]) for action in item["all"]])
    candidate_benefit = sorted((float(action["stage1_p_benefit"]) for action in challengers), reverse=True)
    p_differences = [float(action["stage1_p_benefit"]) - float(incumbent["stage1_p_benefit"]) for action in challengers]
    top2 = candidate_benefit[:2]
    if not len(scores):
        scores = np.asarray([0.0], dtype="float64")
    return {
        "incumbent_stage1_score": float(incumbent["stage1_score"]),
        "incumbent_stage1_p_benefit": float(incumbent["stage1_p_benefit"]),
        "incumbent_stage1_p_harm": float(incumbent["stage1_p_harm"]),
        "top1_stage1_score": float(item["best"]["stage1_score"]),
        "top2_stage1_score": float(item["second"]["stage1_score"]),
        "top1_top2_margin": float(item["best"]["stage1_score"] - item["second"]["stage1_score"]),
        "mean_stage1_score": float(scores.mean()),
        "std_stage1_score": float(scores.std()),
        "median_stage1_score": float(np.median(scores)),
        "top1_minus_median": float(scores.max() - np.median(scores)),
        "max_candidate_stage1_p_benefit": max(candidate_benefit, default=0.0),
        "mean_candidate_stage1_p_benefit": float(np.mean(candidate_benefit)) if candidate_benefit else 0.0,
        "top2_candidate_stage1_p_benefit": float(np.mean(top2)) if top2 else 0.0,
        "max_p_benefit_difference": max(p_differences, default=0.0),
        "max_candidate_incoming_neural_max": _max_feature(challengers, "incoming_neural_max"),
        "max_candidate_incoming_neural_mean": _max_feature(challengers, "incoming_neural_mean"),
        "max_candidate_minus_incumbent_incoming_neural_max": max((feature_value(action, "incoming_neural_max") - feature_value(incumbent, "incoming_neural_max") for action in challengers), default=0.0),
        "max_candidate_minus_incumbent_incoming_neural_mean": max((feature_value(action, "incoming_neural_mean") - feature_value(incumbent, "incoming_neural_mean") for action in challengers), default=0.0),
    }


def selector_features(candidate: dict[str, Any], incumbent: dict[str, Any], all_actions: list[dict[str, Any]], rank: int) -> dict[str, float]:
    scores = _finite([float(action["stage1_score"]) for action in all_actions])
    score = float(candidate["stage1_score"])
    mean_score = float(scores.mean()) if len(scores) else 0.0
    std_score = float(scores.std()) if len(scores) else 0.0
    return {
        "candidate_stage1_p_benefit": float(candidate["stage1_p_benefit"]),
        "candidate_stage1_p_harm": float(candidate["stage1_p_harm"]),
        "candidate_stage1_score": score,
        "p_benefit_difference": float(candidate["stage1_p_benefit"] - incumbent["stage1_p_benefit"]),
        "p_harm_difference": float(candidate["stage1_p_harm"] - incumbent["stage1_p_harm"]),
        "candidate_minus_incumbent_stage1_score": score - float(incumbent["stage1_score"]),
        "stage1_rank": float(rank),
        "candidate_score_z": (score - mean_score) / std_score if std_score else 0.0,
        "candidate_score_percentile": float(np.mean(scores <= score)) if len(scores) else 0.0,
        "candidate_incoming_neural_max": feature_value(candidate, "incoming_neural_max"),
        "candidate_incoming_neural_mean": feature_value(candidate, "incoming_neural_mean"),
        "candidate_incoming_neural_top2_mean": feature_value(candidate, "incoming_neural_top2_mean"),
        "candidate_incoming_neural_second": feature_value(candidate, "incoming_neural_second"),
        "candidate_diff_neural_max": feature_value(candidate, "diff_neural_max"),
        "candidate_diff_neural_mean": feature_value(candidate, "diff_neural_mean"),
        "candidate_diff_neural_second": feature_value(candidate, "diff_neural_second"),
        "candidate_minus_incumbent_incoming_neural_max": feature_value(candidate, "incoming_neural_max") - feature_value(incumbent, "incoming_neural_max"),
        "candidate_minus_incumbent_incoming_neural_mean": feature_value(candidate, "incoming_neural_mean") - feature_value(incumbent, "incoming_neural_mean"),
        "candidate_minus_incumbent_incoming_neural_top2_mean": feature_value(candidate, "incoming_neural_top2_mean") - feature_value(incumbent, "incoming_neural_top2_mean"),
        "candidate_minus_incumbent_incoming_neural_second": feature_value(candidate, "incoming_neural_second") - feature_value(incumbent, "incoming_neural_second"),
        "candidate_minus_incumbent_diff_neural_max": feature_value(candidate, "diff_neural_max") - feature_value(incumbent, "diff_neural_max"),
        "candidate_minus_incumbent_diff_neural_mean": feature_value(candidate, "diff_neural_mean") - feature_value(incumbent, "diff_neural_mean"),
        "candidate_minus_incumbent_diff_neural_second": feature_value(candidate, "diff_neural_second") - feature_value(incumbent, "diff_neural_second"),
    }


def assert_feature_schema(names: tuple[str, ...], forbidden: set[str]) -> None:
    if any(name in forbidden or any(token in name.lower() for token in ("label", "gain", "gold", "delta_recall", "resulting_recall")) for name in names):
        raise AssertionError("outcome field leaked into model feature schema")


def selector_variability(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    result = {}
    for name in SELECTOR_NAMES:
        result[name] = sum(bool(values) and (max(float(row["features"][name]) for row in values) - min(float(row["features"][name]) for row in values) > 1e-12) for values in grouped.values())
    if any(value == 0 for value in result.values()):
        raise AssertionError(f"selector feature is query-constant everywhere: {[name for name, value in result.items() if value == 0]}")
    return result


def build_rows(ranked: dict[str, dict[str, Any]], threshold: float | None, *, labeled: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]
        incumbent = item["best"]
        challengers = item["all"][1:TOP_K + 1]
        keep = keep_recall(incumbent, threshold)
        gate = {"query_id": query_id, "fold": int(incumbent["fold"]), "features": gate_features(item), "incumbent": incumbent, "baseline_top5": list(incumbent.get("baseline_top5", [])), "anchor_execute": anchor_execute(incumbent, threshold)}
        if labeled:
            gate["target"] = int(any(float(action["baseline_recall"]) + float(action["gain"]) > keep for action in challengers))
            gate["keep_recall"] = keep
        gate_rows.append(gate)
        for rank, candidate in enumerate(challengers, start=1):
            row = {"query_id": query_id, "fold": int(candidate["fold"]), "features": selector_features(candidate, incumbent, item["all"], rank), "action": candidate, "incumbent": incumbent, "stage1_rank": rank, "anchor_execute": gate["anchor_execute"]}
            if labeled:
                row["target"] = int(float(candidate["baseline_recall"]) + float(candidate["gain"]) > keep)
                row["delta_vs_keep"] = float(candidate["baseline_recall"]) + float(candidate["gain"]) - keep
                row["keep_recall"] = keep
            selector_rows.append(row)
    assert_feature_schema(GATE_NAMES, OUTCOME_KEYS)
    assert_feature_schema(SELECTOR_NAMES, OUTCOME_KEYS)
    if not labeled:
        if any(any(key in row for key in OUTCOME_KEYS) for row in gate_rows + selector_rows):
            raise AssertionError("outcome leaked into inference rows")
    if selector_rows:
        selector_variability(selector_rows)
    return gate_rows, selector_rows


def gate_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row["features"][name]) for name in GATE_NAMES] for row in rows], dtype="float32")


def selector_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row["features"][name]) for name in SELECTOR_NAMES] for row in rows], dtype="float32")


def selector_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["query_id"]) for row in rows)
    return np.asarray([1.0 / counts[str(row["query_id"])] for row in rows], dtype="float32")


def _fit_xgb(rows: list[dict[str, Any]], kind: str):
    import xgboost as xgb
    if xgb.__version__ != "3.2.0":
        raise RuntimeError(f"xgboost==3.2.0 required, found {xgb.__version__}")
    if kind == "gate":
        x, y, weights, config = gate_matrix(rows), np.asarray([int(row["target"]) for row in rows], dtype="int32"), np.ones(len(rows), dtype="float32"), GATE_CONFIG
    else:
        x, y, weights, config = selector_matrix(rows), np.asarray([int(row["target"]) for row in rows], dtype="int32"), selector_weights(rows), SELECTOR_CONFIG
    if len(set(y.tolist())) < 2:
        raise ValueError(f"{kind} training split requires both target classes")
    model = xgb.XGBClassifier(**config)
    model.fit(x, y, sample_weight=weights, verbose=False)
    return model


def _positive_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    values = model.predict_proba(matrix)
    positions = {int(label): index for index, label in enumerate(model.classes_)}
    return values[:, positions[1]] if 1 in positions else np.zeros(len(matrix), dtype="float32")


def _rank_stage1(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], train_folds: set[int], valid_folds: set[int]) -> dict[str, dict[str, Any]]:
    train = [row for row in actions if int(row["fold"]) in train_folds]
    valid = [row for row in actions if int(row["fold"]) in valid_folds]
    if not train or not valid:
        raise ValueError("empty Stage1 split")
    model = fit_stage1(train, names, config)
    return rank_actions(model, valid, names, float(config["harm_utility_weight"]))


def inner_stage1_plan(outer_train_folds: tuple[int, ...], held_inner: int) -> dict[str, Any]:
    outer = tuple(sorted(outer_train_folds))
    if len(outer) != 3 or held_inner not in outer:
        raise ValueError("inner plan requires three outer-training folds")
    validation_train = tuple(fold for fold in outer if fold != held_inner)
    training_features = {fold: tuple(other for other in validation_train if other != fold) for fold in validation_train}
    if any(held_inner in folds for folds in training_features.values()):
        raise AssertionError("held inner fold leaked into gate/selector training Stage1 features")
    return {"outer_train_folds": outer, "held_inner": int(held_inner), "validation_stage1_train_folds": validation_train, "training_feature_stage1_train_folds": training_features, "held_inner_absent_from_training_features": all(held_inner not in folds for folds in training_features.values())}


def outer_stage1_plan(held_outer: int) -> dict[str, Any]:
    train_folds = tuple(fold for fold in FOLDS if fold != held_outer)
    if held_outer in train_folds or len(train_folds) != len(FOLDS) - 1:
        raise AssertionError("held outer fold leaked into Stage1 training plan")
    return {"held_outer": int(held_outer), "stage1_train_folds": train_folds, "held_outer_absent": held_outer not in train_folds}


def outer_stage1_input(actions: list[dict[str, Any]], outer_train: tuple[int, ...], held_outer: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in actions if int(row["fold"]) in set(outer_train)]
    valid_labeled = [row for row in actions if int(row["fold"]) == held_outer]
    valid_clean = strip_outcomes(valid_labeled)
    if not train or not valid_clean:
        raise ValueError("outer Stage1 train/validation partition is empty")
    if {int(row["fold"]) for row in train} != set(outer_train):
        raise AssertionError("outer Stage1 training partition is missing a train fold")
    if {int(row["fold"]) for row in valid_clean} != {held_outer}:
        raise AssertionError("outer Stage1 validation partition has an unexpected fold")
    if any(any(key in row for key in OUTCOME_KEYS) for row in valid_clean):
        raise AssertionError("held outer outcome leaked into Stage1 input")
    if any(int(row["fold"]) == held_outer for row in train):
        raise AssertionError("held outer row reached Stage1 training partition")
    if any(any(key not in row for key in ("label", "gain", "baseline_recall")) for row in train):
        raise AssertionError("outer-train labels were removed from Stage1 training input")
    return [*train, *valid_clean], valid_labeled


def predict_components(gate_model: Any, selector_model: Any, gate_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate_prob = _positive_probability(gate_model, gate_matrix(gate_rows))
    selector_prob = _positive_probability(selector_model, selector_matrix(selector_rows)) if selector_rows else np.zeros(0, dtype="float32")
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(selector_rows, selector_prob):
        by_query[str(row["query_id"])].append({**row, "selector_probability": float(score)})
    output = []
    for gate, probability in zip(gate_rows, gate_prob):
        candidates = by_query.get(str(gate["query_id"]), [])
        best = sorted(candidates, key=lambda row: (-float(row["selector_probability"]), int(row["stage1_rank"]), str(row["action"].get("incoming_doc_id", ""))))[0] if candidates else None
        output.append({"query_id": gate["query_id"], "fold": int(gate["fold"]), "baseline_top5": list(gate.get("baseline_top5", [])), "incumbent": gate["incumbent"], "anchor_execute": bool(gate["anchor_execute"]), "keep_recall": float(gate.get("keep_recall", 0.0)), "gate_probability": float(probability), "best_challenger": best["action"] if best else None, "best_challenger_probability": float(best["selector_probability"]) if best else 0.0, "best_challenger_stage1_rank": int(best["stage1_rank"]) if best else None})
    return output


def action_top5(baseline_top5: list[str], action: dict[str, Any] | None) -> list[str]:
    top5 = list(baseline_top5)
    if action is not None:
        drop_rank = int(action["drop_rank"])
        if drop_rank not in (4, 5):
            raise AssertionError("only rank4/rank5 actions may modify top5")
        top5[drop_rank - 1] = str(action["incoming_doc_id"])
    if len(top5) != 5 or len(set(top5)) != 5 or top5[:3] != list(baseline_top5)[:3]:
        raise AssertionError("invalid or unprotected top5")
    return top5


def keep_selected_action(row: dict[str, Any]) -> dict[str, Any] | None:
    return row["incumbent"] if row["anchor_execute"] else None


def prediction_delta_vs_keep(row: dict[str, Any]) -> float:
    if row["execute"]:
        if row["selected"] is None:
            raise AssertionError("executed residual has no selected action")
        policy_value = float(row["incumbent"]["baseline_recall"]) + float(row["selected"]["gain"])
    else:
        policy_value = float(row["keep_recall"])
    return policy_value - float(row["keep_recall"])


def apply_thresholds(predictions: list[dict[str, Any]], gate_threshold: float, selector_threshold: float) -> list[dict[str, Any]]:
    result = []
    for row in predictions:
        candidate = row["best_challenger"]
        execute = bool(candidate is not None and float(row["gate_probability"]) >= gate_threshold and float(row["best_challenger_probability"]) >= selector_threshold)
        selected = candidate if execute else keep_selected_action(row)
        baseline_top5 = list(row.get("baseline_top5", row["incumbent"].get("baseline_top5", [])))
        top5 = action_top5(baseline_top5, selected)
        result.append({**row, "execute": execute, "selected": selected, "selected_top5": top5})
    return result


def policy_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot score empty prediction set")
    before = after = keep_after = 0.0
    positive_residual_gain = negative_residual_loss = 0.0
    counts = Counter()
    per_fold: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    benefit_before = benefit_after = 0
    positive_opportunities = positive_captured = 0
    for row in predictions:
        incumbent = row["incumbent"]; selected = row["selected"]
        baseline = float(incumbent["baseline_recall"]); keep_value = float(row["keep_recall"])
        if row["execute"]:
            if selected is None:
                raise AssertionError("residual execution requires a selected challenger")
            selected_value = baseline + float(selected["gain"])
        else:
            selected_value = keep_value
        residual_delta = selected_value - keep_value
        if not row["execute"] and abs(residual_delta) > 1e-12:
            raise AssertionError("non-residual KEEP row has nonzero residual delta")
        before += baseline; keep_after += keep_value; after += selected_value
        per_fold[int(row["fold"])].append((baseline, keep_value, selected_value))
        if row["best_challenger"] is not None and float(row["best_challenger"].get("gain", 0.0)) - (keep_value - baseline) > 0:
            positive_opportunities += 1
        if row["execute"]:
            counts["correction"] += 1
            counts["positive" if residual_delta > 0 else "zero" if residual_delta == 0 else "negative"] += 1
            positive_captured += residual_delta > 0
            positive_residual_gain += max(residual_delta, 0.0)
            negative_residual_loss += min(residual_delta, 0.0)
        else:
            counts["keep"] += 1
        keep_action = keep_selected_action(row)
        benefit_before += bool(keep_action is not None and str(keep_action.get("label", "")) == "BENEFIT")
        benefit_after += bool(selected is not None and str(selected.get("label", "")) == "BENEFIT")
    fold = {}
    for key, values in sorted(per_fold.items()):
        fold[str(key)] = {"baseline_macro_recall": float(np.mean([value[0] for value in values])), "keep_macro_recall": float(np.mean([value[1] for value in values])), "policy_macro_recall": float(np.mean([value[2] for value in values]))}
        fold[str(key)]["delta_vs_keep"] = fold[str(key)]["policy_macro_recall"] - fold[str(key)]["keep_macro_recall"]
    policy = after / len(predictions); keep = keep_after / len(predictions); baseline = before / len(predictions); corrections = counts["correction"]
    return {"query_count": len(predictions), "baseline_macro_recall": baseline, "keep_macro_recall": keep, "policy_macro_recall": policy, "delta_vs_baseline": policy - baseline, "delta_vs_keep": policy - keep, "correction_count": corrections, "keep_count": counts["keep"], "positive_delta_corrections": counts["positive"], "zero_delta_corrections": counts["zero"], "negative_delta_corrections": counts["negative"], "beneficial_correction_precision": counts["positive"] / corrections if corrections else 0.0, "harmful_correction_rate": counts["negative"] / corrections if corrections else 0.0, "positive_residual_recall_captured": positive_captured, "positive_residual_recall_gain_captured": positive_residual_gain, "negative_residual_recall_loss_introduced": negative_residual_loss, "positive_opportunity_query_count": positive_opportunities, "positive_capture_rate": positive_captured / positive_opportunities if positive_opportunities else 0.0, "per_fold": fold, "min_fold_delta_vs_keep": min(value["delta_vs_keep"] for value in fold.values()), "negative_fold_count_vs_keep": sum(value["delta_vs_keep"] < 0 for value in fold.values()), "BENEFIT_selected_before": int(benefit_before), "BENEFIT_selected_after": int(benefit_after)}


def threshold_candidates(values: list[float]) -> list[float]:
    finite = _finite(values)
    return [float("inf")] if not len(finite) else [float("inf"), *sorted({float(np.quantile(finite, quantile)) for quantile in THRESHOLD_QUANTILES})]


def choose_threshold(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gate_values = [float(row["gate_probability"]) for row in predictions]
    selector_values = [float(row["best_challenger_probability"]) for row in predictions if row["best_challenger"] is not None]
    keep_predictions = apply_thresholds(predictions, float("inf"), float("inf"))
    keep_metrics = policy_metrics(keep_predictions)
    candidates = []
    for gate_threshold in threshold_candidates(gate_values):
        for selector_threshold in threshold_candidates(selector_values):
            selected = apply_thresholds(predictions, gate_threshold, selector_threshold)
            metrics = policy_metrics(selected)
            eligible = bool(np.isfinite(gate_threshold) and np.isfinite(selector_threshold) and metrics["delta_vs_keep"] > 0.0 and metrics["negative_fold_count_vs_keep"] == 0)
            candidates.append({"gate_threshold": gate_threshold, "selector_threshold": selector_threshold, "eligible_vs_keep": eligible, **metrics})
    stable = [row for row in candidates if row["eligible_vs_keep"]]
    if not stable:
        return {"selected_policy_type": "KEEP_STAGE1", "selection_reason": "no stable finite threshold pair beats KEEP_STAGE1", "gate_threshold": float("inf"), "selector_threshold": float("inf"), "keep_reference": keep_metrics, "candidates": candidates}
    chosen = max(stable, key=lambda row: (float(row["policy_macro_recall"]), -int(row["negative_fold_count_vs_keep"]), -int(row["negative_delta_corrections"]), float(row["beneficial_correction_precision"]), -int(row["correction_count"]), float(row["gate_threshold"]), float(row["selector_threshold"])))
    return {"selected_policy_type": "OPPORTUNITY_GATED_DELTA_SELECTOR", "selection_reason": "stable positive inner-OOF macro Recall versus KEEP_STAGE1", "gate_threshold": float(chosen["gate_threshold"]), "selector_threshold": float(chosen["selector_threshold"]), "keep_reference": keep_metrics, "selected_candidate": chosen, "candidates": candidates}


def nested_inner_oof(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], outer_train_folds: tuple[int, ...], threshold: float | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    outer = tuple(sorted(outer_train_folds))
    for held_inner in outer:
        plan = inner_stage1_plan(outer, held_inner)
        validation_ranked = _rank_stage1(actions, names, config, set(plan["validation_stage1_train_folds"]), {held_inner})
        training_ranked: dict[str, dict[str, Any]] = {}
        training_provenance = {}
        for scored_fold, stage1_train_folds in plan["training_feature_stage1_train_folds"].items():
            training_ranked.update(_rank_stage1(actions, names, config, set(stage1_train_folds), {scored_fold}))
            training_provenance[str(scored_fold)] = list(stage1_train_folds)
        train_gate, train_selector = build_rows(training_ranked, threshold, labeled=True)
        valid_gate, valid_selector = build_rows(validation_ranked, threshold, labeled=True)
        gate_model = _fit_xgb(train_gate, "gate")
        selector_model = _fit_xgb(train_selector, "selector")
        predictions.extend(predict_components(gate_model, selector_model, valid_gate, valid_selector))
        provenance[str(held_inner)] = {**plan, "training_feature_stage1_train_folds": training_provenance, "held_inner_absent_from_stage1_generators": all(held_inner not in folds for folds in training_provenance.values())}
    return predictions, provenance


def _attach_labels(predictions: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["query_id"]), str(row["incoming_doc_id"]), int(row["drop_rank"])): row for row in actions}
    result = []
    for prediction in predictions:
        def labeled(action: dict[str, Any] | None) -> dict[str, Any] | None:
            if action is None:
                return None
            source = by_key[(str(prediction["query_id"]), str(action["incoming_doc_id"]), int(action["drop_rank"]))]
            return {**action, **{key: source[key] for key in OUTCOME_KEYS if key in source}}
        labeled_incumbent = labeled(prediction["incumbent"])
        if labeled_incumbent is None:
            raise AssertionError("prediction incumbent could not be labeled")
        keep_value = float(labeled_incumbent["baseline_recall"]) + (float(labeled_incumbent["gain"]) if prediction["anchor_execute"] else 0.0)
        result.append({**prediction, "incumbent": labeled_incumbent, "best_challenger": labeled(prediction["best_challenger"]), "keep_recall": keep_value})
    return result


def oracle_macro(ranked: dict[str, dict[str, Any]], threshold: float | None, k: int) -> float:
    values = []
    for item in ranked.values():
        incumbent = item["best"]; keep = keep_recall(incumbent, threshold); candidates = item["all"][1:k + 1]
        values.append(max([keep, *[float(action["baseline_recall"]) + float(action["gain"]) for action in candidates]]))
    return float(np.mean(values)) if values else 0.0


def keep_predictions(ranked: dict[str, dict[str, Any]], threshold: float | None) -> list[dict[str, Any]]:
    result = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]; incumbent = item["best"]
        result.append({"query_id": query_id, "fold": int(incumbent["fold"]), "baseline_top5": list(incumbent["baseline_top5"]), "incumbent": incumbent, "anchor_execute": anchor_execute(incumbent, threshold), "keep_recall": keep_recall(incumbent, threshold), "gate_probability": 0.0, "best_challenger": None, "best_challenger_probability": 0.0})
    return apply_thresholds(result, float("inf"), float("inf"))


def opportunity_diagnostics(ranked: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    selected_by_query = {str(row["query_id"]): row for row in predictions}
    queries_with_benefit = selected_before = selected_after = recovered = lost = 0
    positive_action_count = positive_query_count = 0
    for query_id, item in ranked.items():
        incumbent = item["best"]
        keep = keep_recall(incumbent, threshold)
        candidates = item["all"][1:TOP_K + 1]
        positive = [action for action in candidates if float(action["baseline_recall"]) + float(action["gain"]) > keep]
        if any(str(action.get("label", "")) == "BENEFIT" for action in item["all"]):
            queries_with_benefit += 1
        if any(float(action["gain"]) - (keep - float(incumbent["baseline_recall"])) > 0 for action in candidates):
            positive_query_count += 1
        positive_action_count += len(positive)
        prediction = selected_by_query[str(query_id)]
        keep_action = keep_selected_action(prediction)
        before_benefit = bool(keep_action is not None and str(keep_action.get("label", "")) == "BENEFIT")
        selected_benefit = bool(prediction["selected"] is not None and str(prediction["selected"].get("label", "")) == "BENEFIT")
        selected_before += before_benefit
        selected_after += selected_benefit
        recovered += (not before_benefit) and selected_benefit
        lost += before_benefit and (not selected_benefit)
    return {"queries_with_any_BENEFIT": queries_with_benefit, "BENEFIT_selected_before": selected_before, "BENEFIT_selected_after": selected_after, "benefit_recovered": recovered, "benefit_lost": lost, "positive_opportunity_action_count_top20": positive_action_count, "positive_opportunity_query_count_top20": positive_query_count}


def outer_run(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], threshold: float | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    all_predictions: list[dict[str, Any]] = []
    threshold_reports: dict[str, Any] = {}
    outer_reports: dict[int, dict[str, Any]] = {}
    for held_outer in FOLDS:
        outer_plan = outer_stage1_plan(held_outer)
        outer_train = tuple(outer_plan["stage1_train_folds"])
        inner_predictions, provenance = nested_inner_oof(actions, names, config, outer_train, threshold)
        selected_threshold = choose_threshold(inner_predictions)
        threshold_reports[str(held_outer)] = {key: value for key, value in selected_threshold.items() if key != "candidates"}
        final_ranked: dict[str, dict[str, Any]] = {}
        final_provenance = {"held_outer": int(held_outer), "held_outer_absent": bool(outer_plan["held_outer_absent"]), "scored_fold_stage1_train_folds": {}}
        for scored_fold in outer_train:
            train_folds = set(outer_train) - {scored_fold}
            final_ranked.update(_rank_stage1(actions, names, config, train_folds, {scored_fold}))
            if held_outer in train_folds:
                raise AssertionError("held outer fold leaked into final Stage1 training")
            final_provenance["scored_fold_stage1_train_folds"][str(scored_fold)] = sorted(train_folds)
        final_gate, final_selector = build_rows(final_ranked, threshold, labeled=True)
        gate_model = _fit_xgb(final_gate, "gate"); selector_model = _fit_xgb(final_selector, "selector")
        stage1_outer_input, valid_actions = outer_stage1_input(actions, outer_train, held_outer)
        valid_ranked = _rank_stage1(stage1_outer_input, names, config, set(outer_train), {held_outer})
        valid_gate, valid_selector = build_rows(valid_ranked, threshold, labeled=False)
        valid_predictions = predict_components(gate_model, selector_model, valid_gate, valid_selector)
        valid_predictions = _attach_labels(valid_predictions, valid_actions)
        valid_predictions = apply_thresholds(valid_predictions, float(selected_threshold["gate_threshold"]), float(selected_threshold["selector_threshold"]))
        metrics = policy_metrics(valid_predictions)
        keep_metrics = policy_metrics(keep_predictions({query_id: {"best": row["incumbent"], "second": row["incumbent"], "all": [row["incumbent"]]} for query_id, row in ((str(row["query_id"]), row) for row in valid_predictions)}, threshold))
        outer_reports[held_outer] = {"baseline_macro_recall": metrics["baseline_macro_recall"], "keep_macro_recall": keep_metrics["policy_macro_recall"], "new_policy_macro_recall": metrics["policy_macro_recall"], "delta_vs_keep": metrics["policy_macro_recall"] - keep_metrics["policy_macro_recall"], "gate_threshold": selected_threshold["gate_threshold"], "selector_threshold": selected_threshold["selector_threshold"], "queries_passing_gate": sum(float(row["gate_probability"]) >= float(selected_threshold["gate_threshold"]) for row in valid_predictions), "queries_failing_gate": sum(float(row["gate_probability"]) < float(selected_threshold["gate_threshold"]) for row in valid_predictions), "final_stage1_provenance": final_provenance, "inner_stage1_provenance": provenance, "selected_policy_type": selected_threshold["selected_policy_type"]}
        all_predictions.extend(valid_predictions)
    return all_predictions, threshold_reports, outer_reports


def _toy_action(query_id: str, doc: str, fold: int, score: float, gain: float, label: str, neural: float, baseline: float = .5) -> dict[str, Any]:
    return {"query_id": query_id, "fold": fold, "incoming_doc_id": doc, "drop_rank": 5, "baseline_top5": [f"{query_id}-{index}" for index in range(5)], "baseline_recall": baseline, "gain": gain, "label": label, "stage1_score": score, "stage1_p_benefit": max(score, 0.0), "stage1_p_harm": max(-score, 0.0), "features": {"incoming_neural_max": neural, "incoming_neural_mean": neural, "incoming_neural_top2_mean": neural, "incoming_neural_second": neural, "diff_neural_max": neural, "diff_neural_mean": neural, "diff_neural_second": neural}}


def self_test() -> dict[str, bool]:
    incumbent = _toy_action("q", "keep", 1, .8, 0.0, "NEUTRAL", 0.1)
    positive = _toy_action("q", "positive", 1, .7, .5, "NEUTRAL", .9)
    neutral = _toy_action("q", "neutral", 1, .6, 0.0, "HARM", .2)
    harm = _toy_action("q", "harm", 1, .5, -.5, "BENEFIT", -.2)
    neutral["stage1_p_harm"] = .1
    harm["stage1_p_harm"] = .2
    ranked = {"q": {"best": {**incumbent}, "second": {**positive}, "all": [{**incumbent}, {**positive}, {**neutral}, {**harm}]}}
    gate_rows, selector_rows = build_rows(ranked, .9, labeled=True)
    no_positive = {"q": {"best": {**incumbent}, "second": {**neutral}, "all": [{**incumbent}, {**neutral}, {**harm}]}}
    no_positive_gate, _ = build_rows(no_positive, .9, labeled=True)
    toy_prediction = {"query_id": "q", "fold": 1, "baseline_top5": incumbent["baseline_top5"], "incumbent": incumbent, "anchor_execute": False, "keep_recall": .5, "gate_probability": .9, "best_challenger": positive, "best_challenger_probability": .8}
    gate_blocks = not apply_thresholds([toy_prediction], 1.0, .5)[0]["execute"]
    selector_blocks = not apply_thresholds([toy_prediction], .5, 1.0)[0]["execute"]
    stable_rows = []
    for index, fold in enumerate((1, 2, 3)):
        stable_rows.append({**toy_prediction, "query_id": f"stable-{index}", "fold": fold, "incumbent": {**incumbent, "query_id": f"stable-{index}"}, "best_challenger": {**positive, "query_id": f"stable-{index}", "gain": .1}, "gate_probability": .9, "best_challenger_probability": .9})
    negative_rows = [*stable_rows[:2], {**stable_rows[2], "best_challenger": {**stable_rows[2]["best_challenger"], "gain": -.1}}]
    zero_rows = [{**stable_rows[0], "best_challenger": {**stable_rows[0]["best_challenger"], "gain": 0.0}}]
    stable_choice = choose_threshold(stable_rows); negative_choice = choose_threshold(negative_rows); zero_choice = choose_threshold(zero_rows)
    residual = policy_metrics(apply_thresholds([{**toy_prediction, "keep_recall": .6, "incumbent": {**incumbent, "gain": .1}, "best_challenger": {**positive, "gain": .2}}], .5, .5))
    anchor_abstain = apply_thresholds([{**toy_prediction, "incumbent": {**incumbent, "gain": .5}, "anchor_execute": False, "keep_recall": .5}], float("inf"), float("inf"))
    anchor_abstain_metrics = policy_metrics(anchor_abstain)
    anchor_execute = apply_thresholds([{**toy_prediction, "incumbent": {**incumbent, "incoming_doc_id": "X", "gain": .5}, "anchor_execute": True, "keep_recall": 1.0}], float("inf"), float("inf"))
    anchor_execute_metrics = policy_metrics(anchor_execute)
    residual_execute = apply_thresholds([{**toy_prediction, "anchor_execute": False, "keep_recall": .5}], .5, .5)
    residual_execute_metrics = policy_metrics(residual_execute)
    mixed_keep_rows = [
        {**toy_prediction, "query_id": "mixed-execute", "incumbent": {**incumbent, "query_id": "mixed-execute", "gain": .5}, "anchor_execute": True, "keep_recall": 1.0, "best_challenger": None, "gate_probability": 0.0, "best_challenger_probability": 0.0},
        {**toy_prediction, "query_id": "mixed-abstain", "incumbent": {**incumbent, "query_id": "mixed-abstain", "gain": .5}, "anchor_execute": False, "keep_recall": .5, "best_challenger": None, "gate_probability": 0.0, "best_challenger_probability": 0.0},
    ]
    mixed_keep_metrics = policy_metrics(apply_thresholds(mixed_keep_rows, float("inf"), float("inf")))
    mixed_keep_reference = abs(mixed_keep_metrics["keep_macro_recall"] - .75) < 1e-12 and abs(mixed_keep_metrics["policy_macro_recall"] - .75) < 1e-12
    no_selected_serialization = bool(json.dumps({"selected_action": anchor_abstain[0]["selected"], "selected_top5": anchor_abstain[0]["selected_top5"], "delta_vs_keep": prediction_delta_vs_keep(anchor_abstain[0])}))
    provenance = all(held not in folds for held in (1, 2, 3) for folds in inner_stage1_plan((1, 2, 3), held)["training_feature_stage1_train_folds"].values())
    outer_plan = outer_stage1_plan(4)
    fold0_rejected = False
    try:
        validate_actions([{"query_id": "fold0", "fold": 0}])
    except ValueError:
        fold0_rejected = True
    oracle_toy = {"q": {"best": {**incumbent, "baseline_recall": .5, "gain": 0.0}, "all": [{**incumbent}, {**positive}]}}
    oracle_monotonic = oracle_macro(oracle_toy, None, 20) + 1e-12 >= oracle_macro(oracle_toy, None, 10)
    partition_actions = [{"query_id": f"partition-{fold}", "fold": fold, "label": "NEUTRAL", "gain": 0.0, "baseline_recall": .5, "features": {"f": float(fold)}} for fold in (1, 2, 3, 4)]
    partition_input, partition_valid_labeled = outer_stage1_input(partition_actions, (1, 2, 3), 4)
    partition_train = [row for row in partition_input if int(row["fold"]) in {1, 2, 3}]
    partition_valid = [row for row in partition_input if int(row["fold"]) == 4]
    outer_partition_regression = bool(partition_train and partition_valid and {int(row["fold"]) for row in partition_train} == {1, 2, 3} and {int(row["fold"]) for row in partition_valid} == {4} and all(not any(key in row for key in OUTCOME_KEYS) for row in partition_valid) and all(int(row["fold"]) != 4 for row in partition_train) and all(all(key in row for key in ("label", "gain", "baseline_recall")) for row in partition_train) and any(int(row["fold"]) == 4 for row in partition_valid_labeled))
    weights = selector_weights(selector_rows)
    query_weighted = abs(float(weights.sum()) - 1.0) < 1e-6
    return {"fold0_rejected": fold0_rejected, "gate_positive_target": gate_rows[0]["target"] == 1, "gate_negative_target": no_positive_gate[0]["target"] == 0, "selector_positive_target": selector_rows[0]["target"] == 1, "selector_neutral_target": selector_rows[1]["target"] == 0, "selector_harm_target": selector_rows[2]["target"] == 0, "gate_schema_no_outcomes": not any(name in OUTCOME_KEYS for name in GATE_NAMES), "selector_schema_no_outcomes": not any(name in OUTCOME_KEYS for name in SELECTOR_NAMES), "query_normalized_selector_weights": query_weighted, "no_benefit_weighting": weights[0] == weights[1], "selector_excludes_keep": all(row["action"]["incoming_doc_id"] != "keep" for row in selector_rows), "one_final_selection": len(apply_thresholds([toy_prediction], .5, .5)) == 1, "gate_threshold_blocks": gate_blocks, "selector_threshold_blocks": selector_blocks, "stable_pair_selected": stable_choice["selected_policy_type"] == "OPPORTUNITY_GATED_DELTA_SELECTOR", "negative_inner_fold_rejected": negative_choice["selected_policy_type"] == "KEEP_STAGE1", "no_stable_pair_keeps": zero_choice["selected_policy_type"] == "KEEP_STAGE1", "residual_is_relative_to_keep": residual["delta_vs_keep"] > 0 and residual["delta_vs_baseline"] > residual["delta_vs_keep"], "anchor_abstain_keep": anchor_abstain[0]["selected"] is None and anchor_abstain[0]["selected_top5"] == incumbent["baseline_top5"] and anchor_abstain_metrics["policy_macro_recall"] == .5 and anchor_abstain_metrics["keep_macro_recall"] == .5 and anchor_abstain_metrics["delta_vs_keep"] == 0.0 and anchor_abstain_metrics["correction_count"] == 0, "anchor_execute_keep": anchor_execute[0]["selected"] == anchor_execute[0]["incumbent"] and anchor_execute[0]["selected_top5"][4] == "X" and anchor_execute_metrics["policy_macro_recall"] == 1.0 and anchor_execute_metrics["keep_macro_recall"] == 1.0 and anchor_execute_metrics["delta_vs_keep"] == 0.0 and anchor_execute_metrics["correction_count"] == 0, "residual_challenger_execute": residual_execute[0]["selected"] == residual_execute[0]["best_challenger"] and residual_execute[0]["selected_top5"][4] == "positive" and residual_execute[0]["selected_top5"][:3] == incumbent["baseline_top5"][:3] and abs(residual_execute_metrics["delta_vs_keep"] - .5) < 1e-12, "mixed_keep_reference": mixed_keep_reference, "no_selected_action_serialization": no_selected_serialization, "inner_stage1_provenance": provenance, "outer_fold_provenance_contract": outer_plan["held_outer_absent"] and 4 not in outer_plan["stage1_train_folds"], "outer_stage1_partition": outer_partition_regression, "top_k_20": TOP_K == 20, "oracle_k20_monotonic": oracle_monotonic}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path); parser.add_argument("--v3a-report", type=Path); parser.add_argument("--v3b-report", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "top_k": TOP_K, "gate_config": GATE_CONFIG, "selector_config": SELECTOR_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "folds": list(FOLDS), "fold0_labels_read": False, "gpu_launched": False, "no_submission_created": True}, indent=2)); return
    if any(value is None for value in (args.actions, args.v3a_report, args.v3b_report, args.output_dir)):
        parser.error("--actions, --v3a-report, --v3b-report, and --output-dir are required")
    all_actions = jsonl(args.actions)
    actions = [row for row in all_actions if int(row["fold"]) in FOLDS]
    validate_actions(actions)
    if any(any(key not in row for key in ("label", "gain", "baseline_recall")) for row in actions):
        raise ValueError("folds1--4 actions must be labeled")
    names = feature_names(actions); config = fixed_stage1_config(args.v3b_report)
    report_v3b = json.loads(args.v3b_report.read_text(encoding="utf-8")); threshold = report_v3b["outer_cv"]["best_v3b_raw"].get("action_threshold")
    predictions, threshold_reports, outer_reports = outer_run(actions, names, config, threshold)
    metrics = policy_metrics(predictions)
    baseline_by_query = {str(row["query_id"]): row for row in actions}
    baseline_macro = float(np.mean([float(row["baseline_recall"]) for row in baseline_by_query.values()]))
    ranked_oof = {}
    for held in FOLDS:
        ranked_oof.update(_rank_stage1(actions, names, config, set(FOLDS) - {held}, {held}))
    v3a = v3a_oof(actions, names, load_v3a_config(args.v3a_report))
    fixed_keep = float(np.mean([keep_recall(item["best"], threshold) for item in ranked_oof.values()]))
    benefit_diagnostics = opportunity_diagnostics(ranked_oof, predictions, threshold)
    oracle_k10 = oracle_macro(ranked_oof, threshold, 10); oracle_k20 = oracle_macro(ranked_oof, threshold, TOP_K)
    if abs(oracle_k20 - ORACLE_K20_REFERENCE) > 1e-6:
        raise AssertionError(f"K20 oracle changed: {oracle_k20} != {ORACLE_K20_REFERENCE}")
    if oracle_k20 + 1e-12 < oracle_k10:
        raise AssertionError("K20 oracle is lower than K10")
    selected_types = {str(report["selected_policy_type"]) for report in outer_reports.values()}
    selected_policy = "KEEP_STAGE1" if selected_types == {"KEEP_STAGE1"} else "OPPORTUNITY_GATED_DELTA_SELECTOR"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positive_capture_count = int(sum(bool(row["execute"]) and prediction_delta_vs_keep(row) > 0 for row in predictions))
    positive_capture_rate = positive_capture_count / benefit_diagnostics["positive_opportunity_query_count_top20"] if benefit_diagnostics["positive_opportunity_query_count_top20"] else 0.0
    report = {"status": "OPPORTUNITY_GATED_DELTA_SELECTOR_OOF_COMPLETE", "architecture": "fixed V3B Stage1 -> KEEP_STAGE1 + next TOP20 -> opportunity gate -> challenger selector -> nested threshold -> KEEP or one challenger", "top_k": TOP_K, "gate_feature_schema": list(GATE_NAMES), "selector_feature_schema": list(SELECTOR_NAMES), "selector_feature_within_query_variable_query_count": selector_variability(build_rows(ranked_oof, threshold, labeled=True)[1]), "gate_model_config": GATE_CONFIG, "selector_model_config": SELECTOR_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "anchor_threshold": threshold, "anchor_semantics": "execute Stage1 incumbent iff stage1_score >= fixed V3B action threshold", "baseline": {"macro_recall": baseline_macro}, "v3a": {"macro_recall": v3a["policy_macro_recall"]}, "fixed_v3b_keep": {"macro_recall": fixed_keep}, "opportunity_gated_delta_selector": {**metrics, "delta_vs_baseline": metrics["policy_macro_recall"] - baseline_macro, "delta_vs_v3a": metrics["policy_macro_recall"] - float(v3a["policy_macro_recall"]), "delta_vs_fixed_v3b_keep": metrics["policy_macro_recall"] - fixed_keep}, "per_outer_fold": {str(key): value for key, value in outer_reports.items()}, "threshold_by_outer_fold": threshold_reports, "nested_oof_provenance": {"outer_folds": list(FOLDS), "outer_training_excludes_held_outer": True, "inner_stage1_excludes_held_inner": True, "provenance": {str(key): value.get("inner_stage1_provenance", {}) for key, value in outer_reports.items()}}, "benefit_diagnostics": benefit_diagnostics, "positive_opportunities_top20": benefit_diagnostics["positive_opportunity_action_count_top20"], "positive_opportunity_query_count_top20": benefit_diagnostics["positive_opportunity_query_count_top20"], "positive_capture_count": positive_capture_count, "positive_capture_rate": positive_capture_rate, "queries_passing_gate": sum(value["queries_passing_gate"] for value in outer_reports.values()), "queries_failing_gate": sum(value["queries_failing_gate"] for value in outer_reports.values()), "oracle_references": {"k10_reference": ORACLE_K10_REFERENCE, "k20_reference": ORACLE_K20_REFERENCE, "exact_k10": oracle_k10, "exact_k20": oracle_k20, "k20_matches_reference": abs(oracle_k20 - ORACLE_K20_REFERENCE) <= 1e-6}, "selected_policy_type": selected_policy, "benefit_diagnostic_only": True, "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}
    (args.output_dir / "opportunity_gated_delta_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "opportunity_gated_delta_oof_predictions.jsonl", [{"query_id": row["query_id"], "fold": int(row["fold"]), "gate_probability": float(row["gate_probability"]), "selector_probability": float(row["best_challenger_probability"]), "execute": bool(row["execute"]), "selected_action": row["selected"], "selected_top5": row["selected_top5"], "delta_vs_keep": prediction_delta_vs_keep(row)} for row in predictions])


if __name__ == "__main__":
    main()
