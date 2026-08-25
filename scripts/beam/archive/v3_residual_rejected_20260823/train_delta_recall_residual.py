"""Metric-aligned V3 residual decision learner, folds 1--4 only.

The learner predicts whether a KEEP-vs-challenger decision has positive
delta-Recall.  Stage1 ranking and all threshold decisions are nested OOF.
Fold0 is intentionally never evaluated by this module.
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
    from .train_residual_policy import feature_names
    from .train_residual_policy_v3b import fit_stage1, rank_actions, load_v3a_config, v3a_oof
    from .forensic_stage1_hard_errors import fixed_stage1_config
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from train_residual_policy import feature_names
    from train_residual_policy_v3b import fit_stage1, rank_actions, load_v3a_config, v3a_oof
    from forensic_stage1_hard_errors import fixed_stage1_config


FOLDS = (1, 2, 3, 4)
TOP_K = 10
THRESHOLD_QUANTILES = (.50, .60, .70, .75, .80, .85, .90, .925, .95, .975, .99)
MODEL_CONFIG = {"objective": "binary:logistic", "tree_method": "hist", "n_estimators": 150, "max_depth": 3, "learning_rate": .05, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0, "random_state": 2032, "n_jobs": 16}


def validate_actions(actions: list[dict[str, Any]]) -> None:
    folds_by_query: dict[str, set[int]] = defaultdict(set)
    for row in actions:
        fold = int(row["fold"])
        if fold not in FOLDS:
            raise ValueError(f"delta residual accepts only folds1--4: {row['query_id']} fold={fold}")
        folds_by_query[str(row["query_id"])].add(fold)
    crossed = [query for query, folds in folds_by_query.items() if len(folds) != 1]
    if crossed:
        raise ValueError(f"query belongs to more than one fold: {crossed[:5]}")


def fixed_stage1_ranked(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Produce OOF fixed-V3B Stage1 rankings for the supplied folds."""
    ranked: dict[str, dict[str, Any]] = {}
    for held in sorted({int(row["fold"]) for row in actions}):
        train = [row for row in actions if int(row["fold"]) != held]
        valid = [row for row in actions if int(row["fold"]) == held]
        if not train or not valid:
            raise ValueError(f"empty Stage1 OOF split for fold {held}")
        ranked.update(rank_actions(fit_stage1(train, names, config), valid, names, float(config["harm_utility_weight"])))
    return ranked


def nested_stage1_train_ranked(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], held_outer: int) -> dict[str, dict[str, Any]]:
    inner = tuple(fold for fold in FOLDS if fold != held_outer)
    ranked: dict[str, dict[str, Any]] = {}
    for held in inner:
        train = [row for row in actions if int(row["fold"]) in set(inner) - {held}]
        valid = [row for row in actions if int(row["fold"]) == held]
        if len({int(row["fold"]) for row in train}) != 2 or not valid:
            raise ValueError("inner Stage1 split must train on two outer-train folds")
        ranked.update(rank_actions(fit_stage1(train, names, config), valid, names, float(config["harm_utility_weight"])))
    return ranked


def stage1_training_fold_plan(outer_train_folds: tuple[int, ...], held_inner: int) -> dict[str, Any]:
    """Return and validate the fully nested Stage1 provenance plan."""
    outer = tuple(sorted(int(fold) for fold in outer_train_folds))
    if held_inner not in outer or len(outer) != 3:
        raise ValueError("inner leakage plan requires three outer-training folds and a held inner fold")
    delta_train = tuple(fold for fold in outer if fold != held_inner)
    validation_train = tuple(delta_train)
    training_feature_train = {fold: tuple(other for other in delta_train if other != fold) for fold in delta_train}
    if set(validation_train) != set(delta_train):
        raise AssertionError("inner validation Stage1 provenance is incomplete")
    if any(held_inner in train_folds for train_folds in training_feature_train.values()):
        raise AssertionError("held inner fold leaked into Delta-training Stage1 features")
    if any(fold not in delta_train for train_folds in training_feature_train.values() for fold in train_folds):
        raise AssertionError("non-training fold reached Delta-training Stage1 features")
    return {"outer_train_folds": outer, "held_inner": int(held_inner), "delta_train_folds": delta_train, "validation_stage1_train_folds": validation_train, "training_feature_stage1_train_folds": training_feature_train, "held_inner_absent_from_delta_training_features": all(held_inner not in train_folds for train_folds in training_feature_train.values())}


def leakage_free_inner_delta_oof(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], outer_train_folds: tuple[int, ...], anchor_threshold: float | None) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, dict[str, Any]]]:
    """Build inner Delta OOF rows with fully nested Stage1 feature generation."""
    inner_records: list[dict[str, Any]] = []
    inner_scores: list[float] = []
    provenance: dict[int, dict[str, Any]] = {}
    for held_inner in sorted(outer_train_folds):
        plan = stage1_training_fold_plan(outer_train_folds, held_inner)
        delta_train = set(plan["delta_train_folds"])
        valid_actions = [row for row in actions if int(row["fold"]) == held_inner]
        valid_train_actions = [row for row in actions if int(row["fold"]) in delta_train]
        valid_ranked = rank_actions(fit_stage1(valid_train_actions, names, config), valid_actions, names, float(config["harm_utility_weight"]))
        train_ranked: dict[str, dict[str, Any]] = {}
        for train_fold, stage1_train_folds in plan["training_feature_stage1_train_folds"].items():
            train_actions = [row for row in actions if int(row["fold"]) in set(stage1_train_folds)]
            score_actions = [row for row in actions if int(row["fold"]) == train_fold]
            train_ranked.update(rank_actions(fit_stage1(train_actions, names, config), score_actions, names, float(config["harm_utility_weight"])))
        labeled_train_records, train_names = build_residual_records(train_ranked, names, anchor_threshold, labeled=True)
        unlabeled_valid_records, valid_names = build_residual_records(strip_outcomes(valid_ranked), names, anchor_threshold, labeled=False)
        if train_names != valid_names:
            raise AssertionError("nested inner train/inference residual schemas differ")
        model = fit_delta_model(labeled_train_records, train_names)
        scores = model.predict_proba(inference_matrix(unlabeled_valid_records, valid_names))[:, 1]
        labeled_valid_records, labeled_valid_names = build_residual_records(valid_ranked, names, anchor_threshold, labeled=True)
        if labeled_valid_names != train_names:
            raise AssertionError("nested inner labeled validation schema differs")
        inner_records.extend(labeled_valid_records)
        inner_scores.extend(scores.tolist())
        provenance[held_inner] = plan
    return inner_records, np.asarray(inner_scores, dtype="float32"), provenance


def outer_validation_ranked(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], held_outer: int) -> dict[str, dict[str, Any]]:
    train = [row for row in actions if int(row["fold"]) != held_outer]
    valid = [row for row in actions if int(row["fold"]) == held_outer]
    return rank_actions(fit_stage1(train, names, config), valid, names, float(config["harm_utility_weight"]))


def query_context(item: dict[str, Any]) -> dict[str, float]:
    scores = np.asarray([float(row["stage1_score"]) for row in item["all"]], dtype="float64")
    best, second = item["best"], item["second"]
    return {"top1_stage1_score": float(best["stage1_score"]), "top2_stage1_score": float(second["stage1_score"]), "top1_top2_margin": float(best["stage1_score"] - second["stage1_score"]), "mean_stage1_score": float(scores.mean()), "std_stage1_score": float(scores.std()), "median_stage1_score": float(np.median(scores)), "top1_minus_median": float(scores[0] - np.median(scores))}


def residual_feature_names(action_names: list[str]) -> list[str]:
    names = [*(f"candidate_{name}" for name in action_names), *(f"incumbent_{name}" for name in action_names), *(f"candidate_minus_incumbent_{name}" for name in action_names), "is_keep_stage1", "candidate_stage1_score", "incumbent_stage1_score", "candidate_minus_incumbent_stage1_score", "candidate_stage1_p_benefit", "incumbent_stage1_p_benefit", "p_benefit_difference", "candidate_stage1_p_harm", "incumbent_stage1_p_harm", "p_harm_difference", "stage1_rank", "candidate_score_z", "candidate_score_percentile", *query_context({"all": [{"stage1_score": 0.0}], "best": {"stage1_score": 0.0}, "second": {"stage1_score": 0.0}}).keys()]
    forbidden = ("label", "gain", "gold", "delta_recall", "resulting_recall")
    if any(any(token in name.lower() for token in forbidden) for name in names):
        raise AssertionError("outcome field leaked into residual feature schema")
    return names


def item_vector(candidate: dict[str, Any], incumbent: dict[str, Any], action_names: list[str], is_keep: bool, rank: int, context: dict[str, float], all_scores: np.ndarray) -> list[float]:
    candidate_values = [float(candidate["features"].get(name, 0.0)) for name in action_names]
    incumbent_values = [float(incumbent["features"].get(name, 0.0)) for name in action_names]
    candidate_stage1 = float(candidate["stage1_score"]); incumbent_stage1 = float(incumbent["stage1_score"])
    mean_score = float(all_scores.mean()); std_score = float(all_scores.std())
    values = [*candidate_values, *incumbent_values, *(a - b for a, b in zip(candidate_values, incumbent_values)), float(is_keep), candidate_stage1, incumbent_stage1, candidate_stage1 - incumbent_stage1, float(candidate["stage1_p_benefit"]), float(incumbent["stage1_p_benefit"]), float(candidate["stage1_p_benefit"] - incumbent["stage1_p_benefit"]), float(candidate["stage1_p_harm"]), float(incumbent["stage1_p_harm"]), float(candidate["stage1_p_harm"] - incumbent["stage1_p_harm"]), float(rank), (candidate_stage1 - mean_score) / std_score if std_score else 0.0, float(np.mean(all_scores <= candidate_stage1)), *[float(context[name]) for name in query_context({"all": [{"stage1_score": value} for value in all_scores], "best": {"stage1_score": float(all_scores[0])}, "second": {"stage1_score": float(all_scores[1] if len(all_scores) > 1 else all_scores[0])}}).keys()]]
    return values


def build_residual_records(ranked: dict[str, dict[str, Any]], action_names: list[str], anchor_threshold: float | None, *, labeled: bool) -> tuple[list[dict[str, Any]], list[str]]:
    names = residual_feature_names(action_names); records = []
    for qid, query_id in enumerate(sort_query_ids(ranked)):
        item = ranked[query_id]; incumbent = item["best"]; all_scores = np.asarray([float(row["stage1_score"]) for row in item["all"]], dtype="float64"); context = query_context(item); anchor_execute = anchor_threshold is None or float(incumbent["stage1_score"]) >= float(anchor_threshold)
        keep = {"qid": qid, "query_id": query_id, "fold": int(incumbent["fold"]), "is_keep_stage1": 1, "stage1_rank": 0, "x": item_vector(incumbent, incumbent, action_names, True, 0, context, all_scores), "action": incumbent, "incumbent": incumbent, "anchor_execute": anchor_execute}
        query_records = [keep]
        for rank, candidate in enumerate(item["all"][1:TOP_K + 1], start=1):
            query_records.append({"qid": qid, "query_id": query_id, "fold": int(candidate["fold"]), "is_keep_stage1": 0, "stage1_rank": rank, "x": item_vector(candidate, incumbent, action_names, False, rank, context, all_scores), "action": candidate, "incumbent": incumbent, "anchor_execute": anchor_execute})
        if labeled:
            keep_recall = float(incumbent["baseline_recall"]) + (float(incumbent["gain"]) if anchor_execute else 0.0)
            keep["delta_recall"] = 0.0; keep["target"] = 0
            for record in query_records[1:]:
                challenger_recall = float(record["action"]["baseline_recall"]) + float(record["action"]["gain"])
                record["delta_recall"] = challenger_recall - keep_recall; record["target"] = int(record["delta_recall"] > 0.0)
        records.extend(query_records)
    assert_residual_contract(records, names, labeled=labeled)
    return records, names


def assert_residual_contract(records: list[dict[str, Any]], names: list[str], *, labeled: bool) -> None:
    previous = None; closed: set[int] = set(); keep_counts = Counter()
    for record in records:
        if int(record["fold"]) not in FOLDS: raise AssertionError("fold0/non-training record reached residual learner")
        qid = int(record["qid"])
        if previous is not None and qid != previous: closed.add(previous)
        if qid in closed: raise AssertionError("query records are not contiguous")
        if len(record["x"]) != len(names): raise AssertionError("train/inference schema mismatch")
        keep_counts[qid] += int(record["is_keep_stage1"]); previous = qid
        if not labeled and any(key in record for key in ("target", "delta_recall", "resulting_recall")): raise AssertionError("outcome leaked into inference record")
    if any(value != 1 for value in keep_counts.values()): raise AssertionError("exactly one KEEP_STAGE1 is required per query")


def training_matrix(records: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_residual_contract(records, names, labeled=True)
    x = np.asarray([record["x"] for record in records], dtype="float32"); y = np.asarray([record["target"] for record in records], dtype="int32")
    counts = Counter(str(record["query_id"]) for record in records); weights = np.asarray([1.0 / counts[str(record["query_id"])] for record in records], dtype="float32")
    return x, y, weights


def inference_matrix(records: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    assert_residual_contract(records, names, labeled=False)
    if any({"label", "gain", "gold", "delta_recall", "resulting_recall"}.intersection(record) for record in records): raise AssertionError("outcome field in inference records")
    return np.asarray([record["x"] for record in records], dtype="float32")


def fit_delta_model(records: list[dict[str, Any]], names: list[str]):
    import xgboost as xgb
    if xgb.__version__ != "3.2.0": raise RuntimeError(f"xgboost==3.2.0 required, found {xgb.__version__}")
    x, y, weights = training_matrix(records, names)
    if len(set(y.tolist())) < 2: raise ValueError("residual training split needs positive and nonpositive delta rows")
    model = xgb.XGBClassifier(**MODEL_CONFIG); model.fit(x, y, sample_weight=weights, verbose=False)
    return model


def _group(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, score in zip(records, scores): grouped[str(record["query_id"])].append({**record, "promotion_probability": float(score)})
    return grouped


def select_with_threshold(records: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    selected = []
    for query_id, rows in sorted(_group(records, scores).items()):
        keep = next(row for row in rows if row["is_keep_stage1"])
        challengers = [row for row in rows if not row["is_keep_stage1"]]
        best = sorted(challengers, key=lambda row: (-float(row["promotion_probability"]), int(row["stage1_rank"]), str(row["action"].get("incoming_doc_id", ""))))[0] if challengers else None
        selected.append(best if best is not None and float(best["promotion_probability"]) >= threshold else keep)
    return selected


def keep_gain(record: dict[str, Any]) -> float:
    """Recall gain of the fixed KEEP_STAGE1 anchor for one query."""
    return float(record["incumbent"]["gain"]) if record["anchor_execute"] else 0.0


def selected_policy_gain(record: dict[str, Any]) -> float:
    """Actual selected gain versus the raw baseline."""
    return keep_gain(record) if record["is_keep_stage1"] else float(record["action"]["gain"])


def residual_delta(record: dict[str, Any]) -> float:
    """Actual selected policy gain relative to KEEP_STAGE1."""
    return selected_policy_gain(record) - keep_gain(record)


def labeled_metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    by_fold: dict[int, list[tuple[float, float]]] = defaultdict(list); counts = Counter(); before = after = 0.0; positive_residual_gain = negative_residual_loss = 0.0
    for record in selected:
        baseline = float(record["incumbent"]["baseline_recall"]); gain = selected_policy_gain(record); delta = residual_delta(record); before += baseline; after += baseline + gain; by_fold[int(record["fold"])].append((baseline, baseline + gain))
        is_correction = not record["is_keep_stage1"]
        counts["keep" if not is_correction else "correction"] += 1
        if is_correction:
            counts["positive" if delta > 0 else "zero" if delta == 0 else "negative"] += 1
            positive_residual_gain += max(delta, 0.0); negative_residual_loss += min(delta, 0.0)
    fold_baseline = {str(fold): float(np.mean([before for before, _ in values])) for fold, values in by_fold.items()}
    fold_policy = {str(fold): float(np.mean([after for _, after in values])) for fold, values in by_fold.items()}
    fold_delta = {fold: fold_policy[fold] - fold_baseline[fold] for fold in fold_baseline}
    total = len(selected); swaps = counts["correction"]
    return {"query_count": total, "macro_recall": after / total, "baseline_macro_recall": before / total, "delta_vs_baseline": (after - before) / total, "keep_count": counts["keep"], "correction_count": swaps, "positive_delta_corrections": counts["positive"], "zero_delta_corrections": counts["zero"], "negative_delta_corrections": counts["negative"], "beneficial_correction_precision": counts["positive"] / swaps if swaps else 0.0, "harmful_correction_rate": counts["negative"] / swaps if swaps else 0.0, "positive_residual_recall_gain": positive_residual_gain, "negative_residual_recall_loss": negative_residual_loss, "per_fold_baseline_macro_recall": fold_baseline, "per_fold_policy_macro_recall": fold_policy, "per_fold_delta": fold_delta, "min_fold_delta": min(fold_delta.values()) if fold_delta else 0.0, "negative_fold_count": sum(value < 0 for value in fold_delta.values())}


def threshold_candidates(records: list[dict[str, Any]], scores: np.ndarray) -> list[float]:
    values = np.asarray([float(score) for record, score in zip(records, scores) if not record["is_keep_stage1"]], dtype="float64")
    return [float("inf")] if not len(values) else [float("inf"), *sorted({float(np.quantile(values, q)) for q in THRESHOLD_QUANTILES})]


def choose_threshold(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    candidates = threshold_candidates(records, scores); options = []
    keep_metrics = labeled_metrics(select_with_threshold(records, scores, float("inf")))
    for threshold in candidates:
        selected = select_with_threshold(records, scores, threshold); metrics = labeled_metrics(selected)
        per_fold_delta_vs_keep = {fold: float(metrics["per_fold_policy_macro_recall"][fold]) - float(keep_metrics["per_fold_policy_macro_recall"][fold]) for fold in metrics["per_fold_policy_macro_recall"]}
        delta_vs_keep = float(metrics["macro_recall"]) - float(keep_metrics["macro_recall"])
        eligible = bool(np.isfinite(threshold) and delta_vs_keep > 0.0 and all(value >= 0.0 for value in per_fold_delta_vs_keep.values()))
        options.append({"threshold": threshold, "macro_recall": metrics["macro_recall"], "delta_vs_baseline": metrics["delta_vs_baseline"], "keep_macro_recall": keep_metrics["macro_recall"], "delta_vs_keep": delta_vs_keep, "per_fold_policy_macro_recall": metrics["per_fold_policy_macro_recall"], "per_fold_delta_vs_keep": per_fold_delta_vs_keep, "negative_fold_count_vs_keep": sum(value < 0.0 for value in per_fold_delta_vs_keep.values()), "positive_delta_corrections": metrics["positive_delta_corrections"], "zero_delta_corrections": metrics["zero_delta_corrections"], "negative_delta_corrections": metrics["negative_delta_corrections"], "correction_count": metrics["correction_count"], "beneficial_correction_precision": metrics["beneficial_correction_precision"], "harmful_correction_rate": metrics["harmful_correction_rate"], "eligible_vs_keep": eligible})
    stable = [row for row in options if row["eligible_vs_keep"]]
    if not stable:
        keep_option = next(row for row in options if not np.isfinite(row["threshold"]))
        return {"threshold": float("inf"), "selected_policy_type": "KEEP_STAGE1", "selection_reason": "no stable threshold beats KEEP_STAGE1", "keep_reference": keep_metrics, "selected_candidate": keep_option, "candidates": options}
    chosen = max(stable, key=lambda row: (float(row["macro_recall"]), -int(row["negative_fold_count_vs_keep"]), -int(row["negative_delta_corrections"]), float(row["beneficial_correction_precision"]), -int(row["correction_count"]), float(row["threshold"])))
    return {"threshold": float(chosen["threshold"]), "selected_policy_type": "DELTA_RECALL_RESIDUAL", "selection_reason": "stable positive inner-OOF macro Recall versus KEEP_STAGE1", "keep_reference": keep_metrics, "selected_candidate": chosen, "candidates": options}


def strip_outcomes(ranked: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcome = {"label", "gain", "gold", "baseline_recall", "delta_recall", "resulting_recall"}
    clean = {}
    for query_id, item in ranked.items():
        clean[query_id] = {key: value for key, value in item.items() if key not in {"best", "second", "all"}}
        clean[query_id].update({"best": {key: value for key, value in item["best"].items() if key not in outcome}, "second": {key: value for key, value in item["second"].items() if key not in outcome}, "all": [{key: value for key, value in row.items() if key not in outcome} for row in item["all"]]})
    return clean


def attach_labels(selected: list[dict[str, Any]], ranked: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in selected:
        item = ranked[record["query_id"]]; chosen_id = item["best"]["incoming_doc_id"] if record["is_keep_stage1"] else record["action"]["incoming_doc_id"]; action = next(row for row in item["all"] if str(row["incoming_doc_id"]) == str(chosen_id) and int(row["drop_rank"]) == int(item["best"]["drop_rank"] if record["is_keep_stage1"] else record["action"]["drop_rank"]))
        result.append({**record, "action": action, "incumbent": item["best"]})
    return result


def outer_run(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], anchor_threshold: float | None) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]], dict[int, dict[int, dict[str, Any]]]]:
    selected_all = []; threshold_by_fold = {}; ranked_by_fold = {}; provenance_by_outer: dict[int, dict[int, dict[str, Any]]] = {}
    for held_outer in FOLDS:
        outer_train_folds = tuple(fold for fold in FOLDS if fold != held_outer)
        inner_ranked = nested_stage1_train_ranked(actions, names, config, held_outer)
        inner_oof_records, inner_oof_scores, provenance_by_inner = leakage_free_inner_delta_oof(actions, names, config, outer_train_folds, anchor_threshold)
        provenance_by_outer[held_outer] = provenance_by_inner
        threshold = choose_threshold(inner_oof_records, np.asarray(inner_oof_scores, dtype="float32")); threshold_by_fold[held_outer] = threshold
        final_train_records, final_names = build_residual_records(inner_ranked, names, anchor_threshold, labeled=True)
        final_model = fit_delta_model(final_train_records, final_names)
        valid_ranked = outer_validation_ranked(actions, names, config, held_outer); ranked_by_fold[held_outer] = valid_ranked
        valid_unlabeled, valid_names = build_residual_records(strip_outcomes(valid_ranked), names, anchor_threshold, labeled=False)
        if final_names != valid_names: raise AssertionError("train/inference feature schemas differ")
        valid_scores = final_model.predict_proba(inference_matrix(valid_unlabeled, valid_names))[:, 1]
        selected = select_with_threshold(valid_unlabeled, valid_scores, float(threshold["threshold"]))
        selected_all.extend(attach_labels(selected, valid_ranked))
    return selected_all, threshold_by_fold, ranked_by_fold, provenance_by_outer


def self_test() -> dict[str, bool]:
    toy = {"q": {"best": {"fold": 1, "stage1_score": .8, "stage1_p_benefit": .8, "stage1_p_harm": .0, "features": {"f": 2.0}, "baseline_recall": .5, "gain": 0.0, "label": "NEUTRAL", "incoming_doc_id": "base", "drop_rank": 5}, "second": {"fold": 1, "stage1_score": .5, "stage1_p_benefit": .5, "stage1_p_harm": .0, "features": {"f": 3.0}, "baseline_recall": .5, "gain": .5, "label": "BENEFIT", "incoming_doc_id": "good", "drop_rank": 5}, "all": []}}
    neutral = {**toy["q"]["second"], "stage1_score": .4, "stage1_p_benefit": .4, "features": {"f": 4.0}, "gain": 0.0, "label": "NEUTRAL", "incoming_doc_id": "neutral"}
    harm = {**toy["q"]["second"], "stage1_score": .3, "stage1_p_benefit": .3, "features": {"f": 5.0}, "gain": -.5, "label": "HARM", "incoming_doc_id": "harm"}
    toy["q"]["all"] = [toy["q"]["best"], toy["q"]["second"], neutral, harm]
    records, names = build_residual_records(toy, ["f"], .9, labeled=True); clean, clean_names = build_residual_records(strip_outcomes(toy), ["f"], .9, labeled=False)
    x, y, weights = training_matrix(records, names); no_outcome = inference_matrix(clean, clean_names)
    selected = select_with_threshold(clean, np.asarray([.1, .9, .8, .7]), .5)

    def threshold_rows(specs: list[tuple[int, float, float, float, float]]) -> tuple[list[dict[str, Any]], np.ndarray]:
        rows: list[dict[str, Any]] = []; scores: list[float] = []
        for index, (fold, baseline, keep, challenger, probability) in enumerate(specs):
            incumbent = {"query_id": f"threshold-{index}", "fold": fold, "baseline_recall": baseline, "gain": keep, "incoming_doc_id": f"keep-{index}", "drop_rank": 5}
            keep_row = {"query_id": f"threshold-{index}", "fold": fold, "is_keep_stage1": 1, "anchor_execute": True, "incumbent": incumbent, "action": incumbent, "stage1_rank": 0}
            candidate = {"query_id": f"threshold-{index}", "fold": fold, "baseline_recall": baseline, "gain": challenger, "incoming_doc_id": f"candidate-{index}", "drop_rank": 5}
            candidate_row = {"query_id": f"threshold-{index}", "fold": fold, "is_keep_stage1": 0, "anchor_execute": True, "incumbent": incumbent, "action": candidate, "stage1_rank": 1}
            rows.extend((keep_row, candidate_row)); scores.extend((0.0, probability))
        return rows, np.asarray(scores, dtype="float32")

    # A: candidate is above raw baseline but below KEEP_STAGE1.
    test_a_rows, test_a_scores = threshold_rows([(1, .2, .2, .1, .9)])
    test_a = choose_threshold(test_a_rows, test_a_scores)
    # B: pooled candidate improves, but one inner fold loses to KEEP.
    test_b_rows, test_b_scores = threshold_rows([(1, .5, 0.0, .2, .9), (2, .5, 0.0, .2, .9), (3, .5, 0.0, -.1, .9)])
    test_b = choose_threshold(test_b_rows, test_b_scores)
    # C: candidate beats KEEP on every available inner fold.
    test_c_rows, test_c_scores = threshold_rows([(1, .5, 0.0, .1, .9), (2, .5, 0.0, .1, .9), (3, .5, 0.0, .1, .9)])
    test_c = choose_threshold(test_c_rows, test_c_scores)
    test_d_rows, _ = threshold_rows([(1, .5, .5, 0.0, .9)])
    test_e_rows, _ = threshold_rows([(1, .5, 0.0, .5, .9)])
    test_f_rows, _ = threshold_rows([(1, .5, 0.0, 0.0, .9)])
    test_d = labeled_metrics([test_d_rows[1]]); test_e = labeled_metrics([test_e_rows[1]]); test_f = labeled_metrics([test_f_rows[1]])
    provenance_plan = stage1_training_fold_plan((1, 2, 3), 1)
    provenance_expected = provenance_plan["validation_stage1_train_folds"] == (2, 3) and provenance_plan["training_feature_stage1_train_folds"] == {2: (3,), 3: (2,)}
    provenance_all_inner = all(stage1_training_fold_plan((1, 2, 3), held)["held_inner_absent_from_delta_training_features"] for held in (1, 2, 3))
    fold0_rejected = False
    try:
        validate_actions([{"query_id": "bad", "fold": 0}])
    except ValueError:
        fold0_rejected = True
    required_candidate_fields = {"threshold", "macro_recall", "delta_vs_baseline", "keep_macro_recall", "delta_vs_keep", "per_fold_policy_macro_recall", "per_fold_delta_vs_keep", "positive_delta_corrections", "zero_delta_corrections", "negative_delta_corrections", "eligible_vs_keep"}
    candidate_report_fields = all(required_candidate_fields.issubset(row) for row in test_c["candidates"])
    return {"positive_delta_target": records[1]["target"] == 1 and records[1]["delta_recall"] > 0, "neutral_delta_target": records[2]["target"] == 0 and records[2]["delta_recall"] == 0, "negative_delta_target": records[3]["target"] == 0 and records[3]["delta_recall"] < 0, "keep_zero_delta_target": records[0]["target"] == 0 and records[0]["delta_recall"] == 0, "inference_outcome_free": no_outcome.shape[1] == len(names), "same_schema": names == clean_names, "query_normalized_weights": abs(float(weights.sum()) - 1.0) < 1e-6, "one_keep_per_query": sum(record["is_keep_stage1"] for record in records) == 1, "top10_cap": len(records) <= 11, "max_one_selection": len(selected) == 1, "keep_action_threshold_semantics": records[0]["anchor_execute"] is False, "threshold_a_raw_baseline_but_below_keep_rejected": test_a["selected_policy_type"] == "KEEP_STAGE1" and test_a["selection_reason"] == "no stable threshold beats KEEP_STAGE1", "threshold_b_one_negative_fold_rejected": test_b["selected_policy_type"] == "KEEP_STAGE1", "threshold_c_stable_gain_selected": test_c["selected_policy_type"] == "DELTA_RECALL_RESIDUAL" and bool(np.isfinite(test_c["threshold"])), "residual_negative_correction": test_d["negative_delta_corrections"] == 1 and test_d["zero_delta_corrections"] == 0, "residual_positive_correction": test_e["positive_delta_corrections"] == 1, "residual_neutral_correction": test_f["zero_delta_corrections"] == 1, "candidate_report_fields": candidate_report_fields, "inner_provenance_expected": provenance_expected, "inner_provenance_all_held_folds": provenance_all_inner, "fold0_rejected": fold0_rejected}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path); parser.add_argument("--v3a-report", type=Path); parser.add_argument("--v3b-report", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()};
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "model_config": MODEL_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "folds": list(FOLDS), "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if any(value is None for value in (args.actions, args.v3a_report, args.v3b_report, args.output_dir)): parser.error("--actions, --v3a-report, --v3b-report, and --output-dir are required")
    all_actions = jsonl(args.actions); actions = [row for row in all_actions if int(row["fold"]) in FOLDS]; validate_actions(actions)
    if any("label" not in row or "gain" not in row or "baseline_recall" not in row for row in actions): raise ValueError("folds1--4 must contain labels/gains/baseline_recall")
    names = feature_names(actions); config = fixed_stage1_config(args.v3b_report); threshold = json.loads(args.v3b_report.read_text(encoding="utf-8"))["outer_cv"]["best_v3b_raw"].get("action_threshold")
    selected, thresholds, ranked_by_fold, inner_stage1_provenance = outer_run(actions, names, config, threshold)
    residual_metrics = labeled_metrics(selected)
    baseline = {"query_count": len({str(row["query_id"]) for row in actions}), "macro_recall": float(np.mean([float(row["baseline_recall"]) for row in {str(row["query_id"]): row for row in actions}.values()]))}
    v3a_metrics = v3a_oof(actions, names, load_v3a_config(args.v3a_report)); fixed_rows = []
    for ranked in ranked_by_fold.values():
        for item in ranked.values(): fixed_rows.append({"incumbent": item["best"], "is_keep_stage1": 1, "anchor_execute": threshold is None or float(item["best"]["stage1_score"]) >= float(threshold), "fold": int(item["best"]["fold"])})
    fixed = {"macro_recall": float(np.mean([float(row["incumbent"]["baseline_recall"]) + (float(row["incumbent"]["gain"]) if row["anchor_execute"] else 0.0) for row in fixed_rows]))}
    ranked_items = [item for ranked in ranked_by_fold.values() for item in ranked.values()]
    selected_by_query = {str(row["query_id"]): row for row in selected}
    any_benefit = sum(any(float(action["gain"]) > 0 for action in item["all"]) for item in ranked_items)
    stage1_benefit = sum(str(item["best"]["label"]) == "BENEFIT" for item in ranked_items)
    selected_benefit = sum(str((row["incumbent"] if row["is_keep_stage1"] else row["action"])["label"]) == "BENEFIT" for row in selected)
    positive_opportunities = 0; positive_captured = 0; oracle_total = 0.0; selected_positive_residual_gain = 0.0; selected_negative_residual_loss = 0.0
    for item in ranked_items:
        incumbent = item["best"]; anchor = threshold is None or float(incumbent["stage1_score"]) >= float(threshold); base = float(incumbent["baseline_recall"]); keep_recall = base + (float(incumbent["gain"]) if anchor else 0.0)
        challengers = item["all"][1:TOP_K + 1]
        positive_opportunities += sum(float(action["gain"]) - (float(incumbent["gain"]) if anchor else 0.0) > 0 for action in challengers)
        oracle_total += max([keep_recall, *[base + float(action["gain"]) for action in challengers]])
        row = selected_by_query[str(item["best"]["query_id"])]
        selected_recall = base + selected_policy_gain(row); selected_delta = residual_delta(row)
        positive_captured += int(not row["is_keep_stage1"] and selected_delta > 0)
        selected_positive_residual_gain += max(selected_delta, 0.0); selected_negative_residual_loss += min(selected_delta, 0.0)
    benefit_recovered = sum(str(item["best"]["label"]) != "BENEFIT" and str((selected_by_query[str(item["best"]["query_id"])] ["incumbent"] if selected_by_query[str(item["best"]["query_id"])] ["is_keep_stage1"] else selected_by_query[str(item["best"]["query_id"])] ["action"])["label"]) == "BENEFIT" for item in ranked_items)
    benefit_lost = sum(str(item["best"]["label"]) == "BENEFIT" and str((selected_by_query[str(item["best"]["query_id"])] ["incumbent"] if selected_by_query[str(item["best"]["query_id"])] ["is_keep_stage1"] else selected_by_query[str(item["best"]["query_id"])] ["action"])["label"]) != "BENEFIT" for item in ranked_items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "DELTA_RECALL_RESIDUAL_OOF_COMPLETE", "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True, "architecture": "fixed V3B Stage1 -> KEEP_STAGE1 + next top10 -> selective delta-Recall residual decision", "model_config": MODEL_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "baseline": baseline, "v3a": {"macro_recall": v3a_metrics["policy_macro_recall"]}, "fixed_v3b_stage1": fixed, "delta_recall_residual": {**residual_metrics, "delta_vs_baseline": residual_metrics["delta_vs_baseline"], "delta_vs_v3a": residual_metrics["macro_recall"] - float(v3a_metrics["policy_macro_recall"]), "delta_vs_fixed_stage1": residual_metrics["macro_recall"] - fixed["macro_recall"], "positive_residual_recall_gain_captured": selected_positive_residual_gain, "negative_residual_recall_loss_introduced": selected_negative_residual_loss}, "threshold_by_outer_fold": thresholds, "inner_stage1_feature_provenance": inner_stage1_provenance, "queries_with_any_BENEFIT": any_benefit, "stage1_benefit_top1_before": stage1_benefit, "selected_benefit_top1_after": selected_benefit, "benefit_recovered": benefit_recovered, "benefit_lost": benefit_lost, "positive_challenger_opportunities": positive_opportunities, "positive_challenger_capture_rate": positive_captured / positive_opportunities if positive_opportunities else 0.0, "exact_keep_plus_top10_oracle_macro_recall": oracle_total / len(ranked_items) if ranked_items else 0.0, "feature_schema": residual_feature_names(names)}
    (args.output_dir / "delta_recall_residual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "delta_recall_residual_oof_predictions.jsonl", [{"query_id": row["query_id"], "fold": row["fold"], "selection": "KEEP_STAGE1" if row["is_keep_stage1"] else "CHALLENGER", "correction": not row["is_keep_stage1"], "selected_gain": selected_policy_gain(row), "residual_delta": residual_delta(row)} for row in selected])


if __name__ == "__main__": main()
