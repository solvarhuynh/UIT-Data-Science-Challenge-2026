"""Opportunity-gated Delta selector V2: conditional selector training only.

V1 is intentionally imported, not modified.  V2 keeps its fixed V3B Stage1,
TOP20 window, feature schemas, models, KEEP semantics, and threshold policy.
Only selector *training* excludes queries with no positive residual action.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import jsonl, write_jsonl
    from .train_residual_policy import feature_names
    from .train_residual_policy_v3b import load_v3a_config, v3a_oof
    from .forensic_stage1_hard_errors import fixed_stage1_config
    from . import train_opportunity_gated_delta_selector as v1
except ImportError:
    from common import jsonl, write_jsonl
    from train_residual_policy import feature_names
    from train_residual_policy_v3b import load_v3a_config, v3a_oof
    from forensic_stage1_hard_errors import fixed_stage1_config
    import train_opportunity_gated_delta_selector as v1


FOLDS = v1.FOLDS
TOP_K = v1.TOP_K
THRESHOLD_QUANTILES = v1.THRESHOLD_QUANTILES
GATE_NAMES = v1.GATE_NAMES
SELECTOR_NAMES = v1.SELECTOR_NAMES
GATE_CONFIG = v1.GATE_CONFIG
SELECTOR_CONFIG = v1.SELECTOR_CONFIG
OUTCOME_KEYS = v1.OUTCOME_KEYS
V1_KEEP_REFERENCE = 0.929797619047619
V1_SCORE_REFERENCE = 0.9297976190476189
V1_SELECTOR_TOP1_REFERENCE = 51 / 267


def conditional_selector_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all challengers only for labeled queries with a positive delta."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "target" not in row or "delta_vs_keep" not in row:
            raise AssertionError("conditional selector filtering requires labeled training rows")
        grouped[str(row["query_id"])].append(row)
    included_queries = {query_id for query_id, values in grouped.items() if any(int(row["target"]) == 1 for row in values)}
    filtered = [row for row in rows if str(row["query_id"]) in included_queries]
    counts = Counter("positive" if float(row["delta_vs_keep"]) > 0 else "zero" if float(row["delta_vs_keep"]) == 0 else "negative" for row in filtered)
    total = len(filtered)
    return filtered, {
        "gate_training_query_count": len(grouped),
        "selector_training_query_count": len(included_queries),
        "selector_training_row_count": total,
        "selector_positive_row_count": counts["positive"],
        "selector_zero_row_count": counts["zero"],
        "selector_negative_row_count": counts["negative"],
        "selector_positive_prevalence": counts["positive"] / total if total else 0.0,
        "selector_excluded_no_opportunity_query_count": len(grouped) - len(included_queries),
        "included_query_ids": sorted(included_queries),
    }


def selector_training_rows(gate_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]], *, require_exclusion: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered, stats = conditional_selector_rows(selector_rows)
    if {str(row["query_id"]) for row in gate_rows} != {str(row["query_id"]) for row in selector_rows}:
        raise AssertionError("gate and selector training rows disagree on query population")
    if require_exclusion and int(stats["selector_training_query_count"]) >= int(stats["gate_training_query_count"]):
        raise AssertionError("conditional selector training did not exclude any no-opportunity query")
    all_counts = Counter("positive" if float(row["delta_vs_keep"]) > 0 else "zero" if float(row["delta_vs_keep"]) == 0 else "negative" for row in selector_rows)
    stats["v1_all_query_selector_positive_prevalence"] = all_counts["positive"] / len(selector_rows) if selector_rows else 0.0
    stats["v2_conditional_selector_positive_prevalence"] = stats["selector_positive_prevalence"]
    return filtered, stats


def _stage1_outer_ranked(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], outer_train: tuple[int, ...], held_outer: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stage1_input, valid_labeled = v1.outer_stage1_input(actions, outer_train, held_outer)
    ranked = v1._rank_stage1(stage1_input, names, config, set(outer_train), {held_outer})
    return ranked, valid_labeled


def validation_selector_diagnostics(ranked_clean: dict[str, dict[str, Any]], valid_actions: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float | None, gate_threshold: float, selector_threshold: float) -> dict[str, Any]:
    lookup = {(str(row["query_id"]), str(row["incoming_doc_id"]), int(row["drop_rank"])): row for row in valid_actions}
    by_query = {str(row["query_id"]): row for row in predictions}
    counts = Counter()
    for query_id, item in ranked_clean.items():
        incumbent_clean = item["best"]
        incumbent = lookup[(str(query_id), str(incumbent_clean["incoming_doc_id"]), int(incumbent_clean["drop_rank"]))]
        keep = float(incumbent["baseline_recall"]) + (float(incumbent["gain"]) if v1.anchor_execute(incumbent_clean, threshold) else 0.0)
        candidates = [lookup[(str(query_id), str(action["incoming_doc_id"]), int(action["drop_rank"]))] for action in item["all"][1:TOP_K + 1]]
        opportunity = any(float(action["baseline_recall"]) + float(action["gain"]) > keep for action in candidates)
        prediction = by_query[str(query_id)]
        best = prediction["best_challenger"]
        best_positive = bool(best is not None and float(best["baseline_recall"]) + float(best["gain"]) > keep)
        gate_pass = float(prediction["gate_probability"]) >= float(gate_threshold)
        selector_pass = best is not None and float(prediction["best_challenger_probability"]) >= float(selector_threshold)
        if opportunity:
            counts["opportunity_query_count"] += 1
            counts["selector_best_positive"] += best_positive
            counts["selector_best_zero"] += bool(best is not None and float(best["baseline_recall"]) + float(best["gain"]) == keep)
            counts["selector_best_negative"] += bool(best is not None and float(best["baseline_recall"]) + float(best["gain"]) < keep)
            counts["gate_pass_true_opportunity"] += gate_pass
            if best_positive and gate_pass and selector_pass:
                counts["actionable_positive_capture_candidate"] += 1
            elif not best_positive:
                counts["selector_best_not_positive"] += 1
            elif not gate_pass:
                counts["gate_threshold_blocked"] += 1
            elif not selector_pass:
                counts["selector_threshold_blocked"] += 1
        else:
            counts["nonopportunity_query_count"] += 1
            counts["gate_pass_nonopportunity"] += gate_pass
        counts["gate_pass_total"] += gate_pass
    opportunity_count = counts["opportunity_query_count"]
    gate_pass_total = counts["gate_pass_total"]
    return {
        "opportunity_query_count": opportunity_count,
        "queries_where_selector_best_is_positive": counts["selector_best_positive"],
        "selector_top1_positive_rate": counts["selector_best_positive"] / opportunity_count if opportunity_count else 0.0,
        "selector_best_zero_count": counts["selector_best_zero"],
        "selector_best_negative_count": counts["selector_best_negative"],
        "true_opportunity_query_count": opportunity_count,
        "gate_pass_true_opportunity_count": counts["gate_pass_true_opportunity"],
        "gate_recall_on_opportunity_queries": counts["gate_pass_true_opportunity"] / opportunity_count if opportunity_count else 0.0,
        "gate_pass_nonopportunity_count": counts["gate_pass_nonopportunity"],
        "gate_precision": counts["gate_pass_true_opportunity"] / gate_pass_total if gate_pass_total else 0.0,
        "actionable_positive_capture_candidate_count": counts["actionable_positive_capture_candidate"],
        "miss_decomposition": {"selector_best_not_positive": counts["selector_best_not_positive"], "gate_threshold_blocked": counts["gate_threshold_blocked"], "selector_threshold_blocked": counts["selector_threshold_blocked"]},
    }


def nested_inner_oof_v2(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], outer_train_folds: tuple[int, ...], anchor_threshold: float | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for held_inner in sorted(outer_train_folds):
        plan = v1.inner_stage1_plan(outer_train_folds, held_inner)
        valid_ranked = v1._rank_stage1(actions, names, config, set(plan["validation_stage1_train_folds"]), {held_inner})
        train_ranked: dict[str, dict[str, Any]] = {}
        training_provenance = {}
        for scored_fold, stage1_train_folds in plan["training_feature_stage1_train_folds"].items():
            train_ranked.update(v1._rank_stage1(actions, names, config, set(stage1_train_folds), {scored_fold}))
            training_provenance[str(scored_fold)] = list(stage1_train_folds)
        train_gate, train_selector_all = v1.build_rows(train_ranked, anchor_threshold, labeled=True)
        train_selector, stats = selector_training_rows(train_gate, train_selector_all, require_exclusion=True)
        valid_gate, valid_selector = v1.build_rows(valid_ranked, anchor_threshold, labeled=True)
        gate_model = v1._fit_xgb(train_gate, "gate")
        selector_model = v1._fit_xgb(train_selector, "selector")
        predictions.extend(v1.predict_components(gate_model, selector_model, valid_gate, valid_selector))
        provenance[str(held_inner)] = {**plan, "training_feature_stage1_train_folds": training_provenance, "held_inner_absent_from_stage1_generators": all(held_inner not in folds for folds in training_provenance.values())}
        diagnostics[str(held_inner)] = stats
    return predictions, provenance, diagnostics


def outer_run_v2(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any], anchor_threshold: float | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    predictions_all: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}
    outer_reports: dict[str, Any] = {}
    training_diagnostics: dict[str, Any] = {}
    for held_outer in FOLDS:
        outer_plan = v1.outer_stage1_plan(held_outer)
        outer_train = tuple(outer_plan["stage1_train_folds"])
        inner_predictions, inner_provenance, inner_training = nested_inner_oof_v2(actions, names, config, outer_train, anchor_threshold)
        choice = v1.choose_threshold(inner_predictions)
        thresholds[str(held_outer)] = {key: value for key, value in choice.items() if key != "candidates"}
        final_ranked: dict[str, dict[str, Any]] = {}
        final_provenance = {}
        for scored_fold in outer_train:
            train_folds = set(outer_train) - {scored_fold}
            final_ranked.update(v1._rank_stage1(actions, names, config, train_folds, {scored_fold}))
            final_provenance[str(scored_fold)] = sorted(train_folds)
        final_gate, final_selector_all = v1.build_rows(final_ranked, anchor_threshold, labeled=True)
        final_selector, final_stats = selector_training_rows(final_gate, final_selector_all, require_exclusion=True)
        gate_model = v1._fit_xgb(final_gate, "gate")
        selector_model = v1._fit_xgb(final_selector, "selector")
        valid_ranked_clean, valid_actions = _stage1_outer_ranked(actions, names, config, outer_train, held_outer)
        valid_gate, valid_selector = v1.build_rows(valid_ranked_clean, anchor_threshold, labeled=False)
        raw_predictions = v1.predict_components(gate_model, selector_model, valid_gate, valid_selector)
        labeled_predictions = v1._attach_labels(raw_predictions, valid_actions)
        selected = v1.apply_thresholds(labeled_predictions, float(choice["gate_threshold"]), float(choice["selector_threshold"]))
        metrics = v1.policy_metrics(selected)
        keep = v1.policy_metrics(v1.apply_thresholds(labeled_predictions, float("inf"), float("inf")))
        selector_quality = validation_selector_diagnostics(valid_ranked_clean, valid_actions, labeled_predictions, anchor_threshold, float(choice["gate_threshold"]), float(choice["selector_threshold"]))
        outer_reports[str(held_outer)] = {"baseline_macro_recall": metrics["baseline_macro_recall"], "keep_macro_recall": keep["policy_macro_recall"], "v2_macro_recall": metrics["policy_macro_recall"], "delta_vs_keep": metrics["policy_macro_recall"] - keep["policy_macro_recall"], "gate_threshold": choice["gate_threshold"], "selector_threshold": choice["selector_threshold"], "selected_policy_type": choice["selected_policy_type"], "selector_quality": selector_quality, "inner_stage1_provenance": inner_provenance, "final_stage1_provenance": final_provenance}
        training_diagnostics[str(held_outer)] = {"inner": inner_training, "outer_final": final_stats}
        predictions_all.extend(selected)
    return predictions_all, thresholds, outer_reports, training_diagnostics


def _toy_selector_row(query_id: str, target: int, delta: float, label: str) -> dict[str, Any]:
    return {"query_id": query_id, "fold": 1, "target": target, "delta_vs_keep": delta, "features": {name: float(target) for name in SELECTOR_NAMES}, "action": {"label": label}}


def self_test() -> dict[str, bool]:
    v1_checks = {name: bool(value) for name, value in v1.self_test().items()}
    opportunity_rows = [_toy_selector_row("opportunity", 1, .5, "NEUTRAL"), _toy_selector_row("opportunity", 0, 0.0, "BENEFIT"), _toy_selector_row("opportunity", 0, -.5, "HARM")]
    no_opportunity_rows = [_toy_selector_row("no-opportunity", 0, 0.0, "BENEFIT"), _toy_selector_row("no-opportunity", 0, -.5, "HARM")]
    all_rows = [*opportunity_rows, *no_opportunity_rows]
    gate_rows = [{"query_id": "opportunity"}, {"query_id": "no-opportunity"}]
    filtered, stats = selector_training_rows(gate_rows, all_rows, require_exclusion=True)
    weights = v1.selector_weights(filtered)
    inference_rows = list(all_rows)
    held_validation_rows = list(no_opportunity_rows)
    partition_actions = [{"query_id": f"partition-{fold}", "fold": fold, "label": "NEUTRAL", "gain": 0.0, "baseline_recall": .5, "features": {"f": float(fold)}} for fold in FOLDS]
    partition_input, _ = v1.outer_stage1_input(partition_actions, (1, 2, 3), 4)
    validation_partition = [row for row in partition_input if int(row["fold"]) == 4]
    return {
        "v1_contracts_preserved": all(v1_checks.values()),
        "opportunity_query_included": {str(row["query_id"]) for row in filtered} == {"opportunity"},
        "no_opportunity_query_excluded": "no-opportunity" not in {str(row["query_id"]) for row in filtered},
        "gate_keeps_all_queries": len(gate_rows) == 2 and int(stats["gate_training_query_count"]) == 2,
        "selector_inference_keeps_all_queries": {str(row["query_id"]) for row in inference_rows} == {"opportunity", "no-opportunity"},
        "included_query_normalized_weight": abs(float(weights.sum()) - 1.0) < 1e-6,
        "no_benefit_weighting": weights[0] == weights[1] == weights[2],
        "conditional_class_counts": (int(stats["selector_positive_row_count"]), int(stats["selector_zero_row_count"]), int(stats["selector_negative_row_count"])) == (1, 1, 1),
        "filter_requires_labeled_training_rows": all("target" in row and "delta_vs_keep" in row for row in filtered),
        "held_validation_not_used_for_filter": {str(row["query_id"]) for row in held_validation_rows}.isdisjoint({str(row["query_id"]) for row in filtered}),
        "outer_stage1_partition_preserved": bool(validation_partition) and all(not any(key in row for key in OUTCOME_KEYS) for row in validation_partition),
        "fold0_rejected": v1_checks["fold0_rejected"],
        "top_k_20": TOP_K == 20,
    }


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
        print(json.dumps({"status": "PREFLIGHT_PASS", "top_k": TOP_K, "gate_config": GATE_CONFIG, "selector_config": SELECTOR_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "conditional_selector_training": True, "fold0_labels_read": False, "gpu_launched": False, "no_submission_created": True}, indent=2)); return
    if any(value is None for value in (args.actions, args.v3a_report, args.v3b_report, args.output_dir)):
        parser.error("--actions, --v3a-report, --v3b-report, and --output-dir are required")
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    v1.validate_actions(actions)
    if any(any(key not in row for key in ("label", "gain", "baseline_recall")) for row in actions):
        raise ValueError("folds1--4 actions must be labeled")
    names = feature_names(actions); config = fixed_stage1_config(args.v3b_report)
    report_v3b = json.loads(args.v3b_report.read_text(encoding="utf-8")); anchor_threshold = report_v3b["outer_cv"]["best_v3b_raw"].get("action_threshold")
    predictions, thresholds, per_outer, training = outer_run_v2(actions, names, config, anchor_threshold)
    metrics = v1.policy_metrics(predictions)
    ranked_oof: dict[str, dict[str, Any]] = {}
    for held in FOLDS:
        ranked_oof.update(v1._rank_stage1(actions, names, config, set(FOLDS) - {held}, {held}))
    baseline_by_query = {str(row["query_id"]): row for row in actions}
    baseline = float(np.mean([float(row["baseline_recall"]) for row in baseline_by_query.values()]))
    fixed_keep = float(np.mean([v1.keep_recall(item["best"], anchor_threshold) for item in ranked_oof.values()]))
    v3a = v3a_oof(actions, names, load_v3a_config(args.v3a_report))
    benefit = v1.opportunity_diagnostics(ranked_oof, predictions, anchor_threshold)
    oracle_k20 = v1.oracle_macro(ranked_oof, anchor_threshold, TOP_K)
    if abs(oracle_k20 - v1.ORACLE_K20_REFERENCE) > 1e-6:
        raise AssertionError(f"K20 oracle changed: {oracle_k20} != {v1.ORACLE_K20_REFERENCE}")
    selector_quality = {"opportunity_query_count": sum(value["selector_quality"]["opportunity_query_count"] for value in per_outer.values()), "queries_where_selector_best_is_positive": sum(value["selector_quality"]["queries_where_selector_best_is_positive"] for value in per_outer.values()), "selector_best_zero_count": sum(value["selector_quality"]["selector_best_zero_count"] for value in per_outer.values()), "selector_best_negative_count": sum(value["selector_quality"]["selector_best_negative_count"] for value in per_outer.values())}
    selector_quality["selector_top1_positive_rate"] = selector_quality["queries_where_selector_best_is_positive"] / selector_quality["opportunity_query_count"] if selector_quality["opportunity_query_count"] else 0.0
    gate_quality = {"true_opportunity_query_count": selector_quality["opportunity_query_count"], "gate_pass_true_opportunity_count": sum(value["selector_quality"]["gate_pass_true_opportunity_count"] for value in per_outer.values()), "gate_pass_nonopportunity_count": sum(value["selector_quality"]["gate_pass_nonopportunity_count"] for value in per_outer.values())}
    gate_quality["gate_recall_on_opportunity_queries"] = gate_quality["gate_pass_true_opportunity_count"] / gate_quality["true_opportunity_query_count"] if gate_quality["true_opportunity_query_count"] else 0.0
    gate_total = gate_quality["gate_pass_true_opportunity_count"] + gate_quality["gate_pass_nonopportunity_count"]
    gate_quality["gate_precision"] = gate_quality["gate_pass_true_opportunity_count"] / gate_total if gate_total else 0.0
    miss = {name: sum(value["selector_quality"]["miss_decomposition"][name] for value in per_outer.values()) for name in ("selector_best_not_positive", "gate_threshold_blocked", "selector_threshold_blocked")}
    actionable = sum(value["selector_quality"]["actionable_positive_capture_candidate_count"] for value in per_outer.values())
    selected_types = {value["selected_policy_type"] for value in per_outer.values()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "OPPORTUNITY_GATED_DELTA_SELECTOR_V2_OOF_COMPLETE", "architecture": "V1 architecture with conditional selector training only", "top_k": TOP_K, "gate_feature_schema": list(GATE_NAMES), "selector_feature_schema": list(SELECTOR_NAMES), "gate_model_config": GATE_CONFIG, "selector_model_config": SELECTOR_CONFIG, "threshold_quantiles": THRESHOLD_QUANTILES, "anchor_threshold": anchor_threshold, "v1_reference": {"fixed_v3b_keep": V1_KEEP_REFERENCE, "v1_macro_recall": V1_SCORE_REFERENCE, "selector_top1_positive_rate": V1_SELECTOR_TOP1_REFERENCE}, "baseline": {"macro_recall": baseline}, "v3a": {"macro_recall": v3a["policy_macro_recall"]}, "fixed_v3b_keep": {"macro_recall": fixed_keep}, "v2": {**metrics, "delta_vs_v3b_keep": metrics["policy_macro_recall"] - fixed_keep, "delta_vs_v1": metrics["policy_macro_recall"] - V1_SCORE_REFERENCE}, "per_outer_fold": per_outer, "threshold_by_outer_fold": thresholds, "conditional_selector_training": training, "selector_top1_diagnostics": selector_quality, "gate_diagnostics": gate_quality, "actionable_positive_capture_candidate_count": actionable, "miss_decomposition": miss, "benefit_diagnostics": benefit, "k20_oracle_macro_recall": oracle_k20, "positive_opportunity_count_top20": benefit["positive_opportunity_query_count_top20"], "nested_oof_provenance": {"outer_train_excludes_held_outer": True, "inner_stage1_excludes_held_inner": True, "per_outer": {fold: value["inner_stage1_provenance"] for fold, value in per_outer.items()}}, "selected_policy_type": "KEEP_STAGE1" if selected_types == {"KEEP_STAGE1"} else "OPPORTUNITY_GATED_DELTA_SELECTOR_V2", "benefit_diagnostic_only": True, "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}
    (args.output_dir / "opportunity_gated_delta_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "opportunity_gated_delta_v2_oof_predictions.jsonl", [{"query_id": row["query_id"], "fold": int(row["fold"]), "gate_probability": float(row["gate_probability"]), "selector_probability": float(row["best_challenger_probability"]), "execute": bool(row["execute"]), "selected_action": row["selected"], "selected_top5": row["selected_top5"], "delta_vs_keep": v1.prediction_delta_vs_keep(row)} for row in predictions])


if __name__ == "__main__":
    main()
