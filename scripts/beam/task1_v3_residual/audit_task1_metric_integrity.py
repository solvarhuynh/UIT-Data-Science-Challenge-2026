"""Folds1--4-only independent metric integrity audit for Task1 V3 policies."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .build_actions import action_label
    from .common import jsonl, recall, sort_query_ids
    from .evaluate_v3_fold0 import metric
    from .train_residual_policy import choose, feature_names, fit_model, utilities, validate_label_split
    from .train_residual_policy_v3b import FOLDS, build_predictions, load_v3a_config, outer_oof_from_stage1_cache, outer_stage1_cache
except ImportError:
    from build_actions import action_label
    from common import jsonl, recall, sort_query_ids
    from evaluate_v3_fold0 import metric
    from train_residual_policy import choose, feature_names, fit_model, utilities, validate_label_split
    from train_residual_policy_v3b import FOLDS, build_predictions, load_v3a_config, outer_oof_from_stage1_cache, outer_stage1_cache


TOLERANCE = 1e-12


class FoldRestrictedQuestions:
    """Expose gold only for folds1--4 and count every requested query id."""
    def __init__(self, questions: dict[str, Any], fold_by_query: dict[str, int]):
        self.questions = questions; self.fold_by_query = fold_by_query; self.gold_query_ids_accessed: set[str] = set(); self.fold0_gold_query_ids_accessed: set[str] = set()

    def __getitem__(self, query_id: str) -> dict[str, Any]:
        query_id = str(query_id)
        if self.fold_by_query.get(query_id) == 0:
            self.fold0_gold_query_ids_accessed.add(query_id)
            raise AssertionError(f"Fold0 gold access prohibited: {query_id}")
        self.gold_query_ids_accessed.add(query_id)
        return self.questions[query_id]


def reference_recall(gold: Any, top5: Any) -> float:
    gold_ids = {str(doc_id) for doc_id in gold}
    predicted_ids = {str(doc_id) for doc_id in top5}
    return len(gold_ids & predicted_ids) / len(gold_ids) if gold_ids else 0.0


def reference_top5(baseline: list[str], action: dict[str, Any] | None) -> list[str]:
    result = [str(doc_id) for doc_id in baseline]
    if action is not None:
        result[int(action["drop_rank"]) - 1] = str(action["incoming_doc_id"])
    return result


def toy_metric_audit() -> dict[str, Any]:
    cases = [(["A"], ["A", "B", "C", "D", "E"], 1.0), (["A", "B"], ["A", "X", "Y", "Z", "W"], .5), (["A", "B", "C"], ["A", "B", "X", "Y", "Z"], 2 / 3), (["A", "B"], ["X", "Y", "Z", "W", "Q"], 0.0), (["A", "B"], ["A", "A", "X", "Y", "Z"], .5)]
    values = []
    for gold, top5, expected in cases:
        independent = reference_recall(gold, top5)
        values.append({"expected": expected, "reference": independent, "common_recall": recall(set(gold), top5), "pass": abs(independent - expected) <= TOLERANCE and abs(independent - recall(set(gold), top5)) <= TOLERANCE})
    base = {"q": {"gold_documents": ["A", "B"], "top5": ["A", "X", "Y", "Z", "W"]}}
    evaluator = metric(base, {"q": ["A", "X", "Y", "Z", "W"]})
    try: metric(base, {"q": ["A", "A", "X", "Y", "Z"]})
    except ValueError: evaluator_duplicate_rejected = True
    else: evaluator_duplicate_rejected = False
    return {"cases": values, "evaluate_v3_metric_multigold_case": evaluator["macro_recall"], "evaluate_v3_duplicate_rejected": evaluator_duplicate_rejected, "all_pass": all(row["pass"] for row in values) and abs(evaluator["macro_recall"] - .5) <= TOLERANCE and evaluator_duplicate_rejected}


def audit_action_labels(actions: list[dict[str, Any]], questions: dict[str, Any]) -> dict[str, Any]:
    checked = baseline_mismatches = gain_mismatches = label_mismatches = 0; max_gain_error = 0.0; examples = []
    for action in actions:
        query_id = str(action["query_id"]); gold = questions[query_id]["answer"]; baseline = [str(doc_id) for doc_id in action["baseline_top5"]]
        before = reference_recall(gold, baseline); after = reference_recall(gold, reference_top5(baseline, action)); gain = after - before
        expected_label = "BENEFIT" if gain > 0 else "HARM" if gain < 0 else "NEUTRAL"
        baseline_bad = abs(before - float(action["baseline_recall"])) > TOLERANCE; gain_bad = abs(gain - float(action["gain"])) > TOLERANCE; label_bad = expected_label != str(action["label"])
        checked += 1; baseline_mismatches += baseline_bad; gain_mismatches += gain_bad; label_mismatches += label_bad; max_gain_error = max(max_gain_error, abs(gain - float(action["gain"])))
        if (baseline_bad or gain_bad or label_bad) and len(examples) < 10:
            examples.append({"query_id": query_id, "baseline_stored": action["baseline_recall"], "baseline_reference": before, "gain_stored": action["gain"], "gain_reference": gain, "label_stored": action["label"], "label_reference": expected_label})
    return {"action_count_checked": checked, "baseline_recall_mismatch_count": baseline_mismatches, "gain_mismatch_count": gain_mismatches, "label_mismatch_count": label_mismatches, "max_abs_gain_error": max_gain_error, "examples": examples}


def v3a_reference_oof(actions: list[dict[str, Any]], questions: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions: by_query[str(action["query_id"])].append(action)
    baseline_predictions: dict[str, list[str]] = {}; v3a_predictions: dict[str, list[str]] = {}
    names = feature_names(actions)
    for fold in FOLDS:
        valid = [row for row in actions if int(row["fold"]) == fold]
        if config["selected_policy_type"] == "NO_SWAP_BASELINE": selected = {query_id: None for query_id in {str(row["query_id"]) for row in valid}}
        else:
            train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {fold}]
            model = fit_model(train, names, int(config["max_leaf_nodes"]), float(config["l2_regularization"]))
            selected = choose(valid, utilities(model, valid, names, float(config["harm_weight"])), float(config["threshold"]))
        for query_id in {str(row["query_id"]) for row in valid}:
            baseline = [str(doc_id) for doc_id in by_query[query_id][0]["baseline_top5"]]
            baseline_predictions[query_id] = baseline; v3a_predictions[query_id] = reference_top5(baseline, selected[query_id])
    def macro(predictions: dict[str, list[str]]) -> float:
        return mean(reference_recall(questions[query_id]["answer"], top5) for query_id, top5 in predictions.items())
    return {"baseline_macro_recall": macro(baseline_predictions), "v3a_macro_recall": macro(v3a_predictions), "baseline_predictions": baseline_predictions, "v3a_predictions": v3a_predictions}


def v3a_report_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("selected_inner_cv_metrics")
    if not isinstance(metrics, dict): raise ValueError("V3A policy report lacks selected_inner_cv_metrics")
    return {"baseline_macro_recall": float(metrics["baseline_macro_recall"]), "v3a_macro_recall": float(metrics["policy_macro_recall"])}


def fixed_v3b_oof_predictions(actions: list[dict[str, Any]], report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8")); config = dict(report["outer_cv"]["best_v3b_raw"])
    required = ("max_leaf_nodes", "benefit_multiplier", "harm_train_multiplier", "harm_utility_weight", "stage2_positive_multiplier", "gate_threshold", "action_threshold", "margin_threshold")
    if any(key not in config for key in required): raise ValueError("V3B report lacks fixed best_v3b_raw configuration")
    names = feature_names(actions); cache = outer_stage1_cache(actions, names, config)
    rows = outer_oof_from_stage1_cache(cache, config, float(config["stage2_positive_multiplier"]))
    return build_predictions(rows, config), config


def direct_prediction_audit(rows: list[dict[str, Any]], questions: FoldRestrictedQuestions, baseline_predictions: dict[str, list[str]], expected_report_score: float) -> dict[str, Any]:
    by_query = {str(row["query_id"]): row for row in rows if int(row["fold"]) in FOLDS}
    if set(by_query) != set(baseline_predictions): raise ValueError("V3B OOF predictions do not match folds1--4 action queries")
    baseline_values = {query_id: reference_recall(questions[query_id]["answer"], top5) for query_id, top5 in baseline_predictions.items()}
    baseline = mean(baseline_values.values())
    policy = mean(reference_recall(questions[query_id]["answer"], row["top5"]) for query_id, row in by_query.items())
    gain_scored = mean(baseline_values[query_id] + (float(row["selected_action"]["gain"]) if row.get("selected_action") is not None else 0.0) for query_id, row in by_query.items())
    return {"baseline_macro_recall": baseline, "v3b_direct_macro_recall": policy, "v3b_gain_scored_macro_recall": gain_scored, "v3b_report_macro_recall": expected_report_score, "direct_top5_vs_gain_scoring_match": abs(policy - gain_scored) <= TOLERANCE, "direct_v3b_vs_report_match": abs(policy - expected_report_score) <= TOLERANCE}


def multi_gold_audit(questions: FoldRestrictedQuestions, baseline: dict[str, list[str]], v3a: dict[str, list[str]]) -> dict[str, Any]:
    query_ids = [query_id for query_id in baseline if len({str(doc_id) for doc_id in questions[query_id]["answer"]}) > 1]
    reference_baseline = [reference_recall(questions[query_id]["answer"], baseline[query_id]) for query_id in query_ids]; production_baseline = [recall({str(doc_id) for doc_id in questions[query_id]["answer"]}, baseline[query_id]) for query_id in query_ids]
    reference_v3a = [reference_recall(questions[query_id]["answer"], v3a[query_id]) for query_id in query_ids]; production_v3a = [recall({str(doc_id) for doc_id in questions[query_id]["answer"]}, v3a[query_id]) for query_id in query_ids]
    errors = [abs(left - right) for left, right in zip(reference_baseline + reference_v3a, production_baseline + production_v3a)]
    return {"multi_gold_query_count": len(query_ids), "reference_baseline_recall": mean(reference_baseline) if reference_baseline else 0.0, "production_baseline_recall": mean(production_baseline) if production_baseline else 0.0, "baseline_match": all(error <= TOLERANCE for error in (abs(left - right) for left, right in zip(reference_baseline, production_baseline))), "reference_v3a_recall": mean(reference_v3a) if reference_v3a else 0.0, "production_v3a_recall": mean(production_v3a) if production_v3a else 0.0, "v3a_match": all(error <= TOLERANCE for error in (abs(left - right) for left, right in zip(reference_v3a, production_v3a))), "max_query_level_abs_error": max(errors) if errors else 0.0}


def self_test() -> dict[str, bool]:
    return {"reference_toys": bool(toy_metric_audit()["all_pass"]), "duplicate_does_not_increase_recall": reference_recall(["A", "B"], ["A", "A", "X", "Y", "Z"]) == .5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--v3a-report", type=Path, required=True)
    parser.add_argument("--v3b-report", type=Path, required=True)
    parser.add_argument("--v3b-oof-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = self_test()
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False}, indent=2)); return
    all_actions = jsonl(args.actions); train_actions = [row for row in all_actions if int(row["fold"]) in FOLDS]; fold0_actions = [row for row in all_actions if int(row["fold"]) == 0]
    validate_label_split(train_actions, fold0_actions)
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig")); fold_by_query = {str(row["query_id"]): int(row["fold"]) for row in all_actions}; train_questions = FoldRestrictedQuestions(questions, fold_by_query)
    action_audit = audit_action_labels(train_actions, train_questions); v3a = v3a_reference_oof(train_actions, train_questions, load_v3a_config(args.v3a_report)); production_v3a = v3a_report_metrics(args.v3a_report)
    if args.v3b_oof_predictions is not None and args.v3b_oof_predictions.is_file(): v3b_rows = jsonl(args.v3b_oof_predictions); v3b_reconstruction = "explicit_oof_predictions"
    else: v3b_rows, fixed_config = fixed_v3b_oof_predictions(train_actions, args.v3b_report); v3b_reconstruction = {"mode": "fixed_config_crossfit", "config": fixed_config}
    v3b_report = json.loads(args.v3b_report.read_text(encoding="utf-8")); direct_v3b = direct_prediction_audit(v3b_rows, train_questions, v3a["baseline_predictions"], float(v3b_report["outer_cv"]["best_v3b_raw"]["policy_macro_recall"])); multi = multi_gold_audit(train_questions, v3a["baseline_predictions"], v3a["v3a_predictions"])
    v3a_compare = {"reference_baseline_macro_recall": v3a["baseline_macro_recall"], "production_baseline_macro_recall": production_v3a["baseline_macro_recall"], "baseline_cv_match": abs(v3a["baseline_macro_recall"] - production_v3a["baseline_macro_recall"]) <= TOLERANCE, "reference_v3a_macro_recall": v3a["v3a_macro_recall"], "production_v3a_macro_recall": production_v3a["v3a_macro_recall"], "v3a_cv_match": abs(v3a["v3a_macro_recall"] - production_v3a["v3a_macro_recall"]) <= TOLERANCE}
    toy = toy_metric_audit(); passed = bool(toy["all_pass"]) and all(action_audit[key] == 0 for key in ("baseline_recall_mismatch_count", "gain_mismatch_count", "label_mismatch_count")) and v3a_compare["baseline_cv_match"] and v3a_compare["v3a_cv_match"] and direct_v3b["direct_top5_vs_gain_scoring_match"] and direct_v3b["direct_v3b_vs_report_match"] and multi["baseline_match"] and multi["v3a_match"] and multi["max_query_level_abs_error"] <= TOLERANCE and not train_questions.fold0_gold_query_ids_accessed
    report = {"status": "METRIC_AUDIT_PASS" if passed else "METRIC_AUDIT_FAIL", "fold0_labels_used": False, "fold0_gold_used": False, "gold_query_ids_accessed": len(train_questions.gold_query_ids_accessed), "fold0_gold_query_ids_accessed": len(train_questions.fold0_gold_query_ids_accessed), "metric_formula_match": toy, "action_labels": action_audit, "v3a_reference_vs_production_report": v3a_compare, "v3b_fixed_config_reconstruction": v3b_reconstruction, "v3b_direct_top5": direct_v3b, "multi_gold": multi}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
