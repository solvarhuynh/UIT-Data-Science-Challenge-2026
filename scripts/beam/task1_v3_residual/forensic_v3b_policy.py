"""Fold1--4-only forensic for the V3A action ranking and abstention boundary."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import jsonl
    from .train_residual_policy import choose, feature_names, fit_model, utilities, validate_label_split
    from .train_residual_policy_v3b import FOLDS, load_v3a_config
except ImportError:
    from common import jsonl
    from train_residual_policy import choose, feature_names, fit_model, utilities, validate_label_split
    from train_residual_policy_v3b import FOLDS, load_v3a_config


def fold_forensic(actions: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        by_query[str(row["query_id"])].append(row)
    counts = Counter(str(row["label"]) for row in actions)
    benefit_per_query = {query_id: sum(str(row["label"]) == "BENEFIT" for row in rows) for query_id, rows in by_query.items()}
    return {
        "query_count": len(by_query),
        "queries_with_any_BENEFIT": sum(count > 0 for count in benefit_per_query.values()),
        "queries_without_BENEFIT": sum(count == 0 for count in benefit_per_query.values()),
        "queries_with_multiple_BENEFIT_actions": sum(count > 1 for count in benefit_per_query.values()),
        "BENEFIT_action_count": int(counts["BENEFIT"]), "HARM_action_count": int(counts["HARM"]), "NEUTRAL_action_count": int(counts["NEUTRAL"]),
    }


def v3a_oof_forensic(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> dict[str, Any]:
    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions: all_rows[str(row["query_id"])].append(row)
    output: dict[int, dict[str, Any]] = {}
    for fold in FOLDS:
        valid = [row for row in actions if int(row["fold"]) == fold]
        if config["selected_policy_type"] == "NO_SWAP_BASELINE":
            top, selected = {query_id: None for query_id in {str(row["query_id"]) for row in valid}}, {query_id: None for query_id in {str(row["query_id"]) for row in valid}}
        else:
            train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {fold}]
            model = fit_model(train, names, int(config["max_leaf_nodes"]), float(config["l2_regularization"]))
            values = utilities(model, valid, names, float(config["harm_weight"]))
            top = choose(valid, values, float("-inf"))
            selected = choose(valid, values, float(config["threshold"]))
        selected_labels = Counter(); top_benefit = top_not_benefit = 0; benefit_queries = 0
        swaps = beneficial = harmful = neutral = abstained = 0; ranking_error = false_abstention = dangerous_harm_gate = unnecessary_neutral_gate = 0
        for query_id in {str(row["query_id"]) for row in valid}:
            has_benefit = any(str(row["label"]) == "BENEFIT" for row in all_rows[query_id])
            top_action, selected_action = top[query_id], selected[query_id]
            if has_benefit:
                benefit_queries += 1
                if top_action is not None and str(top_action["label"]) == "BENEFIT": top_benefit += 1
                else: top_not_benefit += 1; ranking_error += 1
            if selected_action is None:
                abstained += 1
                if top_action is not None and str(top_action["label"]) == "BENEFIT": false_abstention += 1
                continue
            label = str(selected_action["label"]); selected_labels[label] += 1; swaps += 1
            beneficial += label == "BENEFIT"; harmful += label == "HARM"; neutral += label == "NEUTRAL"
            dangerous_harm_gate += top_action is not None and str(top_action["label"]) == "HARM"
            unnecessary_neutral_gate += top_action is not None and str(top_action["label"]) == "NEUTRAL"
        output[fold] = {
            "selected_action_is_BENEFIT": int(selected_labels["BENEFIT"]), "selected_action_is_HARM": int(selected_labels["HARM"]), "selected_action_is_NEUTRAL": int(selected_labels["NEUTRAL"]), "abstained_queries": abstained,
            "queries_with_any_BENEFIT": benefit_queries, "queries_where_stage1_top1_is_BENEFIT": top_benefit, "queries_with_any_BENEFIT_but_top1_not_BENEFIT": top_not_benefit,
            "benefit_query_recall_at_top1": top_benefit / benefit_queries if benefit_queries else 0.0,
            "swap_count": swaps, "beneficial_swaps": beneficial, "harmful_swaps": harmful, "neutral_swaps": neutral,
            "stage1_ranking_failure_queries": ranking_error, "queries_where_top1_is_BENEFIT_but_policy_abstains": false_abstention,
            "queries_where_top1_is_HARM_and_policy_executes": dangerous_harm_gate, "queries_where_top1_is_NEUTRAL_and_policy_executes": unnecessary_neutral_gate,
        }
    pooled = Counter()
    for value in output.values(): pooled.update(value)
    benefit_queries = pooled["queries_with_any_BENEFIT"]
    return {"per_fold": {str(fold): value for fold, value in output.items()}, "pooled": {**dict(pooled), "benefit_query_recall_at_top1": pooled["queries_where_stage1_top1_is_BENEFIT"] / benefit_queries if benefit_queries else 0.0}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--v3a-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "uses_folds": list(FOLDS), "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    actions = jsonl(args.actions)
    train = [row for row in actions if int(row["fold"]) in FOLDS]
    fold0 = [row for row in actions if int(row["fold"]) == 0]
    validate_label_split(train, fold0)
    report = {"status": "V3B_FORENSIC_COMPLETE", "fold0_labels_read": False, "per_fold_actions": {str(fold): fold_forensic([row for row in train if int(row["fold"]) == fold]) for fold in FOLDS}, "v3a_current_oof": v3a_oof_forensic(train, feature_names(actions), load_v3a_config(args.v3a_report)), "gpu_launched": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
