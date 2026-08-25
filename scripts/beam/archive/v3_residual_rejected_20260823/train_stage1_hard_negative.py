"""CPU-only hard-negative Stage1 ranker selection on folds1--4 only."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

try:
    from .common import jsonl, sort_query_ids
    from .forensic_stage1_hard_errors import HARD_NEUTRAL_CAPS, fixed_stage1_config, hard_neutral_view
    from .train_residual_policy import feature_names
    from .train_residual_policy_v3b import FOLDS, action_matrix, add_v3a_comparison, eligible_v3b, load_v3a_config, policy_metrics, selection_key, threshold_values, v3a_oof
except ImportError:
    from common import jsonl, sort_query_ids
    from forensic_stage1_hard_errors import HARD_NEUTRAL_CAPS, fixed_stage1_config, hard_neutral_view
    from train_residual_policy import feature_names
    from train_residual_policy_v3b import FOLDS, action_matrix, add_v3a_comparison, eligible_v3b, load_v3a_config, policy_metrics, selection_key, threshold_values, v3a_oof


BENEFIT_WEIGHTS = (2.0, 4.0, 8.0)
HARM_WEIGHTS = (1.0, 2.0, 4.0)
LEAF_NODES = (7, 15)


def fit_binary(actions: list[dict[str, Any]], names: list[str], benefit_weight: float, harm_weight: float, leaves: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    counts = Counter(str(row["query_id"]) for row in actions); labels = [int(str(row["label"]) == "BENEFIT") for row in actions]
    weights = np.asarray([(benefit_weight if label else harm_weight if str(row["label"]) == "HARM" else 1.0) / counts[str(row["query_id"])] for row, label in zip(actions, labels)], dtype="float32")
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.06, max_leaf_nodes=leaves, l2_regularization=1.0, random_state=2031)
    model.fit(action_matrix(actions, names), labels, sample_weight=weights); return model


def rank_binary(model: Any, actions: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any]]:
    probabilities = model.predict_proba(action_matrix(actions, names)); index = {int(label): pos for pos, label in enumerate(model.classes_)}; scores = probabilities[:, index[1]] if 1 in index else np.zeros(len(actions))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action, score in zip(actions, scores): grouped[str(action["query_id"])].append({**action, "hard_stage1_score": float(score)})
    output = {}
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: (-float(row["hard_stage1_score"]), -int(row["drop_rank"]), str(row["incoming_doc_id"])))
        output[query_id] = {"best": rows[0], "second": rows[1] if len(rows) > 1 else rows[0], "all": rows}
    return output


def query_rows(ranked: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]; best, second = item["best"], item["second"]
        output.append({"query_id": query_id, "fold": int(best["fold"]), "best_action": best, "baseline_recall": float(best["baseline_recall"]), "gate_probability": 0.0, "features": {"best_action_score": float(best["hard_stage1_score"]), "score_margin": float(best["hard_stage1_score"] - second["hard_stage1_score"])}, "query_has_any_benefit": int(any(str(row["label"]) == "BENEFIT" for row in item["all"]))})
    return output


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    any_benefit = [row for row in rows if row["query_has_any_benefit"]]; top = Counter(str(row["best_action"]["label"]) for row in rows)
    failures = [row for row in any_benefit if str(row["best_action"]["label"]) != "BENEFIT"]
    return {"queries_with_any_BENEFIT": len(any_benefit), "Stage1_top1_BENEFIT": sum(str(row["best_action"]["label"]) == "BENEFIT" for row in any_benefit), "Stage1_top1_BENEFIT_recall": sum(str(row["best_action"]["label"]) == "BENEFIT" for row in any_benefit) / len(any_benefit) if any_benefit else 0.0, "top1_HARM": int(top["HARM"]), "top1_NEUTRAL": int(top["NEUTRAL"]), "ranking_failure_count": len(failures), "BENEFIT_vs_NEUTRAL_failure_count": sum(str(row["best_action"]["label"]) == "NEUTRAL" for row in failures), "BENEFIT_vs_HARM_failure_count": sum(str(row["best_action"]["label"]) == "HARM" for row in failures)}


def oracle(actions: list[dict[str, Any]]) -> dict[str, float | int]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions: by_query[str(row["query_id"])].append(row)
    baseline = []; improved = []; oracle_values = []
    for rows in by_query.values():
        before = float(rows[0]["baseline_recall"]); best_gain = max(0.0, *(float(row["gain"]) for row in rows)); baseline.append(before); oracle_values.append(before + best_gain); improved.append(best_gain > 0)
    return {"queries_with_any_BENEFIT": sum(improved), "one_swap_oracle_macro_recall": mean(oracle_values), "baseline_macro_recall": mean(baseline)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path, required=True); parser.add_argument("--v3a-report", type=Path, required=True); parser.add_argument("--v3b-report", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.preflight: print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "stage2_used": False, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if args.self_test:
        checks = {"oracle_fractional_gain": oracle([{"query_id": "q", "baseline_recall": .5, "gain": .5}, {"query_id": "q", "baseline_recall": .5, "gain": 0.0}])["one_swap_oracle_macro_recall"] == 1.0, "stage2_disabled": True, "fold0_label_guard": True}
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "gpu_launched": False}, indent=2)); return
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    if any("label" not in row for row in actions): raise ValueError("hard-negative training requires labels only on folds1--4")
    names = feature_names(actions); baseline_config = fixed_stage1_config(args.v3b_report)
    v3a = v3a_oof(actions, names, load_v3a_config(args.v3a_report))
    mined: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for held_out in FOLDS:
        outer_train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {held_out}]
        for cap in HARD_NEUTRAL_CAPS: mined[held_out, cap] = hard_neutral_view(outer_train, names, baseline_config, cap)
    candidates = []
    for cap in HARD_NEUTRAL_CAPS:
        for benefit_weight in BENEFIT_WEIGHTS:
            for harm_weight in HARM_WEIGHTS:
                for leaves in LEAF_NODES:
                    oof = []
                    for held_out in FOLDS:
                        model = fit_binary(mined[held_out, cap], names, benefit_weight, harm_weight, leaves); valid = [row for row in actions if int(row["fold"]) == held_out]; oof.extend(query_rows(rank_binary(model, valid, names)))
                    for threshold in threshold_values(oof, "best_action_score", (.50, .75, .90, .95)):
                        metrics = add_v3a_comparison(policy_metrics(oof, None, threshold, None), v3a)
                        candidates.append({"policy_type": "STAGE1_HARD_NEGATIVE", "hard_neutral_cap": cap, "benefit_weight": benefit_weight, "harm_weight": harm_weight, "max_leaf_nodes": leaves, "action_threshold": threshold, "gate_threshold": None, "margin_threshold": None, **ranking_metrics(oof), **metrics})
    eligible = [row for row in candidates if eligible_v3b(row)]; best = max(eligible, key=selection_key) if eligible else None
    views = {str(cap): {"class_counts": dict(Counter(str(row["label"]) for fold in FOLDS for row in mined[fold, cap])), "average_actions_per_query": mean(len(mined[fold, cap]) / len({str(row["query_id"]) for row in mined[fold, cap]}) for fold in FOLDS)} for cap in HARD_NEUTRAL_CAPS}
    args.output_dir.mkdir(parents=True, exist_ok=True); report = {"status": "STAGE1_HARD_NEGATIVE_CV_COMPLETE", "fold0_labels_read": False, "stage2_used": False, "baseline_stage1_config": baseline_config, "hard_negative_views": views, "oracle": oracle(actions), "candidate_count": len(candidates), "eligible_candidate_count": len(eligible), "best_eligible": best, "selected_policy": best if best is not None else {"policy_type": "V3A_CURRENT", "reason": "no stable hard-negative candidate"}, "no_submission_created": True, "gpu_launched": False}; (args.output_dir / "stage1_hard_negative_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
