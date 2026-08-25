"""Folds1--4 OOF forensic and deterministic hard-neutral mining for Stage1."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

try:
    from .common import jsonl, sort_query_ids, write_jsonl
    from .train_residual_policy import feature_names
    from .train_residual_policy_v3b import FOLDS, fit_stage1, load_v3a_config, rank_actions
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from train_residual_policy import feature_names
    from train_residual_policy_v3b import FOLDS, fit_stage1, load_v3a_config, rank_actions


HARD_NEUTRAL_CAPS = (8, 12, 16)
MINED_NEUTRAL_ORDER = (
    ("stage1_score", True), ("stage1_p_benefit", True), ("incoming_neural_max", True),
    ("incoming_neural_top2_mean", True), ("incoming_union_rank", False), ("incoming_source_support", True),
)


def fixed_stage1_config(v3b_report: Path) -> dict[str, Any]:
    report = json.loads(v3b_report.read_text(encoding="utf-8")); raw = dict(report["outer_cv"]["best_v3b_raw"])
    return {key: raw[key] for key in ("max_leaf_nodes", "benefit_multiplier", "harm_train_multiplier", "harm_utility_weight")}


def stage1_oof(actions: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ranked: dict[str, dict[str, Any]] = {}
    for held_out in FOLDS:
        train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {held_out}]
        valid = [row for row in actions if int(row["fold"]) == held_out]
        model = fit_stage1(train, names, config)
        ranked.update(rank_actions(model, valid, names, float(config["harm_utility_weight"])))
    return ranked


def summary(values: list[float]) -> dict[str, float | None]:
    if not values: return {"mean": None, "median": None, "p25": None, "p75": None}
    return {"mean": mean(values), "median": median(values), "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75))}


def forensic(ranked: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successes: list[dict[str, Any]] = []; failed_benefits: list[dict[str, Any]] = []; neutral_winners: list[dict[str, Any]] = []; harm_winners: list[dict[str, Any]] = []; pairs = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]; best = item["best"]; benefits = [row for row in item["all"] if str(row["label"]) == "BENEFIT"]
        if not benefits: continue
        best_benefit = sorted(benefits, key=lambda row: (-float(row["stage1_score"]), str(row["incoming_doc_id"])))[0]
        if str(best["label"]) == "BENEFIT": successes.append(best); continue
        failed_benefits.append(best_benefit)
        (neutral_winners if str(best["label"]) == "NEUTRAL" else harm_winners).append(best)
        highest_harm = next((row for row in item["all"] if str(row["label"]) == "HARM"), None); highest_neutral = next((row for row in item["all"] if str(row["label"]) == "NEUTRAL"), None)
        pairs.append({"query_id": query_id, "fold": int(best["fold"]), "competitor_label": best["label"], "benefit_action": best_benefit, "competitor_action": best, "highest_scoring_harm": highest_harm, "highest_scoring_neutral": highest_neutral, "feature_differences": {name: float(best_benefit["features"].get(name, 0.0)) - float(best["features"].get(name, 0.0)) for name in best_benefit["features"]}})
    features = ("stage1_score", "stage1_p_benefit", "stage1_p_harm", *sorted({name for rows in (successes, failed_benefits, neutral_winners, harm_winners) for row in rows for name in row["features"]}))
    def group(rows: list[dict[str, Any]]) -> dict[str, Any]: return {name: summary([float(row.get(name, row["features"].get(name, 0.0))) for row in rows]) for name in features}
    opportunities = len(successes) + len(failed_benefits)
    report = {"queries_with_any_BENEFIT": opportunities, "Stage1_top1_BENEFIT": len(successes), "Stage1_top1_BENEFIT_recall": len(successes) / opportunities if opportunities else 0.0, "ranking_failure_count": len(failed_benefits), "BENEFIT_vs_NEUTRAL_failure_count": len(neutral_winners), "BENEFIT_vs_HARM_failure_count": len(harm_winners), "feature_distributions": {"SUCCESS_BENEFIT": group(successes), "FAILED_BENEFIT": group(failed_benefits), "FAILED_competing_NEUTRAL": group(neutral_winners), "FAILED_competing_HARM": group(harm_winners)}, "isolated_max_forensic": {"BENEFIT_winners": group(successes), "BENEFIT_losers": group(failed_benefits), "NEUTRAL_winners": group(neutral_winners), "HARM_winners": group(harm_winners)}}
    return report, pairs


def hard_neutral_view(actions: list[dict[str, Any]], names: list[str], baseline_config: dict[str, Any], cap: int) -> list[dict[str, Any]]:
    """Mine solely from the training split passed by the caller; never held-out labels."""
    model = fit_stage1(actions, names, baseline_config); ranked = rank_actions(model, actions, names, float(baseline_config["harm_utility_weight"])); selected = []
    for item in ranked.values():
        rows = item["all"]; keep = [row for row in rows if str(row["label"]) != "NEUTRAL"]; neutral = [row for row in rows if str(row["label"]) == "NEUTRAL"]; chosen: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
        for name, descending in MINED_NEUTRAL_ORDER:
            def order_key(row: dict[str, Any]) -> tuple[float, str, int]:
                value = float(row.get(name, row["features"].get(name, 0.0)))
                return (-value if descending else value, str(row["incoming_doc_id"]), int(row["drop_rank"]))
            ordered = sorted(neutral, key=order_key)
            for rank, row in enumerate(ordered[:cap]):
                key = (str(row["incoming_doc_id"]), int(row["drop_rank"])); previous = chosen.get(key)
                if previous is None or rank < previous[0]: chosen[key] = (rank, row)
        hard_neutral = [row for _, row in sorted(chosen.values(), key=lambda item: (item[0], str(item[1]["incoming_doc_id"]), int(item[1]["drop_rank"])))[:cap]]
        selected.extend(keep + hard_neutral)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path, required=True); parser.add_argument("--v3b-report", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight: print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    if any("label" not in row for row in actions): raise ValueError("forensic requires labels only on folds1--4")
    config = fixed_stage1_config(args.v3b_report); report, pairs = forensic(stage1_oof(actions, feature_names(actions), config)); args.output_dir.mkdir(parents=True, exist_ok=True); write_jsonl(args.output_dir / "hard_error_pairs.jsonl", pairs); (args.output_dir / "stage1_hard_error_forensic.json").write_text(json.dumps({**report, "fixed_stage1_config": config, "fold0_labels_read": False}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
