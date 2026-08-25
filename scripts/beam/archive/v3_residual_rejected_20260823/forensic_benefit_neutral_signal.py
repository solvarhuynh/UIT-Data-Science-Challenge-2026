"""Descriptive BENEFIT-vs-NEUTRAL signal forensic for V3 residual actions.

This module uses labels only on folds 1--4 and never creates production
thresholds.  All reported feature statistics are inference-visible values.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

try:
    from .common import jsonl, sort_query_ids, write_jsonl
    from .train_delta_recall_residual import FOLDS, feature_names, fixed_stage1_ranked
    from .forensic_stage1_hard_errors import fixed_stage1_config
except ImportError:
    from common import jsonl, sort_query_ids, write_jsonl
    from train_delta_recall_residual import FOLDS, feature_names, fixed_stage1_ranked
    from forensic_stage1_hard_errors import fixed_stage1_config


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p25": float(np.quantile(values, .25)),
        "p75": float(np.quantile(values, .75)),
    }


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    values = np.asarray([*positive, *negative], dtype="float64")
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype="int32")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, values))
    except ValueError:
        return None


def _effect(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    a, b = np.asarray(positive, dtype="float64"), np.asarray(negative, dtype="float64")
    pooled = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _feature_report(positive: list[dict[str, Any]], negative: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        pos = [float(row["features"].get(name, 0.0)) for row in positive]
        neg = [float(row["features"].get(name, 0.0)) for row in negative]
        result[name] = {"positive": _summary(pos), "nonpositive": _summary(neg), "positive_minus_nonpositive": float(np.mean(pos) - np.mean(neg)) if pos and neg else None, "standardized_effect_size": _effect(pos, neg), "roc_auc_positive_vs_nonpositive": _auc(pos, neg)}
    return result


def _sign_stability(rows_by_fold: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]], names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        signs = []
        per_fold = {}
        for fold, (positive, negative) in sorted(rows_by_fold.items()):
            p = [float(row["features"].get(name, 0.0)) for row in positive]
            n = [float(row["features"].get(name, 0.0)) for row in negative]
            delta = float(np.mean(p) - np.mean(n)) if p and n else None
            per_fold[str(fold)] = delta
            if delta is not None and delta != 0:
                signs.append(1 if delta > 0 else -1)
        result[name] = {"per_fold_mean_difference": per_fold, "positive_sign_fraction": float(sum(sign > 0 for sign in signs) / len(signs)) if signs else None, "sign_consistent": bool(signs and all(sign == signs[0] for sign in signs))}
    return result


def action_threshold(v3b_report: Path) -> float | None:
    report = json.loads(Path(v3b_report).read_text(encoding="utf-8"))
    return report["outer_cv"]["best_v3b_raw"].get("action_threshold")


def analyze(ranked: dict[str, dict[str, Any]], names: list[str], anchor_threshold: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    nonpositive: list[dict[str, Any]] = []
    benefit_rows: list[dict[str, Any]] = []
    neutral_rows: list[dict[str, Any]] = []
    residual_deltas: list[float] = []
    by_fold: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"positive": [], "nonpositive": [], "benefit": [], "neutral": []})
    output_rows = []
    for query_id in sort_query_ids(ranked):
        item = ranked[query_id]
        incumbent = item["best"]
        anchor = anchor_threshold is None or float(incumbent["stage1_score"]) >= float(anchor_threshold)
        keep_gain = float(incumbent["gain"]) if anchor else 0.0
        benefits = [row for row in item["all"] if float(row["gain"]) > 0]
        if benefits and str(incumbent["label"]) != "BENEFIT":
            best_benefit = max(benefits, key=lambda row: (float(row["stage1_score"]), -int(row["drop_rank"]), str(row["incoming_doc_id"])))
            output_rows.append({"query_id": query_id, "fold": int(incumbent["fold"]), "incumbent_label": incumbent["label"], "incumbent_stage1_score": incumbent["stage1_score"], "best_benefit_stage1_score": best_benefit["stage1_score"], "stage1_gap": float(best_benefit["stage1_score"]) - float(incumbent["stage1_score"])})
        for rank, row in enumerate(item["all"][1:11], start=1):
            delta = float(row["gain"]) - keep_gain
            residual_deltas.append(delta)
            tagged = {**row, "delta_recall": delta, "residual_rank": rank}
            group = positive if delta > 0 else nonpositive
            group.append(tagged)
            by_fold[int(row["fold"])] ["positive" if delta > 0 else "nonpositive"].append(tagged)
            if str(row["label"]) == "BENEFIT":
                benefit_rows.append(tagged); by_fold[int(row["fold"])] ["benefit"].append(tagged)
            if str(row["label"]) == "NEUTRAL":
                neutral_rows.append(tagged); by_fold[int(row["fold"])] ["neutral"].append(tagged)
    fold_report = {}
    for fold in FOLDS:
        groups = by_fold[fold]
        fold_report[str(fold)] = {"positive_delta_count": len(groups["positive"]), "nonpositive_delta_count": len(groups["nonpositive"]), "benefit_count": len(groups["benefit"]), "neutral_count": len(groups["neutral"]), "positive_vs_nonpositive": _feature_report(groups["positive"], groups["nonpositive"], names), "benefit_vs_neutral": _feature_report(groups["benefit"], groups["neutral"], names)}
    report = {"status": "BENEFIT_NEUTRAL_SIGNAL_FORENSIC", "folds": list(FOLDS), "query_count": len(ranked), "anchor_threshold": anchor_threshold, "anchor_semantics": "execute Stage1 incumbent iff stage1_score >= fixed V3B action threshold", "top10_residual_decision_count": len(positive) + len(nonpositive), "positive_delta_count": len(positive), "nonpositive_delta_count": len(nonpositive), "residual_delta_summary": {"count": len(residual_deltas), "min": float(min(residual_deltas)) if residual_deltas else None, "max": float(max(residual_deltas)) if residual_deltas else None, "mean": float(np.mean(residual_deltas)) if residual_deltas else None}, "fixed_stage1_misses_with_positive_challenger": len(output_rows), "benefit_vs_neutral_rows": len(benefit_rows) + len(neutral_rows), "pooled": {"positive_vs_nonpositive": _feature_report(positive, nonpositive, names), "benefit_vs_neutral": _feature_report(benefit_rows, neutral_rows, names), "sign_stability_positive_vs_nonpositive": _sign_stability({fold: (groups["positive"], groups["nonpositive"]) for fold, groups in by_fold.items()}, names)}, "per_fold": fold_report, "descriptive_only": True, "no_production_threshold_created": True, "fold0_labels_read": False}
    return report, output_rows


def self_test() -> dict[str, bool]:
    statistics_ok = _auc([2.0, 3.0], [0.0, 1.0]) == 1.0 and _summary([])["count"] == 0
    incumbent = {"query_id": "toy", "fold": 1, "stage1_score": 0.8, "gain": 0.5, "label": "BENEFIT", "incoming_doc_id": "incumbent", "drop_rank": 5, "features": {"toy": 1.0}}
    challenger = {"query_id": "toy", "fold": 1, "stage1_score": 0.7, "gain": 0.0, "label": "NEUTRAL", "incoming_doc_id": "challenger", "drop_rank": 5, "features": {"toy": 0.0}}
    toy_ranked = {"toy": {"best": incumbent, "second": challenger, "all": [incumbent, challenger]}}
    without_anchor_key = "anchor_execute" not in toy_ranked["toy"]
    report_a, _ = analyze(toy_ranked, ["toy"], 0.9)
    report_b, _ = analyze(toy_ranked, ["toy"], 0.7)
    anchor_case_a = report_a["anchor_threshold"] == 0.9 and report_a["residual_delta_summary"]["min"] == 0.0 and report_a["residual_delta_summary"]["max"] == 0.0
    anchor_case_b = report_b["anchor_threshold"] == 0.7 and report_b["residual_delta_summary"]["min"] == -0.5 and report_b["residual_delta_summary"]["max"] == -0.5
    return {"statistics_toy": statistics_ok, "analyze_without_anchor_execute": without_anchor_key and anchor_case_a and anchor_case_b, "case_a_anchor_not_executed": anchor_case_a, "case_b_anchor_executed": anchor_case_b, "fold0_not_used": report_a["fold0_labels_read"] is False, "descriptive_only": report_a["descriptive_only"] is True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--v3b-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "gpu_launched": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "fold0_labels_read": False, "thresholds_created": False, "gpu_launched": False}, indent=2)); return
    if any(value is None for value in (args.actions, args.v3b_report, args.output_dir)):
        parser.error("--actions, --v3b-report, and --output-dir are required")
    rows = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    if any("label" not in row or "gain" not in row for row in rows):
        raise ValueError("forensic requires labels/gains on folds1--4 only")
    names = feature_names(rows)
    threshold = action_threshold(args.v3b_report)
    ranked = fixed_stage1_ranked(rows, names, fixed_stage1_config(args.v3b_report))
    report, miss_rows = analyze(ranked, names, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benefit_neutral_signal_report.json").write_text(json.dumps({**report, "fixed_stage1_config": fixed_stage1_config(args.v3b_report)}, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "benefit_neutral_signal_rows.jsonl", miss_rows)


if __name__ == "__main__":
    main()
