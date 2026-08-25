"""Fold0 raw V2 and selective-fusion diagnostic evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .selective_fusion_v2 import fuse_query
except ImportError:
    from selective_fusion_v2 import fuse_query


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def recall(gold: set[str], top5: list[str]) -> float:
    return len(gold & set(top5)) / len(gold) if gold else 0.0


def outcome_metrics(base: dict[str, dict[str, Any]], predictions: dict[str, list[str]]) -> dict[str, Any]:
    values: list[float] = []
    better = worse = neutral = changed = full_gold = 0
    multi: list[float] = []
    for query_id, row in base.items():
        gold = {str(doc_id) for doc_id in row.get("gold_documents", [])}
        base_top = [str(doc_id) for doc_id in row.get("top5", [])]
        top = [str(doc_id) for doc_id in predictions.get(query_id, [])]
        value = recall(gold, top)
        base_value = recall(gold, base_top)
        values.append(value)
        better += value > base_value
        worse += value < base_value
        neutral += value == base_value
        changed += top != base_top
        full_gold += bool(gold and gold.issubset(set(top)))
        if len(gold) > 1:
            multi.append(value)
    return {
        "macro_recall": sum(values) / len(values) if values else 0.0,
        "better": better,
        "worse": worse,
        "neutral": neutral,
        "changed_queries": changed,
        "full_gold_count": full_gold,
        "multi_gold_query_count": len(multi),
        "multi_gold_macro_recall": sum(multi) / len(multi) if multi else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--doc-scores", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fusion-predictions", type=Path)
    parser.add_argument("--thresholds", default="-0.5,0,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--max-neural-rank", type=int, default=20)
    parser.add_argument("--protect-top-ranks", type=int, default=3)
    args = parser.parse_args()

    baseline_all = jsonl(args.baseline)
    baseline = {str(row["query_id"]): row for row in baseline_all if int(row.get("fold", 0)) == 0}
    raw_rows = jsonl(args.raw_predictions)
    raw = {str(row["query_id"]): [str(doc_id) for doc_id in row.get("top5", [])] for row in raw_rows}
    if set(raw) != set(baseline):
        raise ValueError("raw V2 query set does not exactly match baseline fold0")
    if any(len(top) != 5 or len(set(top)) != 5 for top in raw.values()):
        raise ValueError("raw V2 predictions must contain five unique documents")
    scores_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl(args.doc_scores):
        scores_by_query.setdefault(str(row["query_id"]), []).append(row)
    if set(scores_by_query) != set(baseline):
        raise ValueError("V2 doc-score query set does not exactly match baseline fold0")

    baseline_metrics = outcome_metrics(baseline, {qid: [str(doc_id) for doc_id in row.get("top5", [])] for qid, row in baseline.items()})
    raw_metrics = outcome_metrics(baseline, raw)
    candidate_coverages: list[float] = []
    for query_id, row in baseline.items():
        gold = {str(doc_id) for doc_id in row.get("gold_documents", [])}
        scored = {str(item["doc_id"]) for item in scores_by_query[query_id]}
        candidate_coverages.append(len(gold & scored) / len(gold) if gold else 1.0)

    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    diagnostics: list[dict[str, Any]] = []
    best_predictions: dict[str, list[str]] = {}
    best_key: tuple[float, float] | None = None
    boundary_gaps: list[float] = []
    for query_rows in scores_by_query.values():
        ranked = sorted(query_rows, key=lambda row: (-float(row["neural_score"]), int(row.get("union_rank", 10**9)), str(row["doc_id"])))
        if len(ranked) > 5:
            boundary_gaps.append(float(ranked[4]["neural_score"]) - float(ranked[5]["neural_score"]))
    for threshold in thresholds:
        fused: dict[str, list[str]] = {}
        detail_rows: list[dict[str, Any]] = []
        rank_drops = {str(rank): 0 for rank in range(1, 6)}
        swaps = rescued = broken = 0
        for query_id, row in baseline.items():
            top, detail = fuse_query(
                row["top5"],
                scores_by_query[query_id],
                threshold=threshold,
                max_swaps=1,
                max_neural_rank=args.max_neural_rank,
                protect_top_ranks=args.protect_top_ranks,
            )
            fused[query_id] = top
            detail_rows.append({"query_id": query_id, **detail})
            if detail.get("changed"):
                swaps += int(detail.get("swaps", 0))
                rank = detail.get("dropped_rank")
                if rank is not None:
                    rank_drops[str(rank)] += 1
                gold = {str(doc_id) for doc_id in row.get("gold_documents", [])}
                before = recall(gold, [str(doc_id) for doc_id in row["top5"]])
                after = recall(gold, top)
                rescued += before == 0 and after > 0
                broken += before > 0 and after < before
        metrics = outcome_metrics(baseline, fused)
        diagnostics.append(
            {
                "threshold": threshold,
                "policy": {
                    "max_swaps_per_query": 1,
                    "max_neural_rank": args.max_neural_rank,
                    "protect_top_ranks": args.protect_top_ranks,
                },
                **metrics,
                "delta_vs_baseline": metrics["macro_recall"] - baseline_metrics["macro_recall"],
                "swaps": swaps,
                "number_dropped_by_baseline_rank": rank_drops,
                "zero_hit_baseline_queries_rescued": rescued,
                "previously_correct_baseline_queries_broken": broken,
                "max_neural_score_gap_around_top5_boundary": max(boundary_gaps, default=0.0),
            }
        )
        current_key = (metrics["macro_recall"], -threshold)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_predictions = fused

    best = max(diagnostics, key=lambda item: (item["macro_recall"], -item["threshold"])) if diagnostics else None
    report = {
        "status": "V2_EVAL_DIAGNOSTIC_ONLY",
        "fold": 0,
        "raw_v2_macro_recall": raw_metrics["macro_recall"],
        "baseline_fold0_macro_recall_recomputed": baseline_metrics["macro_recall"],
        "raw_v2_delta": raw_metrics["macro_recall"] - baseline_metrics["macro_recall"],
        "raw_v2_better": raw_metrics["better"],
        "raw_v2_worse": raw_metrics["worse"],
        "raw_v2_neutral": raw_metrics["neutral"],
        "raw_v2_changed_queries": raw_metrics["changed_queries"],
        "raw_v2_full_gold_count": raw_metrics["full_gold_count"],
        "raw_v2_multi_gold_macro_recall": raw_metrics["multi_gold_macro_recall"],
        "baseline_multi_gold_macro_recall": baseline_metrics["multi_gold_macro_recall"],
        "candidate_gold_coverage": {
            "mean": sum(candidate_coverages) / len(candidate_coverages) if candidate_coverages else 0.0,
            "zero_queries": sum(value == 0 for value in candidate_coverages),
        },
        "max_neural_score_gap_around_top5_boundary": max(boundary_gaps, default=0.0),
        "selective_fusion": {
            "best_diagnostic_macro_recall": best["macro_recall"] if best else None,
            "best_threshold": best["threshold"] if best else None,
            "best_delta_vs_baseline": best["delta_vs_baseline"] if best else None,
            "best_better": best["better"] if best else None,
            "best_worse": best["worse"] if best else None,
            "best_neutral": best["neutral"] if best else None,
            "best_changed_queries": best["changed_queries"] if best else None,
            "best_swaps": best["swaps"] if best else None,
            "best_number_dropped_by_baseline_rank": best["number_dropped_by_baseline_rank"] if best else None,
            "best_zero_hit_baseline_queries_rescued": best["zero_hit_baseline_queries_rescued"] if best else None,
            "best_previously_correct_baseline_queries_broken": best["previously_correct_baseline_queries_broken"] if best else None,
            "threshold_sweep": diagnostics,
            "diagnostic_only": True,
        },
        "query_count": len(baseline),
        "full_gold_count": raw_metrics["full_gold_count"],
        "training_or_calibration_claim": "none; fold0 sweep is diagnostic and not unbiased model selection",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fusion_predictions and best_predictions:
        rows = [
            {"query_id": query_id, "fold": 0, "top5": top}
            for query_id, top in sorted(best_predictions.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else item[0]))
        ]
        args.fusion_predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
