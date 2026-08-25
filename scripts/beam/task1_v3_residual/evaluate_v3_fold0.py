"""Strict fold0 evaluation and diagnostics for the V3 residual policy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .common import jsonl, recall, sort_query_ids
except ImportError:
    from common import jsonl, recall, sort_query_ids


def metric(base: dict[str, dict[str, Any]], predictions: dict[str, list[str]]) -> dict[str, Any]:
    values = []
    better = worse = neutral = changed = rescues = broken = full = 0
    multi = []
    for query_id, row in base.items():
        gold = {str(doc_id) for doc_id in row["gold_documents"]}
        baseline = [str(doc_id) for doc_id in row["top5"]]
        top5 = predictions[query_id]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise ValueError(f"invalid top5: {query_id}")
        before, after = recall(gold, baseline), recall(gold, top5)
        values.append(after); better += after > before; worse += after < before; neutral += after == before; changed += top5 != baseline
        rescues += before == 0 and after > 0; broken += before > 0 and after < before; full += gold.issubset(set(top5))
        if len(gold) > 1: multi.append(after)
    return {"macro_recall": mean(values), "better": better, "worse": worse, "neutral": neutral, "changed_queries": changed, "swaps": changed, "zero_hit_rescues": rescues, "previously_correct_broken": broken, "full_gold_count": full, "multi_gold_macro_recall": mean(multi) if multi else 0.0}


def oracle_top5(gold: set[str], shortlist: list[str], baseline: list[str]) -> list[str]:
    result = [doc_id for doc_id in shortlist if doc_id in gold]
    result += [doc_id for doc_id in baseline + shortlist if doc_id not in result]
    return result[:5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--policy-predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    baseline = {str(row["query_id"]): row for row in jsonl(args.baseline) if int(row["fold"]) == 0}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(args.features):
        if int(row["fold"]) == 0: by_query[str(row["query_id"])].append(row)
    policy_rows = {str(row["query_id"]): row for row in jsonl(args.policy_predictions)}
    policy = {query_id: [str(doc_id) for doc_id in row["top5"]] for query_id, row in policy_rows.items()}
    if set(baseline) != set(by_query) or set(baseline) != set(policy) or len(baseline) != 1400:
        raise ValueError("fold0 baseline/features/policy query sets must exactly match 1,400 queries")
    base = {qid: {"gold_documents": [str(doc_id) for doc_id in questions[qid]["answer"]], "top5": [str(doc_id) for doc_id in baseline[qid]["top5"]]} for qid in baseline}
    neural: dict[str, list[str]] = {}
    hand: dict[str, list[str]] = {}
    shortlist_oracle: dict[str, list[str]] = {}
    swap_oracle: dict[str, list[str]] = {}
    fp_max_minus_mean: list[float] = []; fp_second: list[float] = []; gold_max_minus_mean: list[float] = []; gold_second: list[float] = []
    for qid in sort_query_ids(base):
        docs = sorted(by_query[qid], key=lambda row: (-float(row["neural_max"]), int(row["union_rank"]), str(row["doc_id"])))
        baseline_top = base[qid]["top5"]
        gold = set(base[qid]["gold_documents"])
        neural[qid] = [str(row["doc_id"]) for row in docs[:5]]
        incoming = next((row for row in docs if str(row["doc_id"]) not in set(baseline_top)), None)
        drop = next(row for row in docs if str(row["doc_id"]) == baseline_top[4])
        hand[qid] = list(baseline_top)
        if incoming is not None and float(incoming["neural_max"]) >= float(drop["neural_max"]): hand[qid][4] = str(incoming["doc_id"])
        shortlist_ids = [str(row["doc_id"]) for row in docs]
        shortlist_oracle[qid] = oracle_top5(gold, shortlist_ids, baseline_top)
        best = (0.0, list(baseline_top))
        for row in docs:
            doc_id = str(row["doc_id"])
            if doc_id in baseline_top: continue
            for rank in (5, 4):
                candidate = list(baseline_top); candidate[rank - 1] = doc_id
                gain = recall(gold, candidate) - recall(gold, baseline_top)
                if gain > best[0]: best = (gain, candidate)
        swap_oracle[qid] = best[1]
        for row in docs:
            if str(row["doc_id"]) in gold:
                gold_max_minus_mean.append(float(row["neural_max_minus_mean"])); gold_second.append(float(row["neural_second"]))
        for row in docs[:5]:
            if str(row["doc_id"]) not in gold:
                fp_max_minus_mean.append(float(row["neural_max_minus_mean"])); fp_second.append(float(row["neural_second"]))
    drop_counts = {"4": 0, "5": 0}
    for row in policy_rows.values():
        selected = row.get("selected_action")
        if selected is not None:
            drop_counts[str(int(selected["drop_rank"]))] += 1
    report = {"status": "V3_FOLD0_EVALUATED", "query_count": len(base), "baseline": metric(base, {qid: row["top5"] for qid, row in base.items()}), "frozen_neural_pure_ranking_diagnostic_only": metric(base, neural), "simple_hand_fusion_diagnostic_only": metric(base, hand), "learned_v3_residual_policy": {**metric(base, policy), "per_drop_rank_counts": drop_counts}, "shortlist_oracle": metric(base, shortlist_oracle), "one_swap_oracle": metric(base, swap_oracle), "false_positive_isolated_max_analysis": {"gold_neural_max_minus_mean_mean": mean(gold_max_minus_mean) if gold_max_minus_mean else None, "top_false_positive_neural_max_minus_mean_mean": mean(fp_max_minus_mean) if fp_max_minus_mean else None, "gold_neural_second_mean": mean(gold_second) if gold_second else None, "top_false_positive_neural_second_mean": mean(fp_second) if fp_second else None}, "top1_to3_protected": True, "fold0_used_for": "evaluation_only", "diagnostic_only_controls": True, "no_submission_created": True}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
