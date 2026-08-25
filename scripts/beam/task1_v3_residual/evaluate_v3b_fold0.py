"""Evaluation-only fold0 report for a frozen V3B policy; never writes a submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .common import jsonl
    from .evaluate_v3_fold0 import metric
except ImportError:
    from common import jsonl
    from evaluate_v3_fold0 import metric


def policy_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in jsonl(path)}


def validate_policy(rows: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], *, require_fold0_count: bool = True) -> dict[str, bool]:
    if set(rows) != set(baseline) or (require_fold0_count and len(rows) != 1400):
        raise ValueError("V3B policy and fold0 baseline must contain the same 1,400 queries")
    for query_id, row in rows.items():
        top5, base = [str(doc) for doc in row["top5"]], [str(doc) for doc in baseline[query_id]["top5"]]
        changed_positions = [index for index in range(5) if base[index] != top5[index]]
        if len(top5) != 5 or len(set(top5)) != 5 or top5[:3] != base[:3] or len(changed_positions) > 1 or any(index not in (3, 4) for index in changed_positions):
            raise ValueError(f"V3B invalid or rank1--3-breaking policy output: {query_id}")
        selected = row.get("selected_action")
        if selected is None:
            if changed_positions:
                raise ValueError(f"V3B abstention changed baseline: {query_id}")
            continue
        drop_rank = int(selected.get("drop_rank", -1))
        incoming = str(selected.get("incoming_doc_id", ""))
        if drop_rank not in (4, 5) or len(changed_positions) != 1 or changed_positions[0] != drop_rank - 1 or top5[drop_rank - 1] != incoming or incoming in base:
            raise ValueError(f"V3B selected_action does not match emitted one-swap top5: {query_id}")
    return {"top1_to3_protected": True, "max_one_swap_per_query": True, "exactly_five_distinct_docs": True, "selected_action_consistency": True}


def swap_diagnostics(base: dict[str, dict[str, Any]], policy: dict[str, dict[str, Any]], questions: dict[str, Any]) -> dict[str, Any]:
    swaps = beneficial = harmful = neutral = rescues = broken = 0
    for query_id, row in policy.items():
        before = [str(doc) for doc in base[query_id]["top5"]]; after = [str(doc) for doc in row["top5"]]
        gold = {str(doc) for doc in questions[query_id]["answer"]}
        # Inline recall keeps this evaluator runnable as a direct script.
        before_score = len(gold & set(before)) / len(gold) if gold else 0.0; after_score = len(gold & set(after)) / len(gold) if gold else 0.0
        if before == after: continue
        swaps += 1; beneficial += after_score > before_score; harmful += after_score < before_score; neutral += after_score == before_score
        rescues += before_score == 0 and after_score > 0; broken += before_score > 0 and after_score < before_score
    return {"swaps": swaps, "beneficial_swaps": beneficial, "harmful_swaps": harmful, "neutral_swaps": neutral, "beneficial_swap_precision": beneficial / swaps if swaps else 0.0, "zero_hit_rescues": rescues, "previously_correct_baseline_queries_broken": broken}


def score_gate(value: float) -> str:
    if value < .925: return "REJECT"
    if value < .930: return "WEAK"
    if value < .935: return "REAL_SIGNAL"
    if value < .940: return "STRONG"
    return "TARGET_REGION"


def self_test() -> dict[str, bool]:
    base = {"q": {"top5": ["a", "b", "c", "d", "e"]}}
    legal = {"q": {"top5": ["a", "b", "c", "d", "x"], "selected_action": {"drop_rank": 5, "incoming_doc_id": "x"}}}
    legal_one_swap = all(validate_policy(legal, base, require_fold0_count=False).values())
    try: validate_policy({"q": {"top5": ["a", "b", "c", "x", "y"], "selected_action": {"drop_rank": 5, "incoming_doc_id": "y"}}}, base, require_fold0_count=False)
    except ValueError: two_position_rejected = True
    else: two_position_rejected = False
    try: validate_policy({"q": {"top5": ["a", "b", "c", "d", "x"], "selected_action": {"drop_rank": 4, "incoming_doc_id": "x"}}}, base, require_fold0_count=False)
    except ValueError: selected_action_mismatch_rejected = True
    else: selected_action_mismatch_rejected = False
    return {"legal_one_swap": legal_one_swap, "two_position_mutation_rejected": two_position_rejected, "selected_action_mismatch_rejected": selected_action_mismatch_rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v3a-predictions", type=Path, required=True)
    parser.add_argument("--v3b-predictions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = self_test()
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "gpu_launched": False}, indent=2)); return
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    baseline = {str(row["query_id"]): row for row in jsonl(args.baseline) if int(row["fold"]) == 0}
    v3a, v3b = policy_rows(args.v3a_predictions), policy_rows(args.v3b_predictions)
    v3a_validation = validate_policy(v3a, baseline); v3b_validation = validate_policy(v3b, baseline)
    base_top5 = {query_id: [str(doc) for doc in row["top5"]] for query_id, row in baseline.items()}
    base = {query_id: {"gold_documents": [str(doc) for doc in questions[query_id]["answer"]], "top5": top5} for query_id, top5 in base_top5.items()}
    baseline_metric = metric(base, base_top5); v3a_metric = metric(base, {query_id: [str(doc) for doc in row["top5"]] for query_id, row in v3a.items()}); v3b_metric = metric(base, {query_id: [str(doc) for doc in row["top5"]] for query_id, row in v3b.items()})
    report = {"status": "V3B_FOLD0_EVALUATED", "query_count": len(base), "baseline": baseline_metric, "v3a": v3a_metric, "v3b": {**v3b_metric, **swap_diagnostics(baseline, v3b, questions)}, "delta_v3b_vs_baseline": v3b_metric["macro_recall"] - baseline_metric["macro_recall"], "delta_v3b_vs_v3a": v3b_metric["macro_recall"] - v3a_metric["macro_recall"], "score_gate": score_gate(v3b_metric["macro_recall"]), "validation": {"v3a": v3a_validation, "v3b": v3b_validation}, "fold0_used_for": "final_evaluation_only_after_selection", "no_submission_created": True, "gpu_launched": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
