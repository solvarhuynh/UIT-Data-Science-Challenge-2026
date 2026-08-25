"""CPU-only folds1--4 audit of exact one-swap Recall set algebra."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import jsonl, write_jsonl
except ImportError:
    from common import jsonl, write_jsonl


FOLDS = (1, 2, 3, 4)
WINDOWS: tuple[int | None, ...] = (10, 20, 50, None)
EPS = 1e-12
OUTCOME_KEYS = {"label", "gain", "resulting_recall", "baseline_recall", "gold"}


def sign(value: float) -> str:
    return "POSITIVE" if value > EPS else "NEGATIVE" if value < -EPS else "ZERO"


def validate_actions(actions: list[dict[str, Any]]) -> None:
    for row in actions:
        if int(row["fold"]) not in FOLDS:
            raise ValueError(f"swap relevance audit accepts folds1--4 only: {row['query_id']} fold={row['fold']}")
        baseline = [str(value) for value in row["baseline_top5"]]
        if len(baseline) != 5:
            raise ValueError(f"invalid baseline top5 for {row['query_id']}")
        if int(row["drop_rank"]) not in (4, 5):
            raise ValueError(f"illegal drop rank for {row['query_id']}")


def action_top5(baseline: list[str], incoming: str, drop_rank: int) -> list[str]:
    top5 = list(baseline)
    if drop_rank not in (4, 5):
        raise AssertionError("only rank4/rank5 replacements are legal")
    top5[drop_rank - 1] = str(incoming)
    if len(top5) != 5 or top5[:3] != list(baseline)[:3]:
        raise AssertionError("one-swap top5 must preserve ranks1--3")
    return top5


def action_record(row: dict[str, Any], gold: set[str], candidate_order: int) -> dict[str, Any]:
    if not gold:
        raise ValueError(f"empty gold answer set is invalid for audited query {row['query_id']}")
    baseline = [str(value) for value in row["baseline_top5"]]
    incoming = str(row["incoming_doc_id"])
    dropped = baseline[int(row["drop_rank"]) - 1]
    actual = float(row["gain"])
    if "resulting_recall" in row:
        if abs((float(row["resulting_recall"]) - float(row["baseline_recall"])) - actual) > EPS:
            raise AssertionError(f"gain/resulting_recall disagreement for {row['query_id']}")
    before = set(baseline); after = set(action_top5(baseline, incoming, int(row["drop_rank"])))
    naive_hit = int(incoming in gold) - int(dropped in gold)
    exact_hit = len(after & gold) - len(before & gold)
    expected = exact_hit / len(gold)
    return {"query_id": str(row["query_id"]), "fold": int(row["fold"]), "incoming_doc_id": incoming, "dropped_doc_id": dropped, "drop_rank": int(row["drop_rank"]), "candidate_order": candidate_order, "gold_count": len(gold), "baseline_hit_count": len(before & gold), "incoming_relevant": int(incoming in gold), "dropped_relevant": int(dropped in gold), "naive_hit_delta": naive_hit, "exact_hit_delta": exact_hit, "actual_delta": actual, "expected_recall_delta": expected, "actual_sign": sign(actual), "naive_sign": sign(naive_hit / len(gold)), "exact_sign": sign(expected), "magnitude_error": abs(actual - expected), "label": str(row.get("label", "MISSING")), "duplicate_effect": naive_hit != exact_hit}


def confusion(rows: list[dict[str, Any]], predicted: str) -> dict[str, dict[str, int]]:
    names = ("POSITIVE", "ZERO", "NEGATIVE")
    counts = Counter((row["actual_sign"], row[predicted]) for row in rows)
    return {actual: {estimate: int(counts[(actual, estimate)]) for estimate in names} for actual in names}


def structure(rows: list[dict[str, Any]], actual_sign: str) -> dict[str, int]:
    values = [row for row in rows if row["actual_sign"] == actual_sign]
    pairs = Counter((int(row["incoming_relevant"]), int(row["dropped_relevant"])) for row in values)
    return {f"incoming_{incoming}|dropped_{dropped}": int(pairs[(incoming, dropped)]) for incoming in (1, 0) for dropped in (0, 1)}


def drop_rank_analysis(rows: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    values = [row for row in rows if int(row["drop_rank"]) == rank]
    count = len(values); signs = Counter(row["actual_sign"] for row in values)
    return {"action_count": count, "positive_count": signs["POSITIVE"], "zero_count": signs["ZERO"], "negative_count": signs["NEGATIVE"], "dropped_relevance_rate": sum(row["dropped_relevant"] for row in values) / count if count else 0.0, "positive_action_rate": signs["POSITIVE"] / count if count else 0.0}


def window_analysis(rows: list[dict[str, Any]], window: int | None) -> dict[str, Any]:
    values = rows if window is None else [row for row in rows if int(row["candidate_order"]) <= window]
    relevant_queries = {row["query_id"] for row in values if row["incoming_relevant"]}
    positive_queries = {row["query_id"] for row in values if row["actual_sign"] == "POSITIVE"}
    incoming_docs = {(row["query_id"], row["incoming_doc_id"]) for row in values}
    relevant_incoming_docs = {(row["query_id"], row["incoming_doc_id"]) for row in values if row["incoming_relevant"]}
    return {"action_count": len(values), "incoming_unique_candidate_count": len(incoming_docs), "incoming_relevant_action_row_count": sum(row["incoming_relevant"] for row in values), "incoming_relevant_unique_candidate_count": len(relevant_incoming_docs), "incoming_relevant_candidate_query_coverage": len(relevant_queries), "positive_action_count": sum(row["actual_sign"] == "POSITIVE" for row in values), "positive_opportunity_query_count": len(positive_queries)}


def query_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    output = []
    for query_id, values in grouped.items():
        first = values[0]
        baseline_rank_relevant = {str(rank): int(next(row["dropped_relevant"] for row in values if int(row["drop_rank"]) == rank)) for rank in (4, 5)}
        item = {"query_id": query_id, "fold": first["fold"], "gold_count": first["gold_count"], "baseline_relevant_hit_count": first["baseline_hit_count"], "rank4_relevant": baseline_rank_relevant["4"], "rank5_relevant": baseline_rank_relevant["5"]}
        for window in WINDOWS:
            subset = values if window is None else [row for row in values if row["candidate_order"] <= window]
            key = "ALL" if window is None else str(window)
            item[f"relevant_incoming_action_rows_{key}"] = sum(row["incoming_relevant"] for row in subset)
            item[f"relevant_incoming_unique_candidates_{key}"] = len({row["incoming_doc_id"] for row in subset if row["incoming_relevant"]})
            item[f"positive_actions_{key}"] = sum(row["actual_sign"] == "POSITIVE" for row in subset)
        output.append(item)
    distributions = {"gold_count": Counter(row["gold_count"] for row in output), "baseline_relevant_hit_count": Counter(row["baseline_relevant_hit_count"] for row in output)}
    top20 = [row for row in output if row["positive_actions_20"] > 0]
    structure_counts = Counter()
    for row in top20:
        r4, r5 = bool(row["rank4_relevant"]), bool(row["rank5_relevant"])
        if not r4 and not r5: structure_counts["both_rank4_rank5_irrelevant"] += 1
        elif not r4: structure_counts["only_rank4_irrelevant"] += 1
        elif not r5: structure_counts["only_rank5_irrelevant"] += 1
        else: structure_counts["both_rank4_rank5_relevant"] += 1
        if not r4: structure_counts["relevant_incoming_plus_irrelevant_rank4"] += 1
        if not r5: structure_counts["relevant_incoming_plus_irrelevant_rank5"] += 1
    return {"query_count": len(output), "gold_count_distribution": {str(key): int(value) for key, value in distributions["gold_count"].items()}, "baseline_relevant_hit_count_distribution": {str(key): int(value) for key, value in distributions["baseline_relevant_hit_count"].items()}, "top20_opportunity_structure": {key: int(value) for key, value in structure_counts.items()}}, output


def raw_baseline_perfect_relevance_oracle(rows: list[dict[str, Any]], window: int) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_order"] <= window:
            grouped[row["query_id"]].append(row)
    values = []
    for actions in grouped.values():
        gold_count = int(actions[0]["gold_count"])
        if gold_count <= 0:
            raise AssertionError("raw-baseline oracle requires non-empty gold")
        before = float(actions[0]["baseline_hit_count"]) / gold_count
        values.append(max([before, *[before + float(action["expected_recall_delta"]) for action in actions]]))
    return sum(values) / len(values) if values else 0.0


def candidate_multiplicity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((row["query_id"], row["incoming_doc_id"]) for row in rows)
    distribution = Counter(counts.values())
    return {"pair_count": len(counts), "min_action_rows_per_query_incoming": min(counts.values()) if counts else 0, "max_action_rows_per_query_incoming": max(counts.values()) if counts else 0, "action_rows_per_query_incoming_distribution": {str(key): int(value) for key, value in sorted(distribution.items())}, "all_multiplicities_positive": all(value > 0 for value in counts.values())}


def positive_relevance_supported(rows: list[dict[str, Any]]) -> bool:
    positive = [row for row in rows if row["actual_sign"] == "POSITIVE"]
    return bool(positive) and all(row["incoming_relevant"] and not row["dropped_relevant"] for row in positive)


def analyze(actions: list[dict[str, Any]], questions: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_actions(actions)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        by_query[str(row["query_id"])].append(row)
    records = []
    for query_id, values in by_query.items():
        if query_id not in questions:
            raise KeyError(f"gold source missing query {query_id}")
        gold = {str(value) for value in questions[query_id].get("answer", [])}
        order = sorted({str(row["incoming_doc_id"]): float(row.get("features", {}).get("incoming_union_rank", float("inf"))) for row in values}.items(), key=lambda item: (item[1], item[0]))
        rank = {doc_id: index for index, (doc_id, _) in enumerate(order, start=1)}
        records.extend(action_record(row, gold, rank[str(row["incoming_doc_id"])]) for row in values)
    exact_match = [row for row in records if row["actual_sign"] == row["exact_sign"]]
    naive_match = [row for row in records if row["actual_sign"] == row["naive_sign"]]
    mismatches = [row for row in records if row["magnitude_error"] > EPS]
    actual_counts = Counter(row["actual_sign"] for row in records)
    label_counts = Counter((row["actual_sign"], row["label"]) for row in records)
    query_report, _ = query_summary(records)
    raw_union_top20_oracle = raw_baseline_perfect_relevance_oracle(records, 20)
    positive_supported = positive_relevance_supported(records)
    report = {"status": "SWAP_RELEVANCE_ALGEBRA_AUDIT_COMPLETE", "folds": list(FOLDS), "gold_source": "data/raw/btc/LegalIR/train.json: questions[query_id].answer, folds1--4 only; empty answers are rejected", "ordering_source": "incoming_union_rank secondary diagnostic only (no Stage1 fitting or persisted fixed-V3B action ordering)", "fixed_v3b_top20_oracle_checked": False, "fixed_v3b_top20_oracle_unavailable_reason": "exact fixed-V3B Stage1 ordering is not available in this model-free audit", "action_count": len(records), "query_count": len(by_query), "gold_count_distribution": query_report["gold_count_distribution"], "actual_delta_distribution": {key: int(value) for key, value in actual_counts.items()}, "exact_sign_confusion_matrix": confusion(records, "exact_sign"), "naive_sign_confusion_matrix": confusion(records, "naive_sign"), "exact_sign_accuracy": len(exact_match) / len(records) if records else 0.0, "naive_sign_accuracy": len(naive_match) / len(records) if records else 0.0, "exact_magnitude_accuracy": (len(records) - len(mismatches)) / len(records) if records else 0.0, "max_abs_delta_error": max((row["magnitude_error"] for row in records), default=0.0), "mean_abs_delta_error": sum(row["magnitude_error"] for row in records) / len(records) if records else 0.0, "mismatch_count": len(mismatches), "mismatch_query_count": len({row["query_id"] for row in mismatches}), "per_fold_mismatch_count": {str(fold): sum(row["fold"] == fold and row["magnitude_error"] > EPS for row in records) for fold in FOLDS}, "naive_vs_exact_corrected_count": sum(row["actual_sign"] != row["naive_sign"] and row["actual_sign"] == row["exact_sign"] for row in records), "positive_action_relevance_cross_tab": structure(records, "POSITIVE"), "zero_action_relevance_cross_tab": structure(records, "ZERO"), "negative_action_relevance_cross_tab": structure(records, "NEGATIVE"), "drop_rank4_analysis": drop_rank_analysis(records, 4), "drop_rank5_analysis": drop_rank_analysis(records, 5), "incoming_relevance_analysis": {"overall": sum(row["incoming_relevant"] for row in records) / len(records) if records else 0.0, **{name.lower(): sum(row["incoming_relevant"] for row in records if row["actual_sign"] == name) / actual_counts[name] if actual_counts[name] else 0.0 for name in ("POSITIVE", "ZERO", "NEGATIVE")}}, "union_top10_analysis": window_analysis(records, 10), "union_top20_analysis": window_analysis(records, 20), "union_top50_analysis": window_analysis(records, 50), "union_all_analysis": window_analysis(records, None), "query_level_analysis": query_report, "union_top20_opportunity_structure": query_report["top20_opportunity_structure"], "action_row_multiplicity": candidate_multiplicity(records), "action_label_cross_tab": {f"{delta}|{label}": int(label_counts[(delta, label)]) for delta in ("POSITIVE", "ZERO", "NEGATIVE") for label in ("BENEFIT", "NEUTRAL", "HARM")}, "raw_baseline_perfect_relevance_union_top20_oracle": raw_union_top20_oracle, "exact_relevance_algebra_valid": len(mismatches) == 0, "positive_actions_are_relevant_in_irrelevant_out": positive_supported, "relevance_decomposition_supported": len(mismatches) == 0 and positive_supported, "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "model_trained": False, "no_submission_created": True}
    return report, mismatches


def self_test() -> dict[str, bool]:
    gold = {"A", "C"}; baseline = ["A", "B", "D", "E", "F"]
    def row(incoming: str, drop_rank: int, gain: float, fold: int = 1, resulting: float | None = None) -> dict[str, Any]:
        value = {"query_id": "q", "fold": fold, "incoming_doc_id": incoming, "drop_rank": drop_rank, "baseline_top5": baseline, "baseline_recall": .5, "gain": gain, "label": "NEUTRAL", "features": {"incoming_union_rank": 1.0}}
        if resulting is not None: value["resulting_recall"] = resulting
        return value
    positive = action_record(row("C", 5, .5), gold, 1)
    positive_rank4 = action_record(row("C", 4, .5), gold, 1)
    negative = action_record({**row("X", 4, -.5), "baseline_top5": ["B", "D", "E", "A", "F"]}, gold, 1)
    both_relevant = action_record(row("C", 4, 0.0), {"A", "E", "C"}, 1); both_irrel = action_record(row("X", 4, 0.0), gold, 1)
    duplicate = action_record(row("A", 5, 0.0), gold, 1)
    agreement = action_record(row("C", 5, .5, resulting=1.0), gold, 1)
    fold0_rejected = False
    try: validate_actions([row("C", 5, .5, fold=0)])
    except ValueError: fold0_rejected = True
    oracle_rows = [positive, negative]
    multiplicity_rows = [{**positive, "query_id": "unique", "incoming_doc_id": "C"}, {**positive_rank4, "query_id": "unique", "incoming_doc_id": "C"}, {**negative, "query_id": "unique", "incoming_doc_id": "X"}]
    empty_gold_rejected = False
    try: action_record(row("C", 5, .5), set(), 1)
    except ValueError: empty_gold_rejected = True
    report, _ = analyze([row("C", 5, .5), row("C", 4, .5)], {"q": {"answer": ["A", "C"]}})
    return {"relevant_in_irrelevant_out_positive": positive["exact_hit_delta"] == 1, "irrelevant_in_relevant_out_negative": negative["exact_hit_delta"] == -1, "both_relevant_zero": both_relevant["exact_hit_delta"] == 0, "both_irrelevant_zero": both_irrel["exact_hit_delta"] == 0, "duplicate_set_guard": duplicate["naive_hit_delta"] == 1 and duplicate["exact_hit_delta"] == 0, "unique_candidate_counting": window_analysis(multiplicity_rows, None)["incoming_relevant_unique_candidate_count"] == 1 and window_analysis(multiplicity_rows, None)["incoming_relevant_action_row_count"] == 2 and candidate_multiplicity(multiplicity_rows)["action_rows_per_query_incoming_distribution"] == {"1": 1, "2": 1}, "rank4_rank5_only": action_top5(baseline, "C", 4)[3] == "C" and action_top5(baseline, "C", 5)[4] == "C", "top1_to3_protected": action_top5(baseline, "C", 5)[:3] == baseline[:3], "gain_resulting_agree": agreement["magnitude_error"] == 0.0, "fold0_rejected": fold0_rejected, "empty_gold_rejected": empty_gold_rejected, "zero_positive_fails_closed": not positive_relevance_supported([both_irrel, duplicate]), "no_model_training": "xgboost" not in globals() and "torch" not in globals(), "gold_denominator": abs(positive["expected_recall_delta"] - .5) < EPS, "raw_baseline_oracle_not_fixed_v3b_comparison": report["fixed_v3b_top20_oracle_checked"] is False and "fixed-V3B Stage1 ordering" in report["fixed_v3b_top20_oracle_unavailable_reason"], "raw_baseline_oracle_monotonic": raw_baseline_perfect_relevance_oracle(oracle_rows, 20) + EPS >= raw_baseline_perfect_relevance_oracle(oracle_rows, 10)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path); parser.add_argument("--questions", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False, "model_trained": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "top_windows": [10, 20, 50, "ALL"], "model_trained": False, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if any(value is None for value in (args.actions, args.questions, args.output_dir)): parser.error("--actions, --questions, and --output-dir are required")
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    report, mismatches = analyze(actions, questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "swap_relevance_algebra_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if mismatches: write_jsonl(args.output_dir / "swap_relevance_algebra_mismatches.jsonl", mismatches)


if __name__ == "__main__": main()
