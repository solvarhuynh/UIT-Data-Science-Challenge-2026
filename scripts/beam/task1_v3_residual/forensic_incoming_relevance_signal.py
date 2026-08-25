"""Model-free folds1--4 audit of cached incoming-document relevance signals."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .common import jsonl
except ImportError:
    from common import jsonl


FOLDS = (1, 2, 3, 4)
EXPECTED = {"action_count": 172338, "unique_candidate_count": 86169, "relevant_candidate_count": 320, "opportunity_query_count": 301, "nonopportunity_query_count": 5299}
OUTCOME_FIELDS = ("label", "gain", "baseline_recall", "resulting_recall", "gold", "delta_recall")


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def require_actions(actions: list[dict[str, Any]]) -> None:
    for row in actions:
        if int(row["fold"]) not in FOLDS:
            raise ValueError(f"incoming relevance forensic accepts folds1--4 only: {row['query_id']} fold={row['fold']}")
        if int(row["drop_rank"]) not in (4, 5):
            raise ValueError(f"illegal drop rank for {row['query_id']}")


def feature_classification(actions: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    names = sorted({name for row in actions for name in row.get("features", {})})
    for row in actions:
        groups[(str(row["query_id"]), str(row["incoming_doc_id"]))].append(row)
    diagnostics: dict[str, dict[str, Any]] = {}
    categories = {"candidate_invariant": [], "drop_dependent": [], "outcome_or_label": list(OUTCOME_FIELDS), "unknown": []}
    for name in names:
        equal, compared = 0, 0
        for key, copies in groups.items():
            if len(copies) != 2:
                raise ValueError(f"expected exactly two action copies for {key}, found {len(copies)}")
            ranks = {int(row["drop_rank"]) for row in copies}
            if ranks != {4, 5}:
                raise ValueError(f"expected rank4/rank5 copies for {key}, found {sorted(ranks)}")
            compared += 1
            if copies[0].get("features", {}).get(name) == copies[1].get("features", {}).get(name):
                equal += 1
        invariant = equal == compared
        diagnostics[name] = {"pair_count": compared, "equal_pair_count": equal, "different_pair_count": compared - equal, "within_candidate_invariance_rate": equal / compared if compared else 0.0}
        if name.startswith("incoming_") and invariant:
            categories["candidate_invariant"].append(name)
        elif name.startswith(("dropped_", "diff_")) or name == "dropped_baseline_rank" or (name.startswith("incoming_") and not invariant):
            categories["drop_dependent"].append(name)
        else:
            categories["unknown"].append(name)
    return categories, diagnostics


def semantics(name: str) -> dict[str, Any]:
    suffix = name.removeprefix("incoming_")
    higher = {"neural_max", "neural_second", "neural_mean", "neural_min", "neural_std", "neural_max_minus_mean", "neural_top2_mean", "bm25_max", "bm25_mean", "reciprocal_union_rank", "source_support", "has_bge_support"}
    lower = {"union_rank", "min_source_rank", "bm25_rank", "adaptive_k500_rank", "knn_char_rank", "knn_word_rank"}
    if suffix in higher:
        return {"definition_source": "score_frozen_features.py:48-59", "expected_direction": "higher_better", "direction_verified": True}
    if suffix in lower:
        return {"definition_source": "score_frozen_features.py:55-59 and build_shortlist_features.py:32-36", "expected_direction": "lower_better", "direction_verified": True}
    return {"definition_source": "build_actions.py: BASE_FEATURES copied from incoming frozen candidate", "expected_direction": "unknown", "direction_verified": False}


def collapse(actions: list[dict[str, Any]], questions: dict[str, Any], candidate_features: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        grouped[(str(row["query_id"]), str(row["incoming_doc_id"]))].append(row)
    candidates = []
    for (query_id, incoming), copies in grouped.items():
        if query_id not in questions:
            raise KeyError(f"gold source missing query {query_id}")
        gold = {str(value) for value in questions[query_id].get("answer", [])}
        if not gold:
            raise ValueError(f"empty gold answer set is invalid for audited query {query_id}")
        if len(copies) != 2 or {int(row["drop_rank"]) for row in copies} != {4, 5}:
            raise ValueError(f"cannot safely deduplicate action copies for {(query_id, incoming)}")
        first = copies[0]
        values = {name: first.get("features", {}).get(name) for name in candidate_features}
        if any(values[name] != copies[1].get("features", {}).get(name) for name in candidate_features):
            raise AssertionError(f"candidate-invariant feature changed across action copies for {(query_id, incoming)}")
        candidates.append({"query_id": query_id, "fold": int(first["fold"]), "incoming_doc_id": incoming, "incoming_relevant": int(incoming in gold), "features": values})
    return candidates


def assert_expected(actions: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    by_query = defaultdict(list)
    for row in candidates:
        by_query[row["query_id"]].append(row)
    actual = {"action_count": len(actions), "unique_candidate_count": len(candidates), "relevant_candidate_count": sum(row["incoming_relevant"] for row in candidates), "opportunity_query_count": sum(any(row["incoming_relevant"] for row in rows) for rows in by_query.values()), "nonopportunity_query_count": sum(not any(row["incoming_relevant"] for row in rows) for rows in by_query.values())}
    if actual != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "actual": actual})


def ranks(rows: list[dict[str, Any]], name: str, direction: str) -> tuple[list[dict[str, Any]], int]:
    missing = sum(numeric(row["features"].get(name)) is None for row in rows)
    def key(row: dict[str, Any]) -> tuple[float, str]:
        value = numeric(row["features"].get(name))
        oriented = -math.inf if value is None else (value if direction == "higher_better" else -value)
        return (-oriented, str(row["incoming_doc_id"]))
    return sorted(rows, key=key), missing


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    ordered = sorted(values)
    def pick(p: float) -> float: return ordered[round((len(ordered) - 1) * p)]
    return {"min": ordered[0], "p10": pick(.10), "p25": pick(.25), "median": median(ordered), "p75": pick(.75), "p90": pick(.90), "max": ordered[-1]}


def opportunity_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else .5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def signal_metrics(candidates: list[dict[str, Any]], name: str, direction: str) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates: by_query[row["query_id"]].append(row)
    top_hits = {k: 0 for k in (1, 2, 3, 5, 10)}; reciprocal = []; best_ranks = []
    concordant = discordant = ties = 0; macro_concordance = []; best_opportunity = []; best_nonopportunity = []; missing = 0
    for rows in by_query.values():
        ranked, row_missing = ranks(rows, name, direction); missing += row_missing
        raw_values = [numeric(row["features"].get(name)) for row in ranked]
        oriented = [(-math.inf if value is None else value if direction == "higher_better" else -value) for value in raw_values]
        relevant = [index for index, row in enumerate(ranked, start=1) if row["incoming_relevant"]]
        top_score = oriented[0] if oriented else -math.inf
        if relevant:
            first = min(relevant); reciprocal.append(1.0 / first); best_ranks.append(first)
            for k in top_hits: top_hits[k] += int(first <= k)
            query_concordant = query_discordant = query_ties = 0
            for rel in (row for row in ranked if row["incoming_relevant"]):
                rel_value = numeric(rel["features"].get(name)); rel_score = -math.inf if rel_value is None else rel_value if direction == "higher_better" else -rel_value
                for irrel in (row for row in ranked if not row["incoming_relevant"]):
                    irrel_value = numeric(irrel["features"].get(name)); irrel_score = -math.inf if irrel_value is None else irrel_value if direction == "higher_better" else -irrel_value
                    if rel_score > irrel_score: query_concordant += 1
                    elif rel_score < irrel_score: query_discordant += 1
                    else: query_ties += 1
            concordant += query_concordant; discordant += query_discordant; ties += query_ties
            pairs = query_concordant + query_discordant + query_ties
            macro_concordance.append((query_concordant + .5 * query_ties) / pairs if pairs else 0.0)
            best_opportunity.append(top_score)
        else:
            best_nonopportunity.append(top_score)
    opportunities = len(reciprocal); pair_total = concordant + discordant + ties
    return {"feature": name, "direction": direction, "missing_count": missing, "top1_relevant_query_count": top_hits[1], "top1_relevant_rate": top_hits[1] / opportunities if opportunities else 0.0, "hit_at_2": top_hits[2] / opportunities if opportunities else 0.0, "hit_at_3": top_hits[3] / opportunities if opportunities else 0.0, "hit_at_5": top_hits[5] / opportunities if opportunities else 0.0, "hit_at_10": top_hits[10] / opportunities if opportunities else 0.0, "mrr_first_relevant": sum(reciprocal) / opportunities if opportunities else 0.0, "mean_best_relevant_rank": sum(best_ranks) / opportunities if opportunities else None, "median_best_relevant_rank": median(best_ranks) if best_ranks else None, "pairwise": {"concordant": concordant, "discordant": discordant, "ties": ties, "micro_concordance": (concordant + .5 * ties) / pair_total if pair_total else 0.0, "macro_query_concordance": sum(macro_concordance) / len(macro_concordance) if macro_concordance else 0.0}, "top_candidate_oriented_score_distribution": {"opportunity": quantiles(best_opportunity), "nonopportunity": quantiles(best_nonopportunity)}, "query_opportunity_auc": opportunity_auc(best_opportunity, best_nonopportunity)}


def analyze(actions: list[dict[str, Any]], questions: dict[str, Any], enforce_expected: bool = True) -> dict[str, Any]:
    require_actions(actions)
    classification, invariance = feature_classification(actions)
    candidates = collapse(actions, questions, classification["candidate_invariant"])
    if enforce_expected: assert_expected(actions, candidates)
    semantics_by_feature = {name: semantics(name) for name in classification["candidate_invariant"]}
    results = []
    for name in classification["candidate_invariant"]:
        info = semantics_by_feature[name]
        directions = [info["expected_direction"]] if info["direction_verified"] else ["higher_better", "lower_better"]
        for direction in directions:
            result = signal_metrics(candidates, name, direction)
            results.append({**result, "direction_verified": info["direction_verified"], "within_candidate_invariance_rate": invariance[name]["within_candidate_invariance_rate"]})
    verified = [row for row in results if row["direction_verified"]]
    best = max(verified, key=lambda row: (row["top1_relevant_rate"], row["mrr_first_relevant"], row["pairwise"]["macro_query_concordance"], row["feature"])) if verified else None
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates: by_query[row["query_id"]].append(row)
    opportunity_groups = [rows for rows in by_query.values() if any(row["incoming_relevant"] for row in rows)]
    random_expected = sum(sum(row["incoming_relevant"] for row in rows) / len(rows) for rows in opportunity_groups) / len(opportunity_groups) if opportunity_groups else 0.0
    promising = bool(best and best["top1_relevant_rate"] >= .30 and best["pairwise"]["macro_query_concordance"] >= .65)
    moderate = bool(best and .25 <= best["top1_relevant_rate"] < .30)
    insufficient = bool(verified and all(row["top1_relevant_rate"] < .25 for row in verified))
    multiplicity = Counter((row["query_id"], row["incoming_doc_id"]) for row in actions)
    return {"status": "INCOMING_RELEVANCE_SIGNAL_FORENSIC_COMPLETE", "folds": list(FOLDS), "gold_source": "data/raw/btc/LegalIR/train.json: questions[query_id].answer, folds1--4 only", "action_source": "existing frozen V3 residual actions; no rebuild", "action_count_before_dedup": len(actions), "unique_candidate_count": len(candidates), "action_multiplicity_distribution": {str(key): int(value) for key, value in sorted(Counter(multiplicity.values()).items())}, "relevant_candidate_count": sum(row["incoming_relevant"] for row in candidates), "opportunity_query_count": len(opportunity_groups), "nonopportunity_query_count": len(by_query) - len(opportunity_groups), "candidate_positive_prevalence": sum(row["incoming_relevant"] for row in candidates) / len(candidates) if candidates else 0.0, "feature_classification": classification, "feature_invariance_diagnostics": invariance, "feature_source_semantics": semantics_by_feature, "signal_results": sorted(results, key=lambda row: (-row["top1_relevant_rate"], -row["mrr_first_relevant"], -row["pairwise"]["macro_query_concordance"], row["feature"], row["direction"])), "random_expected_top1_rate": random_expected, "best_signal": best["feature"] if best else None, "best_signal_top1_relevant_rate": best["top1_relevant_rate"] if best else None, "best_signal_mrr": best["mrr_first_relevant"] if best else None, "best_signal_macro_pairwise_concordance": best["pairwise"]["macro_query_concordance"] if best else None, "best_signal_query_opportunity_auc_if_available": best["query_opportunity_auc"] if best else None, "cached_signal_promising": promising, "cached_signal_moderate": moderate, "cached_signal_insufficient": insufficient, "fixed_v3b_stage1_ordering_used": False, "fold0_labels_read": False, "fold0_evaluated": False, "model_trained": False, "gpu_launched": False, "retrieval_rerun": False, "neural_scoring_rerun": False, "submission_created": False}


def self_test() -> dict[str, bool]:
    base = {"query_id": "q1", "fold": 1, "incoming_doc_id": "R", "baseline_top5": ["a", "b", "c", "d", "e"], "features": {"incoming_neural_max": .9, "incoming_union_rank": 2, "dropped_neural_max": .1, "diff_neural_max": .8}}
    rows = [{**base, "drop_rank": 4}, {**base, "drop_rank": 5, "features": {**base["features"], "dropped_neural_max": .2, "diff_neural_max": .7}}, {**base, "query_id": "q1", "incoming_doc_id": "N", "drop_rank": 4, "features": {"incoming_neural_max": .2, "incoming_union_rank": 1, "dropped_neural_max": .1, "diff_neural_max": .1}}, {**base, "query_id": "q1", "incoming_doc_id": "N", "drop_rank": 5, "features": {"incoming_neural_max": .2, "incoming_union_rank": 1, "dropped_neural_max": .2, "diff_neural_max": 0}}, {**base, "query_id": "q2", "incoming_doc_id": "X", "drop_rank": 4, "features": {"incoming_neural_max": .1, "incoming_union_rank": 2, "dropped_neural_max": .1, "diff_neural_max": 0}}, {**base, "query_id": "q2", "incoming_doc_id": "X", "drop_rank": 5, "features": {"incoming_neural_max": .1, "incoming_union_rank": 2, "dropped_neural_max": .2, "diff_neural_max": -.1}}]
    questions = {"q1": {"answer": ["R"]}, "q2": {"answer": ["Z"]}}
    classes, diagnostics = feature_classification(rows); candidates = collapse(rows, questions, classes["candidate_invariant"]); neural = signal_metrics(candidates, "incoming_neural_max", "higher_better"); union = signal_metrics(candidates, "incoming_union_rank", "lower_better")
    fold0_rejected = False; empty_gold_rejected = False; expected_rejected = False
    try: require_actions([{**rows[0], "fold": 0}])
    except ValueError: fold0_rejected = True
    try: collapse(rows[:2], {"q1": {"answer": []}}, classes["candidate_invariant"])
    except ValueError: empty_gold_rejected = True
    try: assert_expected(rows, candidates)
    except AssertionError: expected_rejected = True
    tied = [{"query_id": "t", "incoming_doc_id": "a", "incoming_relevant": 1, "features": {"f": 1}}, {"query_id": "t", "incoming_doc_id": "b", "incoming_relevant": 0, "features": {"f": 1}}]
    tie_metrics = signal_metrics(tied, "f", "higher_better")
    multi = [{"query_id": "m", "incoming_doc_id": "a", "incoming_relevant": 0, "features": {"f": 3}}, {"query_id": "m", "incoming_doc_id": "b", "incoming_relevant": 1, "features": {"f": 2}}, {"query_id": "m", "incoming_doc_id": "c", "incoming_relevant": 1, "features": {"f": 1}}]
    multi_metrics = signal_metrics(multi, "f", "higher_better")
    analytic_random = sum(row["incoming_relevant"] for row in candidates if row["query_id"] == "q1") / sum(1 for row in candidates if row["query_id"] == "q1")
    return {"deduplicates_two_action_copies": len(candidates) == 3, "gold_only_target": next(row["incoming_relevant"] for row in candidates if row["incoming_doc_id"] == "R") == 1, "candidate_invariant_detected": "incoming_neural_max" in classes["candidate_invariant"] and diagnostics["incoming_neural_max"]["within_candidate_invariance_rate"] == 1.0, "drop_dependent_rejected": "dropped_neural_max" in classes["drop_dependent"], "outcome_label_rejected": "gain" in classes["outcome_or_label"], "no_drop_value_averaging": next(row["features"]["incoming_neural_max"] for row in candidates if row["incoming_doc_id"] == "R") == .9, "higher_better_ranking": neural["top1_relevant_rate"] == 1.0, "lower_better_ranking": union["top1_relevant_rate"] == 0.0, "tie_pairwise_handled": tie_metrics["pairwise"]["ties"] == 1 and tie_metrics["pairwise"]["micro_concordance"] == .5, "multiple_relevant_top1_and_mrr": multi_metrics["top1_relevant_rate"] == 0.0 and multi_metrics["mrr_first_relevant"] == .5, "random_expected_analytic": abs(analytic_random - .5) < 1e-12, "empty_gold_rejected": empty_gold_rejected, "fold0_rejected": fold0_rejected, "expected_counts_fail_closed": expected_rejected, "no_model_training": "sklearn" not in globals() and "xgboost" not in globals() and "torch" not in globals(), "no_threshold_optimization": "threshold" not in globals(), "no_linear_combination_search": "Linear" not in globals()}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--actions", type=Path); parser.add_argument("--questions", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()): raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False, "model_trained": False}, indent=2)); return
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "folds": list(FOLDS), "model_trained": False, "fold0_labels_read": False, "gpu_launched": False}, indent=2)); return
    if any(value is None for value in (args.actions, args.questions, args.output_dir)): parser.error("--actions, --questions, and --output-dir are required")
    actions = [row for row in jsonl(args.actions) if int(row["fold"]) in FOLDS]
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    report = analyze(actions, questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "incoming_relevance_signal_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
