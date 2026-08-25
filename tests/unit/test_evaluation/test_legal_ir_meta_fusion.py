"""Leakage-safe split regression for the optional interpretable meta-ranker."""

from __future__ import annotations

from udsc2026.evaluation.legal_ir_meta_fusion import nested_oof_meta_ranker
from udsc2026.evaluation.legal_ir_recovery import metrics


def test_meta_ranker_trains_on_four_folds_and_scores_fifth() -> None:
    gold = {f"q{index}": [f"g{index}"] for index in range(10)}
    folds = {f"q{index}": index % 5 for index in range(10)}
    strong = {
        query_id: [documents[0], "a", "b", "c", "d", "e"]
        for query_id, documents in gold.items()
    }
    weak = {
        query_id: ["a", "b", "c", "d", documents[0], "e"]
        for query_id, documents in gold.items()
    }

    predictions, reports = nested_oof_meta_ranker(
        gold=gold,
        folds=folds,
        sources={"strong": strong, "weak": weak},
        source_depth=6,
        negatives_per_query=5,
    )

    assert metrics(sorted(gold), gold, predictions)["recall"] == 1.0
    assert len(reports) == 5
    assert all(row["training_query_count"] == 8 for row in reports)
