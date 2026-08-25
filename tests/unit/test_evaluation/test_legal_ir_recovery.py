"""Strict OOF regression tests for Task1 recovery fusion."""

from __future__ import annotations

from udsc2026.evaluation.legal_ir_recovery import (
    FusionConfig,
    metrics,
    nested_oof_fusion,
    weighted_rrf,
)


def test_weighted_rrf_is_deterministic_and_distinct() -> None:
    config = FusionConfig((("a", 1.0), ("b", 1.0)), 0)
    result = weighted_rrf(
        {"a": ["d1", "d1", "d2", "d3"], "b": ["d2", "d1", "d4"]},
        config,
    )

    assert result[:2] == ["d1", "d2"]
    assert len(result) == len(set(result))


def test_nested_oof_selects_without_using_heldout_labels() -> None:
    gold = {f"q{index}": [f"g{index}"] for index in range(5)}
    folds = {f"q{index}": index for index in range(5)}
    good = {qid: [docs[0], "x1", "x2", "x3", "x4"] for qid, docs in gold.items()}
    bad = {qid: ["z1", "z2", "z3", "z4", "z5"] for qid in gold}
    configs = [
        FusionConfig((("good", 1.0),), 0),
        FusionConfig((("bad", 1.0),), 0),
    ]

    predictions, fold_rows = nested_oof_fusion(
        gold=gold,
        folds=folds,
        sources={"good": good, "bad": bad},
        configs=configs,
    )

    assert metrics(sorted(gold), gold, predictions)["recall"] == 1.0
    assert all(row["selection_query_count"] == 4 for row in fold_rows)
    assert all(row["selected_config"]["weights"] == {"good": 1.0} for row in fold_rows)
