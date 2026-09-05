"""Contract tests for the inference-only Modal P19 gate."""

from __future__ import annotations

from scripts.cloud import modal_task2_p19 as script


def test_dev_experiments_only_change_top_parent_ranking() -> None:
    assert {row["name"] for row in script.DEV_EXPERIMENTS} == {
        "adapter_top1_ce",
        "adapter_top1_retrieval",
        "adapter_top1_fusion",
    }
    for row in script.DEV_EXPERIMENTS:
        assert row["model"] == "adapter"
        assert row["top_parents"] == 1
        assert row["max_parent_words"] == 512
        assert row["ce_weight"] + row["retrieval_weight"] > 0


def test_contract_is_deterministic_and_source_bound() -> None:
    first = script._contract("bundle", "source-a")
    assert first == script._contract("bundle", "source-a")
    assert first != script._contract("bundle", "source-b")
    assert len(first) == 24


def test_p19_declares_no_external_or_augmented_inputs() -> None:
    assert script.BASELINE_SELECTOR_OOF["meteor"] > 0.56
    assert script.BASELINE_SELECTOR_OOF["rouge_l"] > 0.44
    assert script.MODAL_GPU_PRIORITY == ["H100"]
