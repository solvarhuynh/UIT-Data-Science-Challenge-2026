"""Tests for Task 2 scorer-compatible dependency boundaries."""

from __future__ import annotations

import pytest

from udsc2026.evaluation.legal_qa_scoring import (
    fast_exact_meteor,
    official_rouge_l,
)


def test_fast_meteor_identical_text_is_nearly_one() -> None:
    score = fast_exact_meteor("mot hai ba", "mot hai ba")

    assert score == pytest.approx(1.0 - 0.5 * (1 / 3) ** 3)


def test_rouge_l_uses_organizer_ascii_tokenization() -> None:
    # rouge-score's default tokenizer strips non-ASCII characters. This test
    # pins the same unintuitive behavior used by the organizer program.
    assert official_rouge_l("Điều 1 quy định", "Điều 1 quy định") == 1.0
    assert official_rouge_l("abc def", "abc xyz") == pytest.approx(0.5)
