"""Regression tests for before/after rerank comparison invariants."""

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.evaluator import validate_rerank_candidate_pools
from udsc2026.evaluation.models import PredictionSample


def _baseline() -> list[PredictionSample]:
    return [
        PredictionSample(
            question_id="q1",
            hits=[
                RetrievalHit(
                    chunk_id="a",
                    doc_id="doc",
                    text="A",
                    hybrid_score=0.8,
                ),
                RetrievalHit(
                    chunk_id="b",
                    doc_id="doc",
                    text="B",
                    hybrid_score=0.7,
                ),
            ],
        )
    ]


def _after(
    first_score: float = 0.9,
    second_score: float = 0.8,
) -> list[PredictionSample]:
    return [
        PredictionSample(
            question_id="q1",
            hits=[
                RetrievalHit(
                    chunk_id="b",
                    doc_id="doc",
                    text="B",
                    hybrid_score=0.7,
                    rerank_score=first_score,
                    final_score=first_score,
                    rank=1,
                ),
                RetrievalHit(
                    chunk_id="a",
                    doc_id="doc",
                    text="A",
                    hybrid_score=0.8,
                    rerank_score=second_score,
                    final_score=second_score,
                    rank=2,
                ),
            ],
        )
    ]


def test_accepts_a_complete_sorted_rerank_output() -> None:
    validate_rerank_candidate_pools(_baseline(), _after())


def test_rejects_missing_rerank_scores() -> None:
    after = _after()
    after[0].hits[0] = (
        after[0].hits[0].model_copy(update={"rerank_score": None, "final_score": None})
    )

    with pytest.raises(ValueError, match="must provide"):
        validate_rerank_candidate_pools(_baseline(), after)


def test_rejects_rank_that_disagrees_with_list_order() -> None:
    after = _after()
    after[0].hits[1] = after[0].hits[1].model_copy(update={"rank": 3})

    with pytest.raises(ValueError, match="expected 2"):
        validate_rerank_candidate_pools(_baseline(), after)


def test_rejects_inconsistent_final_score() -> None:
    after = _after()
    after[0].hits[0] = after[0].hits[0].model_copy(update={"final_score": 0.1})

    with pytest.raises(ValueError, match="inconsistent final_score"):
        validate_rerank_candidate_pools(_baseline(), after)


def test_rejects_unsorted_rerank_scores() -> None:
    with pytest.raises(ValueError, match="not sorted"):
        validate_rerank_candidate_pools(_baseline(), _after(0.5, 0.8))


def test_rejects_empty_rerank_output_for_non_empty_baseline() -> None:
    after = [PredictionSample(question_id="q1", hits=[])]

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_rerank_candidate_pools(_baseline(), after)


def test_accepts_empty_rerank_output_for_empty_baseline() -> None:
    empty = [PredictionSample(question_id="q1", hits=[])]
    validate_rerank_candidate_pools(empty, empty)


def test_equal_scores_must_preserve_baseline_order() -> None:
    with pytest.raises(ValueError, match="preserve baseline order"):
        validate_rerank_candidate_pools(_baseline(), _after(0.8, 0.8))

    stable_after = _after(0.8, 0.8)
    stable_after[0].hits.reverse()
    stable_after[0].hits = [
        hit.model_copy(update={"rank": rank})
        for rank, hit in enumerate(stable_after[0].hits, start=1)
    ]
    validate_rerank_candidate_pools(_baseline(), stable_after)
