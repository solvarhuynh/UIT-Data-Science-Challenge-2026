"""Boundary validation tests for shared retrieval and QA contracts."""

import math

import pytest
from pydantic import ValidationError

from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit


@pytest.mark.parametrize("field", ["chunk_id", "doc_id", "text"])
def test_retrieval_hit_rejects_blank_identity_fields(field: str) -> None:
    values = {"chunk_id": "chunk", "doc_id": "doc", "text": "legal text"}
    values[field] = " "

    with pytest.raises(ValidationError):
        RetrievalHit(**values)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_retrieval_hit_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalHit(
            chunk_id="chunk",
            doc_id="doc",
            text="legal text",
            rerank_score=score,
        )


def test_retrieval_hit_rejects_unknown_nested_fields() -> None:
    with pytest.raises(ValidationError, match="point"):
        RetrievalHit(
            chunk_id="chunk",
            doc_id="doc",
            text="legal text",
            point="Điểm a",
        )


def test_qa_response_rejects_non_finite_confidence() -> None:
    with pytest.raises(ValidationError):
        QAResponse(
            answer="Câu trả lời",
            used_prompt_version="legal_qa_v1",
            confidence=math.nan,
        )
