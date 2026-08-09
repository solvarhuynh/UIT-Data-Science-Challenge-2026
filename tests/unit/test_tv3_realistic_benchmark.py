"""Regression tests for the safe TV3/TV5 benchmark bridge."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.tv3.benchmark_realistic_qa import (
    _rerank_candidates,
    _resolve_local_model,
    _truncate_contexts,
)

from udsc2026.contracts.retrieval import RetrievalHit


def _hits(count: int, *, text_size: int = 32) -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id=f"parent-{index}",
            parent_id=f"parent-{index}",
            doc_id=f"doc-{index}",
            text=f"candidate-{index}-" + ("x" * text_size),
            score=1.0 / (index + 1),
            metadata={"source_rank": index + 1},
        )
        for index in range(count)
    ]


class _FailingReranker:
    def rerank(self, *_args: object, **_kwargs: object) -> list[RetrievalHit]:
        raise OSError("model unavailable")


class _RecordingReranker:
    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    def rerank(
        self,
        _question: str,
        *,
        candidates: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        self.seen_lengths = [len(hit.text) for hit in candidates]
        return [
            hit.model_copy(update={"rank": rank, "rerank_score": float(rank)})
            for rank, hit in enumerate(reversed(candidates[:top_n]), start=1)
        ]


def test_reranker_failure_is_strict_by_default() -> None:
    with pytest.raises(RuntimeError, match="TV5 reranking failed"):
        _rerank_candidates(
            _FailingReranker(),
            "Câu hỏi pháp luật",
            _hits(6),
            top_k=3,
            allow_fallback=False,
        )


def test_explicit_reranker_fallback_never_leaks_candidate_pool() -> None:
    candidates = _hits(20)

    output, applied, error = _rerank_candidates(
        _FailingReranker(),
        "Câu hỏi pháp luật",
        candidates,
        top_k=3,
        allow_fallback=True,
    )

    assert [hit.chunk_id for hit in output] == [
        "parent-0",
        "parent-1",
        "parent-2",
    ]
    assert applied is False
    assert error == "OSError: model unavailable"


def test_reranking_happens_before_prompt_truncation() -> None:
    reranker = _RecordingReranker()
    candidates = _hits(4, text_size=200)

    reranked, applied, error = _rerank_candidates(
        reranker,
        "Câu hỏi pháp luật",
        candidates,
        top_k=2,
        allow_fallback=False,
    )
    prompt_contexts = _truncate_contexts(reranked, max_context_len=40)

    assert reranker.seen_lengths == [len(hit.text) for hit in candidates]
    assert all(len(hit.text) <= 43 for hit in prompt_contexts)
    assert applied is True
    assert error is None


def test_context_truncation_preserves_parent_provenance() -> None:
    original = _hits(1, text_size=200)[0]

    output = _truncate_contexts([original], max_context_len=40)[0]

    assert output.parent_id == original.parent_id
    assert output.metadata == original.metadata
    assert output.text.endswith("...")
    assert original.text != output.text


def test_local_model_only_overrides_the_default_identifier(tmp_path: Path) -> None:
    local_model = tmp_path / "model"
    local_model.mkdir()

    assert _resolve_local_model("org/default", "org/default", local_model) == str(
        local_model
    )
    assert (
        _resolve_local_model("custom/model", "org/default", local_model)
        == "custom/model"
    )
