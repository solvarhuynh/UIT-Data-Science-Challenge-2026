"""Tests for TV1's query route and end-to-end orchestration boundary."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from udsc2026.api.app import create_app
from udsc2026.api.cache import InMemoryCache
from udsc2026.api.orchestrator import RAGOrchestrator
from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit


class _Retriever:
    """Deterministic blocking retriever used to verify threadpool orchestration."""

    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None,
    ) -> list[RetrievalHit]:
        """Return one legal chunk and record the invocation."""
        del query, top_k, filters
        self.calls += 1
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="law-1",
                text="Điều 1 quy định quyền của công dân.",
                law_name="Luật mẫu",
                article="1",
                score=0.9,
            )
        ]


class _QAEngine:
    """Deterministic async QA engine used without model weights."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_answer(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        trace_id: str | None = None,
    ) -> QAResponse:
        """Return a response tied to the contexts received from the pipeline."""
        del question
        self.calls += 1
        return QAResponse(
            answer="Câu trả lời đã được kiểm chứng.",
            used_prompt_version=prompt_version,
            retrieval_hits=contexts,
            trace_id=trace_id,
        )


async def _post(app: Any, body: dict[str, object]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v1/query", json=body)


@pytest.mark.asyncio
async def test_query_runs_pipeline_then_returns_cached_result() -> None:
    """A repeated equivalent query must avoid retrieval and QA work."""
    app = create_app()
    retriever = _Retriever()
    qa_engine = _QAEngine()
    app.state.orchestrator = RAGOrchestrator(
        retriever=retriever,
        qa_engine=qa_engine,
        cache=InMemoryCache(),
    )
    payload = {"question": "Quyền của công dân là gì?", "top_k": 5, "top_n": 1}

    first = await _post(app, payload)
    second = await _post(app, payload)

    assert first.status_code == 200
    assert first.json()["cache_hit"] is False
    assert first.json()["answer"] == "Câu trả lời đã được kiểm chứng."
    assert first.json()["retrieval_hits"][0]["chunk_id"] == "chunk-1"
    assert set(first.json()["latency_ms"]) == {
        "cache",
        "retrieval",
        "rerank",
        "generation",
        "total",
    }
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert retriever.calls == 1
    assert qa_engine.calls == 1


@pytest.mark.asyncio
async def test_query_rejects_invalid_rerank_limit() -> None:
    """Request validation must fail before constructing model dependencies."""
    response = await _post(
        create_app(),
        {"question": "Câu hỏi hợp lệ", "top_k": 2, "top_n": 3},
    )

    assert response.status_code == 422
