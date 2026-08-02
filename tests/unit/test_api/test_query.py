"""Tests for TV1's query route and end-to-end orchestration boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from udsc2026.api.app import create_app
from udsc2026.api.cache import InMemoryCache
from udsc2026.api.orchestrator import RAGOrchestrator
from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import MockLLMClient
from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.qa.qa_engine import QAEngine


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
        self.prompt_versions: list[str] = []
        self.rag_templates: list[str] = []

    async def generate_answer(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        rag_template: str = "default_rag_v1",
        trace_id: str | None = None,
    ) -> QAResponse:
        """Return a response tied to the contexts received from the pipeline."""
        del question
        self.calls += 1
        self.prompt_versions.append(prompt_version)
        self.rag_templates.append(rag_template)
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
    assert first.json()["rag_template"] == "default_rag_v1"
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
    assert second.json()["rag_template"] == "default_rag_v1"
    assert retriever.calls == 1
    assert qa_engine.calls == 1
    assert qa_engine.prompt_versions == ["legal_qa_v1"]
    assert qa_engine.rag_templates == ["default_rag_v1"]


@pytest.mark.asyncio
async def test_query_propagates_rag_template_and_separates_cache_entries() -> None:
    """Different RAG templates must reach QA and never share cached answers."""
    app = create_app()
    retriever = _Retriever()
    qa_engine = _QAEngine()
    app.state.orchestrator = RAGOrchestrator(
        retriever=retriever,
        qa_engine=qa_engine,
        cache=InMemoryCache(),
    )
    base_payload = {
        "question": "Quyền của công dân là gì?",
        "top_k": 5,
        "top_n": 1,
    }

    v1_first = await _post(
        app,
        {**base_payload, "rag_template": "default_rag_v1"},
    )
    v2_first = await _post(
        app,
        {**base_payload, "rag_template": "default_rag_v2"},
    )
    v1_cached = await _post(
        app,
        {**base_payload, "rag_template": "default_rag_v1"},
    )
    v2_cached = await _post(
        app,
        {**base_payload, "rag_template": "default_rag_v2"},
    )

    assert [
        v1_first.status_code,
        v2_first.status_code,
        v1_cached.status_code,
        v2_cached.status_code,
    ] == [200, 200, 200, 200]
    assert v1_first.json()["cache_hit"] is False
    assert v2_first.json()["cache_hit"] is False
    assert v1_cached.json()["cache_hit"] is True
    assert v2_cached.json()["cache_hit"] is True
    assert v1_first.json()["rag_template"] == "default_rag_v1"
    assert v2_first.json()["rag_template"] == "default_rag_v2"
    assert v1_cached.json()["rag_template"] == "default_rag_v1"
    assert v2_cached.json()["rag_template"] == "default_rag_v2"
    assert retriever.calls == 2
    assert qa_engine.calls == 2
    assert qa_engine.rag_templates == ["default_rag_v1", "default_rag_v2"]


@pytest.mark.asyncio
async def test_query_separates_cache_entries_by_prompt_version() -> None:
    """Different system prompts must never share a cached answer."""
    app = create_app()
    retriever = _Retriever()
    qa_engine = _QAEngine()
    app.state.orchestrator = RAGOrchestrator(
        retriever=retriever,
        qa_engine=qa_engine,
        cache=InMemoryCache(),
    )
    base_payload = {
        "question": "Quyền của công dân là gì?",
        "top_k": 5,
        "top_n": 1,
    }

    v1_first = await _post(
        app,
        {**base_payload, "prompt_version": "legal_qa_v1"},
    )
    v2_first = await _post(
        app,
        {**base_payload, "prompt_version": "legal_qa_v2"},
    )
    v1_cached = await _post(
        app,
        {**base_payload, "prompt_version": "legal_qa_v1"},
    )
    v2_cached = await _post(
        app,
        {**base_payload, "prompt_version": "legal_qa_v2"},
    )

    assert [
        v1_first.status_code,
        v2_first.status_code,
        v1_cached.status_code,
        v2_cached.status_code,
    ] == [200, 200, 200, 200]
    assert v1_first.json()["cache_hit"] is False
    assert v2_first.json()["cache_hit"] is False
    assert v1_cached.json()["cache_hit"] is True
    assert v2_cached.json()["cache_hit"] is True
    assert retriever.calls == 2
    assert qa_engine.calls == 2
    assert qa_engine.prompt_versions == ["legal_qa_v1", "legal_qa_v2"]


@pytest.mark.asyncio
async def test_query_runs_real_qa_engine_with_packaged_v2_templates() -> None:
    """TV1 and TV3 integrate with the packaged v2 system and RAG templates."""
    prompts_root = Path(__file__).resolve().parents[3] / "prompts"
    qa_engine = QAEngine(
        llm_client=MockLLMClient(
            fixed_response=("Công dân có quyền theo quy định. [Luật mẫu, Điều 1]")
        ),  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(prompts_root=prompts_root),
    )
    app = create_app()
    app.state.orchestrator = RAGOrchestrator(
        retriever=_Retriever(),
        qa_engine=qa_engine,
        cache=InMemoryCache(),
    )

    response = await _post(
        app,
        {
            "question": "Quyền của công dân là gì?",
            "top_k": 5,
            "top_n": 1,
            "prompt_version": "legal_qa_v2",
            "rag_template": "default_rag_v2",
        },
    )

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "legal_qa_v2"
    assert response.json()["rag_template"] == "default_rag_v2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rag_template",
    ["../../secret", "default_rag_v3", "", 123],
)
async def test_query_rejects_unknown_or_invalid_rag_template(
    rag_template: object,
) -> None:
    """Only the two packaged RAG templates may cross the API boundary."""
    response = await _post(
        create_app(),
        {"question": "Câu hỏi hợp lệ", "rag_template": rag_template},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_version",
    ["../../secret", "legal_qa_v3", "", 123],
)
async def test_query_rejects_unknown_or_invalid_prompt_version(
    prompt_version: object,
) -> None:
    """Only the two packaged system prompts may cross the API boundary."""
    response = await _post(
        create_app(),
        {"question": "Câu hỏi hợp lệ", "prompt_version": prompt_version},
    )

    assert response.status_code == 422


def test_query_openapi_publishes_supported_prompt_profiles() -> None:
    """OpenAPI must expose stable defaults and every packaged prompt profile."""
    schemas = create_app().openapi()["components"]["schemas"]
    request_schema = schemas["QueryRequest"]
    response_schema = schemas["QueryResponse"]

    assert request_schema["required"] == ["question"]
    assert request_schema["properties"]["prompt_version"] == {
        "default": "legal_qa_v1",
        "enum": ["legal_qa_v1", "legal_qa_v2"],
        "title": "Prompt Version",
        "type": "string",
    }
    assert request_schema["properties"]["rag_template"] == {
        "default": "default_rag_v1",
        "enum": ["default_rag_v1", "default_rag_v2"],
        "title": "Rag Template",
        "type": "string",
    }
    assert response_schema["properties"]["rag_template"]["enum"] == [
        "default_rag_v1",
        "default_rag_v2",
    ]


@pytest.mark.asyncio
async def test_query_rejects_invalid_rerank_limit() -> None:
    """Request validation must fail before constructing model dependencies."""
    response = await _post(
        create_app(),
        {"question": "Câu hỏi hợp lệ", "top_k": 2, "top_n": 3},
    )

    assert response.status_code == 422
