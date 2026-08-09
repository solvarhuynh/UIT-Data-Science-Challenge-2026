"""End-to-end RAG pipeline orchestration independent from FastAPI transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Protocol

from udsc2026.api.cache import CacheClient
from udsc2026.contracts.api import LatencyBreakdown, QueryRequest, QueryResponse
from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit

LOGGER = logging.getLogger(__name__)


class Retriever(Protocol):
    """Search interface exposed by the retrieval pipeline."""

    def search(
        self, query: str, top_k: int, filters: dict[str, str | int | list[str]] | None
    ) -> list[RetrievalHit]:
        """Return retrieval candidates for the query."""


class Reranker(Protocol):
    """Reranking interface exposed by TV5's cross-encoder module."""

    def rerank(
        self, query: str, candidates: list[RetrievalHit], top_n: int
    ) -> list[RetrievalHit]:
        """Return the best reranked candidates."""


class QAEngine(Protocol):
    """Async answer-generation interface exposed by TV3."""

    async def generate_answer(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        rag_template: str = "default_rag_v1",
        trace_id: str | None = None,
    ) -> QAResponse:
        """Generate a cited answer from supplied contexts."""


class RAGOrchestrator:
    """Coordinate cache, retrieval, reranking and QA with measurable stages."""

    def __init__(
        self,
        retriever: Retriever,
        qa_engine: QAEngine,
        cache: CacheClient,
        *,
        reranker: Reranker | None = None,
        cache_ttl_seconds: int = 300,
        retriever_version: str = "hybrid_v1",
        reranker_version: str = "disabled",
        parent_context_version: str = "disabled",
    ) -> None:
        """Initialize pipeline dependencies without coupling to FastAPI."""
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        self._retriever = retriever
        self._qa_engine = qa_engine
        self._cache = cache
        self._reranker = reranker
        self._cache_ttl_seconds = cache_ttl_seconds
        self._retriever_version = retriever_version
        self._reranker_version = reranker_version
        self._parent_context_version = parent_context_version

    async def answer(self, request: QueryRequest) -> QueryResponse:
        """Execute the full pipeline and return a stable query response."""
        total_started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        cache_started = time.perf_counter()
        key = self._cache_key(request)
        cached = await self._cache.get(key)
        cache_ms = self._elapsed_ms(cache_started)
        if cached is not None:
            response = QueryResponse.model_validate(cached)
            return response.model_copy(
                update={
                    "cache_hit": True,
                    "trace_id": trace_id,
                    "latency_ms": LatencyBreakdown(
                        cache=cache_ms,
                        retrieval=0.0,
                        rerank=0.0,
                        generation=0.0,
                        total=self._elapsed_ms(total_started),
                    ),
                }
            )

        retrieval_started = time.perf_counter()
        hits = await asyncio.to_thread(
            self._retriever.search, request.question, request.top_k, request.filters
        )
        retrieval_ms = self._elapsed_ms(retrieval_started)

        rerank_started = time.perf_counter()
        if self._reranker is not None:
            hits = await asyncio.to_thread(
                self._reranker.rerank, request.question, hits, request.top_n
            )
        else:
            hits = hits[: request.top_n]
        rerank_ms = self._elapsed_ms(rerank_started)

        generation_started = time.perf_counter()
        qa_response = await self._qa_engine.generate_answer(
            request.question,
            hits,
            prompt_version=request.prompt_version,
            rag_template=request.rag_template,
            trace_id=trace_id,
        )
        generation_ms = self._elapsed_ms(generation_started)
        response = QueryResponse(
            answer=qa_response.answer,
            citations=qa_response.citations,
            retrieval_hits=qa_response.retrieval_hits or hits,
            latency_ms=LatencyBreakdown(
                cache=cache_ms,
                retrieval=retrieval_ms,
                rerank=rerank_ms,
                generation=generation_ms,
                total=self._elapsed_ms(total_started),
            ),
            cache_hit=False,
            prompt_version=qa_response.used_prompt_version,
            rag_template=request.rag_template,
            warnings=qa_response.warnings,
            trace_id=qa_response.trace_id or trace_id,
        )
        await self._cache.set(
            key,
            response.model_dump(mode="json"),
            self._cache_ttl_seconds,
        )
        self._log_quality_signals(response)
        return response

    def _cache_key(self, request: QueryRequest) -> str:
        payload: Mapping[str, object] = {
            "question": request.question,
            "filters": request.filters,
            "top_k": request.top_k,
            "top_n": request.top_n,
            "prompt_version": request.prompt_version,
            "rag_template": request.rag_template,
            "retriever_version": self._retriever_version,
            "reranker_version": self._reranker_version,
            "parent_context_version": self._parent_context_version,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"udsc2026:query:{digest}"

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """Return non-negative elapsed milliseconds rounded for transport."""
        return round(max(0.0, (time.perf_counter() - started) * 1000), 3)

    @staticmethod
    def _log_quality_signals(response: QueryResponse) -> None:
        """Emit structured quality signals consumed by log aggregation/benchmarking."""
        LOGGER.info(
            (
                "rag_query trace_id=%s prompt_version=%s rag_template=%s "
                "cache_hit=%s hits=%d citations=%d warnings=%d total_ms=%.3f"
            ),
            response.trace_id,
            response.prompt_version,
            response.rag_template,
            response.cache_hit,
            len(response.retrieval_hits),
            len(response.citations),
            len(response.warnings),
            response.latency_ms.total,
        )
        if not response.retrieval_hits or not response.citations or response.warnings:
            LOGGER.warning(
                (
                    "rag_quality_signal trace_id=%s empty_hits=%s "
                    "missing_citations=%s warnings=%s"
                ),
                response.trace_id,
                not response.retrieval_hits,
                not response.citations,
                response.warnings,
            )
