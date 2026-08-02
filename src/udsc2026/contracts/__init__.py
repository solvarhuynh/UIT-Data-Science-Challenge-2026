"""Stable IO contracts between RAG lifecycle stages."""

from udsc2026.contracts.api import LatencyBreakdown, QueryRequest, QueryResponse
from udsc2026.contracts.chunk import LegalChunk, LegalParent
from udsc2026.contracts.health import HealthResponse, ReadinessResponse
from udsc2026.contracts.qa import Citation, QAResponse
from udsc2026.contracts.retrieval import RetrievalHit

__all__ = [
    "Citation",
    "HealthResponse",
    "LatencyBreakdown",
    "LegalChunk",
    "LegalParent",
    "QAResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadinessResponse",
    "RetrievalHit",
]
