"""Stable IO contracts between RAG lifecycle stages."""

from udsc2026.contracts.chunk import LegalChunk, LegalParent
from udsc2026.contracts.qa import Citation, QAResponse
from udsc2026.contracts.retrieval import RetrievalHit

__all__ = ["Citation", "LegalChunk", "LegalParent", "QAResponse", "RetrievalHit"]