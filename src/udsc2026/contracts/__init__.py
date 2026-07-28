"""Stable IO contracts between RAG lifecycle stages."""

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.contracts.chunk import LegalChunk

__all__ = ["LegalChunk", "RetrievalHit"]
