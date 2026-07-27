"""Shared retrieval contracts used across Dense, Hybrid, QA, and Rerank."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """Single source of truth for retrieval results passed through the RAG pipeline."""

    chunk_id: str
    doc_id: str
    text: str
    score: Optional[float] = None
    source: Optional[str] = None
    law_name: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    rank: Optional[int] = None
