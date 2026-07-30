"""Shared retrieval contracts used across Dense, Hybrid, QA, and Rerank."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalHit(BaseModel):
    """Single source of truth for retrieval results passed through the RAG pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: Optional[float] = Field(default=None, allow_inf_nan=False)
    source: Optional[str] = None
    law_name: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    sparse_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    hybrid_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    rerank_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    final_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    rank: Optional[int] = Field(default=None, gt=0)
