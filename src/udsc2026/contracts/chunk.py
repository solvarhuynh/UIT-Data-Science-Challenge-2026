"""Contract for law-document chunks exchanged between ingestion and retrieval."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    """LegalChunk là input contract từ TV4 (ingestion) cho TV2 (retrieval indexing); nếu TV4 đổi schema JSONL, cập nhật class này."""

    chunk_id: str
    doc_id: str
    text: str
    law_name: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    effective_date: Optional[str] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
