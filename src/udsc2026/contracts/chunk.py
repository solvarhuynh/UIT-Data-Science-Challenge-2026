"""Contracts for law-document chunks exchanged between ingestion and retrieval."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    """Input contract from TV4 ingestion to TV2 retrieval indexing."""

    chunk_id: str
    parent_id: Optional[str] = None
    doc_id: str
    text: str
    parent_text: Optional[str] = None
    law_name: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    effective_date: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalParent(BaseModel):
    """Full article context persisted separately from search-oriented children."""

    parent_id: str
    doc_id: str
    text: str
    law_name: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
