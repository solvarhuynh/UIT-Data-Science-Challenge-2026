"""Result and validation contracts for parent-child legal chunking."""

from typing import List

from pydantic import BaseModel, Field

from udsc2026.contracts import LegalChunk, LegalParent


class ChunkingResult(BaseModel):
    """All retrieval children and article parents produced from one document."""

    doc_id: str
    article_count: int = 0
    chunks: List[LegalChunk] = Field(default_factory=list)
    parents: List[LegalParent] = Field(default_factory=list)
    requires_manual_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Audit summary emitted before chunks are handed to retrieval."""

    document_count: int
    article_count: int
    chunk_count: int
    parent_count: int
    manual_review_document_count: int
    empty_chunk_count: int
    oversized_chunk_count: int
    missing_metadata_count: int
    unicode_error_count: int
    duplicate_chunk_id_count: int
    orphan_chunk_count: int
    invalid_json_record_count: int
    empty_chunk_ids: List[str] = Field(default_factory=list)
    oversized_chunk_ids: List[str] = Field(default_factory=list)
    missing_metadata_chunk_ids: List[str] = Field(default_factory=list)
    unicode_error_chunk_ids: List[str] = Field(default_factory=list)
    duplicate_chunk_ids: List[str] = Field(default_factory=list)
    orphan_chunk_ids: List[str] = Field(default_factory=list)
    invalid_json_chunk_ids: List[str] = Field(default_factory=list)
    manual_review_documents: List[str] = Field(default_factory=list)
