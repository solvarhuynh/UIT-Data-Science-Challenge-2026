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
    structured_chunk_count: int = 0
    fallback_chunk_count: int = 0
    supplemental_chunk_count: int = 0
    empty_source_placeholder_chunk_count: int = 0
    source_nonempty_line_count: int = 0
    assigned_source_line_count: int = 0
    supplemented_source_line_count: int = 0
    unassigned_source_line_count: int = 0
    source_character_count: int = 0
    assigned_source_character_count: int = 0
    supplemented_source_character_count: int = 0
    repaired_split_article_count: int = 0
    suspected_split_article_count: int = 0
    source_content_empty: bool = False
    structure_status: str = "structured"
    source_family: str = "unknown"
    structure_warnings: List[str] = Field(default_factory=list)
    source_unique_token_count: int = 0
    missing_source_token_count: int = 0
    source_intraline_bigram_count: int = 0
    missing_source_bigram_count: int = 0
    missing_source_token_examples: List[str] = Field(default_factory=list)
    missing_source_bigram_examples: List[str] = Field(default_factory=list)


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
    structured_document_count: int = 0
    fallback_document_count: int = 0
    empty_source_document_count: int = 0
    empty_document_output_count: int = 0
    structured_chunk_count: int = 0
    fallback_chunk_count: int = 0
    supplemental_chunk_count: int = 0
    empty_source_placeholder_chunk_count: int = 0
    source_nonempty_line_count: int = 0
    assigned_source_line_count: int = 0
    supplemented_source_line_count: int = 0
    unassigned_source_line_count: int = 0
    source_character_count: int = 0
    assigned_source_character_count: int = 0
    supplemented_source_character_count: int = 0
    source_assignment_coverage_ratio: float = 1.0
    repaired_split_article_count: int = 0
    suspected_split_article_count: int = 0
    empty_document_output_ids: List[str] = Field(default_factory=list)
    empty_source_document_ids: List[str] = Field(default_factory=list)
    suspected_split_article_documents: List[str] = Field(default_factory=list)
    quality_gate_passed: bool = False
    quality_gate_failures: List[str] = Field(default_factory=list)
    partial_structure_document_count: int = 0
    partial_structure_documents: List[str] = Field(default_factory=list)
    structure_warning_document_count: int = 0
    structure_warning_documents: List[str] = Field(default_factory=list)
    source_unique_token_count: int = 0
    missing_source_token_count: int = 0
    source_intraline_bigram_count: int = 0
    missing_source_bigram_count: int = 0
    missing_source_token_documents: List[str] = Field(default_factory=list)
    missing_source_bigram_documents: List[str] = Field(default_factory=list)
