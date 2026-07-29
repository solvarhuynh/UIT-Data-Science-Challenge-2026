"""Legal-unit parent-child chunking and JSONL delivery utilities."""

from udsc2026.ingestion.chunking.chunker import (
    chunk_clean_document,
    chunk_legal_article,
    chunk_legal_structure,
    split_by_sentence_with_overlap,
    token_len,
)
from udsc2026.ingestion.chunking.models import ChunkingResult, ValidationReport
from udsc2026.ingestion.chunking.validation import validate_chunking_results
from udsc2026.ingestion.chunking.writer import write_chunking_outputs

__all__ = [
    "ChunkingResult",
    "ValidationReport",
    "chunk_clean_document",
    "chunk_legal_article",
    "chunk_legal_structure",
    "split_by_sentence_with_overlap",
    "token_len",
    "validate_chunking_results",
    "write_chunking_outputs",
]
