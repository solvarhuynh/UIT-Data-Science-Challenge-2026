"""Validation for JSONL-ready parent-child chunking results."""

import json
import unicodedata
from collections import Counter
from typing import Iterable, List

from udsc2026.ingestion.chunking.chunker import token_len
from udsc2026.ingestion.chunking.models import ChunkingResult, ValidationReport


_REQUIRED_METADATA = ("law_name", "article", "source")


def validate_chunking_results(
    results: Iterable[ChunkingResult], chunk_size: int = 512
) -> ValidationReport:
    """Return auditable counts; validation never silently drops a bad chunk."""
    result_list = list(results)
    chunks = [chunk for result in result_list for chunk in result.chunks]
    parent_by_id = {
        parent.parent_id
        for result in result_list
        for parent in result.parents
    }
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_ids = sorted(
        chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1
    )
    empty_ids = [chunk.chunk_id for chunk in chunks if not chunk.text.strip()]
    oversized_ids = [
        chunk.chunk_id for chunk in chunks if token_len(chunk.text) > chunk_size
    ]
    missing_ids = [
        chunk.chunk_id
        for chunk in chunks
        if any(chunk.metadata.get(key) in (None, "") for key in _REQUIRED_METADATA)
    ]
    unicode_ids = [
        chunk.chunk_id
        for chunk in chunks
        if unicodedata.normalize("NFC", chunk.text) != chunk.text
    ]
    orphan_ids = [
        chunk.chunk_id
        for chunk in chunks
        if not chunk.parent_id or chunk.parent_id not in parent_by_id
    ]
    invalid_json_ids = [
        chunk.chunk_id for chunk in chunks if not _is_json_serializable(chunk)
    ]
    review_documents = [
        result.doc_id for result in result_list if result.requires_manual_review
    ]
    return ValidationReport(
        document_count=len(result_list),
        article_count=sum(result.article_count for result in result_list),
        chunk_count=len(chunks),
        parent_count=sum(len(result.parents) for result in result_list),
        manual_review_document_count=len(review_documents),
        empty_chunk_count=len(empty_ids),
        oversized_chunk_count=len(oversized_ids),
        missing_metadata_count=len(missing_ids),
        unicode_error_count=len(unicode_ids),
        duplicate_chunk_id_count=len(duplicate_ids),
        orphan_chunk_count=len(orphan_ids),
        invalid_json_record_count=len(invalid_json_ids),
        empty_chunk_ids=empty_ids,
        oversized_chunk_ids=oversized_ids,
        missing_metadata_chunk_ids=missing_ids,
        unicode_error_chunk_ids=unicode_ids,
        duplicate_chunk_ids=duplicate_ids,
        orphan_chunk_ids=orphan_ids,
        invalid_json_chunk_ids=invalid_json_ids,
        manual_review_documents=review_documents,
    )


def _is_json_serializable(record: object) -> bool:
    """Check the exact Pydantic payload shape written as one JSONL line."""
    try:
        json.loads(json.dumps(record.model_dump(), ensure_ascii=False))
    except (TypeError, ValueError):
        return False
    return True
