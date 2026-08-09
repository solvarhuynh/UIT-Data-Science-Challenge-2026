"""Validation for JSONL-ready parent-child chunking results."""

import json
import unicodedata
from collections import Counter
from typing import Iterable, List

from pydantic import BaseModel

from udsc2026.contracts import LegalChunk
from udsc2026.ingestion.chunking.chunker import token_len
from udsc2026.ingestion.chunking.models import ChunkingResult, ValidationReport

_BASE_REQUIRED_METADATA = ("law_name", "source")


def validate_chunking_results(
    results: Iterable[ChunkingResult], chunk_size: int = 192
) -> ValidationReport:
    """Return auditable counts and an explicit corpus quality gate."""

    result_list = list(results)
    chunks = [chunk for result in result_list for chunk in result.chunks]
    parent_by_id = {
        parent.parent_id for result in result_list for parent in result.parents
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
        chunk.chunk_id for chunk in chunks if _has_missing_required_metadata(chunk)
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
    empty_output_ids = [result.doc_id for result in result_list if not result.chunks]
    empty_source_ids = [
        result.doc_id for result in result_list if result.source_content_empty
    ]
    suspected_documents = [
        result.doc_id
        for result in result_list
        if result.suspected_split_article_count > 0
    ]
    partial_documents = [
        result.doc_id for result in result_list if result.structure_status == "partial"
    ]
    warning_documents = [
        result.doc_id for result in result_list if result.structure_warnings
    ]
    missing_token_documents = [
        result.doc_id for result in result_list if result.missing_source_token_count
    ]
    missing_bigram_documents = [
        result.doc_id for result in result_list if result.missing_source_bigram_count
    ]
    source_line_count = sum(result.source_nonempty_line_count for result in result_list)
    assigned_line_count = sum(
        result.assigned_source_line_count for result in result_list
    )
    quality_failures = build_quality_failures(
        empty_ids=empty_ids,
        oversized_ids=oversized_ids,
        missing_ids=missing_ids,
        unicode_ids=unicode_ids,
        duplicate_ids=duplicate_ids,
        orphan_ids=orphan_ids,
        invalid_json_ids=invalid_json_ids,
        empty_output_ids=empty_output_ids,
        empty_source_ids=empty_source_ids,
        unassigned_line_count=sum(
            result.unassigned_source_line_count for result in result_list
        ),
        missing_source_token_count=sum(
            result.missing_source_token_count for result in result_list
        ),
        missing_source_bigram_count=sum(
            result.missing_source_bigram_count for result in result_list
        ),
    )
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
        structured_document_count=sum(
            result.article_count > 0 for result in result_list
        ),
        fallback_document_count=sum(
            result.fallback_chunk_count > 0 for result in result_list
        ),
        empty_source_document_count=len(empty_source_ids),
        empty_document_output_count=len(empty_output_ids),
        structured_chunk_count=sum(
            result.structured_chunk_count for result in result_list
        ),
        fallback_chunk_count=sum(result.fallback_chunk_count for result in result_list),
        supplemental_chunk_count=sum(
            result.supplemental_chunk_count for result in result_list
        ),
        empty_source_placeholder_chunk_count=sum(
            result.empty_source_placeholder_chunk_count for result in result_list
        ),
        source_nonempty_line_count=source_line_count,
        assigned_source_line_count=assigned_line_count,
        supplemented_source_line_count=sum(
            result.supplemented_source_line_count for result in result_list
        ),
        unassigned_source_line_count=sum(
            result.unassigned_source_line_count for result in result_list
        ),
        source_character_count=sum(
            result.source_character_count for result in result_list
        ),
        assigned_source_character_count=sum(
            result.assigned_source_character_count for result in result_list
        ),
        supplemented_source_character_count=sum(
            result.supplemented_source_character_count for result in result_list
        ),
        source_assignment_coverage_ratio=(
            assigned_line_count / source_line_count if source_line_count else 1.0
        ),
        repaired_split_article_count=sum(
            result.repaired_split_article_count for result in result_list
        ),
        suspected_split_article_count=sum(
            result.suspected_split_article_count for result in result_list
        ),
        empty_document_output_ids=empty_output_ids,
        empty_source_document_ids=empty_source_ids,
        suspected_split_article_documents=suspected_documents,
        quality_gate_passed=not quality_failures,
        quality_gate_failures=quality_failures,
        partial_structure_document_count=len(partial_documents),
        partial_structure_documents=partial_documents,
        structure_warning_document_count=len(warning_documents),
        structure_warning_documents=warning_documents,
        source_unique_token_count=sum(
            result.source_unique_token_count for result in result_list
        ),
        missing_source_token_count=sum(
            result.missing_source_token_count for result in result_list
        ),
        source_intraline_bigram_count=sum(
            result.source_intraline_bigram_count for result in result_list
        ),
        missing_source_bigram_count=sum(
            result.missing_source_bigram_count for result in result_list
        ),
        missing_source_token_documents=missing_token_documents,
        missing_source_bigram_documents=missing_bigram_documents,
    )


def _has_missing_required_metadata(chunk: LegalChunk) -> bool:
    if any(chunk.metadata.get(key) in (None, "") for key in _BASE_REQUIRED_METADATA):
        return True
    structure_type = chunk.metadata.get("structure_type", "article")
    if structure_type == "article" and chunk.metadata.get("article") in (None, ""):
        return True
    if structure_type in {"unstructured_fallback", "document_context"}:
        return not bool(
            chunk.metadata.get("fallback_chunking")
            or chunk.metadata.get("supplemental_context")
        )
    if structure_type == "empty_source_placeholder":
        return not bool(chunk.metadata.get("source_content_empty"))
    return False


def build_quality_failures(
    *,
    empty_ids: List[str],
    oversized_ids: List[str],
    missing_ids: List[str],
    unicode_ids: List[str],
    duplicate_ids: List[str],
    orphan_ids: List[str],
    invalid_json_ids: List[str],
    empty_output_ids: List[str],
    empty_source_ids: List[str],
    unassigned_line_count: int,
    missing_source_token_count: int,
    missing_source_bigram_count: int,
) -> List[str]:
    """Return blocking corpus-quality failure labels for nonzero checks."""

    checks = (
        ("empty_chunks", empty_ids),
        ("oversized_chunks", oversized_ids),
        ("missing_required_metadata", missing_ids),
        ("non_nfc_unicode", unicode_ids),
        ("duplicate_chunk_ids", duplicate_ids),
        ("orphan_parent_links", orphan_ids),
        ("invalid_json_records", invalid_json_ids),
        ("documents_without_chunks", empty_output_ids),
        ("official_empty_source_content", empty_source_ids),
    )
    failures = [name for name, values in checks if values]
    if unassigned_line_count:
        failures.append("unassigned_source_lines")
    if missing_source_token_count:
        failures.append("missing_source_tokens_in_children")
    if missing_source_bigram_count:
        failures.append("missing_source_intraline_bigrams_in_children")
    return failures


def _is_json_serializable(record: BaseModel) -> bool:
    """Check the exact Pydantic payload shape written as one JSONL line."""

    try:
        json.loads(json.dumps(record.model_dump(), ensure_ascii=False))
    except (TypeError, ValueError):
        return False
    return True
