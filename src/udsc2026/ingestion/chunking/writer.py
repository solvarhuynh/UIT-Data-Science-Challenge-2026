"""JSONL persistence for retrieval children, parent contexts, and validation."""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

from pydantic import BaseModel

from udsc2026.ingestion.chunking.chunker import chunk_clean_document
from udsc2026.ingestion.chunking.models import ChunkingResult, ValidationReport
from udsc2026.ingestion.chunking.validation import (
    build_quality_failures,
    validate_chunking_results,
)
from udsc2026.ingestion.cleaners.models import CleanDocument

DEFAULT_CHUNKS_DIR = Path("data/processed/chunks")
DEFAULT_PARENTS_DIR = Path("data/processed/parents")
DEFAULT_REPORT_PATH = Path("data/processed/metadata/validation_report.json")
DEFAULT_REVIEW_PATH = Path("data/processed/metadata/manual_review_documents.json")


def write_chunking_outputs(
    documents: Iterable[CleanDocument],
    chunks_dir: Optional[Union[str, Path]] = None,
    parents_dir: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    review_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
) -> Tuple[List[ChunkingResult], ValidationReport]:
    """Chunk documents, save one JSONL pair per document, and write a report."""
    document_list = list(documents)
    _ensure_unique_document_ids(document_list)
    results = [
        chunk_clean_document(document, chunk_size, chunk_overlap)
        for document in document_list
    ]
    chunk_target = Path(chunks_dir) if chunks_dir else DEFAULT_CHUNKS_DIR
    parent_target = Path(parents_dir) if parents_dir else DEFAULT_PARENTS_DIR
    chunk_target.mkdir(parents=True, exist_ok=True)
    parent_target.mkdir(parents=True, exist_ok=True)
    for result in results:
        _write_jsonl(chunk_target / "{0}.jsonl".format(result.doc_id), result.chunks)
        _write_jsonl(parent_target / "{0}.jsonl".format(result.doc_id), result.parents)

    report = validate_chunking_results(results, chunk_size=chunk_size)
    _write_json(
        Path(report_path) if report_path else DEFAULT_REPORT_PATH, report.model_dump()
    )
    _write_json(
        Path(review_path) if review_path else DEFAULT_REVIEW_PATH,
        report.manual_review_documents,
    )
    return results, report


def write_chunking_outputs_streaming(
    documents: Iterable[CleanDocument],
    chunks_dir: Optional[Union[str, Path]] = None,
    parents_dir: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    review_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[int, ValidationReport]:
    """Chunk and validate one document at a time to keep corpus RAM bounded.

    Unlike :func:`write_chunking_outputs`, this function does not retain every
    ``LegalChunk`` in memory until the corpus has finished.  It is intended for
    full BTC runs, while the original function remains useful to callers that
    need the returned per-document objects.
    """
    chunk_target = Path(chunks_dir) if chunks_dir else DEFAULT_CHUNKS_DIR
    parent_target = Path(parents_dir) if parents_dir else DEFAULT_PARENTS_DIR
    chunk_target.mkdir(parents=True, exist_ok=True)
    parent_target.mkdir(parents=True, exist_ok=True)

    document_count = 0
    article_count = 0
    chunk_count = 0
    parent_count = 0
    chunked_document_count = 0
    manual_review_documents: List[str] = []
    empty_chunk_ids: List[str] = []
    oversized_chunk_ids: List[str] = []
    missing_metadata_chunk_ids: List[str] = []
    unicode_error_chunk_ids: List[str] = []
    duplicate_chunk_ids: Set[str] = set()
    orphan_chunk_ids: List[str] = []
    invalid_json_chunk_ids: List[str] = []
    structured_document_count = 0
    fallback_document_count = 0
    structured_chunk_count = 0
    fallback_chunk_count = 0
    supplemental_chunk_count = 0
    empty_source_placeholder_chunk_count = 0
    source_nonempty_line_count = 0
    assigned_source_line_count = 0
    supplemented_source_line_count = 0
    unassigned_source_line_count = 0
    source_character_count = 0
    assigned_source_character_count = 0
    supplemented_source_character_count = 0
    repaired_split_article_count = 0
    suspected_split_article_count = 0
    empty_document_output_ids: List[str] = []
    empty_source_document_ids: List[str] = []
    suspected_split_article_documents: List[str] = []
    partial_structure_documents: List[str] = []
    structure_warning_documents: List[str] = []
    source_unique_token_count = 0
    missing_source_token_count = 0
    source_intraline_bigram_count = 0
    missing_source_bigram_count = 0
    missing_source_token_documents: List[str] = []
    missing_source_bigram_documents: List[str] = []
    seen_document_ids: Set[str] = set()
    seen_chunk_ids: Set[str] = set()

    for document in documents:
        if document.doc_id in seen_document_ids:
            raise ValueError("Duplicate doc_id values: {0}".format(document.doc_id))
        seen_document_ids.add(document.doc_id)

        result = chunk_clean_document(document, chunk_size, chunk_overlap)
        _write_jsonl(chunk_target / "{0}.jsonl".format(result.doc_id), result.chunks)
        _write_jsonl(parent_target / "{0}.jsonl".format(result.doc_id), result.parents)
        partial = validate_chunking_results([result], chunk_size=chunk_size)

        document_count += partial.document_count
        article_count += partial.article_count
        chunk_count += partial.chunk_count
        parent_count += partial.parent_count
        chunked_document_count += int(bool(result.chunks))
        manual_review_documents.extend(partial.manual_review_documents)
        empty_chunk_ids.extend(partial.empty_chunk_ids)
        oversized_chunk_ids.extend(partial.oversized_chunk_ids)
        missing_metadata_chunk_ids.extend(partial.missing_metadata_chunk_ids)
        unicode_error_chunk_ids.extend(partial.unicode_error_chunk_ids)
        duplicate_chunk_ids.update(partial.duplicate_chunk_ids)
        orphan_chunk_ids.extend(partial.orphan_chunk_ids)
        invalid_json_chunk_ids.extend(partial.invalid_json_chunk_ids)
        structured_document_count += partial.structured_document_count
        fallback_document_count += partial.fallback_document_count
        structured_chunk_count += partial.structured_chunk_count
        fallback_chunk_count += partial.fallback_chunk_count
        supplemental_chunk_count += partial.supplemental_chunk_count
        empty_source_placeholder_chunk_count += (
            partial.empty_source_placeholder_chunk_count
        )
        source_nonempty_line_count += partial.source_nonempty_line_count
        assigned_source_line_count += partial.assigned_source_line_count
        supplemented_source_line_count += partial.supplemented_source_line_count
        unassigned_source_line_count += partial.unassigned_source_line_count
        source_character_count += partial.source_character_count
        assigned_source_character_count += partial.assigned_source_character_count
        supplemented_source_character_count += (
            partial.supplemented_source_character_count
        )
        repaired_split_article_count += partial.repaired_split_article_count
        suspected_split_article_count += partial.suspected_split_article_count
        empty_document_output_ids.extend(partial.empty_document_output_ids)
        empty_source_document_ids.extend(partial.empty_source_document_ids)
        suspected_split_article_documents.extend(
            partial.suspected_split_article_documents
        )
        partial_structure_documents.extend(partial.partial_structure_documents)
        structure_warning_documents.extend(partial.structure_warning_documents)
        source_unique_token_count += partial.source_unique_token_count
        missing_source_token_count += partial.missing_source_token_count
        source_intraline_bigram_count += partial.source_intraline_bigram_count
        missing_source_bigram_count += partial.missing_source_bigram_count
        missing_source_token_documents.extend(partial.missing_source_token_documents)
        missing_source_bigram_documents.extend(partial.missing_source_bigram_documents)

        for chunk in result.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                duplicate_chunk_ids.add(chunk.chunk_id)
            seen_chunk_ids.add(chunk.chunk_id)

        if progress_callback is not None:
            progress_callback(document_count, document.doc_id)

    quality_failures = build_quality_failures(
        empty_ids=empty_chunk_ids,
        oversized_ids=oversized_chunk_ids,
        missing_ids=missing_metadata_chunk_ids,
        unicode_ids=unicode_error_chunk_ids,
        duplicate_ids=sorted(duplicate_chunk_ids),
        orphan_ids=orphan_chunk_ids,
        invalid_json_ids=invalid_json_chunk_ids,
        empty_output_ids=empty_document_output_ids,
        empty_source_ids=empty_source_document_ids,
        unassigned_line_count=unassigned_source_line_count,
        missing_source_token_count=missing_source_token_count,
        missing_source_bigram_count=missing_source_bigram_count,
    )
    report = ValidationReport(
        document_count=document_count,
        article_count=article_count,
        chunk_count=chunk_count,
        parent_count=parent_count,
        manual_review_document_count=len(manual_review_documents),
        empty_chunk_count=len(empty_chunk_ids),
        oversized_chunk_count=len(oversized_chunk_ids),
        missing_metadata_count=len(missing_metadata_chunk_ids),
        unicode_error_count=len(unicode_error_chunk_ids),
        duplicate_chunk_id_count=len(duplicate_chunk_ids),
        orphan_chunk_count=len(orphan_chunk_ids),
        invalid_json_record_count=len(invalid_json_chunk_ids),
        empty_chunk_ids=empty_chunk_ids,
        oversized_chunk_ids=oversized_chunk_ids,
        missing_metadata_chunk_ids=missing_metadata_chunk_ids,
        unicode_error_chunk_ids=unicode_error_chunk_ids,
        duplicate_chunk_ids=sorted(duplicate_chunk_ids),
        orphan_chunk_ids=orphan_chunk_ids,
        invalid_json_chunk_ids=invalid_json_chunk_ids,
        manual_review_documents=manual_review_documents,
        structured_document_count=structured_document_count,
        fallback_document_count=fallback_document_count,
        empty_source_document_count=len(empty_source_document_ids),
        empty_document_output_count=len(empty_document_output_ids),
        structured_chunk_count=structured_chunk_count,
        fallback_chunk_count=fallback_chunk_count,
        supplemental_chunk_count=supplemental_chunk_count,
        empty_source_placeholder_chunk_count=empty_source_placeholder_chunk_count,
        source_nonempty_line_count=source_nonempty_line_count,
        assigned_source_line_count=assigned_source_line_count,
        supplemented_source_line_count=supplemented_source_line_count,
        unassigned_source_line_count=unassigned_source_line_count,
        source_character_count=source_character_count,
        assigned_source_character_count=assigned_source_character_count,
        supplemented_source_character_count=supplemented_source_character_count,
        source_assignment_coverage_ratio=(
            assigned_source_line_count / source_nonempty_line_count
            if source_nonempty_line_count
            else 1.0
        ),
        repaired_split_article_count=repaired_split_article_count,
        suspected_split_article_count=suspected_split_article_count,
        empty_document_output_ids=empty_document_output_ids,
        empty_source_document_ids=empty_source_document_ids,
        suspected_split_article_documents=suspected_split_article_documents,
        quality_gate_passed=not quality_failures,
        quality_gate_failures=quality_failures,
        partial_structure_document_count=len(partial_structure_documents),
        partial_structure_documents=partial_structure_documents,
        structure_warning_document_count=len(structure_warning_documents),
        structure_warning_documents=structure_warning_documents,
        source_unique_token_count=source_unique_token_count,
        missing_source_token_count=missing_source_token_count,
        source_intraline_bigram_count=source_intraline_bigram_count,
        missing_source_bigram_count=missing_source_bigram_count,
        missing_source_token_documents=missing_source_token_documents,
        missing_source_bigram_documents=missing_source_bigram_documents,
    )
    _write_json(
        Path(report_path) if report_path else DEFAULT_REPORT_PATH, report.model_dump()
    )
    _write_json(
        Path(review_path) if review_path else DEFAULT_REVIEW_PATH,
        report.manual_review_documents,
    )
    return chunked_document_count, report


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    lines = [
        json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_unique_document_ids(documents: Iterable[CleanDocument]) -> None:
    duplicates = sorted(
        doc_id
        for doc_id, count in Counter(document.doc_id for document in documents).items()
        if count > 1
    )
    if duplicates:
        raise ValueError("Duplicate doc_id values: {0}".format(", ".join(duplicates)))
