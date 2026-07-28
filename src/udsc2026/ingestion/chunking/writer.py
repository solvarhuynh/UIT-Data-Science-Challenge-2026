"""JSONL persistence for retrieval children, parent contexts, and validation."""

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.chunking.chunker import chunk_clean_document
from udsc2026.ingestion.chunking.models import ChunkingResult, ValidationReport
from udsc2026.ingestion.chunking.validation import validate_chunking_results


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
    chunk_size: int = 512,
    chunk_overlap: int = 80,
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


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
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
