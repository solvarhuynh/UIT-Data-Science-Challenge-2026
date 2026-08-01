"""Fault-tolerant batch orchestration and audit output for document cleaning."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from udsc2026.ingestion.cleaners.document_cleaner import clean_document
from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.readers.models import RawDocument

DEFAULT_DOCUMENTS_DIR = Path("data/processed/documents")
DEFAULT_ERRORS_PATH = Path("data/processed/metadata/clean_errors.json")


def _error_record(document: RawDocument, error: Exception) -> Dict[str, str]:
    return {
        "doc_id": document.doc_id,
        "source_path": document.source_path,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _write_document(document: CleanDocument, output_dir: Path) -> None:
    output_file = output_dir / "{0}.json".format(document.doc_id)
    output_file.write_text(
        json.dumps(document.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_errors(errors: List[Dict[str, str]], errors_path: Path) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clean_raw_documents(
    raw_documents: Iterable[RawDocument],
    output_dir: Optional[Path] = None,
    errors_path: Optional[Path] = None,
) -> List[CleanDocument]:
    """Clean documents independently, writing audit JSON without failing a batch.

    One audit file is written per successful document to
    ``data/processed/documents/``. Failures are collected in
    ``data/processed/metadata/clean_errors.json`` by default.
    """
    target_dir = output_dir or DEFAULT_DOCUMENTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    error_output_path = errors_path or DEFAULT_ERRORS_PATH
    cleaned_documents: List[CleanDocument] = []
    errors: List[Dict[str, str]] = []

    for raw_document in raw_documents:
        try:
            cleaned_document = clean_document(raw_document)
            _write_document(cleaned_document, target_dir)
            cleaned_documents.append(cleaned_document)
        except Exception as error:  # Continue processing independent documents.
            errors.append(_error_record(raw_document, error))

    _write_errors(errors, error_output_path)
    return cleaned_documents
