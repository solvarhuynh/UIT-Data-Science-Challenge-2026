"""Format dispatch and fault-tolerant directory extraction."""

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

from udsc2026.ingestion.readers.docx_reader import read_docx
from udsc2026.ingestion.readers.json_reader import read_json
from udsc2026.ingestion.readers.models import RawDocument
from udsc2026.ingestion.readers.pdf_reader import read_pdf
from udsc2026.ingestion.readers.txt_reader import read_txt

READERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".json": read_json,
    ".jsonl": read_json,
    ".txt": read_txt,
}

DEFAULT_ERROR_PATH = Path("data/processed_candidate/metadata/extract_errors.json")


def extract_raw_document(
    file_path: str,
) -> Union[RawDocument, Iterator[RawDocument]]:
    """Read one source; JSONL returns a lazy iterator of RawDocument records."""
    ext = Path(file_path).suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        message = "Định dạng không được hỗ trợ: {0} ({1})".format(ext, file_path)
        raise ValueError(message)
    return reader(file_path)


def _error_record(file_path: Path, error: Exception) -> Dict[str, str]:
    return {
        "source_path": str(file_path),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _write_errors(errors: List[Dict[str, str]], errors_path: Path) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_raw_documents(
    raw_directory: str = "data/raw/btc",
    errors_path: Optional[str] = None,
) -> List[RawDocument]:
    """Extract all files in a BTC directory without stopping on individual errors.

    Unsupported/unparseable files and duplicate ``doc_id`` values are written to
    ``data/processed_candidate/metadata/extract_errors.json`` by default.
    Documents whose ids collide are omitted so every returned document has a
    unique identifier.
    """
    source_directory = Path(raw_directory)
    output_path = Path(errors_path) if errors_path else DEFAULT_ERROR_PATH
    documents: List[RawDocument] = []
    errors: List[Dict[str, str]] = []
    seen_doc_ids = set()

    if not source_directory.exists():
        message = "Không tìm thấy thư mục dữ liệu: {0}".format(source_directory)
        error = FileNotFoundError(message)
        _write_errors([_error_record(source_directory, error)], output_path)
        return documents

    files: Iterable[Path] = sorted(
        path for path in source_directory.rglob("*") if path.is_file()
    )
    for file_path in files:
        try:
            extracted = extract_raw_document(str(file_path))
            candidates = (
                (extracted,) if isinstance(extracted, RawDocument) else extracted
            )
            for document in candidates:
                if document.doc_id in seen_doc_ids:
                    raise ValueError("Trùng doc_id: {0}".format(document.doc_id))
                seen_doc_ids.add(document.doc_id)
                documents.append(document)
        except Exception as error:  # Keep extracting independent source files.
            errors.append(_error_record(file_path, error))

    _write_errors(errors, output_path)
    return documents
