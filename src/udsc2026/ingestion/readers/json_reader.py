"""Readers for JSON and JSONL raw sources."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from udsc2026.ingestion.readers._helpers import (
    DATE_FIELDS,
    DOCUMENT_ID_FIELDS,
    TITLE_FIELDS,
    first_mapping,
    get_first_field,
    make_doc_id,
)
from udsc2026.ingestion.readers.models import RawDocument


CONTENT_FIELDS = ("raw_text", "content", "text", "body", "document_text")


def _decode_json(raw_bytes: bytes, file_path: str) -> str:
    """Decode a JSON source according to the UTF encodings allowed by JSON."""
    for encoding in ("utf-8-sig", "utf-16", "utf-32"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf", raw_bytes, 0, len(raw_bytes), file_path)


def _document_mapping(parsed: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(parsed, Mapping):
        return parsed
    if isinstance(parsed, list):
        return first_mapping(parsed)
    return None


def _metadata_from_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    issue_date = get_first_field(mapping, DATE_FIELDS)
    if issue_date is not None:
        metadata["issue_date"] = issue_date
    return metadata


def _raw_document_from_json_record(
    path: Path,
    raw_text: str,
    parsed: Any,
    source_line: Optional[int] = None,
) -> RawDocument:
    """Build a RawDocument from one parsed JSON object or JSONL record."""
    mapping = _document_mapping(parsed)
    source_id = get_first_field(mapping, DOCUMENT_ID_FIELDS)
    if source_id is None and source_line is not None:
        source_id = "{0}_line_{1}".format(path.stem, source_line)
    title_value = get_first_field(mapping, TITLE_FIELDS)
    metadata = _metadata_from_mapping(mapping)
    metadata.update(
        {
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "json_type": type(parsed).__name__,
            "source_extension": path.suffix.lower(),
        }
    )
    if source_line is not None:
        metadata["source_line"] = source_line
    if source_id is not None:
        metadata["source_document_id"] = source_id
    content = get_first_field(mapping, CONTENT_FIELDS)
    if content is not None:
        raw_text = str(content)
        metadata["content_field"] = next(
            field for field in CONTENT_FIELDS if mapping and mapping.get(field) is not None
        )

    return RawDocument(
        doc_id=make_doc_id(str(path), source_id),
        source_path=str(path),
        title=str(title_value).strip() if title_value is not None else None,
        raw_text=raw_text,
        file_format="json",
        metadata=metadata,
    )


def _iter_jsonl_documents(path: Path) -> Iterator[RawDocument]:
    """Stream JSONL records so multi-GB source files are never loaded at once."""
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            raw_text = line.rstrip("\r\n")
            if not raw_text.strip():
                continue
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as error:
                message = "JSONL không hợp lệ tại dòng {0}: {1}".format(
                    line_number, error.msg
                )
                raise ValueError(message) from error
            yield _raw_document_from_json_record(
                path, raw_text, parsed, source_line=line_number
            )


def read_json(file_path: str) -> Union[RawDocument, Iterator[RawDocument]]:
    """Read JSON, or stream one RawDocument per record for a JSONL source.

    ``RawDocument.raw_text`` is a string, so representing a multi-GB JSONL file
    as one object would necessarily load it into memory. JSONL therefore yields
    one RawDocument per source line; callers should consume it lazily.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _iter_jsonl_documents(path)

    raw_text = _decode_json(path.read_bytes(), str(path))
    parsed = json.loads(raw_text)
    document = _raw_document_from_json_record(path, raw_text, parsed)
    if isinstance(parsed, list):
        document.metadata["record_count"] = len(parsed)
    return document
