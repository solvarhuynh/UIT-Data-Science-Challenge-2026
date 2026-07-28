"""Small shared helpers for readers; no text cleaning is performed here."""

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DOCUMENT_ID_FIELDS = ("doc_id", "document_id", "id", "code")
TITLE_FIELDS = ("title", "document_title", "law_name", "name")
DATE_FIELDS = ("issue_date", "issued_date", "publication_date", "date")


def make_doc_id(file_path: str, source_id: Optional[Any] = None) -> str:
    """Build a stable identifier from a supplied document id or the file stem."""
    value = str(source_id).strip() if source_id is not None else Path(file_path).stem
    if not value:
        value = Path(file_path).stem or "document"

    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "document"


def first_nonempty_line(text: str) -> Optional[str]:
    """Return the first non-empty line without changing the original text."""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def first_mapping(values: Iterable[Any]) -> Optional[Mapping[str, Any]]:
    """Return the first mapping in an iterable, if any."""
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def get_first_field(
    mapping: Optional[Mapping[str, Any]], fields: Iterable[str]
) -> Optional[Any]:
    """Read the first non-empty field from a mapping, case-insensitively."""
    if mapping is None:
        return None

    by_lower_name = {str(key).lower(): value for key, value in mapping.items()}
    for field in fields:
        value = by_lower_name.get(field)
        if value is not None and str(value).strip():
            return value
    return None
