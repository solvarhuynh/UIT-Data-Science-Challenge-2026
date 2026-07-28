"""Readers that convert raw BTC source files into :class:`RawDocument`."""

from udsc2026.ingestion.readers.dispatch import (
    extract_raw_document,
    extract_raw_documents,
)
from udsc2026.ingestion.readers.models import RawDocument

__all__ = ["RawDocument", "extract_raw_document", "extract_raw_documents"]
