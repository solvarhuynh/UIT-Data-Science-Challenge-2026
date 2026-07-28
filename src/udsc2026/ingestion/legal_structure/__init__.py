"""Rule-based parsing for Vietnamese legal-document structure."""

from udsc2026.ingestion.legal_structure.models import (
    Article,
    Chapter,
    Clause,
    LegalStructureDocument,
    Point,
    Section,
    StructureEntry,
)
from udsc2026.ingestion.legal_structure.parser import (
    LegalStructureParser,
    infer_law_name,
    parse_legal_document,
    parse_legal_structure,
    parse_legal_text,
)
from udsc2026.ingestion.legal_structure.patterns import PATTERNS

__all__ = [
    "Article",
    "Chapter",
    "Clause",
    "LegalStructureDocument",
    "LegalStructureParser",
    "PATTERNS",
    "Point",
    "Section",
    "StructureEntry",
    "infer_law_name",
    "parse_legal_document",
    "parse_legal_structure",
    "parse_legal_text",
]
