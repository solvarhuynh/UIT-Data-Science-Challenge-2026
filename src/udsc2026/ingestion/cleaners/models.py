"""Pydantic contracts for the cleaning stage."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CleanDocument(BaseModel):
    """A raw document after reversible, parser-oriented text normalization."""

    doc_id: str
    source_path: str
    title: Optional[str] = None
    cleaned_text: str
    file_format: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    abbreviations: Dict[str, str] = Field(default_factory=dict)
    removed_lines: List[str] = Field(default_factory=list)
