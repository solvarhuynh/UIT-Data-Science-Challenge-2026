"""Shared models for the extraction stage."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Unmodified text and source metadata produced by a format-specific reader."""

    doc_id: str
    source_path: str
    title: Optional[str] = None
    raw_text: str
    file_format: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
