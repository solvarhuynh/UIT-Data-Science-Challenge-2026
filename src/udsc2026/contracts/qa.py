"""QA contracts shared across the QA Engine, API, and evaluation layers."""

from typing import List, Optional

from pydantic import BaseModel, Field

from udsc2026.contracts.retrieval import RetrievalHit


class Citation(BaseModel):
    """Represents a single legal citation extracted and validated from an LLM answer.

    A citation is considered verified when it maps back to at least one of the
    ``RetrievalHit`` objects that were supplied as context to the QA Engine.
    Unverified citations are potential hallucinations and must be surfaced as
    warnings rather than silently dropped.
    """

    law_name: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None
    is_verified: bool = False
    warning: Optional[str] = None


class QAResponse(BaseModel):
    """Final output of the QA Engine returned to TV1 (Orchestrator).

    All fields are mandatory so downstream consumers can rely on the schema
    without defensive None-checks for the core fields.
    """

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    used_prompt_version: str
    retrieval_hits: List[RetrievalHit] = Field(default_factory=list)
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    cache_hit: bool = False
    trace_id: Optional[str] = None
