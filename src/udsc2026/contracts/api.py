"""HTTP request and response contracts for the RAG query endpoint."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from udsc2026.contracts.qa import Citation
from udsc2026.contracts.retrieval import RetrievalHit


class QueryRequest(BaseModel):
    """Validated input accepted by the synchronous RAG query endpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=10, ge=1, le=100)
    top_n: int = Field(default=5, ge=1, le=100)
    filters: dict[str, str | int | list[str]] | None = None
    prompt_version: str = Field(default="legal_qa_v1", min_length=1)
    debug: bool = False

    @model_validator(mode="after")
    def validate_rerank_limit(self) -> "QueryRequest":
        """Require the final rerank count not to exceed retrieval candidates."""
        if self.top_n > self.top_k:
            raise ValueError("top_n must not exceed top_k")
        return self


class LatencyBreakdown(BaseModel):
    """Measured duration of each pipeline stage in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    cache: float = Field(ge=0)
    retrieval: float = Field(ge=0)
    rerank: float = Field(ge=0)
    generation: float = Field(ge=0)
    total: float = Field(ge=0)


class QueryResponse(BaseModel):
    """Stable result returned to clients after retrieval, reranking and QA."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: LatencyBreakdown
    cache_hit: bool
    prompt_version: str
    warnings: list[str] = Field(default_factory=list)
    trace_id: str
    status: Literal["ok"] = "ok"
