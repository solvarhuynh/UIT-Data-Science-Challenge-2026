"""Health and readiness response contracts for the backend shell."""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness response that never initializes heavyweight model clients."""

    status: Literal["ok"] = "ok"
    service: str = "udsc2026-backend"
    version: str


class ReadinessResponse(BaseModel):
    """Readiness result for required configuration and configured model paths."""

    status: Literal["ready", "not_ready"]
    checks: Dict[str, bool] = Field(default_factory=dict)
    missing: List[str] = Field(default_factory=list)
