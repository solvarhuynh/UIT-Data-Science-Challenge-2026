"""Validated configuration shared by hybrid retrieval and readiness probes."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from udsc2026.config import load_project_config


class HybridSettings(BaseModel):
    """Strict settings for score fusion and sparse-index loading."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_k: int = Field(default=50, gt=0)
    min_score: float = Field(default=0.0, allow_inf_nan=False)
    top_k: int = Field(default=10, gt=0)
    bm25_index_path: str = Field(
        default="./data/vector_store/bm25/index.json",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "HybridSettings":
        """Require a convex combination so fused scores remain interpretable."""

        if not math.isclose(
            self.dense_weight + self.sparse_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("hybrid dense_weight and sparse_weight must sum to 1")
        return self


def load_hybrid_settings(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> HybridSettings:
    """Load the ``hybrid`` section and apply the BM25 artifact override."""

    project_config = load_project_config(config_path, environ=environ)
    raw_hybrid = project_config.get("hybrid", {})
    if not isinstance(raw_hybrid, Mapping):
        raise ValueError("Configuration field 'hybrid' must be a mapping")

    values = dict(raw_hybrid)
    environment = os.environ if environ is None else environ
    if "BM25_INDEX_PATH" in environment:
        values["bm25_index_path"] = environment["BM25_INDEX_PATH"]
    return HybridSettings.model_validate(values)
