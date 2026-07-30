"""Validated YAML and environment configuration for cross-encoder reranking."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from udsc2026.config import load_project_config, resolve_config_path
from udsc2026.infrastructure.reranker.client import CrossEncoderClient

_ENVIRONMENT_FIELDS = {
    "RERANKER_ENABLED": "enabled",
    "RERANKER_MODEL_PATH": "model_name_or_path",
    "RERANKER_DEVICE": "device",
    "RERANKER_BATCH_SIZE": "batch_size",
    "RERANKER_MAX_LENGTH": "max_length",
    "RERANKER_TOP_N": "top_n",
    "RERANKER_LOCAL_FILES_ONLY": "local_files_only",
}
_INTEGER_FIELDS = {"batch_size", "max_length", "top_n"}
_BOOLEAN_FIELDS = {"enabled", "local_files_only"}


class RerankerSettings(BaseModel):
    """Runtime settings shared by the client, orchestration, and containers."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
        protected_namespaces=(),
    )

    enabled: bool = True
    model_name_or_path: str = Field(
        default="./models/reranker",
        min_length=1,
    )
    device: str | None = "cpu"
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=512, gt=0)
    top_n: int = Field(default=10, gt=0)
    local_files_only: bool = True

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str | None) -> str | None:
        """Treat a blank device as automatic device selection."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def create_client(self) -> CrossEncoderClient:
        """Create a lazy client without importing or loading model weights."""
        return CrossEncoderClient(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            local_files_only=self.local_files_only,
        )


def load_reranker_settings(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RerankerSettings:
    """Load ``reranker`` YAML settings and apply explicit environment overrides."""
    path = resolve_config_path(config_path, environ=environ)
    if not path.is_file():
        raise FileNotFoundError(f"Reranker config file not found: {path}")

    raw_config = load_project_config(path, environ=environ)

    reranker_config = raw_config.get("reranker", {})
    if not isinstance(reranker_config, Mapping):
        raise ValueError("Configuration field 'reranker' must be a mapping")

    values: dict[str, Any] = dict(reranker_config)
    environment = os.environ if environ is None else environ
    for variable, field_name in _ENVIRONMENT_FIELDS.items():
        if variable in environment:
            raw_value = environment[variable]
            if field_name in _INTEGER_FIELDS:
                try:
                    values[field_name] = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{variable} must be a positive integer") from exc
            elif field_name in _BOOLEAN_FIELDS:
                normalized = raw_value.strip().casefold()
                if normalized not in {"true", "false"}:
                    raise ValueError(f"{variable} must be 'true' or 'false'")
                values[field_name] = normalized == "true"
            else:
                values[field_name] = raw_value

    return RerankerSettings.model_validate(values)
