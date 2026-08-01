"""LLM configuration schema for the local Qwen3 generator."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from udsc2026.config import load_project_config

_ENVIRONMENT_FIELDS = {
    "MODEL_LLM_PATH": "model_path",
    "LLM_BACKEND": "backend",
    "LLM_DEVICE": "device",
    "LLM_DTYPE": "dtype",
    "LLM_MAX_NEW_TOKENS": "max_new_tokens",
    "LLM_TEMPERATURE": "temperature",
    "LLM_TOP_P": "top_p",
    "LLM_REPETITION_PENALTY": "repetition_penalty",
    "LLM_STREAM": "stream",
    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
}
_INTEGER_FIELDS = {"max_new_tokens"}
_FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "repetition_penalty",
    "timeout_seconds",
}
_BOOLEAN_FIELDS = {"stream"}


class LLMConfig(BaseModel):
    """Runtime configuration for the local LLM inference backend.

    All generation parameters are centralised here so that experiments can
    swap config objects without touching inference code.  Values are loaded
    from ``configs/`` YAML at application start and injected by TV1.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
        protected_namespaces=(),
    )

    model_path: str = Field(
        default="./models/qwen3-legal",
        min_length=1,
        description="Absolute or relative path to the local model checkpoint.",
    )
    backend: Literal["transformers", "vllm"] = Field(
        default="transformers",
        description="Inference backend.  Use 'vllm' when a dedicated GPU is available.",
    )
    device: Literal["cuda", "cpu", "mps"] = Field(
        default="cpu",
        description="Target device.  Defaults to CPU for broad compatibility.",
    )
    dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Model weight dtype.  bfloat16 is recommended for Qwen3 on CUDA.",
    )
    max_new_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum number of tokens the model may generate per request.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
        description=(
            "Sampling temperature.  Keep low (≤0.2) for legal answers to "
            "minimise hallucination."
        ),
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Nucleus sampling probability threshold.",
    )
    repetition_penalty: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
        allow_inf_nan=False,
        description="Penalise token repetition.  Values > 1 reduce looping output.",
    )
    stream: bool = Field(
        default=False,
        description="Enable token-by-token streaming (requires SSE support in TV1).",
    )
    timeout_seconds: Optional[float] = Field(
        default=60.0,
        gt=0,
        allow_inf_nan=False,
        description="Maximum wall-clock seconds for a single generation call.",
    )


def load_llm_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMConfig:
    """Load the optional ``llm`` section and overlay explicit environment values."""
    raw_config: Any = load_project_config(path, environ=environ)
    raw_llm = raw_config.get("llm", {})
    if not isinstance(raw_llm, Mapping):
        raise ValueError("Configuration field 'llm' must be a mapping")

    values = dict(raw_llm)
    environment = os.environ if environ is None else environ
    for variable, field_name in _ENVIRONMENT_FIELDS.items():
        if variable in environment:
            raw_value = environment[variable]
            if field_name in _INTEGER_FIELDS:
                try:
                    values[field_name] = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{variable} must be an integer") from exc
            elif field_name in _FLOAT_FIELDS:
                try:
                    values[field_name] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{variable} must be a number") from exc
            elif field_name in _BOOLEAN_FIELDS:
                normalized = raw_value.strip().casefold()
                if normalized not in {"true", "false"}:
                    raise ValueError(f"{variable} must be 'true' or 'false'")
                values[field_name] = normalized == "true"
            else:
                values[field_name] = raw_value
    return LLMConfig.model_validate(values)
