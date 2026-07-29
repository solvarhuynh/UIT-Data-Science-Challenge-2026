"""LLM configuration schema for the local Qwen3 generator."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Runtime configuration for the local LLM inference backend.

    All generation parameters are centralised here so that experiments can
    swap config objects without touching inference code.  Values are loaded
    from ``configs/`` YAML at application start and injected by TV1.
    """

    model_path: str = Field(
        default="./models/qwen3-legal",
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
        description=(
            "Sampling temperature.  Keep low (≤0.2) for legal answers to "
            "minimise hallucination."
        ),
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    repetition_penalty: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
        description="Penalise token repetition.  Values > 1 reduce looping output.",
    )
    stream: bool = Field(
        default=False,
        description="Enable token-by-token streaming (requires SSE support in TV1).",
    )
    timeout_seconds: Optional[float] = Field(
        default=60.0,
        description="Maximum wall-clock seconds for a single generation call.",
    )
