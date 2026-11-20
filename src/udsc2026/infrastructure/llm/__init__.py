"""Model loading and validated configuration for answer generation."""

from typing import TYPE_CHECKING, Any

from udsc2026.infrastructure.llm.config import LLMConfig, load_llm_config

__all__ = ["LLMClient", "LLMConfig", "load_llm_config"]

if TYPE_CHECKING:
    from udsc2026.infrastructure.llm.client import LLMClient


def __getattr__(name: str) -> Any:
    """Load the optional model runtime only when a client is requested."""
    if name == "LLMClient":
        from udsc2026.infrastructure.llm.client import LLMClient

        return LLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
