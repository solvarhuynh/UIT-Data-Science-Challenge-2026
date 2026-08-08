"""Embedding infrastructure for local retrieval models."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from udsc2026.infrastructure.embedding.client import EmbeddingClient

__all__ = ["EmbeddingClient"]


def __getattr__(name: str) -> Any:
    """Load the optional sentence-transformers client only when requested."""
    if name == "EmbeddingClient":
        from udsc2026.infrastructure.embedding.client import EmbeddingClient

        return EmbeddingClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


