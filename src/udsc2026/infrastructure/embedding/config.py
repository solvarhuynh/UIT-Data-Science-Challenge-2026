"""Backward-compatible access to the shared embedding configuration."""

from typing import Any

from udsc2026.infrastructure.config import load_config


def load_embedding_config(path: str = "configs/base.yaml") -> dict[str, Any]:
    """Return embedding settings through the common loader.

    ``path`` is retained for compatibility; shared callers should use
    :func:`udsc2026.infrastructure.config.load_config` directly.
    """
    env = "base" if path.endswith("base.yaml") else "development"
    return load_config(env).get("embedding", {})
