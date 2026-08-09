"""Backward-compatible access to the shared embedding configuration."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from udsc2026.config import load_project_config

_ENVIRONMENT_FIELDS = {
    "MODEL_EMBEDDER_PATH": "embedder_model_path",
    "EMBEDDING_DEVICE": "device",
    "EMBEDDING_BATCH_SIZE": "batch_size",
    "EMBEDDING_MAX_LENGTH": "max_length",
    "EMBEDDING_OUTPUT_DIMENSION": "output_dimension",
    "EMBEDDING_NORMALIZE_EMBEDDINGS": "normalize_embeddings",
    "EMBEDDING_WINDOW_LONG_TEXTS": "window_long_texts",
    "EMBEDDING_WINDOW_OVERLAP_TOKENS": "window_overlap_tokens",
}
_INTEGER_FIELDS = {
    "batch_size",
    "max_length",
    "output_dimension",
    "window_overlap_tokens",
}
_BOOLEAN_FIELDS = {"normalize_embeddings", "window_long_texts"}


def load_embedding_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load embedding YAML settings with explicit runtime overrides."""
    config = cast(
        Mapping[str, Any],
        load_project_config(path, environ=environ),
    )
    raw_embedding = config.get("embedding", {})
    if not isinstance(raw_embedding, Mapping):
        raise ValueError("Configuration field 'embedding' must be a mapping")

    embedding = dict(raw_embedding)
    if "embedder_model_path" not in embedding and "model_path" in embedding:
        embedding["embedder_model_path"] = embedding["model_path"]
    environment = os.environ if environ is None else environ
    for variable, field_name in _ENVIRONMENT_FIELDS.items():
        if variable in environment:
            raw_value = environment[variable]
            if field_name in _BOOLEAN_FIELDS:
                normalized = raw_value.strip().casefold()
                if normalized not in {"true", "false"}:
                    raise ValueError(f"{variable} must be 'true' or 'false'")
                embedding[field_name] = normalized == "true"
            elif field_name in _INTEGER_FIELDS:
                try:
                    embedding[field_name] = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{variable} must be a positive integer") from exc
            else:
                embedding[field_name] = raw_value
    for integer_field in _INTEGER_FIELDS:
        if integer_field in embedding:
            configured_value = embedding[integer_field]
            if isinstance(configured_value, bool) or not isinstance(
                configured_value, int
            ):
                raise ValueError(
                    f"embedding.{integer_field} must be a positive integer"
                )
            if configured_value <= 0:
                raise ValueError(
                    f"embedding.{integer_field} must be a positive integer"
                )
    for text_field in ("embedder_model_path", "device"):
        if text_field in embedding:
            value = embedding[text_field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"embedding.{text_field} must be a non-empty string")
            embedding[text_field] = value.strip()
    for boolean_field in _BOOLEAN_FIELDS:
        if boolean_field in embedding and not isinstance(
            embedding[boolean_field], bool
        ):
            raise ValueError(f"embedding.{boolean_field} must be a boolean")
    if embedding.get("window_overlap_tokens", 0) >= embedding.get("max_length", 256):
        raise ValueError(
            "embedding.window_overlap_tokens must be less than embedding.max_length"
        )
    return embedding
