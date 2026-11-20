"""Factory for selecting a configured vector database backend."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from udsc2026.infrastructure.vector_db.base import (
    VectorDBAdapter,
    validate_collection_name,
)

_ENVIRONMENT_FIELDS = {"FAISS_INDEX_PATH": "faiss_index_path"}


def resolve_vector_db_config(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve vector settings without mutating YAML-derived configuration."""
    raw_vector_db = config.get("vector_db", config)
    if not isinstance(raw_vector_db, Mapping):
        raise ValueError("Configuration field 'vector_db' must be a mapping")

    vector_db = dict(raw_vector_db)
    environment = os.environ if environ is None else environ
    for variable, field_name in _ENVIRONMENT_FIELDS.items():
        if variable in environment:
            vector_db[field_name] = environment[variable]

    backend_value = vector_db.get("type", "faiss")
    if not isinstance(backend_value, str) or not backend_value.strip():
        raise ValueError("vector_db.type must be a non-empty string")
    backend = backend_value.strip().lower()
    vector_db["type"] = backend
    if "FAISS_COLLECTION_NAME" in environment:
        vector_db["collection_name"] = environment["FAISS_COLLECTION_NAME"]
    if "collection_name" in vector_db:
        vector_db["collection_name"] = validate_collection_name(
            vector_db["collection_name"]
        )
    return vector_db


def get_vector_db_adapter(config: dict[str, Any]) -> VectorDBAdapter:
    """Build the configured FAISS adapter."""
    vector_db = resolve_vector_db_config(config)
    backend = vector_db["type"]
    if backend == "faiss":
        from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

        if not vector_db.get("faiss_index_path"):
            raise ValueError("vector_db.faiss_index_path is required for FAISS")
        if not vector_db.get("collection_name"):
            raise ValueError("vector_db.collection_name is required for FAISS")
        index_path = vector_db["faiss_index_path"]
        collection_name = vector_db["collection_name"]
        path = Path(index_path)
        root = str(path.parent) if path.name == collection_name else index_path
        return FaissAdapter(root, collection_name)
    raise ValueError(f"Only the FAISS vector database is supported: {backend}")
