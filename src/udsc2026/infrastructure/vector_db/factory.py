"""Factory for selecting a configured vector database backend."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from udsc2026.infrastructure.vector_db.base import (
    VectorDBAdapter,
    validate_collection_name,
)

_ENVIRONMENT_FIELDS = {
    "VECTOR_DB_TYPE": "type",
    "QDRANT_URL": "qdrant_url",
    "QDRANT_API_KEY": "api_key",
    "FAISS_INDEX_PATH": "faiss_index_path",
}


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

    backend_value = vector_db.get("type", "qdrant")
    if not isinstance(backend_value, str) or not backend_value.strip():
        raise ValueError("vector_db.type must be a non-empty string")
    backend = backend_value.strip().lower()
    vector_db["type"] = backend
    if backend == "qdrant" and "QDRANT_COLLECTION_NAME" in environment:
        vector_db["collection_name"] = environment["QDRANT_COLLECTION_NAME"]
    elif backend == "faiss":
        if "FAISS_COLLECTION_NAME" in environment:
            vector_db["collection_name"] = environment["FAISS_COLLECTION_NAME"]
        elif "QDRANT_COLLECTION_NAME" in environment:
            # Backward compatibility with the original shared collection variable.
            vector_db["collection_name"] = environment["QDRANT_COLLECTION_NAME"]
    api_key = vector_db.get("api_key")
    if isinstance(api_key, str) and not api_key.strip():
        vector_db["api_key"] = None
    if "collection_name" in vector_db:
        vector_db["collection_name"] = validate_collection_name(
            vector_db["collection_name"]
        )
    return vector_db


def get_vector_db_adapter(config: dict[str, Any]) -> VectorDBAdapter:
    """Build the configured Qdrant or FAISS adapter."""
    vector_db = resolve_vector_db_config(config)
    backend = vector_db["type"]
    if backend == "qdrant":
        from udsc2026.infrastructure.vector_db.qdrant_adapter import QdrantAdapter

        if not vector_db.get("qdrant_url"):
            raise ValueError("vector_db.qdrant_url is required for Qdrant")
        if not vector_db.get("collection_name"):
            raise ValueError("vector_db.collection_name is required for Qdrant")
        return QdrantAdapter(
            vector_db["qdrant_url"],
            vector_db["collection_name"],
            vector_db.get("api_key"),
        )
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
    raise ValueError(f"Unsupported vector database type: {backend}")
