"""Factory for selecting a configured vector database backend."""

from pathlib import Path
from typing import Any

from udsc2026.infrastructure.vector_db.base import VectorDBAdapter


def get_vector_db_adapter(config: dict[str, Any]) -> VectorDBAdapter:
    """Build the configured Qdrant or FAISS adapter."""
    vector_db = config.get("vector_db", config)
    backend = vector_db.get("type", "qdrant").lower()
    if backend == "qdrant":
        from udsc2026.infrastructure.vector_db.qdrant_adapter import QdrantAdapter
        return QdrantAdapter(vector_db["qdrant_url"], vector_db["collection_name"], vector_db.get("api_key"))
    if backend == "faiss":
        from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter
        index_path = vector_db["faiss_index_path"]
        collection_name = vector_db["collection_name"]
        path = Path(index_path)
        root = str(path.parent) if path.name == collection_name else index_path
        return FaissAdapter(root, collection_name)
    raise ValueError(f"Unsupported vector database type: {backend}")
