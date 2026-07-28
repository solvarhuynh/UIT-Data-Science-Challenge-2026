"""Backend-neutral vector database infrastructure."""

from udsc2026.infrastructure.vector_db.base import VectorDBAdapter
from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter

__all__ = ["VectorDBAdapter", "get_vector_db_adapter"]
