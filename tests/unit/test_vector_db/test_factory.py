"""Tests for vector database environment overlay and validation."""

import pytest

from udsc2026.infrastructure.vector_db.factory import resolve_vector_db_config


def test_environment_selects_qdrant_without_mutating_base_config() -> None:
    config = {
        "vector_db": {
            "type": "faiss",
            "qdrant_url": "http://localhost:6333",
            "collection_name": "legal_chunks",
            "faiss_index_path": "./data/vector_store/faiss",
        }
    }

    resolved = resolve_vector_db_config(
        config,
        environ={
            "VECTOR_DB_TYPE": "QDRANT",
            "QDRANT_URL": "http://qdrant:6333",
        },
    )

    assert resolved["type"] == "qdrant"
    assert resolved["qdrant_url"] == "http://qdrant:6333"
    assert config["vector_db"]["type"] == "faiss"


def test_rejects_invalid_vector_config_shape() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_vector_db_config({"vector_db": "qdrant"}, environ={})


def test_backend_specific_collection_environment_is_selected() -> None:
    environment = {
        "VECTOR_DB_TYPE": "faiss",
        "QDRANT_COLLECTION_NAME": "qdrant-collection",
        "FAISS_COLLECTION_NAME": "faiss-collection",
    }

    resolved = resolve_vector_db_config(
        {"vector_db": {"collection_name": "yaml-collection"}},
        environ=environment,
    )

    assert resolved["collection_name"] == "faiss-collection"


def test_blank_qdrant_api_key_is_treated_as_unset() -> None:
    resolved = resolve_vector_db_config(
        {"vector_db": {"type": "qdrant"}},
        environ={"QDRANT_API_KEY": "  "},
    )

    assert resolved["api_key"] is None


@pytest.mark.parametrize(
    "name",
    ["", " ", ".", "..", "../outside", r"a\b", "C:", "legal chunks"],
)
def test_rejects_unsafe_collection_names(name: str) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        resolve_vector_db_config(
            {"vector_db": {"type": "faiss", "collection_name": name}},
            environ={},
        )


@pytest.mark.parametrize("backend", ["", "   ", 123])
def test_rejects_invalid_backend_name(backend: object) -> None:
    with pytest.raises(ValueError, match="type"):
        resolve_vector_db_config({"type": backend}, environ={})
