"""Tests for embedding YAML and environment configuration."""

from pathlib import Path

import pytest

from udsc2026.infrastructure.embedding.config import load_embedding_config


def test_embedding_environment_overrides_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "embedding:\n"
        "  embedder_model_path: ./models/default\n"
        "  device: cpu\n"
        "  batch_size: 32\n"
        "  max_length: 256\n",
        encoding="utf-8",
    )

    config = load_embedding_config(
        str(config_path),
        environ={
            "MODEL_EMBEDDER_PATH": "/app/models/embedder",
            "EMBEDDING_DEVICE": "cuda:0",
            "EMBEDDING_BATCH_SIZE": "8",
            "EMBEDDING_NORMALIZE_EMBEDDINGS": "false",
        },
    )

    assert config["embedder_model_path"] == "/app/models/embedder"
    assert config["device"] == "cuda:0"
    assert config["batch_size"] == 8
    assert config["max_length"] == 256
    assert config["normalize_embeddings"] is False


def test_embedding_rejects_invalid_integer_override(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("embedding: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        load_embedding_config(
            str(config_path),
            environ={"EMBEDDING_BATCH_SIZE": "zero"},
        )


@pytest.mark.parametrize(
    ("yaml_value", "expected_message"),
    [
        ("true", "positive integer"),
        ("0", "positive integer"),
        ("1.9", "positive integer"),
    ],
)
def test_embedding_rejects_invalid_yaml_batch_size(
    tmp_path: Path,
    yaml_value: str,
    expected_message: str,
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        f"embedding:\n  batch_size: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        load_embedding_config(str(config_path), environ={})


def test_embedding_rejects_ambiguous_boolean_override(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("embedding: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        load_embedding_config(
            str(config_path),
            environ={"EMBEDDING_NORMALIZE_EMBEDDINGS": "yes"},
        )
