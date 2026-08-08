"""Tests for deterministic reranker configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.infrastructure.reranker import RerankerSettings, load_reranker_settings


def test_loads_yaml_and_environment_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
reranker:
  enabled: true
  model_name_or_path: ./models/default
  device: cpu
  batch_size: 8
  max_length: 256
  top_n: 5
  local_files_only: true
  use_fp16: false
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_reranker_settings(
        config_path,
        environ={
            "RERANKER_MODEL_PATH": "./models/override",
            "RERANKER_DEVICE": "cuda:0",
            "RERANKER_BATCH_SIZE": "16",
            "RERANKER_LOCAL_FILES_ONLY": "false",
            "RERANKER_USE_FP16": "true",
        },
    )

    assert settings.model_name_or_path == "./models/override"
    assert settings.device == "cuda:0"
    assert settings.batch_size == 16
    assert settings.max_length == 256
    assert settings.top_n == 5
    assert settings.local_files_only is False
    assert settings.use_fp16 is True


def test_settings_create_lazy_configured_client() -> None:
    settings = RerankerSettings(
        model_name_or_path="models/legal-reranker",
        device=None,
        batch_size=4,
        max_length=128,
        top_n=3,
        use_fp16=False,
    )

    client = settings.create_client()

    assert client.model_name_or_path == "models/legal-reranker"
    assert client.device is None
    assert client.batch_size == 4
    assert client.max_length == 128
    assert client.local_files_only is True
    assert client.use_fp16 is False
    assert client.is_loaded is False


def test_rejects_non_mapping_reranker_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("reranker: enabled\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_reranker_settings(config_path, environ={})


def test_rejects_invalid_environment_value(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("reranker: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a positive integer"):
        load_reranker_settings(
            config_path,
            environ={"RERANKER_BATCH_SIZE": "zero"},
        )


@pytest.mark.parametrize(
    "invalid_yaml",
    [
        "reranker:\n  batch_size: true\n",
        "reranker:\n  max_length: true\n",
        "reranker:\n  top_n: true\n",
        "reranker:\n  enabled: 1\n",
        "reranker:\n  local_files_only: 1\n",
        "reranker:\n  use_fp16: 1\n",
    ],
)
def test_rejects_yaml_boolean_integer_coercion(
    tmp_path: Path,
    invalid_yaml: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(invalid_yaml, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_reranker_settings(config_path, environ={})


def test_rejects_ambiguous_boolean_environment_value(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("reranker: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        load_reranker_settings(
            config_path,
            environ={"RERANKER_ENABLED": "yes"},
        )


def test_missing_config_file_has_actionable_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Reranker config file not found"):
        load_reranker_settings(missing_path, environ={})
