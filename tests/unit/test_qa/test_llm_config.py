"""Tests for LLM YAML and environment configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.infrastructure.llm.config import load_llm_config


def test_llm_environment_overrides_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "llm:\n"
        "  model_path: ./models/qwen3-legal\n"
        "  device: cpu\n"
        "  max_new_tokens: 512\n",
        encoding="utf-8",
    )

    config = load_llm_config(
        config_path,
        environ={
            "MODEL_LLM_PATH": "/app/models/qwen3-legal",
            "LLM_DEVICE": "cuda",
            "LLM_MAX_NEW_TOKENS": "768",
            "LLM_STREAM": "true",
        },
    )

    assert config.model_path == "/app/models/qwen3-legal"
    assert config.device == "cuda"
    assert config.max_new_tokens == 768
    assert config.stream is True


def test_llm_environment_rejects_ambiguous_boolean(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("llm: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        load_llm_config(config_path, environ={"LLM_STREAM": "yes"})


def test_llm_config_rejects_unknown_yaml_fields_and_non_finite_values(
    tmp_path: Path,
) -> None:
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text("llm:\n  typo_device: cpu\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="typo_device"):
        load_llm_config(unknown_path, environ={})

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("llm:\n  timeout_seconds: .nan\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="finite number"):
        load_llm_config(invalid_path, environ={})


def test_llm_config_quantization_defaults_to_none(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("llm: {}\n", encoding="utf-8")

    config = load_llm_config(config_path, environ={})

    assert config.quantization == "none"


def test_llm_config_quantization_accepts_4bit_and_8bit(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("llm: {}\n", encoding="utf-8")

    config = load_llm_config(
        config_path,
        environ={"LLM_QUANTIZATION": "4bit"},
    )
    assert config.quantization == "4bit"

    config = load_llm_config(
        config_path,
        environ={"LLM_QUANTIZATION": "8bit"},
    )
    assert config.quantization == "8bit"


def test_llm_config_quantization_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("llm: {}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="quantization"):
        load_llm_config(
            config_path,
            environ={"LLM_QUANTIZATION": "3bit"},
        )
