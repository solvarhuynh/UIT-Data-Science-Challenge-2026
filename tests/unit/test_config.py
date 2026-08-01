"""Cross-component tests for the shared project configuration path."""

from pathlib import Path

import pytest

from udsc2026.config import load_project_config, resolve_config_path
from udsc2026.infrastructure.embedding.config import load_embedding_config
from udsc2026.infrastructure.llm.config import load_llm_config
from udsc2026.infrastructure.reranker.config import load_reranker_settings


def test_all_loaders_honor_shared_environment_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "embedding:\n"
        "  embedder_model_path: ./embedder\n"
        "llm:\n"
        "  model_path: ./llm\n"
        "reranker:\n"
        "  model_name_or_path: ./reranker\n",
        encoding="utf-8",
    )
    environment = {"UDSC2026_CONFIG_PATH": str(config_path)}

    assert resolve_config_path(environ=environment) == config_path
    assert load_embedding_config(environ=environment)["embedder_model_path"] == (
        "./embedder"
    )
    assert load_llm_config(environ=environment).model_path == "./llm"
    assert (
        load_reranker_settings(environ=environment).model_name_or_path == "./reranker"
    )


@pytest.mark.parametrize("contents", ["[]\n", "false\n", "0\n"])
def test_project_config_rejects_falsey_non_mapping_roots(
    tmp_path: Path,
    contents: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_project_config(config_path, environ={})


def test_blank_shared_config_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        resolve_config_path(environ={"UDSC2026_CONFIG_PATH": "  "})
