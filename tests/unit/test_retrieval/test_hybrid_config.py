"""Tests for strict hybrid retrieval configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from udsc2026.retrieval.hybrid.config import load_hybrid_settings


def _write_config(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_loads_valid_hybrid_config_and_bm25_environment_override(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path,
        "hybrid:\n"
        "  dense_weight: 0.7\n"
        "  sparse_weight: 0.3\n"
        "  candidate_k: 40\n"
        "  min_score: 0.2\n"
        "  top_k: 8\n"
        "  bm25_index_path: from-yaml.json\n",
    )

    settings = load_hybrid_settings(
        path,
        environ={"BM25_INDEX_PATH": "from-env.json"},
    )

    assert settings.dense_weight == pytest.approx(0.7)
    assert settings.sparse_weight == pytest.approx(0.3)
    assert settings.candidate_k == 40
    assert settings.min_score == pytest.approx(0.2)
    assert settings.top_k == 8
    assert settings.bm25_index_path == "from-env.json"


@pytest.mark.parametrize(
    "hybrid_yaml",
    [
        "[]",
        "null",
        "{dense_weight: true, sparse_weight: 0.0}",
        "{dense_weight: 0.8, sparse_weight: 0.8}",
        "{dense_weight: .nan, sparse_weight: 0.5}",
        "{candidate_k: 0}",
        "{top_k: false}",
        "{bm25_index_path: ''}",
        "{unknown: value}",
    ],
)
def test_rejects_invalid_hybrid_config(
    tmp_path: Path,
    hybrid_yaml: str,
) -> None:
    path = _write_config(tmp_path, f"hybrid: {hybrid_yaml}\n")

    with pytest.raises(ValueError):
        load_hybrid_settings(path, environ={})
