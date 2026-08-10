"""Focused tests for the standalone TV3 vector-cache builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tv3 import build_vector_cache


def test_cache_builder_defaults_are_repository_relative() -> None:
    args = build_vector_cache._build_parser().parse_args([])

    assert args.parents_dir == build_vector_cache.DEFAULT_PARENTS_DIR
    assert args.cache_dir == build_vector_cache.DEFAULT_CACHE_DIR
    assert args.embedding_model == build_vector_cache.DEFAULT_EMBEDDING_MODEL


def test_cache_builder_only_resolves_the_default_model(tmp_path: Path) -> None:
    local_model = tmp_path / "model"
    local_model.mkdir()

    assert build_vector_cache._resolve_local_model(
        build_vector_cache.DEFAULT_EMBEDDING_MODEL,
        local_dir=local_model,
    ) == str(local_model)
    assert build_vector_cache._resolve_local_model(
        "custom/model",
        local_dir=local_model,
    ) == "custom/model"


def test_cache_builder_falls_back_to_cpu_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents_dir = tmp_path / "parents"
    parents_dir.mkdir()
    cache_dir = tmp_path / "cache"
    observed: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, **kwargs: object) -> None:
            observed["kwargs"] = kwargs
            self._doc_embeddings = type(
                "Embeddings",
                (),
                {"shape": (3, 4)},
            )()

        def __enter__(self) -> "FakeRetriever":
            observed["entered"] = True
            return self

        def __exit__(self, *_exc_info: object) -> None:
            observed["exited"] = True

    monkeypatch.setattr(
        build_vector_cache.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        build_vector_cache,
        "MockHybridRetriever",
        FakeRetriever,
    )

    result = build_vector_cache.main(
        [
            "--parents_dir",
            str(parents_dir),
            "--cache_dir",
            str(cache_dir),
            "--embedding_model",
            "custom/model",
            "--device",
            "cuda",
            "--max_docs",
            "7",
        ]
    )

    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["parents_dir"] == parents_dir.resolve()
    assert kwargs["cache_dir"] == cache_dir.resolve()
    assert kwargs["embedding_model_path"] == "custom/model"
    assert kwargs["device"] == "cpu"
    assert kwargs["max_docs"] == 7
    assert kwargs["enable_dense"] is True
    assert observed["entered"] is True
    assert observed["exited"] is True
    assert result == 0


def test_cache_builder_rejects_negative_max_docs() -> None:
    with pytest.raises(SystemExit, match="2"):
        build_vector_cache.main(["--max_docs", "-1"])


def test_cache_builder_rejects_missing_dense_cache_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents_dir = tmp_path / "parents"
    parents_dir.mkdir()
    observed: dict[str, object] = {}

    class MissingDenseRetriever:
        _doc_embeddings = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "MissingDenseRetriever":
            return self

        def __exit__(self, *_exc_info: object) -> None:
            observed["exited"] = True

    monkeypatch.setattr(
        build_vector_cache,
        "MockHybridRetriever",
        MissingDenseRetriever,
    )

    with pytest.raises(RuntimeError, match="dense vector cache"):
        build_vector_cache.main(
            [
                "--parents_dir",
                str(parents_dir),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--device",
                "cpu",
            ]
        )

    assert observed["exited"] is True
