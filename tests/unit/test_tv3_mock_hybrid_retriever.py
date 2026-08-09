"""Focused regression tests for TV3's experimental hybrid retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from experiments.tv3.mock_hybrid_retriever import (
    MockHybridRetriever,
    _corpus_cache_fingerprint,
    _load_cached_embeddings,
)


class _FakeDiskBM25:
    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc

    def get_scores_and_docs(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        del query_tokens, top_k
        return [(0, 1.0)]

    def get_doc(self, doc_id: int) -> dict[str, Any] | None:
        return dict(self.doc) if doc_id == 0 else None


class _DenseModelMustNotRun:
    def encode(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("BM25-only search invoked the dense model")


def _bare_retriever(
    tmp_path: Path,
    *,
    doc: dict[str, Any] | None = None,
) -> MockHybridRetriever:
    retriever = MockHybridRetriever.__new__(MockHybridRetriever)
    retriever.N = 1
    retriever.parents_dir = tmp_path
    retriever.embedding_model_path = "unused-in-tests"
    retriever.rrf_k = 60
    retriever.alpha = 0.5
    retriever.enable_query_decomposition = False
    retriever._disk_bm25 = _FakeDiskBM25(
        doc or {"doc_id": "parent-1", "text": "Original parent text."}
    )
    retriever._doc_embeddings = None
    retriever._dense_model = None
    return retriever


def test_decompose_queries_preserves_intent_cue_and_deduplicates() -> None:
    retriever = MockHybridRetriever.__new__(MockHybridRetriever)
    query = "Doanh nghiệp nộp hồ sơ trễ thì có bị phạt bao nhiêu theo quy định?"

    sub_queries = retriever._decompose_queries(query)
    normalized = [item.strip().casefold() for item in sub_queries]

    assert query in sub_queries
    assert len(normalized) == len(set(normalized))
    assert any(item != query.casefold() and "bị phạt" in item for item in normalized), (
        sub_queries
    )


@pytest.mark.parametrize(
    ("query", "top_k", "mode"),
    [
        ("", 3, "hybrid"),
        ("   ", 3, "hybrid"),
        ("câu hỏi", 0, "hybrid"),
        ("câu hỏi", -1, "hybrid"),
        ("câu hỏi", 3, "unknown"),
    ],
)
def test_search_rejects_invalid_arguments(
    tmp_path: Path, query: str, top_k: int, mode: str
) -> None:
    retriever = _bare_retriever(tmp_path)

    with pytest.raises(ValueError):
        retriever.search(query, top_k=top_k, mode=mode)


def test_bm25_mode_never_invokes_dense_model(tmp_path: Path) -> None:
    retriever = _bare_retriever(tmp_path)
    retriever._doc_embeddings = np.ones((1, 4), dtype=np.float32)
    retriever._dense_model = _DenseModelMustNotRun()

    results = retriever.search("thuế thu nhập", top_k=1, mode="bm25")

    assert [item["doc_id"] for item in results] == ["parent-1"]


def test_search_keeps_parent_hit_text_unchanged(tmp_path: Path) -> None:
    original_text = "Original parent text returned by the parent index."
    retriever = _bare_retriever(
        tmp_path,
        doc={"doc_id": "parent-1", "text": original_text},
    )
    unrelated_text = "Unrelated context that must never be appended. " * 10
    (tmp_path / "parent-1.jsonl").write_text(
        json.dumps({"text": unrelated_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results = retriever.search("thuế thu nhập", top_k=1, mode="bm25")

    assert results[0]["text"] == original_text


def test_search_with_decomposition_disabled_uses_only_original_query(
    tmp_path: Path,
) -> None:
    retriever = _bare_retriever(tmp_path)
    bm25_calls: list[list[str]] = []
    original_search = retriever._disk_bm25.get_scores_and_docs

    def record_search(
        query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        bm25_calls.append(query_tokens)
        return original_search(query_tokens, top_k)

    def decomposition_must_not_run(_query: str) -> list[str]:
        raise AssertionError("disabled query decomposition was invoked")

    retriever._disk_bm25.get_scores_and_docs = record_search
    retriever._decompose_queries = decomposition_must_not_run

    retriever.search(
        "Doanh nghiệp nộp hồ sơ trễ thì có bị phạt bao nhiêu?",
        top_k=1,
        mode="bm25",
    )

    assert len(bm25_calls) == 1


def _write_fingerprinted_corpus(root: Path, tree_hash: str) -> Path:
    parents_dir = root / "parents"
    parents_dir.mkdir(parents=True)
    (parents_dir / "corpus.jsonl").write_text(
        json.dumps({"doc_id": "same", "text": "same corpus"}) + "\n",
        encoding="utf-8",
    )
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "processing_manifest.json").write_text(
        json.dumps({"processed_corpus_tree_hash": tree_hash}),
        encoding="utf-8",
    )
    return parents_dir


def test_corpus_cache_key_changes_with_processed_tree_hash(tmp_path: Path) -> None:
    first = _write_fingerprinted_corpus(tmp_path / "first", "a" * 64)
    second = _write_fingerprinted_corpus(tmp_path / "second", "b" * 64)

    first_key = _corpus_cache_fingerprint(first, max_docs=0)
    second_key = _corpus_cache_fingerprint(second, max_docs=0)

    assert first_key != second_key


def test_corpus_cache_key_changes_with_max_docs(tmp_path: Path) -> None:
    parents_dir = _write_fingerprinted_corpus(tmp_path, "a" * 64)

    full_key = _corpus_cache_fingerprint(parents_dir, max_docs=0)
    limited_key = _corpus_cache_fingerprint(parents_dir, max_docs=100)

    assert full_key != limited_key


def test_load_cached_embeddings_accepts_valid_npy(tmp_path: Path) -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache_file = tmp_path / "embeddings.npy"
    np.save(cache_file, expected)

    loaded = _load_cached_embeddings(cache_file, expected_rows=3)

    assert loaded is not None
    assert loaded.shape == (3, 4)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, expected)


def test_load_cached_embeddings_infers_legacy_raw_dimension(tmp_path: Path) -> None:
    expected = np.arange(15, dtype=np.float32).reshape(3, 5)
    cache_file = tmp_path / "legacy-raw.npy"
    expected.tofile(cache_file)

    loaded = _load_cached_embeddings(cache_file, expected_rows=3)

    assert loaded is not None
    assert loaded.shape == (3, 5)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, expected)


@pytest.mark.parametrize("cache_kind", ["wrong_rows", "corrupt_npy", "bad_raw"])
def test_load_cached_embeddings_rejects_invalid_cache(
    tmp_path: Path, cache_kind: str
) -> None:
    cache_file = tmp_path / f"{cache_kind}.npy"

    if cache_kind == "wrong_rows":
        np.save(cache_file, np.ones((2, 4), dtype=np.float32))
    elif cache_kind == "corrupt_npy":
        cache_file.write_bytes(b"\x93NUMPYcorrupt")
    else:
        cache_file.write_bytes(b"not-a-valid-float32-matrix")

    assert _load_cached_embeddings(cache_file, expected_rows=3) is None
    assert cache_file.exists()
