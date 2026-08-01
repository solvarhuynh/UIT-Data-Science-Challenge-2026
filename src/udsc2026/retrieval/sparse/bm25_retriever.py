"""Standalone BM25 sparse retriever for legal document chunks."""

import pickle
import logging
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.config import load_config
from udsc2026.retrieval.sparse.tokenizer import tokenize_vi

LOGGER = logging.getLogger(__name__)


class BM25Retriever:
    """Build, persist, and search a Vietnamese BM25 index over legal chunks."""

    def __init__(self, index_path: str | None = None) -> None:
        self.index_path = index_path or "data/vector_store/bm25/index.pkl"
        self._bm25: BM25Okapi | None = None
        self._chunks: list[LegalChunk] = []

    def build_index(self, chunks: list[LegalChunk]) -> None:
        """Build an in-memory BM25 index while retaining original chunk contracts."""
        if not chunks:
            raise ValueError("chunks must not be empty")
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize_vi(chunk.text) for chunk in self._chunks])

    def save(self, path: str | None = None) -> None:
        """Serialize the BM25 index and its source chunks to a local pickle."""
        if self._bm25 is None:
            raise ValueError("index has not been built")
        target = Path(path or self.index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as index_file:
            pickle.dump((self._bm25, self._chunks), index_file)

    def load(self, path: str | None = None) -> None:
        """Load a previously serialized BM25 index and its source chunks."""
        target = Path(path or self.index_path)
        with target.open("rb") as index_file:
            self._bm25, self._chunks = pickle.load(index_file)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return filtered BM25 hits, ranked before truncating to ``top_k``."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._bm25 is None:
            raise ValueError("index has not been built or loaded")
        started = time.perf_counter()
        query_tokens = tokenize_vi(query)
        scores = self._bm25.get_scores(query_tokens)
        eligible = [
            (index, float(score))
            for index, score in enumerate(scores)
            if _matches(self._chunks[index], filters)
        ]
        eligible.sort(key=lambda item: item[1], reverse=True)
        result = [_to_hit(self._chunks[index], score) for index, score in eligible[:top_k]]
        LOGGER.info("sparse search latency_ms=%.2f", (time.perf_counter() - started) * 1000)
        return result


def _matches(chunk: LegalChunk, filters: dict[str, str | int | list[str]] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(chunk, key, chunk.metadata.get(key))
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _to_hit(chunk: LegalChunk, score: float) -> RetrievalHit:
    metadata: dict[str, Any] = dict(chunk.metadata)
    for field in ("chapter", "section", "effective_date", "point"):
        value = getattr(chunk, field)
        if value is not None:
            metadata.setdefault(field, value)
    return RetrievalHit(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text,
                        score=score, sparse_score=score, source=chunk.source,
                        law_name=chunk.law_name, article=chunk.article,
                        clause=chunk.clause, metadata=metadata)


_default_retriever: BM25Retriever | None = None


def search(query: str, top_k: int,
           filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]:
    """Search the lazily loaded default BM25 index."""
    global _default_retriever
    if _default_retriever is None:
        config = load_config()
        index_path = config.get("sparse", {}).get("bm25_index_path")
        _default_retriever = BM25Retriever(index_path)
        _default_retriever.load()
    return _default_retriever.search(query, top_k, filters)
