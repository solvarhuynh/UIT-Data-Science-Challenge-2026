"""Standalone BM25 sparse retriever for legal document chunks."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rank_bm25 import BM25Okapi

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.config import load_config
from udsc2026.retrieval.sparse.tokenizer import tokenize_vi

LOGGER = logging.getLogger(__name__)

_INDEX_SCHEMA_VERSION = 1
_INDEX_FILE_MODE = 0o644


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _tokenize_for_bm25(text: str) -> list[str]:
    """Return searchable, case-insensitive tokens with stable whitespace."""
    normalized_tokens: list[str] = []
    for token in tokenize_vi(text):
        normalized = " ".join(token.split()).casefold()
        if normalized and any(character.isalnum() for character in normalized):
            normalized_tokens.append(normalized)
    return normalized_tokens


class BM25Retriever:
    """Build, persist, and search a Vietnamese BM25 index over legal chunks."""

    def __init__(self, index_path: str | None = None) -> None:
        """Initialize BM25 state and resolve its persistence path."""
        resolved_path = (
            index_path
            or os.getenv("BM25_INDEX_PATH")
            or "data/vector_store/bm25/index.json"
        )
        if not isinstance(resolved_path, str) or not resolved_path.strip():
            raise ValueError("BM25 index_path must be a non-empty string")
        self.index_path = resolved_path.strip()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[LegalChunk] = []
        self._tokenized_chunks: list[list[str]] = []

    def build_index(self, chunks: list[LegalChunk]) -> None:
        """Build an in-memory BM25 index while retaining original chunk contracts."""
        if not chunks:
            raise ValueError("chunks must not be empty")
        chunk_ids: list[str] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, LegalChunk):
                raise TypeError(f"chunks[{index}] must be a LegalChunk")
            if not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{index}].chunk_id must not be blank")
            if not chunk.text.strip():
                raise ValueError(f"chunks[{index}].text must not be blank")
            chunk_ids.append(chunk.chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique")
        tokenized_chunks = [_tokenize_for_bm25(chunk.text) for chunk in chunks]
        empty_token_index = next(
            (index for index, tokens in enumerate(tokenized_chunks) if not tokens),
            None,
        )
        if empty_token_index is not None:
            raise ValueError(
                f"chunks[{empty_token_index}].text has no searchable tokens"
            )
        bm25 = BM25Okapi(tokenized_chunks)
        self._chunks = list(chunks)
        self._tokenized_chunks = tokenized_chunks
        self._bm25 = bm25

    def save(self, path: str | None = None) -> None:
        """Atomically persist versioned source chunks without unsafe deserialization."""
        if self._bm25 is None:
            raise ValueError("index has not been built")
        target = Path(path or self.index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
            descriptor = -1
            with stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, _INDEX_FILE_MODE)
            os.replace(temporary_name, target)
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def load(self, path: str | None = None) -> None:
        """Validate a versioned JSON index and deterministically rebuild BM25."""
        target = Path(path or self.index_path)
        if not target.is_file():
            raise FileNotFoundError(f"BM25 index file does not exist: {target}")
        try:
            with target.open("r", encoding="utf-8-sig") as index_file:
                payload: object = json.load(
                    index_file,
                    parse_constant=_reject_json_constant,
                )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid BM25 index JSON in {target}: {exc.msg}") from exc
        except ValueError as exc:
            raise ValueError(f"invalid BM25 index JSON in {target}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("BM25 index root must be a JSON object")
        if set(payload) != {"schema_version", "chunks"}:
            raise ValueError(
                "BM25 index must contain only 'schema_version' and 'chunks'"
            )
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _INDEX_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported BM25 index schema version: {schema_version!r}"
            )
        raw_chunks = payload["chunks"]
        if not isinstance(raw_chunks, list):
            raise ValueError("BM25 index 'chunks' must be a JSON array")
        try:
            chunks = [
                LegalChunk.model_validate(raw_chunk, strict=True)
                for raw_chunk in raw_chunks
            ]
        except ValidationError as exc:
            raise ValueError(
                f"invalid LegalChunk in BM25 index {target}: {exc}"
            ) from exc
        self.build_index(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return filtered BM25 hits, ranked before truncating to ``top_k``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self._bm25 is None:
            raise ValueError("index has not been built or loaded")
        started = time.perf_counter()
        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        eligible = [
            (index, float(score))
            for index, score in enumerate(scores)
            if _matches(self._chunks[index], filters)
            and query_token_set.intersection(self._tokenized_chunks[index])
        ]
        if not eligible:
            return []
        eligible.sort(key=lambda item: item[1], reverse=True)
        LOGGER.info(
            "sparse search latency_ms=%.2f", (time.perf_counter() - started) * 1000
        )
        results = [
            _to_hit(self._chunks[index], score) for index, score in eligible[:top_k]
        ]
        if len(results) > top_k:
            raise RuntimeError(f"BM25 returned {len(results)} hits for top_k={top_k}")
        return results

    @property
    def indexed_chunk_count(self) -> int:
        """Return how many chunks are represented by the current index."""

        return len(self._chunks) if self._bm25 is not None else 0


def _matches(
    chunk: LegalChunk, filters: dict[str, str | int | list[str]] | None
) -> bool:
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
    return RetrievalHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        score=score,
        sparse_score=score,
        source=chunk.source,
        law_name=chunk.law_name,
        article=chunk.article,
        clause=chunk.clause,
        metadata=metadata,
    )


_default_retriever: BM25Retriever | None = None


def search(
    query: str, top_k: int, filters: dict[str, str | int | list[str]] | None = None
) -> list[RetrievalHit]:
    """Search the lazily loaded default BM25 index."""
    global _default_retriever
    if _default_retriever is None:
        config = load_config()
        index_path = config.get("sparse", {}).get("bm25_index_path")
        _default_retriever = BM25Retriever(index_path)
        _default_retriever.load()
    return _default_retriever.search(query, top_k, filters)
