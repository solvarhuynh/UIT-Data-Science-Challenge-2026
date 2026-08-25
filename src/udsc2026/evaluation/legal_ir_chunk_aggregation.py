"""Deterministic multi-chunk to document aggregation for LegalIR.

The module operates on already retrieved/scored chunks.  It deliberately keeps
every supporting chunk attached to the resulting document so diagnostics can
trace a document score back to the raw candidate cache.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

_RANK_METHOD_RE = re.compile(r"^rank_cap(?P<cap>[1-9]\d*)_k(?P<k>\d+)$")
_VALUE_METHOD_RE = re.compile(
    r"^(?P<source>bge|dense)_(?P<kind>max|mean_top[235]|logsumexp_top[235])$"
)


@dataclass(frozen=True)
class ChunkEvidence:
    """One raw chunk candidate with the fields needed by all aggregators."""

    doc_id: str
    chunk_id: str
    dense_rank: int
    dense_score: float
    bge_score: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, fallback_rank: int
    ) -> "ChunkEvidence":
        """Validate and normalize a raw JSON hit without discarding metadata."""

        doc_id = str(value.get("doc_id", value.get("document_id", ""))).strip()
        chunk_id = str(value.get("chunk_id", value.get("evidence_id", ""))).strip()
        if not doc_id or not chunk_id:
            raise ValueError("every chunk needs non-empty doc_id and chunk_id")
        rank = int(value.get("dense_rank", value.get("rank", fallback_rank)))
        if rank < 1:
            raise ValueError("dense_rank must be positive")
        dense_score = float(value.get("dense_score", value.get("score", 0.0)))
        raw_bge = value.get("bge_score", value.get("rerank_score"))
        bge_score = None if raw_bge is None else float(raw_bge)
        if not math.isfinite(dense_score) or (
            bge_score is not None and not math.isfinite(bge_score)
        ):
            raise ValueError("chunk scores must be finite")
        return cls(
            doc_id=doc_id,
            chunk_id=chunk_id,
            dense_rank=rank,
            dense_score=dense_score,
            bge_score=bge_score,
            provenance=dict(value),
        )


@dataclass(frozen=True)
class DocumentAggregate:
    """One ranked document and the raw evidence that produced its score."""

    doc_id: str
    score: float
    best_dense_rank: int
    supporting_chunks: tuple[ChunkEvidence, ...]
    used_chunk_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable diagnostic representation."""

        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "best_dense_rank": self.best_dense_rank,
            "supporting_chunk_count": len(self.supporting_chunks),
            "used_chunk_ids": list(self.used_chunk_ids),
            "supporting_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "dense_rank": chunk.dense_rank,
                    "dense_score": chunk.dense_score,
                    "bge_score": chunk.bge_score,
                    "provenance": dict(chunk.provenance),
                }
                for chunk in self.supporting_chunks
            ],
        }


def _chunks(values: Iterable[ChunkEvidence | Mapping[str, Any]]) -> list[ChunkEvidence]:
    output: list[ChunkEvidence] = []
    seen: set[str] = set()
    for fallback_rank, value in enumerate(values, 1):
        chunk = (
            value
            if isinstance(value, ChunkEvidence)
            else ChunkEvidence.from_mapping(value, fallback_rank=fallback_rank)
        )
        if chunk.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
        output.append(chunk)
    if not output:
        raise ValueError("cannot aggregate an empty chunk ranking")
    return output


def _source_score(chunk: ChunkEvidence, source: str) -> float:
    if source == "dense":
        return chunk.dense_score
    if source == "bge":
        if chunk.bge_score is None:
            raise ValueError(f"chunk {chunk.chunk_id!r} has no BGE score")
        return chunk.bge_score
    raise ValueError(f"unsupported score source {source!r}")


def _ordered(chunks: Sequence[ChunkEvidence], source: str) -> list[ChunkEvidence]:
    if source == "dense":
        return sorted(
            chunks, key=lambda item: (item.dense_rank, item.doc_id, item.chunk_id)
        )
    return sorted(
        chunks,
        key=lambda item: (
            -_source_score(item, source),
            item.dense_rank,
            item.doc_id,
            item.chunk_id,
        ),
    )


def _value_score(values: Sequence[float], kind: str) -> tuple[float, int]:
    ordered = sorted(values, reverse=True)
    if kind == "max":
        return ordered[0], 1
    prefix, raw_count = kind.rsplit("top", 1)
    count = int(raw_count)
    selected = ordered[:count]
    if prefix == "mean_":
        return sum(selected) / len(selected), len(selected)
    if prefix == "logsumexp_":
        maximum = max(selected)
        return maximum + math.log(
            sum(math.exp(value - maximum) for value in selected)
        ), len(selected)
    raise ValueError(f"unsupported aggregation kind {kind!r}")


def aggregate_chunks(
    values: Iterable[ChunkEvidence | Mapping[str, Any]],
    *,
    method: str,
    rank_source: str = "bge",
) -> list[DocumentAggregate]:
    """Aggregate raw chunks using a named, deterministic document method.

    Supported names are ``rank_capN_kK`` and the score methods
    ``bge_max``, ``bge_mean_top2/top3``, ``bge_logsumexp_top2/top3/top5`` plus
    their dense counterparts (including ``dense_mean_top5``).
    """

    chunks = _chunks(values)
    rank_match = _RANK_METHOD_RE.fullmatch(method)
    value_match = _VALUE_METHOD_RE.fullmatch(method)
    if rank_match:
        cap = int(rank_match.group("cap"))
        rank_k = int(rank_match.group("k"))
        source = rank_source
        ordered = _ordered(chunks, source)
        scores: dict[str, float] = {}
        used: dict[str, list[ChunkEvidence]] = {}
        first_rank: dict[str, int] = {}
        for rank, chunk in enumerate(ordered, 1):
            first_rank.setdefault(chunk.doc_id, rank)
            selected = used.setdefault(chunk.doc_id, [])
            if len(selected) >= cap:
                continue
            selected.append(chunk)
            scores[chunk.doc_id] = scores.get(chunk.doc_id, 0.0) + 1.0 / (rank_k + rank)
    elif value_match:
        source = value_match.group("source")
        kind = value_match.group("kind")
        ordered = _ordered(chunks, source)
        by_doc: dict[str, list[ChunkEvidence]] = {}
        first_rank = {}
        for rank, chunk in enumerate(ordered, 1):
            first_rank.setdefault(chunk.doc_id, rank)
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        scores, used = {}, {}
        for doc_id, supporting in by_doc.items():
            score, count = _value_score(
                [_source_score(chunk, source) for chunk in supporting], kind
            )
            scores[doc_id] = score
            used[doc_id] = supporting[:count]
    else:
        raise ValueError(f"unsupported chunk aggregation method {method!r}")

    supporting_by_doc: dict[str, tuple[ChunkEvidence, ...]] = {}
    for chunk in chunks:
        supporting_by_doc.setdefault(chunk.doc_id, tuple())
        supporting_by_doc[chunk.doc_id] += (chunk,)
    aggregates = [
        DocumentAggregate(
            doc_id=doc_id,
            score=scores[doc_id],
            best_dense_rank=min(
                chunk.dense_rank for chunk in supporting_by_doc[doc_id]
            ),
            supporting_chunks=tuple(
                sorted(
                    supporting_by_doc[doc_id],
                    key=lambda item: (item.dense_rank, item.chunk_id),
                )
            ),
            used_chunk_ids=tuple(chunk.chunk_id for chunk in used[doc_id]),
        )
        for doc_id in scores
    ]
    return sorted(
        aggregates,
        key=lambda item: (-item.score, first_rank[item.doc_id], item.doc_id),
    )


def top_document_ids(
    values: Iterable[ChunkEvidence | Mapping[str, Any]],
    *,
    method: str,
    rank_source: str = "bge",
    limit: int = 5,
) -> list[str]:
    """Return at most ``limit`` distinct document IDs from an aggregation."""

    if limit < 1:
        raise ValueError("limit must be positive")
    result = [
        item.doc_id
        for item in aggregate_chunks(values, method=method, rank_source=rank_source)[
            :limit
        ]
    ]
    if len(result) != len(set(result)):
        raise AssertionError("document aggregation produced duplicate IDs")
    return result
