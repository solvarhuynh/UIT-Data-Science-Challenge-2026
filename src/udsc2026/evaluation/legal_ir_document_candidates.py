"""Document-level candidate union and evidence selection for LegalIR P5."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.reranker import RerankerClient


@dataclass(frozen=True)
class DocumentEvidence:
    """One child or parent evidence item before document union."""

    doc_id: str
    evidence_id: str
    text: str
    source: str
    rank: int
    score: float | None
    is_parent: bool = False
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LegalIRDocumentCandidate:
    """Typed final-stage candidate, capped to one slot per document."""

    doc_id: str
    source_ranks: Mapping[str, int]
    source_scores: Mapping[str, float]
    evidence_chunk_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    best_child_rank: int | None
    best_parent_rank: int | None
    first_seen_rank: int
    metadata: Mapping[str, object]
    rerank_score: float | None = None


def evidence_from_hits(
    source: str, hits: Sequence[RetrievalHit], *, is_parent: bool = False
) -> list[DocumentEvidence]:
    """Adapt unchanged retrieval hits without changing their shared contract."""

    if not source.strip():
        raise ValueError("source must not be blank")
    return [
        DocumentEvidence(
            doc_id=hit.doc_id,
            evidence_id=hit.parent_id if is_parent and hit.parent_id else hit.chunk_id,
            text=hit.text,
            source=source,
            rank=hit.rank if hit.rank is not None else index,
            score=_score(hit),
            is_parent=is_parent,
            metadata={
                "law_name": hit.law_name,
                "article": hit.article,
                "clause": hit.clause,
                **hit.metadata,
            },
        )
        for index, hit in enumerate(hits, 1)
    ]


def union_document_candidates(
    sources: Mapping[str, Sequence[DocumentEvidence]],
    *,
    candidate_depth: int = 200,
    evidence_limit: int = 2,
) -> list[LegalIRDocumentCandidate]:
    """Union all retrieval sources by doc ID before expensive reranking."""

    if candidate_depth not in (50, 100, 150, 200, 300, 500):
        raise ValueError("candidate_depth must be one of 50, 100, 150, 200, 300, 500")
    if evidence_limit not in (1, 2):
        raise ValueError("evidence_limit must be 1 or 2")
    grouped: dict[str, list[DocumentEvidence]] = {}
    for source, items in sources.items():
        if not source.strip():
            raise ValueError("source mapping contains a blank name")
        for item in items:
            if item.source != source:
                raise ValueError("evidence source must match its mapping key")
            if item.rank <= 0:
                raise ValueError("evidence rank must be positive")
            grouped.setdefault(item.doc_id, []).append(item)
    candidates = [
        _candidate(doc_id, items, evidence_limit) for doc_id, items in grouped.items()
    ]
    candidates.sort(key=lambda item: (item.first_seen_rank, item.doc_id))
    return candidates[:candidate_depth]


def format_document_evidence(candidate: LegalIRDocumentCandidate) -> str:
    """Render stable BGE/Qwen-compatible document evidence, skipping empty fields."""

    metadata = candidate.metadata
    sections: list[tuple[str, str]] = []
    name = _first_text(metadata, "law_name", "title", "name")
    if name:
        sections.append(("TÊN VĂN BẢN", name))
    structure = "\n".join(
        value
        for key in ("article", "clause", "point")
        if (value := _first_text(metadata, key))
    )
    if structure:
        sections.append(("CẤU TRÚC", structure))
    for index, text in enumerate(candidate.evidence_texts, 1):
        if text.strip():
            sections.append((f"EVIDENCE {index}", text.strip()))
    return "\n\n".join(f"[{title}]\n{value}" for title, value in sections)


class DocumentReranker(Protocol):
    """Adapter boundary for BGE today and Qwen-based rerankers later."""

    def score_documents(
        self, query: str, candidates: Sequence[LegalIRDocumentCandidate]
    ) -> Sequence[float]:
        """Return one document relevance score per candidate."""

        ...


class CrossEncoderDocumentReranker:
    """BGE-compatible adapter over the existing ``RerankerClient`` protocol."""

    def __init__(self, client: RerankerClient) -> None:
        """Wrap the existing production reranker client."""

        self._client = client

    def score_documents(
        self, query: str, candidates: Sequence[LegalIRDocumentCandidate]
    ) -> Sequence[float]:
        """Score deterministic formatted evidence for each document."""

        return tuple(
            float(score)
            for score in self._client.score(
                query, tuple(format_document_evidence(item) for item in candidates)
            )
        )


def rerank_document_candidates(
    query: str,
    candidates: Sequence[LegalIRDocumentCandidate],
    reranker: DocumentReranker,
) -> list[LegalIRDocumentCandidate]:
    """Attach reranker scores while preserving the document candidate pool."""

    scores = list(reranker.score_documents(query, candidates))
    if len(scores) != len(candidates):
        raise ValueError("reranker must return one score per document candidate")
    updated: list[LegalIRDocumentCandidate] = []
    for candidate, score in zip(candidates, scores):
        if not math.isfinite(score):
            raise ValueError("reranker scores must be finite real numbers")
        updated.append(replace(candidate, rerank_score=float(score)))
    return updated


def final_document_ranking(
    candidates: Sequence[LegalIRDocumentCandidate], *, rrf_k: int = 60
) -> list[LegalIRDocumentCandidate]:
    """Fuse reranker and all source ranks while retaining an internal deep pool."""

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    rerank_order = sorted(
        candidates,
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else -math.inf),
            item.doc_id,
        ),
    )
    rerank_ranks = {item.doc_id: index for index, item in enumerate(rerank_order, 1)}

    def score(item: LegalIRDocumentCandidate) -> float:
        ranks = list(item.source_ranks.values())
        if item.rerank_score is not None:
            ranks.append(rerank_ranks[item.doc_id])
        return sum(1.0 / (rrf_k + rank) for rank in ranks)

    return sorted(
        candidates, key=lambda item: (-score(item), item.first_seen_rank, item.doc_id)
    )


def final_submission_documents(
    candidates: Sequence[LegalIRDocumentCandidate],
) -> list[str]:
    """Return the official top five distinct document IDs only at final output."""

    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if candidate.doc_id not in seen:
            seen.add(candidate.doc_id)
            result.append(candidate.doc_id)
        if len(result) == 5:
            break
    return result


def _candidate(
    doc_id: str, items: Sequence[DocumentEvidence], evidence_limit: int
) -> LegalIRDocumentCandidate:
    by_source: dict[str, list[DocumentEvidence]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    source_ranks = {
        name: min(item.rank for item in values) for name, values in by_source.items()
    }
    source_scores = {
        name: max((item.score or 0.0) for item in values)
        for name, values in by_source.items()
    }
    evidence = sorted(
        items,
        key=lambda item: (
            -(item.score if item.score is not None else -math.inf),
            item.rank,
            item.evidence_id,
        ),
    )[:evidence_limit]
    metadata: dict[str, object] = {}
    for item in sorted(items, key=lambda item: (item.rank, item.evidence_id)):
        metadata.update(
            {
                key: value
                for key, value in (item.metadata or {}).items()
                if value not in (None, "")
            }
        )
    child = [item.rank for item in items if not item.is_parent]
    parent = [item.rank for item in items if item.is_parent]
    return LegalIRDocumentCandidate(
        doc_id,
        source_ranks,
        source_scores,
        tuple(item.evidence_id for item in evidence),
        tuple(item.text for item in evidence),
        min(child) if child else None,
        min(parent) if parent else None,
        min(item.rank for item in items),
        metadata,
    )


def _score(hit: RetrievalHit) -> float | None:
    for value in (hit.rerank_score, hit.final_score, hit.dense_score, hit.score):
        if value is not None:
            return float(value)
    return None


def _first_text(metadata: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
