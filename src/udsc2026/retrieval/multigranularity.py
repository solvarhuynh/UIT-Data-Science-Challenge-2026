"""Versioned Task 1 multi-granularity retrieval primitives.

The builders are pure: they derive embedding input text without modifying the
canonical :class:`LegalChunk` or :class:`LegalParent` records on disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from udsc2026.contracts import LegalChunk, LegalParent

CHILD_RAW_V1 = "child_raw_v1"
CHILD_META_V1 = "child_meta_v1"
PARENT_META_V1 = "parent_meta_v1"
RepresentationVersion = Literal["child_raw_v1", "child_meta_v1", "parent_meta_v1"]


def _value(record: LegalChunk | LegalParent, name: str) -> str:
    direct = getattr(record, name, None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    metadata = record.metadata.get(name)
    return metadata.strip() if isinstance(metadata, str) and metadata.strip() else ""


def _metadata_text(record: LegalChunk | LegalParent, fields: Sequence[str]) -> str:
    labels = {
        "law_name": "Law",
        "chapter": "Chapter",
        "section": "Section",
        "article": "Article",
        "clause": "Clause",
        "point": "Point",
    }
    return "\n".join(
        f"{labels[field]}: {value}"
        for field in fields
        if (value := _value(record, field))
    )


def build_child_retrieval_text(
    chunk: LegalChunk, representation_version: str = CHILD_RAW_V1
) -> str:
    """Return versioned child text while preserving ``chunk.text`` unchanged."""

    if representation_version == CHILD_RAW_V1:
        return chunk.text
    if representation_version != CHILD_META_V1:
        raise ValueError(f"unsupported child representation {representation_version!r}")
    prefix = _metadata_text(chunk, ("law_name", "article", "clause", "point"))
    return f"{prefix}\n\n{chunk.text}" if prefix else chunk.text


def build_parent_retrieval_text(
    parent: LegalParent, representation_version: str = PARENT_META_V1
) -> str:
    """Return parent embedding text with parent-level legal metadata."""

    if representation_version != PARENT_META_V1:
        raise ValueError(
            f"unsupported parent representation {representation_version!r}"
        )
    prefix = _metadata_text(parent, ("law_name", "chapter", "section", "article"))
    return f"{prefix}\n\n{parent.text}" if prefix else parent.text


def separate_collection_name(representation_version: RepresentationVersion) -> str:
    """Return a P4 collection name that cannot overwrite the legacy child index."""

    names = {
        CHILD_RAW_V1: "legal_chunks_dek21_v2_child_raw_v1",
        CHILD_META_V1: "legal_chunks_meta_dek21_v2_child_meta_v1",
        PARENT_META_V1: "legal_parents_dek21_v2_parent_meta_v1",
    }
    return names[representation_version]


@dataclass(frozen=True)
class ParentDenseHit:
    """A parent-index result; only ``doc_id`` crosses document-level fusion."""

    parent_id: str
    doc_id: str
    score: float
    rank: int


def parent_hits_to_document_ranking(hits: Sequence[ParentDenseHit]) -> list[str]:
    """Collapse ordered parent hits to first-occurrence document IDs."""

    result: list[str] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda item: (item.rank, -item.score, item.parent_id)):
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            result.append(hit.doc_id)
    return result


def fuse_child_parent_documents(
    child_documents: Sequence[str],
    parent_hits: Sequence[ParentDenseHit],
    *,
    rrf_k: int = 60,
) -> list[str]:
    """Fuse child and parent retrieval at document level; never emit parent IDs."""

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    parent_documents = parent_hits_to_document_ranking(parent_hits)
    scores: dict[str, float] = {}
    for ranking in (child_documents, parent_documents):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def retrieval_corpus_hash(records: Iterable[tuple[str, str]]) -> str:
    """Order-independent hash over record ID and derived representation text."""

    accumulator = bytearray(32)
    count = 0
    for record_id, text in records:
        digest = hashlib.sha256(
            json.dumps(
                [record_id, text], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).digest()
        for index, value in enumerate(digest):
            accumulator[index] ^= value
        count += 1
    if not count:
        raise ValueError("retrieval corpus cannot be empty")
    return bytes(accumulator).hex()


def build_index_manifest(
    *,
    representation_version: RepresentationVersion,
    corpus_hash: str,
    model_hash: str,
    embedding_dimension: int,
    record_count: int,
    git_commit: str | None,
) -> Mapping[str, object]:
    """Create the required separate-index manifest contract."""

    if embedding_dimension <= 0 or record_count <= 0:
        raise ValueError("embedding_dimension and record_count must be positive")
    return {
        "schema_version": "legal-ir-multigranularity-index-v1",
        "collection_name": separate_collection_name(representation_version),
        "representation_version": representation_version,
        "corpus_hash": corpus_hash,
        "model_hash": model_hash,
        "embedding_dimension": embedding_dimension,
        "record_count": record_count,
        "git_commit": git_commit,
    }
