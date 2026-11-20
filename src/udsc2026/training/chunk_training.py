"""Inference-matched raw-chunk sampling for conservative Task1 fine-tuning V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RankBands:
    """Inclusive raw retrieval rank boundaries for negative sampling."""

    hard_end: int = 20
    medium_end: int = 100

    def band(self, rank: int) -> str:
        if rank <= self.hard_end:
            return "hard"
        if rank <= self.medium_end:
            return "medium"
        return "easy"


def _compact_chunk(hit: Mapping[str, Any], fallback_rank: int) -> dict[str, Any]:
    doc_id = str(hit.get("doc_id", hit.get("document_id", ""))).strip()
    chunk_id = str(hit.get("chunk_id", hit.get("evidence_id", ""))).strip()
    text = str(hit.get("text", hit.get("evidence", ""))).strip()
    rank = int(hit.get("dense_rank", hit.get("rank", fallback_rank)))
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    if not doc_id or not chunk_id or not text or rank < 1:
        raise ValueError(
            "raw training chunks require doc_id/chunk_id/text/positive rank"
        )
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "dense_rank": rank,
        "dense_score": float(hit.get("dense_score", hit.get("score", 0.0))),
        "law_name": str(hit.get("law_name", metadata.get("law_name", ""))),
        "parent_id": str(hit.get("parent_id", metadata.get("parent_id", ""))),
    }


def build_training_record(
    *,
    query_id: str,
    question: str,
    gold_documents: Sequence[str],
    raw_hits: Sequence[Mapping[str, Any]],
    bands: RankBands = RankBands(),
    positives_per_query: int = 3,
    negatives_per_band: int = 3,
    include_same_law: bool = True,
) -> dict[str, Any] | None:
    """Create one balanced query group from exactly the inference candidate pool."""

    if positives_per_query < 1 or negatives_per_band < 1:
        raise ValueError("sampling limits must be positive")
    gold = set(map(str, gold_documents))
    chunks = [_compact_chunk(hit, rank) for rank, hit in enumerate(raw_hits, 1)]
    positives = [chunk for chunk in chunks if chunk["doc_id"] in gold][
        :positives_per_query
    ]
    if not positives:
        return None
    buckets: dict[str, list[dict[str, Any]]] = {
        "hard": [],
        "medium": [],
        "easy": [],
        "same_law": [],
    }
    positive_laws = {chunk["law_name"] for chunk in positives if chunk["law_name"]}
    seen_negative_docs: dict[str, set[str]] = {name: set() for name in buckets}
    for chunk in chunks:
        if chunk["doc_id"] in gold:
            continue
        band = bands.band(int(chunk["dense_rank"]))
        if (
            len(buckets[band]) < negatives_per_band
            and chunk["doc_id"] not in seen_negative_docs[band]
        ):
            buckets[band].append(chunk)
            seen_negative_docs[band].add(chunk["doc_id"])
        if (
            include_same_law
            and chunk["law_name"] in positive_laws
            and len(buckets["same_law"]) < negatives_per_band
            and chunk["doc_id"] not in seen_negative_docs["same_law"]
        ):
            buckets["same_law"].append(chunk)
            seen_negative_docs["same_law"].add(chunk["doc_id"])
    return {
        "query_id": str(query_id),
        "question": question,
        "gold_documents": sorted(gold),
        "positive_chunks": positives,
        "negative_chunks": buckets,
        "representation": "retrieved_raw_chunks_matching_inference_v1",
    }
