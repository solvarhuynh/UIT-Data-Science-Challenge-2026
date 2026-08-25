"""Anchor-preserving diagnostic fusion for V2 neural document scores."""
from __future__ import annotations

from typing import Any, Iterable


def fuse_query(
    baseline_top5: Iterable[str],
    scored_docs: Iterable[dict[str, Any]],
    threshold: float = 0.0,
    max_swaps: int = 1,
    max_neural_rank: int = 20,
    protect_top_ranks: int = 3,
) -> tuple[list[str], dict[str, Any]]:
    """Return one deterministic, at-most-one-swap anchor-preserving ranking."""
    baseline = [str(doc_id) for doc_id in baseline_top5]
    if len(baseline) != 5 or len(set(baseline)) != 5:
        raise ValueError("baseline top5 must contain five unique documents")
    docs = {str(row["doc_id"]): dict(row) for row in scored_docs}
    neural = sorted(
        (row for row in docs.values() if str(row["doc_id"]) not in set(baseline)),
        key=lambda row: (-float(row["neural_score"]), int(row.get("union_rank", 10**9)), str(row["doc_id"])),
    )
    neural = [row for row in neural if int(row.get("neural_rank", 10**9)) <= max_neural_rank]
    if not neural or max_swaps <= 0:
        return baseline, {"changed": False, "swaps": 0, "dropped_rank": None, "incoming_doc": None, "margin": None}
    incoming = neural[0]
    incoming_score = float(incoming["neural_score"])
    drop_candidates = list(range(4, protect_top_ranks - 1, -1))
    if not drop_candidates:
        drop_candidates = [4]
    for rank_index in drop_candidates:
        dropped_doc = baseline[rank_index]
        dropped_score = float(docs.get(dropped_doc, {}).get("neural_score", float("-inf")))
        margin = incoming_score - dropped_score
        if margin < float(threshold):
            continue
        fused = list(baseline)
        fused[rank_index] = str(incoming["doc_id"])
        return fused, {
            "changed": fused != baseline,
            "swaps": 1,
            "dropped_rank": rank_index + 1,
            "incoming_doc": str(incoming["doc_id"]),
            "margin": margin,
            "incoming_neural_rank": int(incoming.get("neural_rank", 10**9)),
            "incoming_union_rank": int(incoming.get("union_rank", 10**9)),
            "incoming_source_support": int(incoming.get("source_support", 0) or 0),
            "incoming_has_bge_support": bool(incoming.get("has_bge_support", False)),
        }
    return baseline, {"changed": False, "swaps": 0, "dropped_rank": None, "incoming_doc": None, "margin": None}
