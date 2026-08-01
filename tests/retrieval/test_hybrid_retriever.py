from unittest.mock import Mock, patch

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever


def test_hybrid_uses_candidate_k_filters_min_score_and_top_k():
    dense = Mock()
    sparse = Mock()
    dense.search.return_value = [
        RetrievalHit(chunk_id="a", doc_id="d", text="a", dense_score=1.0)
    ]
    sparse.search.return_value = [
        RetrievalHit(chunk_id="a", doc_id="d", text="a", sparse_score=1.0)
    ]
    retriever = HybridRetriever(dense, sparse, candidate_k=7, min_score=0.8)
    with patch(
        "udsc2026.retrieval.hybrid.hybrid_retriever.fuse_scores",
        return_value=[
            RetrievalHit(chunk_id="a", doc_id="d", text="a", final_score=0.9),
            RetrievalHit(chunk_id="b", doc_id="d", text="b", final_score=0.2),
        ],
    ) as fuse:
        result = retriever.search("q", 1, {"law_name": "X"})
    dense.search.assert_called_once_with("q", 7, {"law_name": "X"})
    sparse.search.assert_called_once_with("q", 7, {"law_name": "X"})
    fuse.assert_called_once()
    assert [item.chunk_id for item in result] == ["a"]
