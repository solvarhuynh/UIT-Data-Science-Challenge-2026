from unittest.mock import Mock

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.dense.dense_retriever import DenseRetriever


def test_dense_search_encodes_then_searches():
    embedding = Mock()
    embedding.embed_query.return_value = [0.1, 0.2]
    vector_db = Mock()
    vector_db.search.return_value = [
        RetrievalHit(chunk_id="c", doc_id="d", text="x", score=0.8)
    ]
    result = DenseRetriever(embedding, vector_db).search(
        "query", 3, {"article": "Điều 1"}
    )
    embedding.embed_query.assert_called_once_with("query")
    vector_db.search.assert_called_once_with([0.1, 0.2], 3, {"article": "Điều 1"})
    assert result[0].dense_score == 0.8
    assert result[0].rank == 1


def test_dense_search_rejects_backend_overflow():
    embedding = Mock()
    embedding.embed_query.return_value = [0.1, 0.2]
    vector_db = Mock()
    vector_db.search.return_value = [
        RetrievalHit(chunk_id=str(index), doc_id="d", text="x", score=0.8)
        for index in range(6)
    ]
    with pytest.raises(RuntimeError, match="top_k=5"):
        DenseRetriever(embedding, vector_db).search("query", 5)
