from unittest.mock import Mock, patch

import pytest

from udsc2026.contracts import LegalChunk


def chunk(chunk_id="c1", text="Điều 10 hợp đồng", **kwargs):
    return LegalChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        law_name="Bộ luật Lao động",
        article="Điều 10",
        clause="Khoản 1",
        source="sample",
        **kwargs,
    )


def test_qdrant_upsert_and_search(mock_qdrant):
    from udsc2026.infrastructure.vector_db.qdrant_adapter import QdrantAdapter

    client = mock_qdrant.return_value
    result = Mock(
        score=0.91,
        payload={
            "chunk_id": "c1",
            "doc_id": "doc-1",
            "text": "Điều 10 hợp đồng",
            "law_name": "Bộ luật Lao động",
            "article": "Điều 10",
            "clause": "Khoản 1",
            "source": "sample",
            "metadata": {},
        },
    )
    client.search.return_value = [result]
    with patch(
        "udsc2026.infrastructure.vector_db.qdrant_adapter.QdrantClient",
        return_value=client,
    ):
        adapter = QdrantAdapter("http://qdrant", "legal_chunks")
        adapter.upsert([chunk()], [[1.0, 0.0, 0.0, 0.0]])
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["chunk_id"] == "c1"
        assert point.payload["doc_id"] == "doc-1"
        assert point.payload["law_name"] == "Bộ luật Lao động"
        assert point.payload["article"] == "Điều 10"
        assert point.payload["clause"] == "Khoản 1"
        hits = adapter.search([1.0, 0.0, 0.0, 0.0], 1)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score == pytest.approx(0.91)
    assert hits[0].law_name == "Bộ luật Lao động"


def test_faiss_round_trip_and_metadata(tmp_path):
    pytest.importorskip("faiss")
    from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

    adapter = FaissAdapter(str(tmp_path), "legal_chunks")
    adapter.create_collection("legal_chunks", 4)
    adapter.upsert(
        [chunk("near"), chunk("far", text="Thuế thu nhập cá nhân")],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
    )
    hits = adapter.search([0.99, 0.01, 0, 0], 1)
    assert hits[0].chunk_id == "near"
    assert hits[0].law_name == "Bộ luật Lao động"
    assert hits[0].metadata["clause"] == "Khoản 1"
