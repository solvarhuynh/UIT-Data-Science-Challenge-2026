from udsc2026.contracts import LegalChunk
from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever


def chunks():
    return [
        LegalChunk(
            chunk_id="labor", doc_id="d1", text="hợp đồng lao động và người lao động"
        ),
        LegalChunk(
            chunk_id="tax", doc_id="d2", text="thuế thu nhập cá nhân và kê khai thuế"
        ),
        LegalChunk(
            chunk_id="leave", doc_id="d3", text="nghỉ phép hằng năm của người lao động"
        ),
    ]


def test_bm25_ranks_matching_chunk_first():
    retriever = BM25Retriever()
    retriever.build_index(chunks())
    hits = retriever.search("hợp đồng lao động", 2)
    assert hits[0].chunk_id == "labor"
    assert hits[0].rank is None  # BM25 contract does not assign rank itself.


def test_bm25_save_load_round_trip(tmp_path):
    path = tmp_path / "bm25.pkl"
    original = BM25Retriever(str(path))
    original.build_index(chunks())
    original.save()
    loaded = BM25Retriever(str(path))
    loaded.load()
    assert loaded.search("thuế thu nhập cá nhân", 1)[0].chunk_id == "tax"
