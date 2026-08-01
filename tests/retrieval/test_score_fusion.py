from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores, normalize_scores


def hit(chunk_id, score, field, **kwargs):
    return RetrievalHit(
        chunk_id=chunk_id, doc_id="d", text="text", **{field: score}, **kwargs
    )


def test_normalize_scores_minmax_and_equal_values():
    values = normalize_scores(
        [hit("a", 2, "dense_score"), hit("b", 4, "dense_score")], "dense_score"
    )
    assert [item.dense_score for item in values] == [0.0, 1.0]
    equal = normalize_scores(
        [hit("a", 2, "dense_score"), hit("b", 2, "dense_score")], "dense_score"
    )
    assert [item.dense_score for item in equal] == [1.0, 1.0]


def test_fusion_dense_only_keeps_sparse_none():
    result = fuse_scores([hit("a", 0.8, "dense_score")], [])
    assert result[0].sparse_score is None
    assert result[0].hybrid_score == 0.5


def test_fusion_preserves_citation_metadata():
    dense = RetrievalHit(
        chunk_id="a",
        doc_id="d",
        text="text",
        law_name="Luật X",
        article="Điều 10",
        clause="Khoản 2",
        source="src",
        dense_score=0.8,
    )
    sparse = RetrievalHit(chunk_id="a", doc_id="d", text="text", sparse_score=0.7)
    result = fuse_scores([dense], [sparse])[0]
    assert (result.law_name, result.article, result.clause, result.source) == (
        "Luật X",
        "Điều 10",
        "Khoản 2",
        "src",
    )


def test_fusion_weights_change_order_and_assign_rank():
    dense = [hit("dense", 1.0, "dense_score"), hit("sparse", 0.2, "dense_score")]
    sparse = [hit("dense", 0.1, "sparse_score"), hit("sparse", 1.0, "sparse_score")]
    result = fuse_scores(dense, sparse, 0.8, 0.2)
    assert result[0].chunk_id == "dense"
    assert [item.rank for item in result] == [1, 2]
    assert result[0].final_score >= result[1].final_score
