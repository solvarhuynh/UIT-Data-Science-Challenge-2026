"""P4 pure retrieval representation and document-fusion contracts."""

from udsc2026.contracts import LegalChunk, LegalParent
from udsc2026.retrieval.multigranularity import (
    CHILD_META_V1,
    PARENT_META_V1,
    ParentDenseHit,
    build_child_retrieval_text,
    build_index_manifest,
    build_parent_retrieval_text,
    fuse_child_parent_documents,
    parent_hits_to_document_ranking,
    separate_collection_name,
)


def test_retrieval_text_builders_preserve_canonical_text() -> None:
    chunk = LegalChunk(
        chunk_id="c1",
        doc_id="d1",
        text="Nội dung khoản.",
        law_name="Luật mẫu",
        article="Điều 17",
        clause="Khoản 2",
        point="Điểm a",
    )
    parent = LegalParent(
        parent_id="p1",
        doc_id="d1",
        text="Toàn văn điều.",
        law_name="Luật mẫu",
        chapter="Chương I",
        section="Mục 1",
        article="Điều 17",
    )

    child_text = build_child_retrieval_text(chunk, CHILD_META_V1)
    parent_text = build_parent_retrieval_text(parent, PARENT_META_V1)

    assert chunk.text == "Nội dung khoản."
    assert "Law: Luật mẫu" in child_text and "Clause: Khoản 2" in child_text
    assert parent.text == "Toàn văn điều."
    assert "Chapter: Chương I" in parent_text and "Article: Điều 17" in parent_text


def test_parent_results_fuse_only_document_ids() -> None:
    hits = [
        ParentDenseHit("p2", "doc-b", 0.9, 1),
        ParentDenseHit("p3", "doc-b", 0.8, 2),
        ParentDenseHit("p1", "doc-a", 0.7, 3),
    ]

    assert parent_hits_to_document_ranking(hits) == ["doc-b", "doc-a"]
    assert fuse_child_parent_documents(["doc-a", "doc-c"], hits) == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]


def test_p4_index_manifest_uses_separate_versioned_collection() -> None:
    manifest = build_index_manifest(
        representation_version=PARENT_META_V1,
        corpus_hash="a" * 64,
        model_hash="b" * 64,
        embedding_dimension=1024,
        record_count=12,
        git_commit="commit",
    )

    assert manifest["collection_name"] == separate_collection_name(PARENT_META_V1)
    assert manifest["collection_name"].startswith("legal_parents_dek21_v2_")
    assert manifest["collection_name"] != "legal_chunks"
