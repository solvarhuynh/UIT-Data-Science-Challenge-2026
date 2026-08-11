"""Synthetic P5 document-union and evidence-selection regressions."""

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_ir_document_candidates import (
    evidence_from_hits,
    final_document_ranking,
    final_submission_documents,
    format_document_evidence,
    rerank_document_candidates,
    union_document_candidates,
)


def _hit(
    index: int, doc_id: str, *, score: float, parent: bool = False
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{index}",
        parent_id=f"parent-{doc_id}" if parent else None,
        doc_id=doc_id,
        text=f"evidence {index}",
        score=score,
        rank=index,
        law_name="Luật mẫu",
        article="Điều 17",
        clause="Khoản 2",
    )


def test_200_child_chunks_from_three_docs_make_only_three_candidates() -> None:
    hits = [
        _hit(index, f"doc-{index % 3}", score=float(201 - index))
        for index in range(1, 201)
    ]

    candidates = union_document_candidates(
        {"child_dense": evidence_from_hits("child_dense", hits)}, candidate_depth=200
    )

    assert len(candidates) == 3
    assert {item.doc_id for item in candidates} == {"doc-0", "doc-1", "doc-2"}


def test_document_union_preserves_sources_and_top2_evidence_deterministically() -> None:
    child = [_hit(3, "doc-a", score=0.4), _hit(1, "doc-a", score=0.9)]
    parent = [_hit(2, "doc-a", score=0.8, parent=True)]
    candidate = union_document_candidates(
        {
            "child_dense": evidence_from_hits("child_dense", child),
            "parent_dense": evidence_from_hits("parent_dense", parent, is_parent=True),
        },
        candidate_depth=50,
        evidence_limit=2,
    )[0]

    assert candidate.source_ranks == {"child_dense": 1, "parent_dense": 2}
    assert candidate.best_child_rank == 1 and candidate.best_parent_rank == 2
    assert candidate.evidence_chunk_ids == ("chunk-1", "parent-doc-a")
    text = format_document_evidence(candidate)
    assert "[TÊN VĂN BẢN]" in text and "[EVIDENCE 2]" in text


def test_final_output_is_distinct_and_limited_to_five() -> None:
    hits = [_hit(index, f"doc-{index}", score=float(index)) for index in range(1, 8)]
    candidates = union_document_candidates(
        {"child_dense": evidence_from_hits("child_dense", hits)}, candidate_depth=50
    )

    class FakeReranker:
        def score_documents(self, query, values):
            return list(reversed(range(len(values))))

    ranked = final_document_ranking(
        rerank_document_candidates("q", candidates, FakeReranker())
    )
    output = final_submission_documents(ranked)
    assert len(output) == 5
    assert len(output) == len(set(output))
