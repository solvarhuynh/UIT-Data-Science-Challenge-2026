"""CPU-only regression tests for P3 lexical components."""

from __future__ import annotations

from udsc2026.evaluation.legal_ir_lexical import (
    LegalContext,
    build_bm25f_rankings,
    build_citation_rankings,
    build_knn_rankings,
    parse_legal_citations,
)


def test_citation_parser_accepts_unicode_case_and_spacing_variants() -> None:
    signals = parse_legal_citations(
        "ĐIỀU 17, khoản 2, ĐIỂM a; Nghị định 123 / 2020 / NĐ-CP; "
        "Thông tư số 10/2024/TT-BTC; Quyết định 123/QĐ-TTg"
    )

    assert signals.articles == {"17"}
    assert signals.clauses == {"2"}
    assert signals.points == {"a"}
    assert signals.instruments == {
        "nghịđịnh:123/2020/nđ-cp",
        "thôngtư:10/2024/tt-btc",
        "quyếtđịnh:123/qđ-ttg",
    }


def test_citation_ranking_uses_content_not_document_id() -> None:
    contexts = [
        LegalContext("opaque-a", "Văn bản khác."),
        LegalContext("opaque-b", "Theo Nghị định 123/2020/NĐ-CP tại Điều 17."),
    ]
    ranking = build_citation_rankings(
        {"q": "Điều 17 Nghị định 123/2020/NĐ-CP"}, contexts
    )

    assert ranking["q"] == ["opaque-b"]


def test_char_knn_and_bm25f_title_are_separate_components() -> None:
    questions = {"q": "thủ tục đăng ký xe"}
    labeled = [("source", "thủ tục đăng ký xe máy", ["doc-labeled"])]
    contexts = [
        LegalContext("body", "nội dung không liên quan", ""),
        LegalContext("title", "nội dung không liên quan", "Thủ tục đăng ký xe"),
    ]

    assert build_knn_rankings(
        questions, labeled, analyzer="char_wb", ngram_range=(3, 5)
    )["q"] == ["doc-labeled"]
    assert build_bm25f_rankings(questions, contexts, title_weight=0)["q"] == []
    assert build_bm25f_rankings(questions, contexts, title_weight=2)["q"] == ["title"]
