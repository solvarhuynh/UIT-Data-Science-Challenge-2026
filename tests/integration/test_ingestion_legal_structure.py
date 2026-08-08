"""Tests for rule-based Vietnamese legal-document parsing."""

from udsc2026.ingestion.legal_structure import parse_legal_structure
from udsc2026.ingestion.readers.models import RawDocument


def test_parser_builds_hierarchy_and_preserves_recognition_order():
    text = """BỘ LUẬT MINH HỌA 2026
Chương I. QUY ĐỊNH CHUNG
Mục 1: Phạm vi
Điều 10. Quy định mẫu
1. Nội dung khoản một.
a) Nội dung điểm a.
b) Nội dung điểm b.
2. Nội dung khoản hai.
"""

    parsed = parse_legal_structure(text)

    assert parsed.law_name == "BỘ LUẬT MINH HỌA 2026"
    assert parsed.requires_manual_review is False
    assert [entry.level for entry in parsed.entries] == [
        "chapter",
        "section",
        "article",
        "clause",
        "point",
        "point",
        "clause",
    ]
    chapter = parsed.chapters[0]
    article = chapter.sections[0].articles[0]
    assert chapter.identifier == "I"
    assert article.identifier == "10"
    assert article.title == "Quy định mẫu"
    assert [clause.identifier for clause in article.clauses] == ["1", "2"]
    assert [point.identifier for point in article.clauses[0].points] == ["a", "b"]
    assert parsed.entries[4].path == {
        "chapter": "Chương I",
        "section": "Mục 1",
        "article": "Điều 10",
        "clause": "Khoản 1",
        "point": "Điểm a",
    }


def test_parser_uses_document_title_and_accepts_labeled_lower_levels():
    document = RawDocument(
        doc_id="law-1",
        source_path="data/raw/law-1.txt",
        title="Luật có tiêu đề từ metadata",
        raw_text=(
            "DÒNG IN HOA KHÔNG ĐƯỢC ƯU TIÊN\n"
            "Điều 1: Phạm vi điều chỉnh\n"
            "Khoản 1. Quy định chung\n"
            "Điểm đ) Trường hợp đặc biệt"
        ),
        file_format="txt",
    )

    parsed = parse_legal_structure(document)

    assert parsed.doc_id == "law-1"
    assert parsed.law_name == "Luật có tiêu đề từ metadata"
    clause = parsed.articles[0].clauses[0]
    assert clause.identifier == "1"
    assert clause.points[0].identifier == "đ"


def test_parser_does_not_make_preamble_lists_into_legal_clauses():
    parsed = parse_legal_structure("1. Danh sách ở phần mở đầu\na) Một ý mở đầu")

    assert parsed.entries == []
    assert parsed.articles == []
    assert parsed.requires_manual_review is True
    assert parsed.review_reasons == ["no_article_detected"]


def test_parser_accepts_missing_space_in_explicit_lower_level_labels():
    parsed = parse_legal_structure(
        "Điều 1. Phạm vi\n"
        "Khoản1. Quy định bị thiếu khoảng trắng.\n"
        "Điểma) Nội dung điểm a.\n"
        "Tài khoản 1 của khách hàng không phải là khoản pháp luật."
    )

    article = parsed.articles[0]
    assert [clause.identifier for clause in article.clauses] == ["1"]
    assert [point.identifier for point in article.clauses[0].points] == ["a"]
    assert "Tài khoản 1" in article.clauses[0].points[0].content


def test_parser_accepts_mid_line_prefixes_and_accentless_headings():
    parsed = parse_legal_structure(
        "BỘ LUẬT MINH HỌA\n"
        "1. Dieu 1. Pham vi dieu chinh\n"
        "2. Khoan 1. Quy dinh chung\n"
        "3. Diem a) Noi dung diem a."
    )

    article = parsed.articles[0]
    assert article.identifier == "1"
    assert article.title == "Pham vi dieu chinh"
    assert [clause.identifier for clause in article.clauses] == ["1"]
    assert [point.identifier for point in article.clauses[0].points] == ["a"]
