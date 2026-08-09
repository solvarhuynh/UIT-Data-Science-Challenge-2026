"""Tests for rule-based Vietnamese legal-document parsing."""

import pytest

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


def test_parser_repairs_standalone_split_article_and_suffix_identifier():
    parsed = parse_legal_structure(
        "LUẬT MẪU\nĐiều\n12a. Quy định được bổ sung\n1. Nội dung áp dụng."
    )

    article = parsed.articles[0]
    assert article.identifier == "12a"
    assert article.heading == "Điều 12a. Quy định được bổ sung"
    assert article.start_line == 2
    assert article.end_line == 4
    assert parsed.repaired_split_article_count == 1
    assert parsed.suspected_split_article_count == 0
    assert parsed.requires_manual_review is False


def test_parser_repairs_chapter_line_merged_with_split_article_label():
    parsed = parse_legal_structure(
        "Chương I. QUY ĐỊNH CHUNG Điều\n1. Phạm vi\n1. Nội dung."
    )

    assert parsed.chapters[0].identifier == "I"
    assert parsed.chapters[0].articles[0].identifier == "1"
    assert parsed.repaired_split_article_count == 1


def test_parser_does_not_rewrite_prose_that_merely_ends_with_article_word():
    parsed = parse_legal_structure("Nội dung được viện dẫn tại Điều\n7. Danh mục")

    assert parsed.articles == []
    assert parsed.suspected_split_article_count == 1
    assert parsed.requires_manual_review is True


def test_parser_keeps_orphan_point_continuation_in_article_intro():
    parsed = parse_legal_structure(
        "Điều 1. Phạm vi\na) Điểm đứng trước khoản.\nDòng tiếp của điểm.\n"
        "1. Khoản xuất hiện sau."
    )

    article = parsed.articles[0]
    assert "Dòng tiếp của điểm." in article.content
    assert [clause.identifier for clause in article.clauses] == ["1"]


def test_parser_rejects_line_start_article_citations_and_slash_ranges():
    parsed = parse_legal_structure(
        "Điều 21; Điều 43 được viện dẫn.\n"
        "Điều 21 ; Điều 43 cũng được viện dẫn.\n"
        "Điều 18, Mục 3 của văn bản.\n"
        "Điều 61/62 là khoảng tham chiếu.\n"
        "Điều 1… là ký hiệu trong mẫu."
    )

    assert parsed.articles == []
    assert parsed.requires_manual_review is True


def test_parser_rejects_amendment_cross_reference_but_accepts_quoted_heading():
    parsed = parse_legal_structure(
        "Điều 1. Sửa đổi luật\n"
        "Bổ sung Điều 98a vào sau\n"
        "Điều 98 như sau:\n"
        "“Điều 98a. Thuê phương tiện\n"
        "1. Nội dung được bổ sung."
    )

    assert [article.identifier for article in parsed.articles] == ["1", "98a"]
    assert "Điều 98 như sau:" in parsed.articles[0].content
    assert parsed.articles[1].heading.startswith("“Điều 98a.")


def test_parser_does_not_promote_standalone_nhu_sau_cross_reference():
    parsed = parse_legal_structure("Điều 98 như sau:\n“Điều 98đ. Nội dung mới.")

    assert [article.identifier for article in parsed.articles] == ["98đ"]


def test_parser_keeps_line_start_instrument_citation_inside_active_article():
    parsed = parse_legal_structure(
        "Điều 1. Quy định chính\n"
        "Điều 21 của Bộ luật hình sự thì được áp dụng.\n"
        "Điều\n"
        "2. Điều khoản tiếp theo."
    )

    assert [article.identifier for article in parsed.articles] == ["1", "2"]
    assert "Điều 21 của Bộ luật hình sự" in parsed.articles[0].content


def test_parser_allows_punctuated_heading_whose_title_starts_with_cua():
    parsed = parse_legal_structure("Điều 1. Của cải là tài sản theo quy định.")

    assert [article.identifier for article in parsed.articles] == ["1"]


def test_parser_rejects_high_confidence_unpunctuated_citation_continuations():
    citations = (
        "Điều 9 Luật Hòa giải, đối thoại tại Tòa án bao gồm chi phí.",
        "Điều 39 Bộ luật Tố tụng dân sự 2015;",
        "Điều 10 Nghị định này, cơ quan tiếp nhận phải thông báo.",
        "Điều 6 Thông tư này;",
        "Điều 16 và Điều 17 Nghị định này thì áp dụng.",
        "Điều 30 hoặc khoản 1 Điều 31 của Luật.",
        "Điều 3 được sửa đổi, bổ sung như sau:",
        "Điều 5 quy định kèm theo Quyết định số 99-QĐ/TW.",
        "Điều 3 nêu trên.",
    )

    for citation in citations:
        parsed = parse_legal_structure(citation)
        assert parsed.articles == [], citation


def test_parser_does_not_promote_bare_split_number_without_heading_signal():
    parsed = parse_legal_structure("Mẫu thể thức\nĐiều\n1\nNội dung minh họa")

    assert parsed.articles == []
    assert parsed.repaired_split_article_count == 0
    assert parsed.suspected_split_article_count == 1


def test_appendix_heading_closes_last_article_span():
    parsed = parse_legal_structure(
        "Điều 1. Nội dung chính\n1. Quy định.\nPHỤ LỤC I\nBảng dữ liệu phụ lục"
    )

    article = parsed.articles[0]
    assert article.end_line == 2
    assert "PHỤ LỤC" not in article.text
    assert parsed.appendix_boundary_count == 1


def test_numbered_appendix_heading_closes_last_article_span():
    parsed = parse_legal_structure(
        "Điều 1. Nội dung chính\n1. Quy định.\nPHỤ LỤC SỐ 02\nBảng dữ liệu phụ lục"
    )

    article = parsed.articles[0]
    assert article.end_line == 2
    assert "PHỤ LỤC" not in article.text
    assert parsed.appendix_boundary_count == 1


def test_appendix_does_not_reopen_main_hierarchy_on_article_like_rows():
    parsed = parse_legal_structure(
        "Điều 1. Nội dung chính\n1. Quy định.\n"
        "PHỤ LỤC I\nĐiều 2. Ví dụ trong biểu mẫu\n1. Dữ liệu mẫu."
    )

    assert [article.identifier for article in parsed.articles] == ["1"]
    assert parsed.appendix_boundary_count == 1


def test_invalid_appendix_prefix_is_not_accepted_as_roman_identifier():
    parsed = parse_legal_structure(
        "Điều 1. Nội dung chính\nPHỤ LỤC INVALID\nĐiều 2. Nội dung tiếp theo."
    )

    assert [article.identifier for article in parsed.articles] == ["1", "2"]
    assert parsed.appendix_boundary_count == 0


def test_parser_keeps_fraction_codes_and_operators_as_content_not_hierarchy():
    parsed = parse_legal_structure(
        "Điều 1. Bảng tỷ lệ và mã đối chiếu\n"
        "4/5\n12/2023\n8535/8536\n+\n%\n≤\n≥\nKết thúc bảng."
    )

    assert [entry.level for entry in parsed.entries] == ["article"]
    assert [article.identifier for article in parsed.articles] == ["1"]
    for value in ("4/5", "12/2023", "8535/8536", "+", "%", "≤", "≥"):
        assert value in parsed.articles[0].content


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


@pytest.mark.parametrize("label", ["Điều", "Dieu"])
def test_prefixed_explicit_article_is_not_rejected_as_citation(label):
    parsed = parse_legal_structure(
        f"1. {label} 1. Quy định này áp dụng trên toàn quốc.",
    )

    assert [article.identifier for article in parsed.articles] == ["1"]
    assert parsed.requires_manual_review is False


def test_parser_repairs_accentless_split_article_heading():
    parsed = parse_legal_structure(
        "DIEU\n1. Pham vi dieu chinh.",
    )

    assert [article.identifier for article in parsed.articles] == ["1"]
    assert parsed.repaired_split_article_count == 1


@pytest.mark.parametrize(
    "line",
    [
        "Dieu 1 cua Luat nay duoc ap dung.",
        "1. Dieu 1 cua Luat nay duoc ap dung.",
        "- Dieu 1 quy dinh tai van ban nay.",
    ],
)
def test_accentless_article_references_are_not_promoted(line):
    parsed = parse_legal_structure(line)

    assert parsed.articles == []
    assert parsed.requires_manual_review is True


@pytest.mark.parametrize("heading", ["PHU LUC I", "1. PHU LUC I"])
def test_accentless_appendix_is_a_terminal_structure_boundary(heading):
    parsed = parse_legal_structure(
        f"Dieu 1. Noi dung chinh.\n{heading}\nDieu 99. Vi du trong phu luc."
    )

    assert [article.identifier for article in parsed.articles] == ["1"]
    assert parsed.appendix_boundary_count == 1
