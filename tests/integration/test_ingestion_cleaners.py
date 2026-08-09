"""Tests for conservative legal-text cleaning and abbreviation extraction."""

from udsc2026.ingestion.cleaners import (
    clean_document,
    clean_raw_documents,
    clean_text,
    expanded_terms_in_text,
    extract_abbreviations,
)
from udsc2026.ingestion.readers.models import RawDocument


def test_cleaner_normalizes_unicode_noise_and_preserves_legal_structure():
    raw_text = (
        "Đie\u0302\u0300u\u00a01.\u00ad Phạm vi\náp dụng của Luật.\n"
        "1. Nội dung\nquy định."
    )

    cleaned_text, removed_lines = clean_text(raw_text)

    assert removed_lines == []
    assert cleaned_text == ("Điều 1. Phạm vi áp dụng của Luật.\n1. Nội dung quy định.")


def test_cleaner_removes_repeated_headers_footers_and_page_markers():
    raw_text = (
        "CỔNG THÔNG TIN PHÁP LUẬT\nĐiều 1. Nội dung trang một.\n"
        "Trang 1\n\f\n"
        "CỔNG THÔNG TIN PHÁP LUẬT\nĐiều 2. Nội dung trang hai.\nTrang 2"
    )

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text == (
        "CỔNG THÔNG TIN PHÁP LUẬT\nĐiều 1. Nội dung trang một.\f"
        "Điều 2. Nội dung trang hai."
    )
    assert "CỔNG THÔNG TIN PHÁP LUẬT" in removed_lines
    assert "Trang 1" in removed_lines
    assert "Trang 2" in removed_lines


def test_cleaner_preserves_bare_numeric_lines_used_by_legal_tables():
    raw_text = "Phụ lục định mức\n2026\n100\n200\nTổng cộng"

    cleaned_text, removed_lines = clean_text(raw_text)

    assert "2026" in cleaned_text.splitlines()
    assert "100" in cleaned_text.splitlines()
    assert "200" in cleaned_text.splitlines()
    assert removed_lines == []


def test_cleaner_preserves_standalone_legal_labels_for_parser_repair():
    raw_text = "QUY ĐỊNH CHUNG\n\nChương\n\nI\n\nĐiều\n\n1. Phạm vi"

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text == "QUY ĐỊNH CHUNG\nChương I\nĐiều\n1. Phạm vi"
    assert removed_lines == []


def test_cleaner_normalizes_eth_confusable_before_structure_line_merging():
    raw_text = (
        "Chương XXVIII THỦ TỤC ĐỐI VỚI NGƯỜI DƯỚI 18 TUỔI\n"
        "Ðiều 413. Phạm vi áp dụng\n"
        "1. Nội dung quy định tại ðiểm a."
    )

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text.startswith(
        "Chương XXVIII THỦ TỤC ĐỐI VỚI NGƯỜI DƯỚI 18 TUỔI\nĐiều 413. Phạm vi áp dụng"
    )
    assert "điểm a" in cleaned_text
    assert removed_lines == []


def test_cleaner_keeps_numbered_appendix_boundary_separate_from_signature():
    raw_text = "KT. BỘ TRƯỞNG Tạ Anh Tuấn\nPHỤ LỤC SỐ 01\nBảng dữ liệu"

    cleaned_text, removed_lines = clean_text(raw_text)

    assert "Tuấn\nPHỤ LỤC SỐ 01" in cleaned_text
    assert removed_lines == []


def test_cleaner_removes_table_of_contents_but_preserves_draft_status():
    raw_text = "DỰ THẢO\nMỤC LỤC\nChương I .... 1\nĐiều 1. Nội dung pháp luật."

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text == "DỰ THẢO\nĐiều 1. Nội dung pháp luật."
    assert removed_lines == ["MỤC LỤC", "Chương I .... 1"]


def test_cleaner_preserves_fraction_date_code_and_symbol_table_cells():
    raw_text = (
        "Tỷ lệ\n4/5\nKỳ báo cáo\n12/2023\nMã đối chiếu\n8535/8536\n"
        "Toán tử\n+\n%\n≤\n≥\nKết thúc"
    )

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text.splitlines() == [
        "Tỷ lệ",
        "4/5",
        "Kỳ báo cáo",
        "12/2023",
        "Mã đối chiếu",
        "8535/8536",
        "Toán tử",
        "+",
        "%",
        "≤",
        "≥",
        "Kết thúc",
    ]
    assert removed_lines == []


def test_clean_document_records_auditable_removal_reasons():
    raw_document = RawDocument(
        doc_id="law",
        source_path="law.txt",
        raw_text="MỤC LỤC\nChương I .... 1\nĐiều 1. Nội dung.\nTrang 1/2",
        file_format="txt",
    )

    document = clean_document(raw_document)

    assert document.metadata["cleaning"]["removed_line_reason_counts"] == {
        "explicit_page_marker": 1,
        "table_of_contents_entry": 1,
        "table_of_contents_header": 1,
    }


def test_clean_document_returns_contract_and_document_abbreviations():
    raw_document = RawDocument(
        doc_id="law",
        source_path="data/raw/btc/law.txt",
        title=None,
        raw_text=(
            "Bộ luật Lao động (sau đây gọi là BLLĐ) quy định quyền.\n"
            "BLLĐ áp dụng cho người lao động."
        ),
        file_format="txt",
    )

    document = clean_document(raw_document)

    assert document.abbreviations["BLLĐ"] == "Bộ luật Lao động"
    assert document.metadata["cleaning"]["unicode_normalization"] == "NFC"
    assert "BLLĐ" not in document.cleaned_text.replace("BLLĐ", "")


def test_expanded_terms_keeps_original_text_unchanged():
    abbreviations = extract_abbreviations("Bộ luật Lao động (sau đây gọi là BLLĐ)")
    text = "BLLĐ quy định quyền của người lao động."

    assert expanded_terms_in_text(text, abbreviations) == {"BLLĐ": "Bộ luật Lao động"}
    assert text == "BLLĐ quy định quyền của người lao động."


def test_cleaner_preserves_page_boundaries_and_normalizes_old_tones():
    raw_text = "Nội dung trang một\nchưa kết thúc\fNội dung trang hai\nhoà giải."

    cleaned_text, _ = clean_text(raw_text)

    assert cleaned_text == (
        "Nội dung trang một chưa kết thúc\fNội dung trang hai hòa giải."
    )


def test_clean_raw_documents_writes_audit_and_empty_error_report(tmp_path):
    raw_document = RawDocument(
        doc_id="law",
        source_path="data/raw/btc/law.txt",
        raw_text="Điều 1. Nội dung.",
        file_format="txt",
    )
    output_dir = tmp_path / "documents"
    errors_path = tmp_path / "metadata" / "clean_errors.json"

    cleaned_documents = clean_raw_documents(
        [raw_document], output_dir=output_dir, errors_path=errors_path
    )

    assert [document.doc_id for document in cleaned_documents] == ["law"]
    assert (output_dir / "law.json").exists()
    assert errors_path.read_text(encoding="utf-8") == "[]"
