"""Tests for conservative legal-text cleaning and abbreviation extraction."""

from udsc2026.ingestion.cleaners import (
    clean_document,
    clean_text,
    expanded_terms_in_text,
    extract_abbreviations,
)
from udsc2026.ingestion.readers.models import RawDocument


def test_cleaner_normalizes_unicode_noise_and_preserves_legal_structure():
    raw_text = "Đie\u0302u\u00a01.\u00ad Phạm vi\náp dụng của Luật.\n1. Nội dung\nquy định."

    cleaned_text, removed_lines = clean_text(raw_text)

    assert removed_lines == []
    assert cleaned_text == "Điều 1. Phạm vi áp dụng của Luật.\n1. Nội dung quy định."


def test_cleaner_removes_repeated_headers_footers_and_page_markers():
    raw_text = (
        "CỔNG THÔNG TIN PHÁP LUẬT\nĐiều 1. Nội dung trang một.\nTrang 1\n\f\n"
        "CỔNG THÔNG TIN PHÁP LUẬT\nĐiều 2. Nội dung trang hai.\nTrang 2"
    )

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text == "Điều 1. Nội dung trang một.\n\nĐiều 2. Nội dung trang hai."
    assert "CỔNG THÔNG TIN PHÁP LUẬT" in removed_lines
    assert "Trang 1" in removed_lines
    assert "Trang 2" in removed_lines


def test_cleaner_removes_table_of_contents_and_watermark_only():
    raw_text = "DỰ THẢO\nMỤC LỤC\nChương I .... 1\nĐiều 1. Nội dung pháp luật."

    cleaned_text, removed_lines = clean_text(raw_text)

    assert cleaned_text == "Điều 1. Nội dung pháp luật."
    assert removed_lines == ["DỰ THẢO", "MỤC LỤC", "Chương I .... 1"]


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
    abbreviations = extract_abbreviations(
        "Bộ luật Lao động (sau đây gọi là BLLĐ)"
    )
    text = "BLLĐ quy định quyền của người lao động."

    assert expanded_terms_in_text(text, abbreviations) == {
        "BLLĐ": "Bộ luật Lao động"
    }
    assert text == "BLLĐ quy định quyền của người lao động."
