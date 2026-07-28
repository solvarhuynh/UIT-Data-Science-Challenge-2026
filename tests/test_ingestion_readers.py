"""Tests for the raw-data extraction readers."""

import json

import pytest

from udsc2026.ingestion.readers import (
    RawDocument,
    extract_raw_document,
    extract_raw_documents,
)


def test_read_txt_detects_encoding_and_preserves_newlines(tmp_path):
    path = tmp_path / "van_ban.txt"
    path.write_bytes("LUAT THU NGHIEM\r\nDieu 1".encode("utf-8"))

    document = extract_raw_document(str(path))

    assert isinstance(document, RawDocument)
    assert document.doc_id == "van_ban"
    assert document.file_format == "txt"
    assert document.raw_text == "LUAT THU NGHIEM\r\nDieu 1"
    assert document.title == "LUAT THU NGHIEM"
    assert document.metadata["encoding"]


def test_read_json_uses_source_id_and_keeps_source_text(tmp_path):
    path = tmp_path / "luat.json"
    raw_text = '{\n  "document_id": "LAW-2026",\n  "title": "Luat mau"\n}\n'
    path.write_text(raw_text, encoding="utf-8")

    document = extract_raw_document(str(path))

    assert document.doc_id == "law_2026"
    assert document.title == "Luat mau"
    assert document.raw_text == path.read_bytes().decode("utf-8")
    assert document.metadata["source_document_id"] == "LAW-2026"


def test_read_jsonl_reports_record_count(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": "one"}\n{"id": "two"}\n', encoding="utf-8")

    documents = list(extract_raw_document(str(path)))

    assert [document.doc_id for document in documents] == ["one", "two"]
    assert [document.metadata["source_line"] for document in documents] == [1, 2]
    assert all(document.file_format == "json" for document in documents)


def test_extract_raw_documents_logs_errors_and_duplicate_ids(tmp_path):
    raw_directory = tmp_path / "btc"
    raw_directory.mkdir()
    (raw_directory / "first.json").write_text('{"id": "same"}', encoding="utf-8")
    (raw_directory / "second.json").write_text('{"id": "same"}', encoding="utf-8")
    (raw_directory / "broken.json").write_text("{", encoding="utf-8")
    errors_path = tmp_path / "metadata" / "extract_errors.json"

    documents = extract_raw_documents(str(raw_directory), str(errors_path))

    assert [document.doc_id for document in documents] == ["same"]
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    assert len(errors) == 2
    assert {item["error_type"] for item in errors} == {"JSONDecodeError", "ValueError"}


def test_unsupported_extension_raises_clear_error(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="không được hỗ trợ"):
        extract_raw_document(str(path))


def test_read_docx_keeps_paragraph_and_table_text(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "luat.docx"
    source = docx.Document()
    source.add_paragraph("LUAT MAU")
    table = source.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Dieu 1"
    table.cell(0, 1).text = "Noi dung"
    source.save(path)

    document = extract_raw_document(str(path))

    assert document.file_format == "docx"
    assert "LUAT MAU" in document.raw_text
    assert "Dieu 1\tNoi dung" in document.raw_text
    assert document.metadata["table_count"] == 1


def test_read_pdf_crops_header_footer_and_keeps_layout(tmp_path, monkeypatch):
    from udsc2026.ingestion.readers import pdf_reader

    class CroppedPage:
        def extract_text(self, **kwargs):
            assert kwargs == {"layout": True}
            return "Noi dung trang"

    class Page:
        height = 100
        width = 200

        def crop(self, bounding_box):
            assert bounding_box == (0, 10, 200, 90)
            return CroppedPage()

    class Pdf:
        pages = [Page()]
        metadata = {"Title": "Luat mau"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class PdfPlumber:
        @staticmethod
        def open(_file_path):
            return Pdf()

    path = tmp_path / "luat.pdf"
    path.write_bytes(b"%PDF-test")
    monkeypatch.setattr(pdf_reader, "_require_pdfplumber", lambda: PdfPlumber())

    document = pdf_reader.read_pdf(str(path))

    assert document.raw_text == "Noi dung trang"
    assert document.metadata["content_crop"] == {
        "top_percent": 10,
        "bottom_percent": 10,
    }
    assert document.metadata["ocr_status"] == "not_required"
