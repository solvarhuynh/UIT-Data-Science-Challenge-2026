"""Tests for legal-unit parent-child chunking and JSONL delivery."""

import json

import pytest

from udsc2026.ingestion.chunking import (
    chunk_clean_document,
    split_by_sentence_with_overlap,
    validate_chunking_results,
    write_chunking_outputs,
)
from udsc2026.ingestion.cleaners.models import CleanDocument


def _document() -> CleanDocument:
    return CleanDocument(
        doc_id="doc001",
        source_path="data/raw/btc/doc001.txt",
        title="Bộ luật Minh họa 2026",
        cleaned_text=(
            "Chương II. QUY ĐỊNH MẪU\n"
            "Điều 10. Trách nhiệm\n"
            "1. Người lao động thực hiện quy định BLLĐ.\n"
            "a) Báo cáo đúng thời hạn.\n"
            "b) Lưu hồ sơ đầy đủ.\n"
            "2. Người sử dụng lao động hỗ trợ."
        ),
        file_format="txt",
        abbreviations={"BLLĐ": "Bộ luật Lao động"},
        metadata={"effective_date": "2026-01-01"},
    )


def test_chunking_links_legal_children_to_full_article_parent():
    result = chunk_clean_document(_document())

    assert result.article_count == 1
    assert len(result.parents) == 1
    assert result.parents[0].parent_id == "doc001_article_10"
    assert len(result.chunks) == 4
    assert {chunk.parent_id for chunk in result.chunks} == {"doc001_article_10"}
    assert all(chunk.parent_text is None for chunk in result.chunks)
    point_chunk = next(chunk for chunk in result.chunks if chunk.point == "Điểm a")
    assert point_chunk.chunk_id == "doc001_article_10_clause_1_point_a"
    assert point_chunk.metadata["expanded_terms"] == {}
    clause_chunk = next(chunk for chunk in result.chunks if chunk.clause == "Khoản 1")
    assert clause_chunk.metadata["expanded_terms"] == {"BLLĐ": "Bộ luật Lao động"}
    assert clause_chunk.metadata["chapter"] == "Chương II"
    assert clause_chunk.metadata["effective_date"] == "2026-01-01"


def test_article_without_clause_gets_a_distinct_body_child():
    document = CleanDocument(
        doc_id="single",
        source_path="data/raw/single.txt",
        title="Luật mẫu",
        cleaned_text="Điều 1. Nội dung duy nhất.",
        file_format="txt",
    )

    result = chunk_clean_document(document)

    assert result.parents[0].parent_id == "single_article_1"
    assert result.chunks[0].chunk_id == "single_article_1_body"
    assert result.chunks[0].parent_id == "single_article_1"
    assert result.chunks[0].clause is None


def test_sentence_split_preserves_boundaries_and_limits_overlap():
    parts = split_by_sentence_with_overlap(
        "Một hai ba. Bốn năm sáu. Bảy tám chín.",
        chunk_size=4,
        chunk_overlap=3,
    )

    assert parts == ["Một hai ba.", "ba. Bốn năm sáu.", "sáu. Bảy tám chín."]


def test_sentence_larger_than_limit_falls_back_to_word_boundaries():
    parts = split_by_sentence_with_overlap(
        "một hai ba bốn năm sáu bảy tám chín mười",
        chunk_size=4,
        chunk_overlap=1,
    )

    assert parts == [
        "một hai ba bốn",
        "bốn năm sáu bảy",
        "bảy tám chín mười",
    ]
    assert all(len(part.split()) <= 4 for part in parts)


def test_writer_emits_jsonl_and_validation_report(tmp_path):
    chunks_dir = tmp_path / "chunks"
    parents_dir = tmp_path / "parents"
    report_path = tmp_path / "metadata" / "validation_report.json"

    results, report = write_chunking_outputs(
        [_document()],
        chunks_dir=chunks_dir,
        parents_dir=parents_dir,
        report_path=report_path,
        review_path=tmp_path / "metadata" / "manual_review.json",
    )

    child_lines = (chunks_dir / "doc001.jsonl").read_text(encoding="utf-8").splitlines()
    parent_lines = (
        (parents_dir / "doc001.jsonl").read_text(encoding="utf-8").splitlines()
    )
    children = [json.loads(line) for line in child_lines]
    parent = json.loads(parent_lines[0])
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(child_lines) == len(results[0].chunks)
    assert {child["parent_id"] for child in children} == {parent["parent_id"]}
    assert all(child["metadata"]["article"] == "Điều 10" for child in children)
    assert report.chunk_count == 4
    assert report_payload["missing_metadata_count"] == 0
    assert report_payload["orphan_chunk_count"] == 0
    assert report_payload["invalid_json_record_count"] == 0


def test_manual_review_document_gets_auditable_fallback_chunks():
    document = CleanDocument(
        doc_id="review",
        source_path="data/raw/review.txt",
        title="Văn bản chưa rõ",
        cleaned_text="Nội dung không có cấu trúc Điều.",
        file_format="txt",
    )

    result = chunk_clean_document(document)
    report = validate_chunking_results([result])

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id.startswith("review_document_part_")
    assert result.chunks[0].metadata["fallback_chunking"] is True
    assert result.requires_manual_review is True
    assert report.manual_review_documents == ["review"]


def test_validation_reports_orphan_child_parent_link():
    result = chunk_clean_document(_document())
    result.chunks[0].parent_id = "missing_parent"

    report = validate_chunking_results([result])

    assert report.orphan_chunk_ids == [result.chunks[0].chunk_id]


def test_writer_rejects_duplicate_document_ids_before_overwriting(tmp_path):
    with pytest.raises(ValueError, match="Duplicate doc_id values: doc001"):
        write_chunking_outputs(
            [_document(), _document()],
            chunks_dir=tmp_path / "chunks",
            parents_dir=tmp_path / "parents",
        )
