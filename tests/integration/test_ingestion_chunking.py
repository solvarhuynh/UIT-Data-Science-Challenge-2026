"""Tests for legal-unit parent-child chunking and JSONL delivery."""

import json

import pytest

from udsc2026.ingestion.chunking import (
    chunk_clean_document,
    split_by_sentence_with_overlap,
    validate_chunking_results,
    write_chunking_outputs,
    write_chunking_outputs_streaming,
)
from udsc2026.ingestion.chunking.chunker import _set_searchable_content_coverage
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
    assert len(result.parents) == 2
    assert result.parents[0].parent_id == "doc001_article_10"
    assert len(result.chunks) == 6
    article_chunks = [
        chunk for chunk in result.chunks if chunk.parent_id == "doc001_article_10"
    ]
    assert len(article_chunks) == 5
    assert all(chunk.parent_text is None for chunk in result.chunks)
    point_chunk = next(chunk for chunk in result.chunks if chunk.point == "Điểm a")
    assert point_chunk.chunk_id == "doc001_article_10_clause_1_point_a"
    assert point_chunk.metadata["expanded_terms"] == {}
    clause_chunk = next(chunk for chunk in result.chunks if chunk.clause == "Khoản 1")
    assert clause_chunk.metadata["expanded_terms"] == {"BLLĐ": "Bộ luật Lao động"}
    assert clause_chunk.metadata["chapter"] == "Chương II"
    assert clause_chunk.metadata["effective_date"] == "2026-01-01"
    assert result.supplemental_chunk_count == 1
    assert result.unassigned_source_line_count == 0
    assert result.assigned_source_line_count == result.source_nonempty_line_count


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


def test_long_sentence_split_keeps_overlap_across_sentence_boundaries():
    parts = split_by_sentence_with_overlap(
        "một hai ba. bốn năm sáu bảy tám. chín mười.",
        chunk_size=4,
        chunk_overlap=1,
    )
    bigrams = {pair for part in parts for pair in zip(part.split(), part.split()[1:])}

    assert ("ba.", "bốn") in bigrams
    assert ("tám.", "chín") in bigrams
    assert all(len(part.split()) <= 4 for part in parts)


def test_exact_limit_sentence_keeps_overlap_with_previous_sentence():
    parts = split_by_sentence_with_overlap(
        "alpha. one two three four.",
        chunk_size=4,
        chunk_overlap=1,
    )
    bigrams = {pair for part in parts for pair in zip(part.split(), part.split()[1:])}

    assert ("alpha.", "one") in bigrams
    assert all(len(part.split()) <= 4 for part in parts)


def test_label_only_clause_is_prefixed_to_first_point_child():
    document = CleanDocument(
        doc_id="label-only-clause",
        source_path="data/raw/label-only-clause.txt",
        title="Luật kiểm thử",
        cleaned_text=("Điều 7. Nội dung\nKhoản 331.\nc) Thông tin bổ sung."),
        file_format="txt",
    )

    result = chunk_clean_document(document)

    point_chunk = next(chunk for chunk in result.chunks if chunk.point == "Điểm c")
    assert point_chunk.text.startswith("Khoản 331. c) Thông tin")
    assert result.missing_source_token_count == 0
    assert result.missing_source_bigram_count == 0


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
    parents = [json.loads(line) for line in parent_lines]
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(child_lines) == len(results[0].chunks)
    assert {child["parent_id"] for child in children} == {
        parent["parent_id"] for parent in parents
    }
    article_children = [
        child for child in children if child["metadata"]["structure_type"] == "article"
    ]
    assert all(child["metadata"]["article"] == "Điều 10" for child in article_children)
    assert report.chunk_count == 6
    assert report_payload["missing_metadata_count"] == 0
    assert report_payload["orphan_chunk_count"] == 0
    assert report_payload["invalid_json_record_count"] == 0
    assert report_payload["source_assignment_coverage_ratio"] == 1.0


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
    assert result.chunks[0].metadata["structure_type"] == "unstructured_fallback"
    assert result.requires_manual_review is True
    assert report.manual_review_documents == ["review"]
    assert report.missing_metadata_count == 0


def test_empty_official_passage_is_explicit_placeholder_and_fails_quality_gate():
    document = CleanDocument(
        doc_id="empty",
        source_path="data/raw/btc/context_empty.json",
        title="Quyet-dinh-kiem-thu",
        cleaned_text="",
        file_format="json",
        metadata={"source_content_empty": True},
    )

    result = chunk_clean_document(document)
    report = validate_chunking_results([result])

    assert result.source_content_empty is True
    assert result.chunks[0].text == "Tên văn bản: Quyet dinh kiem thu"
    assert result.chunks[0].metadata["synthetic_placeholder"] is True
    assert report.empty_source_document_ids == ["empty"]
    assert report.quality_gate_passed is False
    assert "official_empty_source_content" in report.quality_gate_failures


def test_vietnamese_d_suffix_does_not_collide_with_ascii_d():
    document = CleanDocument(
        doc_id="suffix",
        source_path="data/raw/suffix.txt",
        title="Luật hậu tố",
        cleaned_text=(
            "Điều 98d. Hậu tố ASCII.\nNội dung thứ nhất.\n"
            "Điều 98đ. Hậu tố tiếng Việt.\nNội dung thứ hai."
        ),
        file_format="txt",
    )

    result = chunk_clean_document(document)
    parent_ids = {parent.parent_id for parent in result.parents}

    assert "suffix_article_98d" in parent_ids
    assert "suffix_article_98d_vn" in parent_ids


def test_quoted_amendment_article_keeps_coverage_without_fake_reference_parent():
    document = CleanDocument(
        doc_id="amendment",
        source_path="data/raw/amendment.txt",
        title="Luật sửa đổi",
        cleaned_text=(
            "Điều 1. Sửa đổi luật\n"
            "Bổ sung Điều 98a vào sau\n"
            "Điều 98 như sau:\n"
            "“Điều 98a. Thuê phương tiện\n"
            "1. Nội dung được bổ sung."
        ),
        file_format="txt",
    )

    result = chunk_clean_document(document)
    parent_ids = {parent.parent_id for parent in result.parents}

    assert "amendment_article_98" not in parent_ids
    assert "amendment_article_98a" in parent_ids
    assert result.missing_source_token_count == 0
    assert result.missing_source_bigram_count == 0


def test_appendix_is_searchable_supplement_not_part_of_article_parent():
    document = CleanDocument(
        doc_id="appendix",
        source_path="data/raw/appendix.txt",
        title="Luật có phụ lục",
        cleaned_text=("Điều 1. Nội dung chính\n1. Quy định.\nPHỤ LỤC I\n100\n200"),
        file_format="txt",
    )

    result = chunk_clean_document(document)
    article_parent = next(parent for parent in result.parents if parent.article)
    supplemental = [
        chunk
        for chunk in result.chunks
        if chunk.metadata["structure_type"] == "document_context"
    ]

    assert "PHỤ LỤC" not in article_parent.text
    assert any("PHỤ LỤC I" in chunk.text for chunk in supplemental)
    assert result.unassigned_source_line_count == 0


def test_tcvn_source_is_routed_to_full_text_fallback_even_with_article_mentions():
    document = CleanDocument(
        doc_id="standard",
        source_path="data/raw/context_standard.json",
        title="Tiêu chuẩn thử nghiệm",
        cleaned_text="Mục thử nghiệm\nĐiều 7. của tiêu chuẩn được viện dẫn.",
        file_format="json",
        metadata={"source_link": "https://thuvienphapluat.vn/tcvn/example"},
    )

    result = chunk_clean_document(document)

    assert result.structure_status == "unstructured_fallback"
    assert result.source_family == "tcvn"
    assert result.fallback_chunk_count == len(result.chunks)
    assert "unstructured_source_family" in result.review_reasons


def test_single_late_article_match_routes_to_fallback_as_low_confidence():
    preamble = "\n".join("Dòng mở đầu {0}.".format(index) for index in range(12))
    document = CleanDocument(
        doc_id="late",
        source_path="data/raw/late.txt",
        title="Tài liệu pha trộn",
        cleaned_text=preamble + "\nĐiều 9. Một tham chiếu ở cuối.",
        file_format="txt",
    )

    result = chunk_clean_document(document)

    assert result.structure_status == "unstructured_fallback"
    assert "low_structure_confidence" in result.review_reasons


def test_sparse_repaired_article_sequence_is_marked_partial_for_review():
    document = CleanDocument(
        doc_id="split",
        source_path="data/raw/split.txt",
        title="Luật có tiêu đề bị ngắt dòng",
        cleaned_text="Điều\n6. Phạm vi áp dụng\n1. Nội dung.",
        file_format="txt",
    )

    result = chunk_clean_document(document)

    assert result.structure_status == "partial"
    assert "sparse_repaired_article_sequence" in result.structure_warnings
    assert result.fallback_chunk_count == 0


def test_searchable_coverage_counts_missing_duplicate_occurrences():
    result = chunk_clean_document(
        CleanDocument(
            doc_id="coverage",
            source_path="data/raw/coverage.txt",
            title="Luật kiểm thử",
            cleaned_text="Điều 1. alpha beta alpha beta",
            file_format="txt",
        )
    )
    for parent in result.parents:
        parent.text = "Điều 1. alpha beta"
    for chunk in result.chunks:
        chunk.text = "Điều 1. alpha beta"

    _set_searchable_content_coverage(result, "Điều 1. alpha beta alpha beta")

    assert result.missing_source_token_count == 2
    assert result.missing_source_bigram_count == 2


def test_fallback_chunking_handles_no_article_preamble_documents():
    document = CleanDocument(
        doc_id="fallback",
        source_path="data/raw/fallback.txt",
        title="Luật mẫu",
        cleaned_text=(
            "Lời nói đầu\n"
            "Văn bản này quy định nguyên tắc chung.\n\n"
            "Phạm vi điều chỉnh\n"
            "Nội dung điều chỉnh của luật được áp dụng trên toàn quốc."
        ),
        file_format="txt",
    )

    result = chunk_clean_document(document)

    assert result.parents
    assert result.chunks
    assert result.article_count == 0
    assert result.review_reasons == ["fallback_chunked_no_article"]
    assert result.requires_manual_review is True
    assert result.parents[0].article in {
        "Lời nói đầu",
        "Phạm vi điều chỉnh",
        "Phần mở đầu",
    }
    assert all(chunk.article == result.parents[0].article for chunk in result.chunks)
    report = validate_chunking_results([result])
    assert report.manual_review_documents == ["fallback"]
    assert report.structured_document_count == 0
    assert report.fallback_document_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "Chuong I Dieu\n1. Pham vi dieu chinh.",
        "Muc 2 Dieu\n3. Noi dung ap dung.",
    ],
)
def test_accentless_container_split_preserves_searchable_coverage(text):
    document = CleanDocument(
        doc_id="accentless-container",
        source_path="data/raw/accentless-container.txt",
        title="Van ban mau",
        cleaned_text=text,
        file_format="txt",
    )

    result = chunk_clean_document(document)

    assert result.missing_source_token_count == 0
    assert result.missing_source_bigram_count == 0


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


def test_streaming_writer_matches_regular_validation_without_retaining_results(
    tmp_path,
):
    regular_results, regular_report = write_chunking_outputs(
        [_document()],
        chunks_dir=tmp_path / "regular" / "chunks",
        parents_dir=tmp_path / "regular" / "parents",
        report_path=tmp_path / "regular" / "metadata" / "report.json",
        review_path=tmp_path / "regular" / "metadata" / "review.json",
    )
    progress = []

    chunked_count, streaming_report = write_chunking_outputs_streaming(
        [_document()],
        chunks_dir=tmp_path / "streaming" / "chunks",
        parents_dir=tmp_path / "streaming" / "parents",
        report_path=tmp_path / "streaming" / "metadata" / "report.json",
        review_path=tmp_path / "streaming" / "metadata" / "review.json",
        progress_callback=lambda count, doc_id: progress.append((count, doc_id)),
    )

    assert chunked_count == 1
    assert streaming_report == regular_report
    assert progress == [(1, "doc001")]
    assert len(regular_results) == 1
