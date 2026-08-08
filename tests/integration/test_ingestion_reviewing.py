"""Tests for manual-review classification and breakdown reporting."""

from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.reviewing import (
    build_manual_review_breakdown,
    build_manual_review_document_report,
    classify_manual_review_document,
)


def test_manual_review_classifier_distinguishes_common_reasons():
    empty_doc = CleanDocument(
        doc_id="empty",
        source_path="data/raw/empty.txt",
        title=None,
        cleaned_text="",
        file_format="txt",
    )
    fallback_doc = CleanDocument(
        doc_id="fallback",
        source_path="data/raw/fallback.txt",
        title="Luật mẫu",
        cleaned_text="Lời nói đầu\nPhạm vi điều chỉnh\nNội dung áp dụng.",
        file_format="txt",
    )
    regex_doc = CleanDocument(
        doc_id="regex",
        source_path="data/raw/regex.txt",
        title=None,
        cleaned_text="BỘ LUẬT MẪU ĐIỀU 1. Nội dung áp dụng trong văn bản.",
        file_format="txt",
    )

    assert classify_manual_review_document(empty_doc)[0] == "cleaner_cleared_text"
    assert classify_manual_review_document(fallback_doc)[0] == "fallback_chunkable_no_article"
    assert classify_manual_review_document(regex_doc)[0] == "regex_recoverable"


def test_breakdown_groups_document_ids_by_primary_reason():
    documents = [
        CleanDocument(
            doc_id="empty",
            source_path="data/raw/empty.txt",
            title=None,
            cleaned_text="",
            file_format="txt",
        ),
        CleanDocument(
            doc_id="fallback",
            source_path="data/raw/fallback.txt",
            title="Luật mẫu",
            cleaned_text="Lời nói đầu\nPhạm vi điều chỉnh\nNội dung áp dụng.",
            file_format="txt",
        ),
        CleanDocument(
            doc_id="regex",
            source_path="data/raw/regex.txt",
            title=None,
            cleaned_text="BỘ LUẬT MẪU ĐIỀU 1. Nội dung áp dụng trong văn bản.",
            file_format="txt",
        ),
    ]

    breakdown = build_manual_review_breakdown(documents, ["empty", "fallback", "regex"])

    buckets = {bucket.reason: bucket for bucket in breakdown.buckets}
    assert buckets["cleaner_cleared_text"].count == 1
    assert buckets["fallback_chunkable_no_article"].count == 1
    assert buckets["regex_recoverable"].count == 1


def test_document_report_lists_details_for_one_reason():
    documents = [
        CleanDocument(
            doc_id="ocr",
            source_path="data/raw/ocr.txt",
            title=None,
            cleaned_text="1234 \n* * *\n5678",
            file_format="txt",
            removed_lines=["* * *"],
        ),
        CleanDocument(
            doc_id="clean",
            source_path="data/raw/clean.txt",
            title=None,
            cleaned_text="",
            file_format="txt",
        ),
    ]

    report = build_manual_review_document_report(
        documents, ["ocr", "clean"], "ocr_noise"
    )

    assert report.reason == "ocr_noise"
    assert report.count == 1
    assert [item.doc_id for item in report.documents] == ["ocr"]