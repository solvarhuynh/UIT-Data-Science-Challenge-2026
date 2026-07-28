"""End-to-end tests for the reproducible legal ingestion pipeline."""

import json

from udsc2026.ingestion import run_ingestion_pipeline


def test_pipeline_chains_raw_clean_chunk_and_audit_outputs(tmp_path):
    raw_directory = tmp_path / "raw" / "btc"
    raw_directory.mkdir(parents=True)
    (raw_directory / "law.txt").write_text(
        "LUẬT MẪU\nĐiều 1. Phạm vi điều chỉnh.\n1. Nội dung áp dụng.",
        encoding="utf-8",
    )
    processed_root = tmp_path / "processed"

    result = run_ingestion_pipeline(raw_directory, processed_root)

    assert result.raw_document_count == 1
    assert result.cleaned_document_count == 1
    assert result.chunked_document_count == 1
    assert result.validation_report.chunk_count == 1
    assert (processed_root / "documents" / "law.json").exists()
    assert (processed_root / "chunks" / "law.jsonl").exists()
    assert (processed_root / "parents" / "law.jsonl").exists()
    assert (processed_root / "metadata" / "extract_errors.json").exists()
    assert (processed_root / "metadata" / "clean_errors.json").exists()
    report = json.loads(
        (processed_root / "metadata" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["chunk_count"] == 1
    assert report["orphan_chunk_count"] == 0
