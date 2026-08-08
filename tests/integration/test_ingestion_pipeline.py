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


def test_pipeline_accepts_btc_contexts_and_writes_manifest_and_orphan_reports(tmp_path):
    raw_directory = tmp_path / "raw" / "btc"
    context_dir = raw_directory / "LegalIR" / "selected-contexts"
    qa_dir = raw_directory / "LegalQA"
    context_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)

    (context_dir / "context_21.json").write_text(
        json.dumps(
            {
                "id": 21,
                "name": "Quyet-dinh-36-2012-QD-TTg",
                "link": "https://example.invalid/context/21",
                "passage": "Điều 1. Nội dung áp dụng.\n1. Trường hợp cụ thể.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (qa_dir / "train.json").write_text(
        json.dumps({"1001": {"question": "Q1", "answer": ["21"]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (qa_dir / "public-official.json").write_text(
        json.dumps({"2001": {"question": "Q2", "answer": ["21"]}}, ensure_ascii=False),
        encoding="utf-8",
    )

    processed_root = tmp_path / "processed"

    result = run_ingestion_pipeline(raw_directory, processed_root)

    assert result.raw_document_count == 1
    assert result.cleaned_document_count == 1
    assert result.chunked_document_count == 1
    assert result.manifest_path is not None
    assert result.orphan_report_path is not None
    assert result.corpus_hash
    assert (processed_root / "documents" / "21.json").exists()
    assert (processed_root / "chunks" / "21.jsonl").exists()
    assert (processed_root / "parents" / "21.jsonl").exists()

    manifest = json.loads(
        (processed_root / "metadata" / "manifest.json").read_text(encoding="utf-8")
    )
    orphan_report = json.loads(
        (processed_root / "metadata" / "orphan_contexts.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "btc-context-v1"
    assert manifest["context_file_count"] == 1
    assert manifest["accepted_context_count"] == 1
    assert manifest["orphan_context_count"] == 0
    assert manifest["document_ids"] == ["21"]
    assert len(manifest["qa_fixtures"]) == 2
    assert {fixture["file_name"] for fixture in manifest["qa_fixtures"]} == {
        "train.json",
        "public-official.json",
    }
    assert orphan_report["orphan_context_count"] == 0
    assert orphan_report["errors"] == []
