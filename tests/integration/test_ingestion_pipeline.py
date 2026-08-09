"""End-to-end tests for the reproducible legal ingestion pipeline."""

import json

from udsc2026.evaluation.synthetic_generator import SyntheticBenchmarkArtifact
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
    assert result.validation_report.chunk_count == 3
    assert result.validation_report.structured_chunk_count == 2
    assert result.validation_report.supplemental_chunk_count == 1
    assert result.validation_report.source_assignment_coverage_ratio == 1.0
    assert result.integrity_gate_passed is True
    assert result.semantic_completeness_gate_passed is True
    assert (processed_root / "documents" / "law.json").exists()
    assert (processed_root / "chunks" / "law.jsonl").exists()
    assert (processed_root / "parents" / "law.jsonl").exists()
    assert (processed_root / "metadata" / "extract_errors.json").exists()
    assert (processed_root / "metadata" / "clean_errors.json").exists()
    assert (processed_root / "metadata" / "disk_audit_report.json").exists()
    assert (processed_root / "metadata" / "processing_manifest.json").exists()
    report = json.loads(
        (processed_root / "metadata" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["chunk_count"] == 3
    assert report["orphan_chunk_count"] == 0
    disk_audit = json.loads(
        (processed_root / "metadata" / "disk_audit_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert disk_audit["cleaned_removed_line_count"] == 0
    assert disk_audit["removed_line_reason_counts"] == {}
    assert disk_audit["removed_line_audit_error_count"] == 0
    assert disk_audit["semantically_risky_removed_line_count"] == 0


def test_pipeline_accepts_btc_contexts_and_writes_manifest_and_orphan_reports(tmp_path):
    raw_directory = tmp_path / "raw" / "btc"
    context_dir = raw_directory / "LegalIR" / "selected-contexts"
    duplicate_context_dir = raw_directory / "LegalQA" / "selected-contexts"
    qa_dir = raw_directory / "LegalQA"
    context_dir.mkdir(parents=True)
    duplicate_context_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    context_payload = json.dumps(
        {
            "id": 21,
            "name": "Quyet-dinh-36-2012-QD-TTg",
            "link": "https://example.invalid/context/21",
            "passage": "Điều 1. Nội dung áp dụng.\n1. Trường hợp cụ thể.",
        },
        ensure_ascii=False,
    )
    (context_dir / "context_21.json").write_text(context_payload, encoding="utf-8")
    (duplicate_context_dir / "context_21.json").write_text(
        context_payload, encoding="utf-8"
    )
    (qa_dir / "train.json").write_text(
        json.dumps({"1001": {"question": "Q1", "answer": "A1"}}, ensure_ascii=False),
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
    assert result.integrity_gate_passed is True
    assert result.semantic_completeness_gate_passed is True
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

    assert manifest["schema_version"] == "btc-context-v2"
    assert manifest["context_file_count"] == 2
    assert manifest["accepted_context_count"] == 1
    assert manifest["physical_context_file_count"] == 2
    assert manifest["unique_context_count"] == 1
    assert manifest["exact_duplicate_context_count"] == 1
    assert manifest["conflicting_duplicate_context_count"] == 0
    assert manifest["orphan_context_count"] == 0
    assert manifest["document_ids"] == ["21"]
    assert len(manifest["qa_fixtures"]) == 2
    assert {fixture["file_name"] for fixture in manifest["qa_fixtures"]} == {
        "train.json",
        "public-official.json",
    }
    assert {fixture["task"] for fixture in manifest["qa_fixtures"]} == {"LegalQA"}
    assert {fixture["relative_path"] for fixture in manifest["qa_fixtures"]} == {
        "LegalQA/train.json",
        "LegalQA/public-official.json",
    }
    assert orphan_report["orphan_context_count"] == 0
    assert orphan_report["errors"] == []
    assert orphan_report["exact_duplicate_context_count"] == 1


def test_pipeline_can_regenerate_a_hash_bound_streaming_benchmark(
    tmp_path, monkeypatch
):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    (raw_directory / "law.txt").write_text(
        "LUáº¬T MáºªU\n"
        "Äiá»u 1. Quyá»n cá»§a ngÆ°á»i lao Ä‘á»™ng.\n"
        "1. NgÆ°á»i lao Ä‘á»™ng cĂ³ quyá»n yĂªu cáº§u báº£o vá»‡ quyá»n lá»£i.",
        encoding="utf-8",
    )
    processed_root = tmp_path / "processed_v3"
    calls = []

    def regenerate(chunks_path, output_path, **kwargs):
        calls.append((chunks_path, output_path, kwargs))
        output_path.parent.mkdir(parents=True)
        output_path.write_text("{}\n" * 100, encoding="utf-8")
        return SyntheticBenchmarkArtifact(
            output_path=str(output_path),
            record_count=100,
            benchmark_sha256="a" * 64,
            source_chunk_count=3,
            source_chunk_corpus_sha256="b" * 64,
        )

    monkeypatch.setattr(
        "udsc2026.evaluation.synthetic_generator.regenerate_synthetic_benchmark",
        regenerate,
    )

    result = run_ingestion_pipeline(
        raw_directory,
        processed_root,
        synthetic_benchmark_count=100,
        synthetic_benchmark_seed=7,
        synthetic_benchmark_require_all_question_types=False,
    )

    benchmark_path = processed_root / "benchmarks" / "synthetic_qa.jsonl"
    assert result.synthetic_benchmark_path == str(benchmark_path)
    assert result.synthetic_benchmark_record_count == 100
    assert result.synthetic_benchmark_sha256 is not None
    assert result.synthetic_benchmark_source_chunk_corpus_sha256 is not None
    assert calls == [
        (
            processed_root / "chunks",
            benchmark_path,
            {
                "target_count": 100,
                "seed": 7,
                "require_all_question_types": False,
            },
        )
    ]
    assert len(benchmark_path.read_text(encoding="utf-8").splitlines()) == 100
    processing_manifest = json.loads(
        (processed_root / "metadata" / "processing_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert processing_manifest["synthetic_benchmark"]["record_count"] == 100
