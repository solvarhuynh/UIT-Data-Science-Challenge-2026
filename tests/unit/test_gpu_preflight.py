"""Tests for GPU preflight validation of immutable processed artifacts."""

import hashlib
import json

import pytest
from scripts.gpu.preflight import _data_check, _processed_corpus_tree_hash


def _write_processed_fixture(root):
    for directory in ("documents", "chunks", "parents", "benchmarks", "metadata"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "documents" / "law.json").write_text("{}", encoding="utf-8")
    (root / "chunks" / "law.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "parents" / "law.jsonl").write_text("{}\n", encoding="utf-8")
    benchmark = root / "benchmarks" / "synthetic_qa.jsonl"
    benchmark.write_text("{}\n", encoding="utf-8")
    benchmark_sha = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    processed_tree_hash = _processed_corpus_tree_hash(root)
    manifest = {
        "counts": {"documents": 1, "chunks": 1, "parents": 1},
        "integrity_gate_passed": True,
        "processed_corpus_tree_hash": processed_tree_hash,
        "synthetic_benchmark": {
            "record_count": 1,
            "benchmark_sha256": benchmark_sha,
            "source_chunk_count": 1,
        },
    }
    audit = {
        "document_count": 1,
        "chunk_count": 1,
        "parent_count": 1,
        "integrity_gate_passed": True,
        "corpus_tree_hash": processed_tree_hash,
        "semantic_completeness_gate_passed": False,
        "semantic_issues": ["official_empty_source_content"],
        "integrity_failures": [],
    }
    (root / "metadata" / "processing_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "metadata" / "disk_audit_report.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    return benchmark


def test_data_preflight_accepts_integrity_pass_with_reported_semantic_gaps(tmp_path):
    _write_processed_fixture(tmp_path)

    result = _data_check(tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["integrity_gate_passed"] is True
    assert result["semantic_completeness_gate_passed"] is False
    assert result["semantic_issues"] == ["official_empty_source_content"]


def test_data_preflight_rejects_benchmark_changed_after_manifest(tmp_path):
    benchmark = _write_processed_fixture(tmp_path)
    benchmark.write_text("{}\n{}\n", encoding="utf-8")

    result = _data_check(tmp_path)

    assert result["ok"] is False
    assert "benchmark_sha256_mismatch" in result["errors"]
    assert "benchmark_record_count_mismatch" in result["errors"]


@pytest.mark.parametrize(
    "relative_path",
    (
        "documents/law.json",
        "parents/law.jsonl",
        "chunks/law.jsonl",
    ),
)
def test_data_preflight_rejects_corpus_file_changed_after_audit(
    tmp_path, relative_path
):
    _write_processed_fixture(tmp_path)
    artifact = tmp_path / relative_path
    artifact.write_bytes(artifact.read_bytes() + b" ")

    result = _data_check(tmp_path)

    assert result["ok"] is False
    assert "processed_corpus_tree_hash_mismatch" in result["errors"]
