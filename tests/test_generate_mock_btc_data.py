"""Tests for the synthetic BTC corpus generator script."""

from pathlib import Path

from scripts.generate_mock_btc_data import generate_mock_data
from udsc2026.ingestion.readers import extract_raw_documents


def test_mock_generator_creates_reader_compatible_corpus(tmp_path):
    output_dir = tmp_path / "btc" / "mock"

    written = generate_mock_data(output_dir)
    documents = extract_raw_documents(str(output_dir), str(tmp_path / "errors.json"))

    assert (output_dir / "mock_btc_law.txt").exists()
    assert (output_dir / "mock_btc_law.json").exists()
    assert (output_dir / "mock_btc_laws.jsonl").exists()
    assert len(written) in (3, 4)
    assert len(documents) == len(written) + 1
    assert any("DỰ THẢO" in document.raw_text for document in documents)
    assert any("(sau đây gọi là BLLĐ)" in document.raw_text for document in documents)
    assert any("a) Người lao động" in document.raw_text for document in documents)


def test_mock_generator_preserves_unrelated_files_in_target_directory(tmp_path):
    output_dir = tmp_path / "btc" / "mock"
    output_dir.mkdir(parents=True)
    preserved_file = output_dir / "real_btc_file.txt"
    preserved_file.write_text("Không được xoá file này.", encoding="utf-8")

    generate_mock_data(output_dir)

    assert preserved_file.read_text(encoding="utf-8") == "Không được xoá file này."


def test_default_output_is_resolved_from_repository_not_current_directory():
    expected = Path(__file__).resolve().parent.parent / "data" / "raw" / "btc" / "mock"

    from scripts.generate_mock_btc_data import DEFAULT_OUTPUT_DIR

    assert DEFAULT_OUTPUT_DIR == expected
