"""Tests for the synthetic BTC corpus generator script."""

import json
import os
from pathlib import Path

import pytest
from scripts import generate_mock_btc_data as generator

from udsc2026.ingestion.readers import extract_raw_documents


def test_mock_generator_creates_reader_compatible_corpus(tmp_path: Path) -> None:
    output_dir = tmp_path / "btc" / "mock"

    written = generator.generate_mock_data(output_dir)
    documents = extract_raw_documents(str(output_dir), str(tmp_path / "errors.json"))

    assert (output_dir / "mock_btc_law.txt").exists()
    assert (output_dir / "mock_btc_law.json").exists()
    assert (output_dir / "mock_btc_laws.jsonl").exists()
    assert len(written) in (3, 4)
    assert len(documents) == len(written) + 1
    assert any("DỰ THẢO" in document.raw_text for document in documents)
    assert any("(sau đây gọi là BLLĐ)" in document.raw_text for document in documents)
    assert any("a) Người lao động" in document.raw_text for document in documents)


def test_mock_generator_does_not_follow_preexisting_hard_link(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "mock"
    output_dir.mkdir()
    victim = tmp_path / "victim.txt"
    original = "must remain unchanged"
    victim.write_text(original, encoding="utf-8")
    linked_output = output_dir / "mock_btc_law.txt"
    try:
        os.link(victim, linked_output)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")

    generator.generate_mock_data(output_dir)

    assert victim.read_text(encoding="utf-8") == original
    assert linked_output.read_text(encoding="utf-8") == generator._BASE_TEXT
    assert not linked_output.samefile(victim)


def test_mock_generator_rejects_symbolic_link_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "mock"
    output_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    linked_output = output_dir / "mock_btc_law.txt"
    try:
        os.symlink(victim, linked_output)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        generator.generate_mock_data(output_dir)

    assert victim.read_text(encoding="utf-8") == "must remain unchanged"


def test_mock_generator_preserves_unrelated_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "btc" / "mock"
    output_dir.mkdir(parents=True)
    preserved_file = output_dir / "real_btc_file.txt"
    preserved_file.write_text("Không được xoá file này.", encoding="utf-8")

    generator.generate_mock_data(output_dir)

    assert preserved_file.read_text(encoding="utf-8") == "Không được xoá file này."


def test_default_output_is_resolved_from_repository() -> None:
    expected = Path(__file__).resolve().parent.parent / "data" / "raw" / "btc" / "mock"

    assert generator.DEFAULT_OUTPUT_DIR == expected


def test_json_output_exposes_compatible_identity_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "mock"

    generator.generate_mock_data(output_dir)
    record = json.loads((output_dir / "mock_btc_law.json").read_text(encoding="utf-8"))

    assert record["id"] == record["document_id"] == "mock-json-law"
    assert record["effective_date"] == "2026-01-01"
    assert record["metadata"]["dataset"] == "synthetic-development-only"
