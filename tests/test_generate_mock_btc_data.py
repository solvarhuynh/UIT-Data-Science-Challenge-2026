"""Tests for the synthetic BTC corpus generator script."""

import importlib.util
import os
from pathlib import Path

import pytest

from udsc2026.ingestion.readers import extract_raw_documents


def _load_generator_module():
    path = Path("scripts/generate_mock_btc_data.py")
    spec = importlib.util.spec_from_file_location("mock_btc_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mock_generator_creates_reader_compatible_corpus(tmp_path):
    generator = _load_generator_module()
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


def test_mock_generator_does_not_follow_preexisting_hard_link(tmp_path: Path) -> None:
    generator = _load_generator_module()
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
