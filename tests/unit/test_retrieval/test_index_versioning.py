import importlib.util
from pathlib import Path

import pytest

from udsc2026.contracts.chunk import LegalChunk


def _load_index_script():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "data_prep"
        / "index_chunks.py"
    )
    spec = importlib.util.spec_from_file_location("index_chunks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chunks():
    return [
        LegalChunk(chunk_id="b", doc_id="doc", text="beta"),
        LegalChunk(chunk_id="a", doc_id="doc", text="alpha"),
    ]


def test_corpus_hash_is_stable_independent_of_input_order():
    module = _load_index_script()
    assert module.calculate_corpus_hash(_chunks()) == module.calculate_corpus_hash(
        list(reversed(_chunks()))
    )


def test_manifest_match_requires_all_version_keys(tmp_path):
    module = _load_index_script()
    manifest_path = tmp_path / "manifest.json"
    expected = {
        "corpus_hash": "corpus",
        "model_hash": "model",
        "vector_db_type": "faiss",
        "collection_name": "legal_chunks",
    }
    manifest_path.write_text(__import__("json").dumps(expected), encoding="utf-8")
    assert module.is_current_manifest(manifest_path, expected)
    changed = dict(expected, model_hash="changed")
    assert not module.is_current_manifest(manifest_path, changed)


def test_batch_size_cli_override_accepts_positive_integer():
    module = _load_index_script()

    assert module.parse_args(["--batch-size", "128"]).batch_size == 128


def test_batch_size_cli_defaults_to_config_value():
    module = _load_index_script()

    args = module.parse_args([])
    assert args.batch_size is None
    assert args.error_report == Path("artifacts/tv2/diagnostics/index_errors.json")
    assert module.resolve_batch_size(None, {"batch_size": 64}) == 64
    assert module.resolve_batch_size(128, {"batch_size": 64}) == 128


def test_configured_batch_size_must_be_positive():
    module = _load_index_script()

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        module.resolve_batch_size(None, {"batch_size": 0})


@pytest.mark.parametrize("invalid_value", ["0", "-1", "not-an-integer"])
def test_batch_size_cli_rejects_invalid_values(invalid_value: str):
    module = _load_index_script()

    with pytest.raises(SystemExit):
        module.parse_args(["--batch-size", invalid_value])
