"""Model-free integration test for the TV5 reranker command."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_reranker.py"
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "tv5"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_reranker_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    """Injectable replacement that avoids downloading or loading a model."""

    init_kwargs: dict[str, object] = {}

    def __init__(self, model: str, **kwargs: object) -> None:
        type(self).init_kwargs = {"model": model, **kwargs}

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        del query, batch_size
        return [float(len(document)) for document in documents]


def test_cli_run_writes_complete_reproducible_bundle(tmp_path: Path) -> None:
    script = _load_script()
    args = script.build_parser().parse_args(
        [
            "--benchmark",
            str(FIXTURES / "dev_benchmark.jsonl"),
            "--candidates",
            str(FIXTURES / "predictions_before.json"),
            "--model",
            "BAAI/bge-reranker-v2-m3",
            "--device",
            "cuda:0",
            "--fp16",
            "--allow-remote-model",
            "--candidate-k",
            "3",
            "--top-n",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    written = script.run(args, client_factory=FakeClient)

    assert len(written) == 9
    assert all(path.is_file() for path in written)
    assert FakeClient.init_kwargs["use_fp16"] is True
    assert FakeClient.init_kwargs["local_files_only"] is False
    manifest = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    assert manifest["model"]["name_or_path"] == "BAAI/bge-reranker-v2-m3"
    assert manifest["evaluation"]["sample_count"] == 12
    assert manifest["evaluation"]["candidate_pair_count"] == 31
    assert len(manifest["inputs"]["benchmark_sha256"]) == 64
    comparison = json.loads(
        (tmp_path / "evaluation" / "comparison.json").read_text("utf-8")
    )
    assert comparison["report_type"] == "comparison"


def test_cli_rejects_top_n_larger_than_candidate_pool(tmp_path: Path) -> None:
    script = _load_script()
    args = script.build_parser().parse_args(
        [
            "--benchmark",
            str(FIXTURES / "dev_benchmark.jsonl"),
            "--candidates",
            str(FIXTURES / "predictions_before.json"),
            "--top-n",
            "5",
            "--candidate-k",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    try:
        script.run(args, client_factory=FakeClient)
    except ValueError as exc:
        assert "--top-n must not exceed --candidate-k" in str(exc)
    else:
        raise AssertionError("invalid candidate limit was accepted")
    assert list(tmp_path.iterdir()) == []
