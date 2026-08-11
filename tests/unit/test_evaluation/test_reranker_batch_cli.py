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

    assert len(written) == 10
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
    label_status = json.loads(
        (tmp_path / "evaluation" / "label_status.json").read_text("utf-8")
    )
    assert label_status["generic_chunk_metrics"] == "available"


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _unlabeled_benchmark() -> list[dict[str, object]]:
    return [
        {
            "question_id": "q1",
            "question": "Câu hỏi có nhãn document riêng?",
            "answer": "LABEL_NOT_AVAILABLE",
            "gold_chunk_ids": ["LABEL_NOT_AVAILABLE"],
            "gold_citations": [],
            "difficulty": "easy",
            "question_type": "public_official_unlabeled",
            "metadata": {"unlabeled_public_question": True},
        }
    ]


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "question_id": "q1",
            "hits": [
                {"chunk_id": "wrong-chunk", "doc_id": "wrong", "text": "x"},
                {
                    "chunk_id": "gold-chunk",
                    "doc_id": "gold-doc",
                    "text": "the known gold document has a longer text",
                },
            ],
        }
    ]


def test_unlabeled_transport_benchmark_skips_zero_chunk_metrics(tmp_path: Path) -> None:
    script = _load_script()
    benchmark = tmp_path / "unlabeled.json"
    candidates = tmp_path / "candidates.json"
    _write_json(benchmark, _unlabeled_benchmark())
    _write_json(candidates, _candidates())
    args = script.build_parser().parse_args(
        [
            "--benchmark",
            str(benchmark),
            "--candidates",
            str(candidates),
            "--candidate-k",
            "2",
            "--top-n",
            "2",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    script.run(args, client_factory=FakeClient)

    status = json.loads(
        (tmp_path / "output" / "evaluation" / "label_status.json").read_text("utf-8")
    )
    assert status["status"] == "unlabeled_transport"
    assert status["generic_chunk_metrics"] == "skipped"
    assert not (tmp_path / "output" / "evaluation" / "before.json").exists()


def test_document_reference_fixture_produces_nonzero_legal_ir_metric(
    tmp_path: Path,
) -> None:
    script = _load_script()
    benchmark = tmp_path / "unlabeled.json"
    candidates = tmp_path / "candidates.json"
    references = tmp_path / "references.json"
    _write_json(benchmark, _unlabeled_benchmark())
    _write_json(candidates, _candidates())
    _write_json(
        references,
        {"q1": {"question": "Câu hỏi có nhãn document riêng?", "answer": ["gold-doc"]}},
    )
    args = script.build_parser().parse_args(
        [
            "--benchmark",
            str(benchmark),
            "--candidates",
            str(candidates),
            "--candidate-k",
            "2",
            "--top-n",
            "2",
            "--legal-ir-references",
            str(references),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    script.run(args, client_factory=FakeClient)

    report = json.loads(
        (tmp_path / "output" / "evaluation" / "legal_ir_after.json").read_text("utf-8")
    )
    assert report["official_recall"] == 1.0
    assert report["official_precision"] == 0.5
