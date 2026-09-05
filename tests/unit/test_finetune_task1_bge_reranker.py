"""Model-free contract tests for the Kaggle-only P13 BGE fine-tuning runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "training" / "finetune_task1_bge_reranker.py"
SPEC = importlib.util.spec_from_file_location("p13_finetune", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = tmp_path / "train.json"
    folds = tmp_path / "folds.json"
    candidates = tmp_path / "candidates.jsonl"
    negatives = tmp_path / "negatives"
    train_payload: dict[str, dict[str, object]] = {}
    fold_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for fold in range(5):
        query_id = f"q{fold}"
        train_payload[query_id] = {
            "question": f"question {fold}",
            "answer": [f"gold-{fold}"],
        }
        fold_rows.append({"fold": fold, "validation_ids": [query_id]})
        candidate_rows.append(
            {
                "question_id": query_id,
                "hits": [
                    {
                        "doc_id": f"gold-{fold}",
                        "chunk_id": f"gold-{fold}-c",
                        "rank": 1,
                        "text": f"GOLD evidence {fold}",
                    },
                    {
                        "doc_id": f"wrong-{fold}",
                        "chunk_id": f"wrong-{fold}-c",
                        "rank": 2,
                        "text": f"wrong legal evidence {fold}",
                    },
                ],
            }
        )
    train.write_text(json.dumps(train_payload), encoding="utf-8")
    folds.write_text(
        json.dumps({"schema_version": "legal-ir-strict-cv-v2", "folds": fold_rows}),
        encoding="utf-8",
    )
    candidates.write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    negatives.mkdir()
    for training_fold in range(5):
        rows = []
        for source_fold in range(5):
            if source_fold == training_fold:
                continue
            rows.append(
                {
                    "query_id": f"q{source_fold}",
                    "query": f"question {source_fold}",
                    "positive_doc": f"gold-{source_fold}",
                    "positive_evidence": f"GOLD evidence {source_fold}",
                    "negative_doc": f"wrong-{source_fold}",
                    "negative_evidence": f"wrong legal evidence {source_fold}",
                    "negative_type": "semantic_confuser",
                    "fold": training_fold,
                    "query_validation_fold": source_fold,
                    "provenance": {"law_name": "Law Test"},
                }
            )
        (negatives / f"fold_{training_fold}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return train, folds, candidates, negatives


class FakeBackend:
    """Records document inputs without importing or loading neural weights."""

    score_inputs: list[tuple[str, tuple[str, ...]]] = []

    def __init__(self, *_: object, **__: object) -> None:
        self.trained = False

    @property
    def model_revision(self) -> str:
        return "fake-bge-revision"

    def prepare_training(self, examples: list[object], epochs: int) -> None:
        assert examples and epochs == 2

    def fit_epoch(self, examples: list[object]) -> float:
        assert {example.label for example in examples} == {0.0, 1.0}
        self.trained = True
        return 0.25

    def score(self, query: str, documents: list[str]) -> list[float]:
        type(self).score_inputs.append((query, tuple(documents)))
        return [10.0 if "GOLD evidence" in document else 0.0 for document in documents]

    def save_checkpoint(self, directory: Path, metadata: dict[str, object]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mock_checkpoint.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def close(self) -> None:
        pass


def _fake_factory(*args: object, **kwargs: object) -> FakeBackend:
    del args, kwargs
    return FakeBackend()


def test_fold_dataset_rejects_validation_label_leakage() -> None:
    record = {
        "query_id": "q0",
        "query": "question",
        "positive_doc": "gold",
        "positive_evidence": "gold evidence",
        "negative_doc": "wrong",
        "negative_evidence": "wrong evidence",
        "negative_type": "semantic_confuser",
        "fold": 0,
    }
    with pytest.raises(ValueError, match="leakage"):
        MODULE.build_fold_dataset(
            [record],
            fold=0,
            fold_map={"q0": 0},
            ablation="semi-hard-plus-hard",
            same_law_boost=2,
            max_training_pairs=0,
        )


def test_mock_runner_writes_strict_oof_bundle_with_fixed_candidates(
    tmp_path: Path,
) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--output-dir",
            str(tmp_path / "output"),
            "--epochs",
            "2",
        ]
    )
    FakeBackend.score_inputs.clear()
    outputs = MODULE.run(args, backend_factory=_fake_factory)

    assert outputs == [tmp_path / "output"]
    decision = json.loads(
        (tmp_path / "output" / "final_decision.json").read_text(encoding="utf-8")
    )
    assert decision["base_model"] == "BAAI/bge-reranker-v2-m3"
    assert (tmp_path / "output" / "fold_0" / "train_ids.json").is_file()
    assert (tmp_path / "output" / "fold_0" / "checkpoint").is_dir()
    oof = [
        json.loads(line)
        for line in (tmp_path / "output" / "oof_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(oof) == 5
    assert all(len(row["baseline_top5"]) <= 5 for row in oof)
    by_query: dict[str, set[tuple[str, ...]]] = {}
    for query, documents in FakeBackend.score_inputs:
        by_query.setdefault(query, set()).add(documents)
    assert all(len(documents) == 1 for documents in by_query.values())


def test_dry_run_is_model_free_and_serializes_fold_manifests(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--output-dir",
            str(tmp_path / "dry"),
            "--dry-run",
        ]
    )

    MODULE.run(args, backend_factory=lambda *_args, **_kwargs: pytest.fail("loaded"))

    decision = json.loads(
        (tmp_path / "dry" / "final_decision.json").read_text(encoding="utf-8")
    )
    assert decision["status"] == "NOT_RUN"
    manifest = json.loads(
        (tmp_path / "dry" / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "DRY_RUN"
    assert (tmp_path / "dry" / "fold_4" / "dataset_manifest.json").is_file()


def test_config_validation_rejects_nonpositive_learning_rate(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--learning-rate",
            "0",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="learning-rate"):
        MODULE.run(args)


def test_cpu_diagnostic_uses_requested_tiny_subset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    args = MODULE.build_parser().parse_args(
        [
            "--train", str(train),
            "--folds", str(folds),
            "--candidates", str(candidates),
            "--negatives-dir", str(negatives),
            "--base-model", "models/reranker",
            "--output-dir", str(tmp_path / "cpu-diagnostic"),
            "--device", "cpu",
            "--diagnostic-only",
            "--folds-to-run", "0",
            "--max-training-pairs", "20",
            "--max-validation-queries", "1",
        ]
    )
    MODULE.run(args, backend_factory=_fake_factory)
    assert "diagnostic_device = cpu" in capsys.readouterr().out


def test_explicit_cuda_requires_available_cuda() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA is required"):
        MODULE._resolve_device("cuda", dry_run=False)


def test_parser_accepts_deeper_candidate_pool_for_high_recall_oof() -> None:
    args = MODULE.build_parser().parse_args(
        ["--candidate-depth", "300", "--inference-batch-size", "24"]
    )
    assert args.candidate_depth == 300
    assert args.inference_batch_size == 24


def test_training_pair_cap_is_query_balanced_and_deterministic() -> None:
    records = [
        {
            "query_id": query_id,
            "positive_doc": f"gold-{query_id}",
            "negative_doc": f"wrong-{query_id}-{index}",
            "negative_type": "semantic_confuser",
            "_copy_index": 0,
        }
        for query_id in ("q1", "q2", "q3")
        for index in range(3)
    ]

    first = MODULE._balanced_training_subset(records, limit=3, fold=2)
    second = MODULE._balanced_training_subset(records, limit=3, fold=2)

    assert first == second
    assert {row["query_id"] for row in first} == {"q1", "q2", "q3"}


def test_resume_rejects_changed_fold_dataset_before_overwrite(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    output = tmp_path / "resume-output"
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--output-dir",
            str(output),
        ]
    )
    MODULE.run(args, backend_factory=_fake_factory)
    fold_zero = negatives / "fold_0.jsonl"
    changed = fold_zero.read_text(encoding="utf-8").replace(
        '"negative_doc": "wrong-1"', '"negative_doc": "changed-1"', 1
    )
    fold_zero.write_text(changed, encoding="utf-8")
    args.resume = True

    with pytest.raises(ValueError, match="dataset changed"):
        MODULE.run(args, backend_factory=_fake_factory)


def test_resume_retrains_fold_when_checkpoint_is_missing(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    output = tmp_path / "resume-missing-checkpoint"
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--output-dir",
            str(output),
        ]
    )
    MODULE.run(args, backend_factory=_fake_factory)
    checkpoint = output / "fold_0" / "checkpoint" / "mock_checkpoint.json"
    checkpoint.unlink()
    args.resume = True

    MODULE.run(args, backend_factory=_fake_factory)

    assert checkpoint.is_file()


def test_completed_resume_preserves_base_model_revision(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _write_fixture_inputs(tmp_path)
    output = tmp_path / "completed-resume"
    args = MODULE.build_parser().parse_args(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--candidates",
            str(candidates),
            "--negatives-dir",
            str(negatives),
            "--output-dir",
            str(output),
        ]
    )
    MODULE.run(args, backend_factory=_fake_factory)
    args.resume = True

    MODULE.run(args, backend_factory=_fake_factory)

    manifest = json.loads(
        (output / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base_model"]["revision"] == "fake-bge-revision"


def test_gpu_backend_close_releases_training_state() -> None:
    empty_cache_calls: list[bool] = []
    backend = object.__new__(MODULE.TorchBGERerankerBackend)
    backend._optimizer = object()
    backend._scheduler = object()
    backend._scaler = object()
    backend._model = object()
    backend._tokenizer = object()
    backend._torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(empty_cache=lambda: empty_cache_calls.append(True))
    )

    backend.close()

    assert backend._optimizer is None
    assert backend._scheduler is None
    assert backend._scaler is None
    assert not hasattr(backend, "_model")
    assert not hasattr(backend, "_tokenizer")
    assert empty_cache_calls == [True]
