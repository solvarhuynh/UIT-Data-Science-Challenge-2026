"""Model-free tests for the Task 2 Qwen LoRA pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "finetune_task2_qwen_lora.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task2_qwen_lora", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_model_fingerprint_ignores_volatile_beam_manifest(
    tmp_path: Path,
) -> None:
    script = _load_script()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / ".beam_model_manifest.json"
    manifest.write_text('{"completed_at_utc":"first"}', encoding="utf-8")
    before = script._base_model_fingerprint(tmp_path)
    manifest.write_text('{"completed_at_utc":"second"}', encoding="utf-8")

    assert script._base_model_fingerprint(tmp_path) == before


def test_prepare_is_scoped_and_orders_teacher_contexts() -> None:
    script = _load_script()
    questions = {
        "train": {"question": "Mức phạt?", "answer": "Phạt 20 triệu."},
        "heldout": {"question": "Thời hạn?", "answer": "Ba ngày."},
    }
    labels = [
        {
            "question_id": "heldout",
            "parent_text": "SECRET",
            "reference_overlap": 1.0,
        },
        {
            "question_id": "train",
            "parent_text": "context low",
            "reference_overlap": 0.2,
            "candidate_rank": 1,
            "doc_id": "d1",
            "parent_id": "p1",
        },
        {
            "question_id": "train",
            "parent_text": "context high",
            "reference_overlap": 0.8,
            "candidate_rank": 5,
            "doc_id": "d2",
            "parent_id": "p2",
        },
    ]

    records, report = script.prepare_records(
        questions,
        ["train"],
        labels,
        contexts_per_question=2,
    )

    assert [row["text"] for row in records[0]["contexts"]] == [
        "context low",
        "context high",
    ]
    assert "SECRET" not in repr(records)
    assert report["ignored_out_of_scope_label_rows"] == 1


def test_prepare_reports_missing_candidate_labels() -> None:
    script = _load_script()
    records, report = script.prepare_records(
        {"q": {"question": "Q?", "answer": "A"}},
        ["q"],
        [],
        contexts_per_question=1,
    )

    assert records == []
    assert report["missing_label_questions"] == ["q"]


def test_rank_fusion_can_balance_ce_and_retrieval() -> None:
    script = _load_script()
    parents = [
        {"parent_id": "ce", "rank": 1, "candidate_rank": 10},
        {"parent_id": "retrieval", "rank": 3, "candidate_rank": 1},
    ]

    ce_first = script.fuse_parent_rankings(
        parents,
        ce_weight=1.0,
        retrieval_weight=0.0,
        rrf_k=5,
    )
    retrieval_first = script.fuse_parent_rankings(
        parents,
        ce_weight=0.0,
        retrieval_weight=1.0,
        rrf_k=5,
    )

    assert ce_first[0]["parent_id"] == "ce"
    assert retrieval_first[0]["parent_id"] == "retrieval"
    with pytest.raises(ValueError, match="at least one"):
        script.fuse_parent_rankings(
            parents,
            ce_weight=0.0,
            retrieval_weight=0.0,
            rrf_k=5,
        )


def test_exact_overlay_rejects_ambiguous_training_answers() -> None:
    script = _load_script()
    public = {"public": {"question": "  CÂU hỏi giống ? "}}
    known = {
        "a": {"question": "Câu hỏi giống?", "answer": "A"},
        "b": {"question": "CÂU HỎI GIỐNG ?", "answer": "B"},
    }

    assert script.exact_known_answers(public, known) == {}


def test_chat_prompt_extracts_transformers_five_batch_encoding() -> None:
    script = _load_script()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["enable_thinking"] is False
            assert messages[-1]["role"] == "user"
            return {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}

    assert script._chat_prompt_ids(Tokenizer(), "Q?", "context") == [10, 20, 30]


def test_prediction_alignment_enforces_coverage_and_order() -> None:
    script = _load_script()
    predictions = [
        {"id": "q1", "answer": "one"},
        {"id": "q2", "answer": "two"},
    ]

    assert script.align_predictions(predictions, ["q2", "q1"]) == [
        {"id": "q2", "answer": "two"},
        {"id": "q1", "answer": "one"},
    ]
    with pytest.raises(ValueError, match="coverage mismatch"):
        script.align_predictions(predictions, ["q1", "missing"])


def test_train_cli_accepts_two_gpu_data_parallel() -> None:
    script = _load_script()

    args = script.build_parser().parse_args(
        [
            "train",
            "--train-data",
            "train.jsonl",
            "--model-dir",
            "model",
            "--output-dir",
            "output",
            "--initial-adapter",
            "adapter",
            "--batch-size",
            "2",
            "--data-parallel",
            "--resume",
        ]
    )

    assert args.data_parallel is True
    assert args.batch_size == 2
    assert args.resume is True
    assert args.initial_adapter == Path("adapter")


def test_smoke_cli_and_zero_rrf_are_supported() -> None:
    script = _load_script()
    train = script.build_parser().parse_args(
        [
            "train",
            "--train-data",
            "train.jsonl",
            "--model-dir",
            "model",
            "--output-dir",
            "output",
            "--smoke-only",
        ]
    )
    generate = script.build_parser().parse_args(
        [
            "generate",
            "--questions",
            "questions.json",
            "--rankings",
            "rankings.jsonl",
            "--model-dir",
            "model",
            "--base-model-dir",
            "base-model",
            "--output",
            "predictions.json",
            "--diagnostics",
            "diagnostics.json",
            "--rrf-k",
            "0",
        ]
    )

    assert train.smoke_only is True
    assert generate.rrf_k == 0
    assert generate.base_model_dir == Path("base-model")


def test_latest_epoch_checkpoint_requires_adapter_and_training_state(
    tmp_path: Path,
) -> None:
    script = _load_script()
    first = tmp_path / "checkpoint-epoch-1"
    first.mkdir()
    (first / "adapter_model.safetensors").write_bytes(b"adapter")
    (first / "training_state.pt").write_bytes(b"state")
    incomplete = tmp_path / "checkpoint-epoch-2"
    incomplete.mkdir()
    (incomplete / "adapter_model.safetensors").write_bytes(b"adapter")

    assert script._latest_epoch_checkpoint(tmp_path) == (1, first)
