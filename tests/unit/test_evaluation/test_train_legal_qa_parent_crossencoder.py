"""Model-free tests for the LegalQA parent cross-encoder training command."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "training" / "train_legal_qa_parent_crossencoder.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_legal_qa_parent_crossencoder_cli",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PickleTorch(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")

    @staticmethod
    def save(payload: object, stream: Any) -> None:
        pickle.dump(payload, stream)

    @staticmethod
    def load(path: Path, **_: object) -> object:
        with path.open("rb") as stream:
            return pickle.load(stream)


class _CheckpointModel:
    def save_pretrained(
        self,
        directory: Path,
        *,
        safe_serialization: bool,
    ) -> None:
        assert safe_serialization is True
        _write_json(directory / "config.json", {"model_type": "roberta"})
        (directory / "model.safetensors").write_bytes(b"model")


class _CheckpointTokenizer:
    def save_pretrained(self, directory: Path) -> None:
        _write_json(directory / "tokenizer_config.json", {"model_max_length": 256})


def _checkpoint_state(
    script: ModuleType,
    contract: dict[str, object],
    *,
    epoch: int,
    best_epoch: int = 1,
    best_metric: float = 0.75,
) -> dict[str, object]:
    history = [
        {"epoch": number, "train_loss": 1.0 / number} for number in range(1, epoch + 1)
    ]
    return {
        "schema_version": script.TRAINING_STATE_SCHEMA_VERSION,
        "training_contract": contract,
        "epoch": epoch,
        "global_updates": epoch,
        "updates_per_epoch": 1,
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "metrics": history[-1],
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_epoch": epoch},
        "scaler": {},
        "python_rng_state": (3, (1, 2, 3), None),
        "torch_rng_state": b"cpu-rng",
        "cuda_rng_states": [],
        "loader_generator_state": b"loader-rng",
    }


def _train_args(
    script: ModuleType,
    root: Path,
    *,
    resume: bool = False,
) -> Any:
    train_data = root / "train.jsonl"
    model_dir = root / "model"
    train_data.write_text('{"row":1}\n', encoding="utf-8")
    model_dir.mkdir(exist_ok=True)
    _write_json(model_dir / "config.json", {"model_type": "roberta"})
    (model_dir / "model.safetensors").write_bytes(b"base-model")
    argv = [
        "train",
        "--train-data",
        str(train_data),
        "--fit-all",
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(root / "output"),
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--gradient-accumulation",
        "2",
    ]
    if resume:
        argv.append("--resume")
    return script.build_parser().parse_args(argv)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parent(
    parent_id: str,
    doc_id: str,
    text: str,
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "doc_id": doc_id,
        "text": text,
        "law_name": "Luật thử nghiệm",
        "article": "Điều 1",
        "metadata": {},
    }


def _hit(chunk_id: str, parent_id: str, doc_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "doc_id": doc_id,
        "text": "child text",
        "rerank_score": 0.9,
        "metadata": {"parent_id": parent_id},
    }


def test_prepare_removes_normalized_eval_duplicates_and_uses_real_parents(
    tmp_path: Path,
) -> None:
    script = _load_script()
    parents = tmp_path / "parents"
    parents.mkdir()
    parent_texts = {
        "train-pos": "Phạt tiền từ 10 đến 20 triệu đồng.",
        "train-neg": "Hồ sơ đăng ký gồm hai biểu mẫu khác nhau.",
        "eval-pos": "Người lao động được nghỉ ba ngày làm việc.",
        "eval-neg": "Quy định về đăng ký phương tiện giao thông.",
        "dup-pos": "Nội dung chỉ dùng cho hàng bị loại.",
        "dup-neg": "Nội dung âm chỉ dùng cho hàng bị loại.",
    }
    for doc_id, text in parent_texts.items():
        _write_jsonl(
            parents / f"{doc_id}.jsonl",
            [_parent(f"{doc_id}-parent", doc_id, text)],
        )

    questions = tmp_path / "questions.json"
    _write_json(
        questions,
        {
            "train-duplicate": {
                "question": "Câu hỏi giống?",
                "answer": "Nội dung chỉ dùng cho hàng bị loại.",
            },
            "train-good": {
                "question": "Mức phạt là bao nhiêu?",
                "answer": "Phạt tiền từ 10 đến 20 triệu đồng.",
            },
            "eval": {
                "question": "  CÂU hỏi giống ? ",
                "answer": "Người lao động được nghỉ ba ngày làm việc.",
            },
        },
    )
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {
                "question_id": "train-duplicate",
                "hits": [
                    _hit("dup-p", "dup-pos-parent", "dup-pos"),
                    _hit("dup-n", "dup-neg-parent", "dup-neg"),
                ],
            },
            {
                "question_id": "train-good",
                "hits": [
                    _hit("train-p", "train-pos-parent", "train-pos"),
                    _hit("train-n", "train-neg-parent", "train-neg"),
                ],
            },
            {
                "question_id": "eval",
                "hits": [
                    _hit("eval-p", "eval-pos-parent", "eval-pos"),
                    _hit("eval-n", "eval-neg-parent", "eval-neg"),
                ],
            },
        ],
    )
    train_ids = tmp_path / "train_ids.json"
    eval_ids = tmp_path / "eval_ids.json"
    _write_json(train_ids, ["train-duplicate", "train-good"])
    _write_json(eval_ids, ["eval"])
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval.jsonl"
    audit_output = tmp_path / "audit.json"
    args = script.build_parser().parse_args(
        [
            "prepare",
            "--questions",
            str(questions),
            "--candidates",
            str(candidates),
            "--train-ids",
            str(train_ids),
            "--eval-ids",
            str(eval_ids),
            "--parents-dir",
            str(parents),
            "--hard-negatives",
            "1",
            "--min-positive-overlap",
            "0",
            "--negative-gap",
            "0",
            "--train-output",
            str(train_output),
            "--eval-output",
            str(eval_output),
            "--audit-output",
            str(audit_output),
        ]
    )

    written = script.run_prepare(args)

    assert written == [train_output, eval_output, audit_output]
    train_rows = [
        json.loads(line) for line in train_output.read_text("utf-8").splitlines()
    ]
    eval_rows = [
        json.loads(line) for line in eval_output.read_text("utf-8").splitlines()
    ]
    assert {row["question_id"] for row in train_rows} == {"train-good"}
    assert {row["question_id"] for row in eval_rows} == {"eval"}
    assert {row["label"] for row in train_rows} == {0, 1}
    positive = next(row for row in train_rows if row["label"] == 1)
    assert positive["parent_text"] == parent_texts["train-pos"]
    assert positive["anchor_text"] == "child text"
    assert "answer" not in positive
    report = json.loads(audit_output.read_text("utf-8"))
    strict = report["strict_isolation"]
    assert strict["removed_train_exact_eval_duplicates"] == 1
    assert strict["removed_train_ids"] == ["train-duplicate"]
    assert report["train"]["discard_rate"] == 0.0
    assert report["train"]["best_parent_overlap"]["median"] == 1.0


def test_audit_rejects_normalized_question_leakage() -> None:
    script = _load_script()
    base = {
        "schema_version": 1,
        "question": "Cùng một câu hỏi?",
        "doc_id": "doc",
        "parent_text": "parent",
    }
    train_rows = [
        {**base, "question_id": "train", "parent_id": "p", "label": 1},
        {**base, "question_id": "train", "parent_id": "n", "label": 0},
    ]
    eval_rows = [
        {
            **base,
            "question": "  CÙNG một câu hỏi ? ",
            "question_id": "eval",
            "parent_id": "p2",
            "label": 1,
        },
        {
            **base,
            "question": "  CÙNG một câu hỏi ? ",
            "question_id": "eval",
            "parent_id": "n2",
            "label": 0,
        },
    ]

    with pytest.raises(ValueError, match="normalized questions overlap"):
        script.audit_training_rows(train_rows, eval_rows)


def test_reference_overlap_prefers_verbatim_relevant_parent() -> None:
    script = _load_script()
    reference = "Phạt tiền từ 10 đến 20 triệu đồng"

    relevant = script.reference_overlap(
        reference,
        "Theo quy định, phạt tiền từ 10 đến 20 triệu đồng.",
    )
    unrelated = script.reference_overlap(
        reference,
        "Hồ sơ gồm đơn đề nghị và giấy chứng nhận.",
    )

    assert 0 < relevant <= 1
    assert unrelated == 0
    assert relevant > unrelated


def test_hard_model_length_limit_is_256() -> None:
    script = _load_script()

    script._validate_model_length(256)
    with pytest.raises(ValueError, match="hard limit 256"):
        script._validate_model_length(257)


def test_split_ids_uses_organizer_order_and_reports_exact_duplicates(
    tmp_path: Path,
) -> None:
    script = _load_script()
    questions = tmp_path / "questions.json"
    _write_json(
        questions,
        {
            "eval-1": {"question": "Câu thứ nhất", "answer": "A"},
            "eval-2": {"question": "Câu trùng?", "answer": "B"},
            "train-1": {"question": "Câu thứ ba", "answer": "C"},
            "train-dup": {"question": " CÂU trùng ? ", "answer": "D"},
        },
    )
    train_output = tmp_path / "train.json"
    eval_output = tmp_path / "eval.json"
    audit_output = tmp_path / "audit.json"
    args = script.build_parser().parse_args(
        [
            "split-ids",
            "--questions",
            str(questions),
            "--eval-count",
            "2",
            "--train-output",
            str(train_output),
            "--eval-output",
            str(eval_output),
            "--audit-output",
            str(audit_output),
        ]
    )

    script.run_split_ids(args)

    assert json.loads(eval_output.read_text("utf-8")) == ["eval-1", "eval-2"]
    assert json.loads(train_output.read_text("utf-8")) == [
        "train-1",
        "train-dup",
    ]
    report = json.loads(audit_output.read_text("utf-8"))
    assert report["ordering"] == "organizer_json_insertion_order"
    assert report["train_normalized_exact_eval_duplicates"] == 1
    assert report["duplicate_train_ids"] == ["train-dup"]


def test_sigmoid_is_stable_for_large_logits() -> None:
    script = _load_script()

    assert script._sigmoid(1000.0) == 1.0
    assert script._sigmoid(-1000.0) == 0.0


def test_listwise_groups_preserve_questions_and_require_both_labels() -> None:
    script = _load_script()
    rows = [
        {"question_id": "q2", "label": 1},
        {"question_id": "q2", "label": 0},
        {"question_id": "q1", "label": 0},
        {"question_id": "q1", "label": 1},
    ]

    groups = script.group_training_rows(rows)

    assert [[row["question_id"] for row in group] for group in groups] == [
        ["q2", "q2"],
        ["q1", "q1"],
    ]
    with pytest.raises(ValueError, match="needs positive and negative"):
        script.group_training_rows([{"question_id": "q", "label": 1}])


def test_train_parser_exposes_listwise_objective() -> None:
    script = _load_script()

    args = script.build_parser().parse_args(
        [
            "train",
            "--train-data",
            "train.jsonl",
            "--eval-data",
            "eval.jsonl",
            "--objective",
            "listwise",
            "--target-temperature",
            "0.15",
            "--resume",
        ]
    )

    assert args.objective == "listwise"
    assert args.target_temperature == 0.15
    assert args.resume is True


def test_training_contract_tracks_semantics_not_paths_or_gpu_microbatch(
    tmp_path: Path,
) -> None:
    script = _load_script()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_args = _train_args(script, first)
    second_args = _train_args(script, second)
    second_args.batch_size = 4
    second_args.gradient_accumulation = 4
    second_args.device = "cpu"
    second_args.num_workers = 3

    first_contract = script._training_contract(first_args)
    second_contract = script._training_contract(second_args)

    assert first_contract == second_contract
    assert first_contract == {
        "schema_version": script.TRAINING_CONTRACT_VERSION,
        "training_data_sha256": [script._sha256(first_args.train_data[0])],
        "evaluation_data_sha256": [],
        "preverified_input_archive_sha256": None,
        "base_model_fingerprint": script._model_fingerprint(first_args.model_dir),
        "fit_all": True,
        "epochs": 2,
        "effective_batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "max_length": 256,
        "objective": "bce",
        "target_temperature": 0.1,
        "freeze_layers": 6,
        "freeze_embeddings": True,
        "amp": True,
        "seed": 2026,
        "label_profile": script.LABEL_PROFILE,
    }
    second_args.learning_rate = 3e-5
    assert script._training_contract(second_args) != first_contract
    second_args.learning_rate = first_args.learning_rate
    second_args.train_data[0].write_text('{"row":2}\n', encoding="utf-8")
    assert script._training_contract(second_args) != first_contract


def test_preverified_archive_member_hashes_skip_large_data_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    args = _train_args(script, tmp_path)
    archive_hash = "a" * 64
    train_hash = "b" * 64
    args.preverified_input_archive_sha256 = archive_hash
    args.preverified_train_data_sha256 = [train_hash]

    original_sha256 = script._sha256

    def reject_data_rehash(path: Path) -> str:
        if path in args.train_data:
            raise AssertionError("preverified training data must not be rehashed")
        return original_sha256(path)

    monkeypatch.setattr(script, "_sha256", reject_data_rehash)
    contract = script._training_contract(args)
    assert contract["training_data_sha256"] == [train_hash]
    assert contract["preverified_input_archive_sha256"] == archive_hash

    args.preverified_input_archive_sha256 = None
    with pytest.raises(ValueError, match="require --preverified-input"):
        script._training_contract(args)


def test_checkpoint_commit_is_atomic_and_resume_state_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    torch = _PickleTorch()
    contract = {"schema_version": script.TRAINING_CONTRACT_VERSION}
    checkpoint = tmp_path / "checkpoint-latest"
    first_state = _checkpoint_state(script, contract, epoch=1)
    script._save_checkpoint(
        torch,
        _CheckpointModel(),
        _CheckpointTokenizer(),
        checkpoint,
        first_state,
    )
    assert script._checkpoint_is_complete(checkpoint)
    assert (
        script._load_training_state(
            torch,
            checkpoint,
            contract,
            total_epochs=2,
        )["epoch"]
        == 1
    )
    assert script._resume_position(first_state, updates_per_epoch=1) == (2, 1)

    # Beam Volume metadata can transiently claim the committed directory does
    # not exist. Saving must still attempt rename(2) directly and replace the
    # non-empty checkpoint instead of failing with ENOTEMPTY.
    original_exists = Path.exists
    second_state = _checkpoint_state(script, contract, epoch=2)
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "exists",
            lambda path: False if path == checkpoint else original_exists(path),
        )
        script._save_checkpoint(
            torch,
            _CheckpointModel(),
            _CheckpointTokenizer(),
            checkpoint,
            second_state,
        )
    assert (
        script._load_training_state(
            torch,
            checkpoint,
            contract,
            total_epochs=2,
        )["epoch"]
        == 2
    )

    class BrokenModel:
        @staticmethod
        def save_pretrained(*_: object, **__: object) -> None:
            raise RuntimeError("staged model failure")

    with pytest.raises(RuntimeError, match="staged model failure"):
        script._save_checkpoint(
            torch,
            BrokenModel(),
            _CheckpointTokenizer(),
            checkpoint,
            second_state,
        )
    persisted = script._load_training_state(
        torch,
        checkpoint,
        contract,
        total_epochs=2,
    )
    assert persisted["epoch"] == 2
    assert not list(tmp_path.glob(".checkpoint-latest.*.tmp"))

    with pytest.raises(ValueError, match="contract differs"):
        script._load_training_state(
            torch,
            checkpoint,
            {"schema_version": "different"},
            total_epochs=2,
        )
    (checkpoint / "training_state.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="state is corrupt"):
        script._load_training_state(
            torch,
            checkpoint,
            contract,
            total_epochs=2,
        )


def test_completed_manifest_is_validated_then_resume_is_a_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    torch = _PickleTorch()
    args = _train_args(script, tmp_path, resume=True)
    contract = script._training_contract(args)
    output = args.output_dir
    best = output / "checkpoint-final"
    latest = output / "checkpoint-latest"
    best_state = _checkpoint_state(script, contract, epoch=1)
    latest_state = _checkpoint_state(script, contract, epoch=2)
    script._save_checkpoint(
        torch,
        _CheckpointModel(),
        _CheckpointTokenizer(),
        best,
        best_state,
    )
    script._save_checkpoint(
        torch,
        _CheckpointModel(),
        _CheckpointTokenizer(),
        latest,
        latest_state,
    )
    manifest = {
        "schema_version": script.SCHEMA_VERSION,
        "status": "COMPLETE",
        "history": latest_state["history"],
        "best_metric": latest_state["best_metric"],
        "best_epoch": latest_state["best_epoch"],
        "global_updates": latest_state["global_updates"],
        "checkpoint": best.name,
        "latest_checkpoint": latest.name,
        "training_contract": contract,
    }
    script._write_json(output / "training_manifest.json", manifest)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def unexpected_model_load(*_: object, **__: object) -> None:
        raise AssertionError("a completed matching run must not load the model")

    monkeypatch.setattr(script, "_load_model_stack", unexpected_model_load)

    assert script.run_train(args) == [best, output / "training_manifest.json"]

    args.learning_rate = 3e-5
    with pytest.raises(ValueError, match="contract differs"):
        script.run_train(args)


def test_resume_position_starts_at_the_next_epoch() -> None:
    script = _load_script()
    state = _checkpoint_state(
        script,
        {"schema_version": script.TRAINING_CONTRACT_VERSION},
        epoch=2,
    )
    state["global_updates"] = 14
    state["updates_per_epoch"] = 7
    state["scheduler"] = {"last_epoch": 14}

    assert script._resume_position(state, updates_per_epoch=7) == (3, 14)
    with pytest.raises(ValueError, match="updates-per-epoch differs"):
        script._resume_position(state, updates_per_epoch=8)
