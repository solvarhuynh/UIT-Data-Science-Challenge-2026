"""Model-free tests for the LegalQA parent cross-encoder training command."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "training"
    / "train_legal_qa_parent_crossencoder.py"
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
