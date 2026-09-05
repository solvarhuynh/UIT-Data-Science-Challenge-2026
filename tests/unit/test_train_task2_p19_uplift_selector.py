"""Focused tests for the CPU-only P19 uplift selector."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from udsc2026.evaluation.legal_qa_candidates import build_answer_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/training/train_task2_p19_uplift_selector.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("task2_p19_selector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_numeric_features_are_reference_free_and_fixed_width() -> None:
    script = _load_script()
    features = script.numeric_features(
        "Mức phạt và thời hạn là bao nhiêu?",
        "Phạt 5 triệu đồng trong 10 ngày.",
        "Điều 1 quy định thời hạn 10 ngày và mức phạt 5 triệu đồng.",
    )
    assert len(features) == 9 + len(script._CUES)
    assert all(isinstance(value, float) for value in features)


def test_train_then_select_preserves_exact_known_answer(tmp_path: Path) -> None:
    script = _load_script()
    questions = {
        str(index): {
            "question": f"Câu hỏi pháp luật số {index}?",
            "answer": f"Đáp án chính thức số {index}.",
        }
        for index in range(4)
    }
    qwen = [
        {"id": question_id, "answer": f"Trả lời mô hình số {question_id}."}
        for question_id in questions
    ]
    extractive = [
        {
            "id": question_id,
            "answer": (
                f"Điều {question_id}. Dữ liệu pháp lý chính thức cho câu hỏi "
                f"số {question_id}. Quy định áp dụng trực tiếp."
            ),
        }
        for question_id in questions
    ]
    bank: list[dict[str, object]] = []
    for question_id, record in questions.items():
        candidates = build_answer_candidates(
            record["question"],
            next(row["answer"] for row in qwen if row["id"] == question_id),
            next(row["answer"] for row in extractive if row["id"] == question_id),
        )
        for profile_index, candidate in enumerate(candidates):
            bank.append(
                {
                    "id": question_id,
                    "profile": candidate.profile,
                    "answer": candidate.answer,
                    "meteor": 0.4 + 0.001 * profile_index,
                    "rouge_l": 0.5,
                }
            )

    questions_path = tmp_path / "questions.json"
    qwen_path = tmp_path / "qwen.json"
    extractive_path = tmp_path / "extractive.json"
    bank_path = tmp_path / "bank.jsonl"
    selector_path = tmp_path / "selector.joblib"
    _write_json(questions_path, questions)
    _write_json(qwen_path, qwen)
    _write_json(extractive_path, extractive)
    bank_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bank),
        encoding="utf-8",
    )
    train_args = argparse.Namespace(
        questions=questions_path,
        candidate_bank=bank_path,
        output=selector_path,
        baseline="raw_prefix_352",
        seed=2026,
        estimators=100,
        ensemble_size=1,
    )
    script.run_train(train_args)

    public_questions = {
        "public": {
            "question": questions["0"]["question"],
        }
    }
    public_questions_path = tmp_path / "public_questions.json"
    public_qwen_path = tmp_path / "public_qwen.json"
    public_extractive_path = tmp_path / "public_extractive.json"
    output_path = tmp_path / "predictions.json"
    diagnostics_path = tmp_path / "diagnostics.json"
    _write_json(public_questions_path, public_questions)
    _write_json(
        public_qwen_path,
        [{"id": "public", "answer": "Một câu trả lời mô hình."}],
    )
    _write_json(
        public_extractive_path,
        [
            {
                "id": "public",
                "answer": "Điều 0. Dữ liệu pháp lý chính thức. Quy định áp dụng.",
            }
        ],
    )
    select_args = argparse.Namespace(
        questions=public_questions_path,
        qwen=public_qwen_path,
        extractive=public_extractive_path,
        selector=selector_path,
        output=output_path,
        diagnostics=diagnostics_path,
        known_answers=questions_path,
    )
    script.run_select(select_args)
    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert predictions == [{"id": "public", "answer": questions["0"]["answer"]}]
    assert diagnostics[0]["profile"] == "exact_known_answer"
