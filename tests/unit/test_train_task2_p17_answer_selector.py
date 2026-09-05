"""Contract tests for the P17 answer-selector helper."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/training/train_task2_p17_answer_selector.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p17_selector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> argparse.Namespace:
    questions = {
        f"q{index}": {
            "question": f"Quy định pháp luật số {index} là gì?",
            "answer": f"tham chiếu {index}",
        }
        for index in range(9)
    }
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(questions), encoding="utf-8")
    ids = list(questions)
    id_path = tmp_path / "ids.json"
    id_path.write_text(json.dumps(ids), encoding="utf-8")
    bank = []
    for question_id in ids:
        bank.extend(
            [
                {
                    "id": question_id,
                    "profile": "short",
                    "answer": "câu ngắn",
                    "meteor": 0.4,
                    "rouge_l": 0.5,
                    "target": 0.4,
                },
                {
                    "id": question_id,
                    "profile": "long",
                    "answer": "câu trả lời dài và đúng hơn",
                    "meteor": 0.8,
                    "rouge_l": 0.7,
                    "target": 0.8,
                },
            ]
        )
    bank_path = tmp_path / "bank.jsonl"
    bank_path.write_text(
        "".join(json.dumps(row) + "\n" for row in bank), encoding="utf-8"
    )
    return argparse.Namespace(
        questions=question_path,
        question_ids=id_path,
        candidate_bank=bank_path,
        output_dir=tmp_path / "prepared",
        folds=3,
        seed=2026,
    )


def test_candidate_document_exposes_length_and_both_ends() -> None:
    script = _load()
    answer = " ".join(f"w{index}" for index in range(200))

    document = script.candidate_document("raw_prefix_352", answer)

    assert "raw prefix 352" in document
    assert "200 từ" in document
    assert "w0" in document
    assert "w199" in document


def test_prepare_creates_isolated_complete_folds(tmp_path: Path) -> None:
    script = _load()
    args = _fixture(tmp_path)

    outputs = script.run_prepare(args)

    manifest = json.loads((args.output_dir / "split_manifest.json").read_text())
    assert manifest["question_count"] == 9
    assert manifest["candidate_count"] == 18
    assert sum(manifest["fold_counts"].values()) == 9
    assert args.output_dir / "fold_0_valid_ids.json" in outputs
    all_rows = (args.output_dir / "all.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(all_rows) == 18
    assert sum(json.loads(row)["label"] for row in all_rows) == 9


def test_oof_selector_rejects_leakage_and_selects_unseen_best(tmp_path: Path) -> None:
    script = _load()
    args = _fixture(tmp_path)
    script.run_prepare(args)
    manifest = json.loads((args.output_dir / "split_manifest.json").read_text())
    bank = script._read_jsonl(args.candidate_bank)
    score_paths = []
    for fold in range(3):
        path = tmp_path / f"scores_{fold}.jsonl"
        rows = [
            {
                "id": row["id"],
                "profile": row["profile"],
                "logit": 1.0 if row["profile"] == "long" else 0.0,
                "fold": fold,
            }
            for row in bank
            if manifest["fold_by_id"][row["id"]] == fold
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        score_paths.append(path)
    output = tmp_path / "selected"

    script.run_select_oof(
        argparse.Namespace(
            candidate_bank=args.candidate_bank,
            split_manifest=args.output_dir / "split_manifest.json",
            scores=score_paths,
            output_dir=output,
        )
    )

    report = json.loads((output / "selector_report.json").read_text())
    assert report["oof_fast_metrics"]["meteor"] == pytest.approx(0.8)
    assert report["full_profile_priors"]["long"] == pytest.approx(0.8)
