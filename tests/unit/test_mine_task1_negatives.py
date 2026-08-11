"""CPU-only tests for Task1 cached negative mining."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "training" / "mine_task1_negatives.py"
SPEC = importlib.util.spec_from_file_location("mine_task1_negatives", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.json"
    folds = tmp_path / "folds.json"
    rankings = tmp_path / "rankings.jsonl"
    train.write_text(
        json.dumps(
            {
                "q1": {"question": "quy định lao động", "answer": ["gold"]},
                "q2": {"question": "quy định bảo hiểm", "answer": ["gold2"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    folds.write_text(
        json.dumps(
            {
                "schema_version": "legal-ir-strict-cv-v2",
                "folds": [
                    {"fold": 0, "validation_ids": ["q1"]},
                    {"fold": 1, "validation_ids": ["q2"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    rankings.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "question_id": "q1",
                    "hits": [
                        {
                            "doc_id": "gold",
                            "rank": 1,
                            "law_name": "Law A",
                            "text": "quy định lao động đúng",
                        },
                        {
                            "doc_id": "same-law",
                            "rank": 2,
                            "law_name": "Law A",
                            "text": "quy định lao động sai điều",
                        },
                        {
                            "doc_id": "ambiguous",
                            "rank": 3,
                            "text": "quy định lao động",
                            "metadata": {"structure_warnings": ["ambiguous_split"]},
                        },
                    ],
                },
                {
                    "question_id": "q2",
                    "hits": [
                        {"doc_id": "gold2", "rank": 1, "text": "bảo hiểm đúng"},
                        {"doc_id": "hard", "rank": 2, "text": "bảo hiểm sai"},
                    ],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return train, folds, rankings


def test_fold_training_records_exclude_held_out_labels(tmp_path: Path) -> None:
    train, folds, rankings = _write_inputs(tmp_path)
    output = tmp_path / "negatives"
    assert (
        MODULE.main(
            [
                "--train",
                str(train),
                "--folds",
                str(folds),
                "--rankings",
                f"hcmute={rankings}",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    fold_zero = [
        json.loads(line)
        for line in (output / "fold_0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert fold_zero and {row["query_id"] for row in fold_zero} == {"q2"}
    assert all(row["query_validation_fold"] != row["fold"] for row in fold_zero)


def test_same_law_and_ambiguous_safety_filters(tmp_path: Path) -> None:
    train, folds, rankings = _write_inputs(tmp_path)
    output = tmp_path / "negatives"
    MODULE.main(
        [
            "--train",
            str(train),
            "--folds",
            str(folds),
            "--rankings",
            f"hcmute={rankings}",
            "--output-dir",
            str(output),
        ]
    )
    fold_one = [
        json.loads(line)
        for line in (output / "fold_1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {(row["negative_doc"], row["negative_type"]) for row in fold_one} == {
        ("same-law", "same_law")
    }
    assert all(row["negative_doc"] != "ambiguous" for row in fold_one)


def test_optional_easy_random_band_is_disabled_by_default_but_classifiable() -> None:
    candidate = MODULE.Candidate(
        doc_id="easy",
        text="văn bản xa hơn",
        law_name="",
        rank=101,
        score=0.01,
        source="hcmute",
        ambiguous=False,
    )
    assert MODULE.classify_negative(
        candidate,
        query="câu hỏi",
        positive_laws=set(),
        hard_rank_max=20,
        semi_min=10,
        semi_max=100,
        easy_rank_min=101,
    ) == ("easy_random", "easy")
