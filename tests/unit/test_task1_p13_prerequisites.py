"""CPU tests for the P13 full-coverage prerequisite checker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluation" / "check_task1_p13_prerequisites.py"
SPEC = importlib.util.spec_from_file_location("p13_prerequisites", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = tmp_path / "train.json"
    folds = tmp_path / "folds.json"
    candidates = tmp_path / "candidates.jsonl"
    negatives = tmp_path / "negatives"
    train.write_text(
        json.dumps(
            {
                f"q{index}": {"question": f"q {index}", "answer": [f"d{index}"]}
                for index in range(5)
            }
        ),
        encoding="utf-8",
    )
    folds.write_text(
        json.dumps(
            {
                "schema_version": "legal-ir-strict-cv-v2",
                "folds": [
                    {"fold": index, "validation_ids": [f"q{index}"]}
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates.write_text(
        "".join(
            json.dumps({"question_id": f"q{index}", "hits": []}) + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    negatives.mkdir()
    for fold in range(5):
        rows = [
            {
                "query_id": f"q{index}",
                "positive_doc": f"d{index}",
                "negative_doc": f"n{index}",
                "negative_type": "semantic_confuser",
            }
            for index in range(5)
            if index != fold
        ]
        (negatives / f"fold_{fold}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return train, folds, candidates, negatives


def test_complete_fixture_reports_exact_coverage(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _fixture(tmp_path)
    report = MODULE.inspect_coverage(
        train=train,
        folds=folds,
        candidates=candidates,
        negatives_dir=negatives,
        expected_query_count=5,
    )
    assert report["coverage_status"] == "complete"
    assert report["total_queries"] == 5
    assert report["candidate_query_ids"] == 5
    assert report["queries_missing_negatives"] == []


def test_missing_candidate_fails_prerequisite(tmp_path: Path) -> None:
    train, folds, candidates, negatives = _fixture(tmp_path)
    candidates.write_text(
        candidates.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    report = MODULE.inspect_coverage(
        train=train,
        folds=folds,
        candidates=candidates,
        negatives_dir=negatives,
        expected_query_count=5,
    )
    assert report["coverage_status"] == "incomplete"
    assert len(report["queries_missing_candidates"]) == 4
