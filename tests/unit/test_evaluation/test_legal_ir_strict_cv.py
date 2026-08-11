"""Tests for deterministic grouped LegalIR out-of-fold partitions."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.build_strict_legal_ir_cv import (
    build_parser,
    build_strict_grouped_folds,
    run,
    validate_strict_grouped_folds,
)

from udsc2026.evaluation.legal_ir import load_warmup


def _write_train(path: Path) -> None:
    payload = {
        "q1": {"question": "Điều 1 áp dụng thế nào?", "answer": ["d1"]},
        "q2": {"question": "điều 1 ÁP DỤNG thế nào !", "answer": ["d1"]},
        "q3": {"question": "Điều 2 là gì?", "answer": ["d2"]},
        "q4": {"question": "Điều 3 là gì?", "answer": ["d3"]},
        "q5": {"question": "Điều 4 là gì?", "answer": ["d4"]},
        "q6": {"question": "Điều 5 là gì?", "answer": ["d5"]},
        "q7": {"question": "Điều 6 là gì?", "answer": ["d6"]},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_grouped_cv_keeps_normalized_duplicates_in_one_fold_and_is_deterministic(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.json"
    _write_train(train)
    samples = load_warmup(train)

    first = build_strict_grouped_folds(samples, fold_count=5, seed=2026)
    second = build_strict_grouped_folds(samples, fold_count=5, seed=2026)

    assert first == second
    assert any({"q1", "q2"}.issubset(set(fold)) for fold in first)
    validation = validate_strict_grouped_folds(samples, first)
    assert validation["folds_disjoint"] is True
    assert validation["validation_union_equals_all_train"] is True
    assert validation["normalized_question_groups_disjoint"] is True


def test_grouped_cv_writes_required_v2_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    _write_train(train)
    output = tmp_path / "strict_cv_v2"
    args = build_parser().parse_args(
        ["--input", str(train), "--output-dir", str(output), "--folds", "5"]
    )

    written = run(args)

    assert [path.name for path in written] == [
        "folds.json",
        "fold_stats.json",
        "manifest.json",
        "README.md",
    ]
    folds = json.loads((output / "folds.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "fold_stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert folds["schema_version"] == "legal-ir-strict-cv-v2"
    assert folds["seed"] == 2026
    assert len(folds["folds"]) == 5
    assert stats["validation"]["validation_union_equals_all_train"] is True
    assert stats["validation"]["normalized_question_groups_disjoint"] is True
    assert manifest["model"]["used"] is False
    assert manifest["corpus"]["root"] == "data/processed_v3"
    assert "exact NFKC" in (output / "README.md").read_text(encoding="utf-8")
