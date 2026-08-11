"""Strict tests for building LegalQA answers from parent CE rankings."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from scripts.submission.build_legal_qa_parent_scores import (
    bounded_parent_window,
    build_predictions,
    load_parent_scores,
    run,
)


def _parent(
    rank: int,
    logit: float,
    *,
    text: str,
    anchor: str,
) -> dict[str, object]:
    return {
        "doc_id": f"doc-{rank}",
        "parent_id": f"parent-{rank}",
        "parent_text": text,
        "anchor_text": anchor,
        "rank": rank,
        "candidate_rank": rank + 2,
        "crossencoder_logit": logit,
    }


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_anchor_window_matches_parent_expander_single_anchor_behavior() -> None:
    parent = "zero one two three four five six seven eight nine"

    assert bounded_parent_window(parent, "four five", 6) == (
        "two three four five six seven"
    )
    assert bounded_parent_window(parent, "missing child text", 2) == ("missing child")


def test_build_preserves_question_order_and_enforces_total_budget() -> None:
    scores = {
        "q1": [
            _parent(1, 2.0, text="q1 parent", anchor="q1 parent"),
        ],
        "q2": [
            _parent(
                1,
                3.0,
                text="zero one two three four five six seven eight nine",
                anchor="four five",
            ),
            _parent(2, 1.0, text="alpha beta gamma", anchor="alpha"),
        ],
    }

    predictions, summary = build_predictions(
        ["q2", "q1"],
        scores,
        top_parents=2,
        max_parent_tokens=6,
        max_total_tokens=8,
    )

    assert [row["id"] for row in predictions] == ["q2", "q1"]
    assert predictions[0]["answer"] == ("two three four five six seven\nalpha beta")
    assert len(predictions[0]["answer"].split()) == 8
    assert summary["question_count"] == 2
    assert summary["parents_used"] == 3


def test_evidence_backed_rrf_can_rescue_retrieval_parent() -> None:
    ce_first = _parent(1, 2.0, text="CE first", anchor="CE first")
    retrieval_first = _parent(
        2,
        1.0,
        text="retrieval first",
        anchor="retrieval first",
    )
    ce_first["candidate_rank"] = 10
    retrieval_first["candidate_rank"] = 1

    predictions, _ = build_predictions(
        ["q1"],
        {"q1": [ce_first, retrieval_first]},
        top_parents=1,
        max_parent_tokens=8,
        max_total_tokens=8,
        ce_weight=0.4,
        retrieval_weight=0.6,
        rrf_k=5,
    )

    assert predictions == [{"id": "q1", "answer": "retrieval first"}]


def test_score_loader_rejects_rank_logit_contradiction(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    _write_scores(
        scores,
        [
            {
                "question_id": "q1",
                "parents": [
                    _parent(1, 1.0, text="first", anchor="first"),
                    _parent(2, 2.0, text="second", anchor="second"),
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="contradict CE logits"):
        load_parent_scores(scores)


def test_run_is_reference_independent_and_writes_internal_schema(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    question_ids = tmp_path / "question_ids.json"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "candidate.json"
    questions.write_text(
        json.dumps(
            {
                "q2": {"question": "Câu hai?", "answer": "SECRET TWO"},
                "q1": {"question": "Câu một?", "answer": "SECRET ONE"},
                "unused": {"question": "Không đánh giá?", "answer": "SECRET"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    question_ids.write_text('["q2", "q1"]\n', encoding="utf-8")
    _write_scores(
        scores,
        [
            {
                "question_id": "q1",
                "parents": [_parent(1, 2.0, text="nguồn một", anchor="nguồn một")],
            },
            {
                "question_id": "q2",
                "parents": [_parent(1, 2.0, text="nguồn hai", anchor="nguồn hai")],
            },
        ],
    )
    args = Namespace(
        scores=scores,
        questions=questions,
        question_ids=question_ids,
        output=output,
        top_parents=1,
        max_parent_tokens=8,
        max_total_tokens=8,
        ce_weight=1.0,
        retrieval_weight=0.0,
        rrf_k=5,
    )

    summary = run(args)
    predictions = json.loads(output.read_text(encoding="utf-8"))

    assert summary["question_count"] == 2
    assert predictions == [
        {"id": "q2", "answer": "nguồn hai"},
        {"id": "q1", "answer": "nguồn một"},
    ]
    assert "SECRET" not in output.read_text(encoding="utf-8")


def test_build_rejects_incomplete_question_coverage() -> None:
    scores = {
        "q1": [_parent(1, 1.0, text="source", anchor="source")],
    }

    with pytest.raises(ValueError, match="coverage mismatch"):
        build_predictions(
            ["q1", "q2"],
            scores,
            top_parents=1,
            max_parent_tokens=8,
            max_total_tokens=8,
        )
