"""Focused leakage and source-span tests for the third-parent selector."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from nltk.translate.meteor_score import meteor_score

from scripts.submission.supervised_legal_qa_third_parent import (
    CandidateDataset,
    CandidateMeta,
    FEATURE_NAMES,
    FeatureContext,
    Hit,
    ParentTextStore,
    QueryRanking,
    SupervisedParentPrior,
    add_oof_prior_features,
    build_candidate_dataset,
    contiguous_parent_span,
    enumerate_parent_candidates,
    fast_exact_meteor,
    fit_selector,
    load_rankings,
    validate_strict_split,
)


class _IdentityStemmer:
    def stem(self, word: str) -> str:
        return word


class _EmptyWordNet:
    def synsets(self, word: str) -> list[object]:
        return []


def _hit(
    chunk: str,
    parent: str,
    dense_rank: int,
    bge_rank: int,
) -> Hit:
    return Hit(
        chunk_id=chunk,
        doc_id=f"doc-{parent}",
        parent_id=parent,
        text=f"anchor {chunk}",
        article=f"Điều {parent[-1]}",
        clause="",
        law_name="Luật thử nghiệm",
        section_heading="",
        dense_rank=dense_rank,
        bge_rank=bge_rank,
        dense_score=1.0 / dense_rank,
        bge_score=1.0 / bge_rank,
    )


def test_fast_proxy_matches_nltk_exact_only_profile() -> None:
    reference = "Điều 4 quy định mức phạt tiền là 5 triệu đồng."
    prediction = "Theo Điều 4 mức phạt tiền là 5 triệu đồng."
    expected = meteor_score(
        [reference.split()],
        prediction.split(),
        stemmer=_IdentityStemmer(),
        wordnet=_EmptyWordNet(),
    )

    assert fast_exact_meteor(reference, prediction) == pytest.approx(expected)


def test_parent_enumeration_keeps_two_distinct_old_rrf_parents() -> None:
    ranking = QueryRanking(
        "q1",
        (
            _hit("c1", "p1", 1, 1),
            _hit("c1-duplicate-parent", "p1", 2, 2),
            _hit("c2", "p2", 3, 3),
            _hit("c3", "p3", 4, 4),
        ),
    )

    base, candidates = enumerate_parent_candidates(ranking)

    assert [item.key[1] for item in base] == ["p1", "p2"]
    assert [item.key[1] for item in candidates] == ["p3"]


def test_ranking_loader_streams_and_merges_common_pool(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.jsonl"
    bge_path = tmp_path / "bge.jsonl"
    dense_hits = [
        {"chunk_id": "c1", "doc_id": "d1", "dense_score": 0.9},
        {"chunk_id": "c2", "doc_id": "d2", "dense_score": 0.8},
    ]
    bge_hits = [
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "text": "nguá»“n hai",
            "rerank_score": 4.0,
            "metadata": {"parent_id": "p2", "article": "Äiá»u 2"},
        },
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "nguá»“n má»™t",
            "rerank_score": 3.0,
            "metadata": {"parent_id": "p1", "article": "Äiá»u 1"},
        },
    ]
    dense_path.write_text(
        json.dumps({"question_id": "q1", "hits": dense_hits}) + "\n",
        encoding="utf-8",
    )
    bge_path.write_text(
        json.dumps({"question_id": "q1", "hits": bge_hits}) + "\n",
        encoding="utf-8",
    )

    ranking = load_rankings(dense_path, bge_path, ["q1"])["q1"]

    assert [hit.chunk_id for hit in ranking.hits] == ["c2", "c1"]
    assert [hit.dense_rank for hit in ranking.hits] == [2, 1]
    assert ranking.hits[0].parent_id == "p2"
    assert ranking.hits[0].bge_score == 4.0


def test_contiguous_span_contains_only_one_source_subsequence() -> None:
    source = " ".join(f"w{index}" for index in range(20))
    span = contiguous_parent_span(source, "w9 w10", 6)

    assert len(span.split()) == 6
    assert span in source
    assert "[…]" not in span


def test_strict_split_drops_train_question_text_duplicated_in_eval() -> None:
    questions = {
        "q1": {"question": "Mức phạt là bao nhiêu?", "answer": "a"},
        "q2": {"question": "  MỨC PHẠT LÀ BAO NHIÊU? ", "answer": "b"},
        "q3": {"question": "Điều kiện vay vốn?", "answer": "c"},
    }

    filtered, removed = validate_strict_split(questions, ["q1", "q3"], ["q2"])

    assert filtered == ["q3"]
    assert removed == ["q1"]


def test_strict_split_does_not_bypass_exhaustive_id_coverage() -> None:
    questions = {
        "q1": {"question": "Mức phạt?", "answer": "a"},
        "q2": {"question": "Vay vốn?", "answer": "b"},
        "q3": {"question": "Thời hạn?", "answer": "c"},
    }

    with pytest.raises(ValueError, match="exactly cover questions"):
        validate_strict_split(questions, ["q1"], ["q2"])


def test_supervised_prior_fits_only_explicitly_allowed_questions() -> None:
    dataset = CandidateDataset(
        features=np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32),
        labels=np.asarray([0.2, 0.4], dtype=np.float32),
        sample_weights=np.ones(2, dtype=np.float32),
        metas=[
            CandidateMeta("q1", "doc-a", "Điều 1"),
            CandidateMeta("q2", "doc-b", "Điều 2"),
        ],
        groups={"q1": (0, 1), "q2": (1, 2)},
        base_answers={},
        candidate_spans=None,
    )
    questions = {
        "q1": {"question": "mức phạt hành chính", "answer": "a"},
        "q2": {"question": "điều kiện vay vốn", "answer": "b"},
    }
    prior = SupervisedParentPrior()
    prior.fit(dataset, questions, {"q1"})

    same_doc, forbidden_doc = prior.features_for_group(
        "mức phạt hành chính",
        [
            CandidateMeta("eval", "doc-a", "Điều 1"),
            CandidateMeta("eval", "doc-b", "Điều 2"),
        ],
    )

    assert same_doc[0] > 0.99 and same_doc[1] > 0.99
    assert forbidden_doc == (0.0, 0.0)


def test_tiny_training_dataset_fits_without_retaining_candidate_answers(
    tmp_path: Path,
) -> None:
    questions = {
        "q1": {"question": "mức phạt hành chính", "answer": "anchor c3"},
        "q2": {"question": "điều kiện vay vốn", "answer": "anchor d4"},
    }
    rankings = {
        "q1": QueryRanking(
            "q1",
            tuple(_hit(f"c{n}", f"p{n}", n, n) for n in range(1, 5)),
        ),
        "q2": QueryRanking(
            "q2",
            tuple(_hit(f"d{n}", f"x{n}", n, n) for n in range(1, 5)),
        ),
    }
    args = Namespace(
        base_dense_weight=0.2,
        base_bge_weight=0.8,
        base_rrf_k=60,
        candidate_depth=0,
        base_parent_tokens=32,
        third_parent_tokens=16,
        learning_rate=0.1,
        max_iter=2,
        max_leaf_nodes=3,
        min_samples_leaf=1,
        l2_regularization=0.0,
        seed=2026,
    )
    context = FeatureContext.fit(rankings, ["q1", "q2"])
    dataset = build_candidate_dataset(
        questions,
        rankings,
        ["q1", "q2"],
        ParentTextStore(tmp_path, max_cached_documents=2),
        context,
        args,
        labeled=True,
        retain_answer_spans=False,
    )
    add_oof_prior_features(dataset, questions, ["q1", "q2"], folds=2)
    model = fit_selector(dataset, args)

    assert dataset.features.shape == (4, len(FEATURE_NAMES))
    assert dataset.candidate_spans is None
    assert model.predict(dataset.features).shape == (4,)
