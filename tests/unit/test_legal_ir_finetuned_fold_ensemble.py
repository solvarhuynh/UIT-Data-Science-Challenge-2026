from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from udsc2026.evaluation.legal_ir_document_candidates import LegalIRDocumentCandidate

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "submission"
    / "build_legal_ir_finetuned_fold_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location("finetuned_fold_ensemble", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate(doc_id: str, rank: int) -> LegalIRDocumentCandidate:
    return LegalIRDocumentCandidate(
        doc_id=doc_id,
        source_ranks={"cached_candidates": rank},
        source_scores={"cached_candidates": 1.0 / rank},
        evidence_chunk_ids=(f"{doc_id}:1",),
        evidence_texts=(f"evidence {doc_id}",),
        best_child_rank=rank,
        best_parent_rank=None,
        first_seen_rank=rank,
        metadata={},
    )


def test_rank_ensemble_averages_folds_and_keeps_dense_anchor() -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2), _candidate("c", 3)]
    rankings = [["b", "a", "c"], ["b", "c", "a"], ["b", "a", "c"]]

    result = MODULE._ensemble_documents(
        candidates,
        rankings,
        rrf_k=60,
        dense_weight=1.0,
        reranker_weight=1.0,
    )

    assert result[0] == "b"
    assert set(result) == {"a", "b", "c"}


def test_rank_ensemble_rejects_changed_candidate_pool() -> None:
    with pytest.raises(ValueError, match="preserve"):
        MODULE._ensemble_documents(
            [_candidate("a", 1), _candidate("b", 2)],
            [["a", "missing"]],
            rrf_k=60,
            dense_weight=1.0,
            reranker_weight=1.0,
        )


def test_decision_gate_requires_explicit_complete_promotion(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "task1-p13-decision-v1",
                "status": "REJECT_CHECKPOINT",
                "promotable": False,
                "complete_oof": True,
                "oof": {"fold_count": 5},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a complete promotable"):
        MODULE._load_decision(decision)


def test_exact_label_map_unions_normalized_duplicates(tmp_path: Path) -> None:
    labels = tmp_path / "train.json"
    labels.write_text(
        json.dumps(
            {
                "q1": {"question": "Điều  5?", "answer": ["a"]},
                "q2": {"question": "điều 5", "answer": ["b"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mapping = MODULE._exact_label_map(labels)

    assert mapping[MODULE._normalize_question("ĐIỀU 5!!!")] == ["a", "b"]
