from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluation"
    / "generate_precomputed_legal_ir_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("precomputed_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload(doc_id: str, chunk_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "law_name": f"Law {doc_id}",
        "metadata": {"structure_status": "structured"},
    }


def test_collapse_assigns_dense_document_ranks_and_keeps_top_evidence() -> None:
    result = MODULE.collapse_faiss_row(
        [0, 1, 2, 3, 4],
        [0.9, 0.8, 0.7, 0.6, 0.5],
        {
            0: _payload("a", "a1", "A one"),
            1: _payload("a", "a2", "A two"),
            2: _payload("b", "b1", "B one"),
            3: _payload("c", "c1", "C one"),
            4: _payload("b", "b2", "B two"),
        },
        chunk_k=5,
        document_depth=2,
        evidence_limit=2,
    )

    assert result["unique_document_count"] == 3
    assert [row["doc_id"] for row in result["documents"]] == ["a", "b"]
    assert [row["rank"] for row in result["documents"]] == [1, 2]
    assert [row["best_chunk_rank"] for row in result["documents"]] == [1, 3]
    assert [row["evidence_id"] for row in result["documents"][0]["evidence"]] == [
        "a1",
        "a2",
    ]
    assert [row["evidence_id"] for row in result["documents"][1]["evidence"]] == [
        "b1",
        "b2",
    ]


def test_collapse_rejects_missing_payload_and_non_finite_score() -> None:
    with pytest.raises(ValueError, match="missing payload"):
        MODULE.collapse_faiss_row(
            [9],
            [0.5],
            {},
            chunk_k=1,
            document_depth=1,
            evidence_limit=1,
        )
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.collapse_faiss_row(
            [0],
            [float("nan")],
            {0: _payload("a", "a1", "text")},
            chunk_k=1,
            document_depth=1,
            evidence_limit=1,
        )


def test_validated_batch_normalizes_without_mutating_source() -> None:
    embeddings = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    result = MODULE._validated_batch(embeddings, 0, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])
    assert embeddings.tolist() == [[3.0, 4.0], [0.0, 2.0]]


def test_validated_batch_rejects_zero_or_non_finite_vectors() -> None:
    with pytest.raises(ValueError, match="zero/invalid norm"):
        MODULE._validated_batch(np.zeros((1, 2), dtype=np.float32), 0, 1)
    with pytest.raises(ValueError, match="invalid"):
        MODULE._validated_batch(
            np.asarray([[float("inf"), 1.0]], dtype=np.float32), 0, 1
        )


def test_parser_rejects_invalid_depth() -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--document-depth", "0"])


def test_question_loader_requires_labels_unless_public_mode(tmp_path: Path) -> None:
    questions = tmp_path / "public.json"
    questions.write_text(
        json.dumps({"q1": {"question": "Câu hỏi", "answer": None}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no usable gold"):
        MODULE._load_questions(questions)
    ids, gold = MODULE._load_questions(questions, allow_unlabeled=True)

    assert ids == ["q1"]
    assert gold == {"q1": []}
