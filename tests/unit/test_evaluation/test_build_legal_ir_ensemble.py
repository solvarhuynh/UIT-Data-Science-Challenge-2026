"""Focused CLI tests for the CPU-only LegalIR ensemble builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "submission" / "build_legal_ir_ensemble.py"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _hit(chunk_id: str, document_id: str) -> dict[str, str]:
    return {"chunk_id": chunk_id, "doc_id": document_id}


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    passages = {
        "d1": "Thời hạn đăng ký phương tiện là mười ngày làm việc.",
        "d2": "Hồ sơ đăng ký bao gồm đơn đề nghị và giấy tờ tùy thân.",
        "d3": "Cơ quan có thẩm quyền tiếp nhận và xử lý hồ sơ.",
        "d4": "Quyền và nghĩa vụ của người yêu cầu được quy định tại điều này.",
        # Empty organizer passages are valid corpus documents; BM25 simply gives
        # them no lexical evidence while dense/BGE may still rank their chunks.
        "d5": "",
        "d6": "Quy định đặc biệt xác định chính xác thời hạn là ba ngày.",
    }
    for document_id, passage in passages.items():
        _write_json(
            contexts / f"context_{document_id}.json",
            {"id": document_id, "passage": passage, "link": "https://example.test"},
        )

    questions = tmp_path / "questions.json"
    labels = tmp_path / "labels.json"
    exclusion = tmp_path / "exclude.json"
    dense = tmp_path / "dense.jsonl"
    reranked = tmp_path / "reranked.jsonl"
    output = tmp_path / "predictions.json"
    components = tmp_path / "components.json"
    _write_json(
        questions,
        {
            "q1": {"question": "THỜI HẠN đặc biệt là bao lâu?", "answer": None},
            "q2": {"question": "Hồ sơ đăng ký gồm gì?", "answer": None},
        },
    )
    _write_json(
        labels,
        {
            "source-exact": {
                "question": "Thời hạn đặc biệt là bao lâu!",
                "answer": ["d6"],
            },
            "source-neighbor": {
                "question": "Hồ sơ đăng ký bao gồm giấy tờ nào?",
                "answer": ["d2"],
            },
        },
    )
    _write_json(exclusion, {"source-exact": {}})

    dense_rows = []
    reranked_rows = []
    for question_id in ("q1", "q2"):
        dense_hits = [
            _hit("c1", "d1"),
            _hit("c2", "d1"),
            _hit("c3", "d2"),
            _hit("c4", "d3"),
            _hit("c5", "d4"),
            _hit("c6", "d5"),
            _hit("c7", "d6"),
        ]
        reranked_hits = [
            _hit("c3", "d2"),
            _hit("c4", "d3"),
            _hit("c5", "d4"),
            _hit("c6", "d5"),
            _hit("c1", "d1"),
            _hit("c2", "d1"),
            _hit("c7", "d6"),
        ]
        dense_rows.append({"question_id": question_id, "hits": dense_hits})
        reranked_rows.append({"question_id": question_id, "hits": reranked_hits})
    _write_jsonl(dense, dense_rows)
    _write_jsonl(reranked, reranked_rows)
    return {
        "contexts": contexts,
        "questions": questions,
        "labels": labels,
        "exclusion": exclusion,
        "dense": dense,
        "reranked": reranked,
        "output": output,
        "components": components,
    }


def _run(paths: dict[str, Path], *extra: object) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--questions",
        str(paths["questions"]),
        "--dense",
        str(paths["dense"]),
        "--reranked",
        str(paths["reranked"]),
        "--labeled",
        str(paths["labels"]),
        "--contexts-dir",
        str(paths["contexts"]),
        "--output",
        str(paths["output"]),
        "--components-output",
        str(paths["components"]),
        *map(str, extra),
    ]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_builder_deduplicates_documents_caches_components_and_overlays_exact_labels(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _run(paths, "--exact-label-overlay")

    assert result.returncode == 0, result.stderr
    assert "questions=2 documents_per_question=5 exact_matches=1" in result.stdout
    predictions = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert [row["id"] for row in predictions] == ["q1", "q2"]
    assert predictions[0]["documents"][0] == "d6"
    assert all(
        len(row["documents"]) == len(set(row["documents"])) == 5
        for row in predictions
    )

    components = json.loads(paths["components"].read_text(encoding="utf-8"))
    assert [row["id"] for row in components] == ["q1", "q2"]
    assert components[0]["dense"] == ["d1", "d2", "d3", "d4", "d5", "d6"]
    assert set(components[0]) == {"id", "dense", "bge", "knn", "bm25"}


def test_exclusion_source_disables_self_label_overlay(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _run(
        paths,
        "--exact-label-overlay",
        "--exclude-question-sources",
        paths["exclusion"],
    )

    assert result.returncode == 0, result.stderr
    assert "exact_matches=0" in result.stdout
    predictions = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert predictions[0]["documents"][0] != "d6"


def test_builder_rejects_invalid_component_weights(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _run(paths, "--bm25-weight", "nan")

    assert result.returncode == 2
    assert "component weights must be finite" in result.stderr
    assert not paths["output"].exists()
