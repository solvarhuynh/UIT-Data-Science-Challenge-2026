"""Subprocess coverage for the ingestion-to-BM25 build boundary."""

import json
import subprocess
import sys
from pathlib import Path

from udsc2026.retrieval.sparse import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _chunk(chunk_id: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "doc_id": "law",
        "text": text,
        "metadata": {},
    }


def test_build_bm25_cli_indexes_sorted_chunk_files(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "b.jsonl").write_text(
        json.dumps(_chunk("c2", "Điều 2 về dân sự"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (chunks_dir / "a.jsonl").write_text(
        json.dumps(_chunk("c1", "Điều 1 về lao động"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "vector-store" / "index.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_bm25.py"),
            "--chunks",
            str(chunks_dir),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Indexed 2 chunks" in result.stdout
    restored = BM25Retriever(str(output))
    restored.load()
    assert restored.search("lao động", top_k=1)[0].chunk_id == "c1"


def test_build_bm25_cli_rejects_duplicates_without_output(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.json"
    chunks.write_text(
        json.dumps(
            [_chunk("duplicate", "Điều 1"), _chunk("duplicate", "Điều 2")],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "index.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_bm25.py"),
            "--chunks",
            str(chunks),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "chunk_id values must be unique" in result.stderr
    assert not output.exists()
