"""Security and round-trip tests for versioned BM25 JSON persistence."""

import json
import os
import stat
from pathlib import Path

import pytest

from udsc2026.contracts import LegalChunk
from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever, _tokenize_for_bm25


def _chunks() -> list[LegalChunk]:
    return [
        LegalChunk(
            chunk_id="labor-35",
            parent_id="labor",
            doc_id="labor-law",
            text="Người lao động có quyền đơn phương chấm dứt hợp đồng.",
            law_name="Bộ luật Lao động",
            article="Điều 35",
        ),
        LegalChunk(
            chunk_id="civil-1",
            parent_id="civil",
            doc_id="civil-law",
            text="Bộ luật dân sự quy định quyền dân sự.",
            law_name="Bộ luật Dân sự",
            article="Điều 1",
        ),
    ]


def test_bm25_json_round_trip_preserves_search_and_contracts(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    original = BM25Retriever(str(index_path))
    original.build_index(_chunks())
    before = original.search("đơn phương chấm dứt hợp đồng", top_k=2)
    original.save()

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["chunks"][0]["chunk_id"] == "labor-35"

    restored = BM25Retriever(str(index_path))
    restored.load()
    after = restored.search("đơn phương chấm dứt hợp đồng", top_k=2)

    assert before[0].parent_id == "labor"
    assert after[0].parent_id == "labor"
    assert [hit.model_dump() for hit in after] == [hit.model_dump() for hit in before]


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"schema_version":true,"chunks":[]}',
        '{"schema_version":999,"chunks":[]}',
        '{"schema_version":1,"chunks":"not-a-list"}',
        '{"schema_version":1,"chunks":[],"unexpected":true}',
        '{"schema_version":1,"chunks":[{"chunk_id":1}]}',
    ],
)
def test_bm25_load_rejects_invalid_or_incompatible_index(
    tmp_path: Path,
    payload: str,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        BM25Retriever(str(index_path)).load()


def test_bm25_rejects_duplicate_or_blank_chunks() -> None:
    retriever = BM25Retriever()
    duplicate = _chunks()
    duplicate[1].chunk_id = duplicate[0].chunk_id
    with pytest.raises(ValueError, match="unique"):
        retriever.build_index(duplicate)

    blank = _chunks()
    blank[0].text = " "
    with pytest.raises(ValueError, match="text must not be blank"):
        retriever.build_index(blank)

    punctuation_only = _chunks()
    punctuation_only[0].text = "...?!"
    with pytest.raises(ValueError, match="no searchable tokens"):
        retriever.build_index(punctuation_only)


def test_bm25_uses_environment_index_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_path = tmp_path / "configured-index.json"
    monkeypatch.setenv("BM25_INDEX_PATH", str(configured_path))

    assert BM25Retriever().index_path == str(configured_path)


def test_bm25_load_rejects_non_finite_json_metadata(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        '{"schema_version":1,"chunks":['
        '{"chunk_id":"c1","doc_id":"law","text":"Điều 1",'
        '"metadata":{"score":NaN}}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite"):
        BM25Retriever(str(index_path)).load()


def test_bm25_normalizes_legal_token_case_and_whitespace() -> None:
    assert _tokenize_for_bm25("Điều   10") == ["điều 10"]
    assert _tokenize_for_bm25("đIỀU 10") == ["điều 10"]
    assert _tokenize_for_bm25("Điểm   a") == ["điểm a"]
    assert _tokenize_for_bm25("ĐIỂM Đ") == ["điểm đ"]

    retriever = BM25Retriever()
    chunks = [
        LegalChunk(
            chunk_id="article-10",
            doc_id="law",
            text="Điều   10 quy định quyền của người lao động.",
        ),
        LegalChunk(
            chunk_id="article-20",
            doc_id="law",
            text="Điều 20 quy định nghĩa vụ của người sử dụng lao động.",
        ),
        LegalChunk(
            chunk_id="article-30",
            doc_id="law",
            text="Điều 30 quy định về đối thoại tại nơi làm việc.",
        ),
    ]
    retriever.build_index(chunks)

    hits = retriever.search("đIỀU 10", top_k=3)

    assert hits[0].chunk_id == "article-10"


def test_bm25_returns_no_hits_for_query_without_searchable_tokens() -> None:
    retriever = BM25Retriever()
    retriever.build_index(_chunks())

    assert retriever.search("...?! — ©", top_k=2) == []
    assert retriever.search("xyzqwerty abcfoobar", top_k=2) == []
    assert (
        retriever.search(
            "đơn phương",
            top_k=2,
            filters={"law_name": "Bộ luật Dân sự"},
        )
        == []
    )


def test_bm25_save_requests_container_readable_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "index.json"
    retriever = BM25Retriever(str(index_path))
    retriever.build_index(_chunks())
    applied_modes: list[int] = []
    real_chmod = os.chmod

    def record_chmod(path: str, mode: int) -> None:
        applied_modes.append(mode)
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", record_chmod)

    retriever.save()

    assert applied_modes == [0o644]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_bm25_index_is_readable_by_container_uid_on_posix(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    retriever = BM25Retriever(str(index_path))
    retriever.build_index(_chunks())

    retriever.save()

    assert stat.S_IMODE(index_path.stat().st_mode) == 0o644
