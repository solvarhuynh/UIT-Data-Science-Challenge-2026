"""Integration-style adapter tests with real FAISS and an in-memory Qdrant fake."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from udsc2026.contracts import LegalChunk
from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter
from udsc2026.infrastructure.vector_db.qdrant_adapter import QdrantAdapter


def _chunks() -> list[LegalChunk]:
    return [
        LegalChunk(
            chunk_id="article-1",
            parent_id="law",
            doc_id="law",
            text="Quyền của người lao động.",
            law_name="Bộ luật Lao động",
            article="Điều 1",
            metadata={"topic": "labor"},
        ),
        LegalChunk(
            chunk_id="article-2",
            parent_id="law",
            doc_id="law",
            text="Quyền dân sự.",
            law_name="Bộ luật Dân sự",
            article="Điều 2",
            metadata={"topic": "civil"},
        ),
    ]


def test_faiss_adapter_round_trip_filter_and_delete(tmp_path: Path) -> None:
    unicode_root = tmp_path / "dữ-liệu"
    adapter = FaissAdapter(str(unicode_root), "legal_chunks")
    adapter.create_collection("legal_chunks", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    adapter.create_collection("legal_chunks", vector_size=2)
    assert adapter.index is not None
    assert adapter.index.ntotal == 2

    hits = adapter.search([1.0, 0.0], top_k=2)
    assert [hit.chunk_id for hit in hits] == ["article-1", "article-2"]
    assert hits[0].metadata["parent_id"] == "law"

    restored = FaissAdapter(str(unicode_root), "legal_chunks")
    filtered = restored.search(
        [1.0, 0.0],
        top_k=2,
        filters={"law_name": "Bộ luật Lao động"},
    )
    assert [hit.chunk_id for hit in filtered] == ["article-1"]

    restored.delete_collection("legal_chunks")
    assert not (unicode_root / "legal_chunks").exists()
    assert restored.search([1.0, 0.0], top_k=1) == []


def test_faiss_cosine_normalizes_vectors_and_queries(tmp_path: Path) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2, distance="cosine")
    adapter.upsert(_chunks(), [[10.0, 0.0], [1.0, 1.0]])

    hits = adapter.search([25.0, 0.0], top_k=2)

    assert [hit.chunk_id for hit in hits] == ["article-1", "article-2"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5)
    with pytest.raises(ValueError, match="non-zero"):
        adapter.search([0.0, 0.0], top_k=1)


def test_faiss_dot_distance_does_not_reject_zero_vectors(tmp_path: Path) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2, distance="dot")
    adapter.upsert([_chunks()[0]], [[0.0, 0.0]])

    assert adapter.search([1.0, 0.0], top_k=1)[0].score == pytest.approx(0.0)
    restored = FaissAdapter(str(tmp_path), "legal")
    assert restored.distance == "dot"


def test_faiss_filter_searches_past_unmatched_top_candidates(tmp_path: Path) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.8, 0.6]])

    hits = adapter.search(
        [1.0, 0.0],
        top_k=1,
        filters={"topic": "civil"},
    )

    assert [hit.chunk_id for hit in hits] == ["article-2"]


def test_faiss_rejects_tampered_or_mismatched_artifacts(tmp_path: Path) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    payload_path = tmp_path / "legal" / "payloads.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["index_sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        FaissAdapter(str(tmp_path), "legal")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable")
def test_faiss_artifacts_are_world_readable(tmp_path: Path) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2)

    for name in ("index.faiss", "payloads.json"):
        assert (tmp_path / "legal" / name).stat().st_mode & 0o777 == 0o644


def test_faiss_upsert_replaces_existing_chunk_without_duplicate(
    tmp_path: Path,
) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    replacement = _chunks()[0].model_copy(update={"text": "Nội dung đã cập nhật."})

    adapter.upsert([replacement], [[0.0, 1.0]])

    assert adapter.index is not None
    assert adapter.index.ntotal == 2
    hits = adapter.search([0.0, 1.0], top_k=2)
    replaced_hit = next(hit for hit in hits if hit.chunk_id == "article-1")
    assert replaced_hit.text == "Nội dung đã cập nhật."


@pytest.mark.parametrize(
    "embeddings",
    [
        [[1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ],
)
def test_faiss_invalid_initial_batch_leaves_no_collection(
    tmp_path: Path,
    embeddings: list[list[float]],
) -> None:
    adapter = FaissAdapter(str(tmp_path), "legal")

    with pytest.raises(ValueError):
        adapter.upsert(_chunks(), embeddings)

    assert adapter.index is None
    assert not (tmp_path / "legal").exists()


def test_faiss_persistence_failure_rolls_back_disk_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import udsc2026.infrastructure.vector_db.faiss_adapter as faiss_module

    adapter = FaissAdapter(str(tmp_path), "legal")
    adapter.create_collection("legal", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    index_path = tmp_path / "legal" / "index.faiss"
    payload_path = tmp_path / "legal" / "payloads.json"
    original_files = (index_path.read_bytes(), payload_path.read_bytes())
    original_results = [hit.model_dump() for hit in adapter.search([1.0, 0.0], top_k=2)]

    real_replace = faiss_module.os.replace
    replace_calls = 0

    def fail_second_commit(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("forced FAISS commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(faiss_module.os, "replace", fail_second_commit)
    with pytest.raises(OSError, match="forced FAISS commit failure"):
        adapter.upsert(
            [
                LegalChunk(
                    chunk_id="article-3",
                    doc_id="law",
                    text="Văn bản mới.",
                )
            ],
            [[0.5, 0.5]],
        )

    assert (index_path.read_bytes(), payload_path.read_bytes()) == original_files
    assert [
        hit.model_dump() for hit in adapter.search([1.0, 0.0], top_k=2)
    ] == original_results
    assert {path.name for path in (tmp_path / "legal").iterdir()} == {
        "index.faiss",
        "payloads.json",
    }


def test_faiss_initial_upsert_commit_failure_leaves_no_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import udsc2026.infrastructure.vector_db.faiss_adapter as faiss_module

    adapter = FaissAdapter(str(tmp_path), "legal")
    real_replace = faiss_module.os.replace
    replace_calls = 0

    def fail_payload_commit(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("forced initial FAISS commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(faiss_module.os, "replace", fail_payload_commit)

    with pytest.raises(OSError, match="forced initial FAISS commit failure"):
        adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    assert adapter.index is None
    assert adapter.payloads == {}
    assert not (tmp_path / "legal").exists()


def test_faiss_adapter_never_traverses_collection_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("safe", encoding="utf-8")
    adapter = FaissAdapter(str(tmp_path / "indexes"), "legal")

    with pytest.raises(ValueError, match="collection_name"):
        adapter.create_collection("../outside", vector_size=2)
    with pytest.raises(ValueError, match="collection_name"):
        adapter.delete_collection("../outside")

    assert marker.read_text(encoding="utf-8") == "safe"


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.existing_vectors: models.VectorParams | None = None

    def collection_exists(self, *, collection_name: str) -> bool:
        assert collection_name
        return self.existing_vectors is not None

    def get_collection(self, *, collection_name: str) -> SimpleNamespace:
        assert collection_name
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=self.existing_vectors)
            )
        )

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        self.existing_vectors = kwargs["vectors_config"]

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)

    def search(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.search_calls.append(kwargs)
        return [
            SimpleNamespace(
                payload={
                    "chunk_id": "article-1",
                    "doc_id": "law",
                    "text": "Quyền của người lao động.",
                    "law_name": "Bộ luật Lao động",
                    "article": "Điều 1",
                    "metadata": {"topic": "labor"},
                },
                score=0.9,
            )
        ]

    def delete_collection(self, *, collection_name: str) -> None:
        self.deleted.append(collection_name)


def test_qdrant_adapter_maps_models_payloads_filters_and_deletion() -> None:
    adapter = QdrantAdapter("http://localhost:6333", "legal_chunks")
    fake = _FakeQdrantClient()
    adapter.client = fake  # type: ignore[assignment]

    adapter.create_collection("legal_v2", vector_size=2, distance="dot")
    vector_config = fake.create_calls[0]["vectors_config"]
    assert vector_config.size == 2
    assert vector_config.distance == models.Distance.DOT
    adapter.create_collection("legal_v2", vector_size=2, distance="dot")
    assert len(fake.create_calls) == 1

    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    points = fake.upsert_calls[0]["points"]
    assert len(points) == 2
    assert points[0].payload["chunk_id"] == "article-1"
    assert fake.upsert_calls[0]["wait"] is True

    hits = adapter.search(
        [1.0, 0.0],
        top_k=3,
        filters={"law_name": "Bộ luật Lao động", "article": ["Điều 1"]},
    )
    assert [hit.chunk_id for hit in hits] == ["article-1"]
    assert fake.search_calls[0]["limit"] == 3
    query_filter = fake.search_calls[0]["query_filter"]
    assert isinstance(query_filter, models.Filter)
    assert [condition.key for condition in query_filter.must] == [
        "law_name",
        "article",
    ]

    adapter.search([1.0, 0.0], top_k=3, filters={"topic": "labor"})
    metadata_filter = fake.search_calls[1]["query_filter"]
    assert metadata_filter.must[0].key == "metadata.topic"

    adapter.delete_collection("legal_v2")
    assert fake.deleted == ["legal_v2"]


def test_qdrant_create_rejects_incompatible_existing_collection() -> None:
    adapter = QdrantAdapter("http://localhost:6333", "legal")
    fake = _FakeQdrantClient()
    fake.existing_vectors = models.VectorParams(
        size=3,
        distance=models.Distance.COSINE,
    )
    adapter.client = fake  # type: ignore[assignment]

    with pytest.raises(ValueError, match="incompatible"):
        adapter.create_collection("legal", vector_size=2)

    assert fake.create_calls == []


def test_qdrant_adapter_round_trip_with_local_client() -> None:
    adapter = QdrantAdapter("http://localhost:6333", "legal")
    adapter.client = QdrantClient(":memory:")
    adapter.create_collection("legal", vector_size=2)
    adapter.create_collection("legal", vector_size=2)
    adapter.upsert(_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    hits = adapter.search(
        [1.0, 0.0],
        top_k=2,
        filters={"topic": "labor"},
    )

    assert [hit.chunk_id for hit in hits] == ["article-1"]


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("create_collection", ("legal", 0)),
        ("create_collection", ("../outside", 2)),
        ("search", ([], 1)),
        ("search", ([0.1], True)),
        ("delete_collection", ("../outside",)),
    ],
)
def test_vector_adapters_reject_invalid_boundaries(
    tmp_path: Path,
    method: str,
    args: tuple[Any, ...],
) -> None:
    adapters = [
        FaissAdapter(str(tmp_path), "legal"),
        QdrantAdapter("http://localhost:6333", "legal"),
    ]
    for adapter in adapters:
        with pytest.raises(ValueError):
            getattr(adapter, method)(*args)
