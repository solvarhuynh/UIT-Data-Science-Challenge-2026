"""FAISS implementation with a JSON metadata side-store."""

import json
import logging
import shutil
from pathlib import Path

import faiss
import numpy as np

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.vector_db.base import VectorDBAdapter, chunk_payload, payload_to_hit

logger = logging.getLogger(__name__)


class FaissAdapter(VectorDBAdapter):
    """Store normalized vectors in FAISS and payloads beside the index on disk."""

    def __init__(self, index_path: str, collection_name: str) -> None:
        self.root = Path(index_path) / collection_name
        self.collection_name = collection_name
        self.index: faiss.Index | None = None
        self.payloads: dict[int, dict] = {}
        self._load()

    def _load(self) -> None:
        index_file = self.root / "index.faiss"
        payload_file = self.root / "payloads.json"
        if index_file.exists() and payload_file.exists():
            self.index = faiss.read_index(str(index_file))
            self.payloads = {int(k): v for k, v in json.loads(payload_file.read_text(encoding="utf-8")).items()}

    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        if distance not in {"cosine", "dot"}:
            raise ValueError("FAISS adapter supports cosine/dot distance only")
        self.root = self.root.parent / name
        self.collection_name = name
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(vector_size)
        self.payloads = {}
        self._persist()

    def _persist(self) -> None:
        if self.index is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.root / "index.faiss"))
        (self.root / "payloads.json").write_text(json.dumps(self.payloads, ensure_ascii=False), encoding="utf-8")

    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        if self.index is None:
            self.create_collection(self.collection_name, len(embeddings[0]))
        vectors = np.asarray(embeddings, dtype="float32")
        start = self.index.ntotal
        self.index.add(vectors)
        self.payloads.update({start + i: chunk_payload(chunk) for i, chunk in enumerate(chunks)})
        self._persist()

    def search(self, query_vector: list[float], top_k: int, filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        if filters:
            logger.warning("FAISS backend applies filters after vector search; results may be fewer than top_k")
        scores, ids = self.index.search(np.asarray([query_vector], dtype="float32"), min(top_k, self.index.ntotal))
        hits = []
        for score, item_id in zip(scores[0], ids[0]):
            payload = self.payloads.get(int(item_id))
            if payload and all(payload.get(key) in value if isinstance(value, list) else payload.get(key) == value for key, value in (filters or {}).items()):
                hits.append(payload_to_hit(payload, float(score)))
        return hits

    def delete_collection(self, name: str) -> None:
        target = self.root.parent / name
        if target.exists():
            shutil.rmtree(target)
        if name == self.collection_name:
            self.index = None
            self.payloads = {}
