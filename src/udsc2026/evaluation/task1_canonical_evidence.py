"""Deterministic, model-free evidence resolution from the immutable Task1 corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CanonicalEvidence:
    document_id: str
    evidence_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    title: str | None
    metadata: dict[str, Any]
    source: str

    @property
    def text(self) -> str:
        return "\n\n".join(self.evidence_texts)


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


class CanonicalEvidenceResolver:
    """Resolve document evidence without consulting a retrieval result."""

    def __init__(self, processed_root: Path, *, evidence_limit: int = 2) -> None:
        if evidence_limit not in (1, 2):
            raise ValueError("evidence_limit must be 1 or 2")
        self.root = processed_root
        self.documents = processed_root / "documents"
        self.parents = processed_root / "parents"
        self.evidence_limit = evidence_limit
        self._cache: dict[str, CanonicalEvidence | None] = {}

    def resolve(self, document_id: str) -> CanonicalEvidence | None:
        document_id = str(document_id).strip()
        if document_id in self._cache:
            return self._cache[document_id]
        metadata: dict[str, Any] = {}
        title: str | None = None
        document_path = self.documents / f"{document_id}.json"
        if document_path.is_file():
            payload = json.loads(document_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                title = str(payload.get("law_name") or metadata.get("law_name") or "").strip() or None
        rows = list(_jsonl(self.parents / f"{document_id}.jsonl"))
        rows = [row for row in rows if str(row.get("doc_id", document_id)) == document_id]
        rows.sort(key=lambda row: (str(row.get("parent_id", "")), str(row.get("text", ""))))
        evidence_ids: list[str] = []
        texts: list[str] = []
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            evidence_ids.append(str(row.get("parent_id", f"{document_id}:parent:{len(evidence_ids)+1}")))
            texts.append(text)
            if len(texts) >= self.evidence_limit:
                break
            if not title:
                title = str(row.get("law_name", "")).strip() or None
        if not texts and document_path.is_file():
            payload = json.loads(document_path.read_text(encoding="utf-8-sig"))
            text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if text:
                evidence_ids = [document_id]
                texts = [text]
        result = (
            CanonicalEvidence(document_id, tuple(evidence_ids), tuple(texts), title, metadata, "canonical_corpus")
            if texts
            else None
        )
        self._cache[document_id] = result
        return result


def corpus_fingerprint(processed_root: Path) -> str:
    """Hash canonical file names and bytes, without loading the corpus in memory."""
    digest = hashlib.sha256()
    manifest = processed_root / "metadata" / "manifest.json"
    if manifest.is_file():
        # The ingestion manifest is the canonical corpus fingerprint source;
        # hashing millions of child files would make every local unit test slow.
        with manifest.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    for path in sorted(processed_root.glob("**/*")):
        if path.is_file() and path.suffix.casefold() in {".json", ".jsonl"}:
            digest.update(str(path.relative_to(processed_root)).replace("\\", "/").encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()
