"""Lazy parent lookup and bounded post-rerank context expansion."""

import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from udsc2026.contracts import LegalParent, RetrievalHit

_SAFE_DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}")


class ParentStoreError(RuntimeError):
    """Raised when a parent file is missing, corrupt, or internally inconsistent."""


class ParentContextSettings(BaseModel):
    """Strict runtime settings for bounded parent-context hydration."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    enabled: bool = True
    parents_dir: str = Field(default="data/processed_v3/parents", min_length=1)
    max_cached_documents: int = Field(default=128, gt=0)
    max_parent_tokens: int = Field(default=2048, gt=0)
    max_total_tokens: int = Field(default=7000, gt=0)

    @model_validator(mode="after")
    def validate_token_limits(self) -> "ParentContextSettings":
        """Require each parent window to fit inside the total context budget."""

        if self.max_parent_tokens > self.max_total_tokens:
            raise ValueError("max_parent_tokens must not exceed max_total_tokens")
        return self


class JsonlParentStore:
    """Load one document's parents on demand with a bounded LRU cache."""

    def __init__(
        self, parents_dir: str | Path, *, max_cached_documents: int = 128
    ) -> None:
        if max_cached_documents < 1:
            raise ValueError("max_cached_documents must be positive")
        self._root = Path(parents_dir).resolve()
        self._max_cached_documents = max_cached_documents
        self._cache: OrderedDict[str, Dict[str, LegalParent]] = OrderedDict()
        self._cache_lock = threading.RLock()

    def get(self, doc_id: str, parent_id: str) -> LegalParent:
        """Return one validated parent without allowing path traversal."""

        if _SAFE_DOC_ID.fullmatch(doc_id) is None:
            raise ParentStoreError("unsafe doc_id for parent lookup")
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ParentStoreError("parent_id must not be blank")
        parents = self._load_document(doc_id)
        try:
            return parents[parent_id]
        except KeyError as exc:
            raise ParentStoreError(
                "parent_id {0!r} is missing for doc_id {1!r}".format(parent_id, doc_id)
            ) from exc

    def _load_document(self, doc_id: str) -> Dict[str, LegalParent]:
        with self._cache_lock:
            cached = self._cache.pop(doc_id, None)
            if cached is not None:
                self._cache[doc_id] = cached
                return cached

        path = (self._root / "{0}.jsonl".format(doc_id)).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ParentStoreError("parent path escapes configured root") from exc
        if not path.is_file():
            raise ParentStoreError("parent file does not exist: {0}".format(path))

        parents: Dict[str, LegalParent] = {}
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        continue
                    parent = LegalParent.model_validate(json.loads(raw_line))
                    if parent.doc_id != doc_id:
                        raise ValueError(
                            "line {0} has mismatched doc_id".format(line_number)
                        )
                    if parent.parent_id in parents:
                        raise ValueError(
                            "line {0} repeats parent_id {1!r}".format(
                                line_number, parent.parent_id
                            )
                        )
                    if not parent.text.strip():
                        raise ValueError(
                            "line {0} has empty parent text".format(line_number)
                        )
                    parents[parent.parent_id] = parent
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ParentStoreError(
                "invalid parent file for doc_id {0!r}: {1}".format(doc_id, exc)
            ) from exc
        if not parents:
            raise ParentStoreError("parent file has no records: {0}".format(path))

        with self._cache_lock:
            # Another request may have populated this document while the file was
            # being parsed. Reuse that canonical cache entry when present.
            cached = self._cache.pop(doc_id, None)
            selected = cached if cached is not None else parents
            self._cache[doc_id] = selected
            while len(self._cache) > self._max_cached_documents:
                self._cache.popitem(last=False)
            return selected


class ParentContextExpander:
    """Deduplicate parent hits and create anchor-centered bounded contexts."""

    def __init__(
        self,
        store: JsonlParentStore,
        *,
        max_parent_tokens: int = 2048,
        max_total_tokens: int = 7000,
    ) -> None:
        if max_parent_tokens < 1 or max_total_tokens < 1:
            raise ValueError("parent context token limits must be positive")
        if max_parent_tokens > max_total_tokens:
            raise ValueError("max_parent_tokens must not exceed max_total_tokens")
        self._store = store
        self._max_parent_tokens = max_parent_tokens
        self._max_total_tokens = max_total_tokens

    def expand(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """Expand reranked children while keeping their original order and scores."""

        groups: OrderedDict[Tuple[str, str, str], List[RetrievalHit]] = OrderedDict()
        for hit in hits:
            parent_id = _resolved_parent_id(hit)
            if parent_id is None:
                group_key = ("child", hit.doc_id, hit.chunk_id)
            else:
                group_key = ("parent", hit.doc_id, parent_id)
            groups.setdefault(group_key, []).append(hit)

        expanded: List[RetrievalHit] = []
        remaining = self._max_total_tokens
        for (group_kind, doc_id, group_id), group_hits in groups.items():
            if remaining <= 0:
                break
            parent_id = group_id if group_kind == "parent" else None
            parent = self._safe_parent(doc_id, parent_id, group_hits)
            if parent is not None:
                limit = min(self._max_parent_tokens, remaining)
                parent_text = _bounded_parent_text(
                    parent.text,
                    [hit.text for hit in group_hits],
                    limit,
                )
                if parent_text is not None:
                    token_count = _token_count(parent_text)
                    metadata = dict(group_hits[0].metadata)
                    metadata.pop("point", None)
                    metadata.update(
                        {
                            "parent_context_expanded": True,
                            "parent_context_anchor_chunk_ids": [
                                hit.chunk_id for hit in group_hits
                            ],
                            "parent_context_token_count": token_count,
                            "parent_context_truncated": (
                                token_count < _token_count(parent.text)
                            ),
                        }
                    )
                    expanded.append(
                        group_hits[0].model_copy(
                            update={
                                "parent_id": parent.parent_id,
                                "text": parent_text,
                                "law_name": parent.law_name or group_hits[0].law_name,
                                "article": parent.article or group_hits[0].article,
                                "clause": None,
                                "source": parent.source or group_hits[0].source,
                                "metadata": metadata,
                            }
                        )
                    )
                    remaining -= token_count
                    continue

            for hit in group_hits:
                if remaining <= 0:
                    break
                bounded = _truncate_tokens(hit.text, remaining)
                if not bounded:
                    break
                metadata = dict(hit.metadata)
                metadata["parent_context_expanded"] = False
                metadata["parent_context_fallback"] = True
                expanded.append(
                    hit.model_copy(update={"text": bounded, "metadata": metadata})
                )
                remaining -= _token_count(bounded)
        return expanded

    def _safe_parent(
        self,
        doc_id: str,
        parent_id: Optional[str],
        hits: List[RetrievalHit],
    ) -> Optional[LegalParent]:
        if parent_id is None:
            return None
        try:
            parent = self._store.get(doc_id, parent_id)
        except ParentStoreError:
            return None
        for hit in hits:
            if hit.article and parent.article and hit.article != parent.article:
                return None
        return parent


def _resolved_parent_id(hit: RetrievalHit) -> Optional[str]:
    direct = hit.parent_id.strip() if hit.parent_id else None
    value = hit.metadata.get("parent_id")
    metadata_value = value.strip() if isinstance(value, str) and value.strip() else None
    if direct is not None and metadata_value is not None and direct != metadata_value:
        return None
    return direct or metadata_value


def _bounded_parent_text(
    parent_text: str, anchors: List[str], token_limit: int
) -> Optional[str]:
    parent_tokens = parent_text.split()
    if not parent_tokens or token_limit < 1:
        return None
    if len(parent_tokens) <= token_limit:
        return parent_text.strip()

    folded_parent = [token.casefold() for token in parent_tokens]
    matches: List[Tuple[int, int]] = []
    for anchor in anchors:
        anchor_tokens = anchor.split()
        if not anchor_tokens:
            continue
        folded_anchor = [token.casefold() for token in anchor_tokens]
        start = _find_subsequence(folded_parent, folded_anchor)
        if start is not None:
            matches.append((start, start + len(anchor_tokens)))
    if not matches:
        return None

    per_anchor = max(1, token_limit // len(matches))
    intervals: List[Tuple[int, int]] = []
    for start, end in matches:
        anchor_length = end - start
        extra = max(per_anchor - anchor_length, 0)
        left = max(0, start - extra // 2)
        right = min(len(parent_tokens), end + extra - extra // 2)
        intervals.append((left, right))
    intervals = _merge_intervals(intervals)

    output: List[str] = []
    for start, end in intervals:
        if output and len(output) < token_limit:
            output.append("[…]")
        remaining = token_limit - len(output)
        if remaining <= 0:
            break
        output.extend(parent_tokens[start : min(end, start + remaining)])
    return " ".join(output) if output else None


def _find_subsequence(haystack: List[str], needle: List[str]) -> Optional[int]:
    """Find the first exact token subsequence in linear time using KMP."""

    if not needle:
        return 0
    if len(needle) > len(haystack):
        return None

    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix[index] = matched

    matched = 0
    for index, token in enumerate(haystack):
        while matched and token != needle[matched]:
            matched = prefix[matched - 1]
        if token == needle[matched]:
            matched += 1
            if matched == len(needle):
                return index - len(needle) + 1
    return None


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _truncate_tokens(text: str, limit: int) -> str:
    return " ".join(text.split()[: max(limit, 0)])


def _token_count(text: str) -> int:
    return len(text.split())
