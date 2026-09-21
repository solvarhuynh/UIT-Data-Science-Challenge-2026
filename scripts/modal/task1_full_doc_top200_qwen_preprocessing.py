"""CPU-only preprocessing helpers for the optimized FULLDOC TOP200 runner.

This module deliberately contains no model or scoring code.  It reuses the
existing canonical selector and only changes how immutable query/document
inputs are loaded and prepared.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path
from typing import Any, Iterable

from scripts.beam.task1_v2.evidence import PreparedDocument, prepare_document, select_true_s2_prepared


def load_questions_once(train_path: Path, query_ids: set[str]) -> dict[str, str]:
    """Load only canonical question text, with one source traversal."""
    result: dict[str, str] = {}
    decoder = json.JSONDecoder()
    wanted = {str(item) for item in query_ids}
    with train_path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            block = handle.read(1 << 20)
            if block:
                buffer += block
            else:
                eof = True

        def trim() -> None:
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        def decode() -> Any:
            nonlocal buffer
            while True:
                trim()
                if not buffer:
                    raise RuntimeError("unexpected EOF in canonical train source")
                try:
                    value, position = decoder.raw_decode(buffer)
                    buffer = buffer[position:]
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        refill()
        trim()
        if not buffer.startswith("{"):
            raise RuntimeError("canonical train source is not a JSON object")
        buffer = buffer[1:]
        while True:
            trim()
            if buffer.startswith("}"):
                break
            query_id = str(decode())
            trim()
            if not buffer.startswith(":"):
                raise RuntimeError("invalid canonical train object")
            buffer = buffer[1:]
            record = decode()
            if query_id in wanted:
                if not isinstance(record, dict) or "question" not in record:
                    raise RuntimeError(f"canonical question missing: {query_id}")
                question = str(record["question"])
                if not question.strip() or question == query_id:
                    raise RuntimeError(f"invalid canonical question text: {query_id}")
                result[query_id] = question
            trim()
            if buffer.startswith(","):
                buffer = buffer[1:]
    if set(result) != wanted:
        raise RuntimeError("canonical question coverage mismatch")
    return result


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_prepare_document_job(item: tuple[str, Path]) -> tuple[str, PreparedDocument, float, float]:
    """Read and prepare one document in one worker without parent row transfer."""
    doc, path = item
    load_start = time.perf_counter()
    rows = load_jsonl_rows(path)
    load_seconds = time.perf_counter() - load_start
    prepare_start = time.perf_counter()
    prepared = prepare_document(rows)
    prepare_seconds = time.perf_counter() - prepare_start
    return doc, prepared, load_seconds, prepare_seconds


def _validate_selected(
    query: str,
    doc: str,
    question: str,
    selected: list[dict[str, Any]],
    expected_units: int,
    expected_ids: list[str] | None,
) -> list[dict[str, Any]]:
    ids = [str(item["chunk_id"]) for item in selected]
    if not (1 <= len(selected) <= 3) or len(ids) != len(set(ids)) or len(selected) != expected_units:
        raise RuntimeError(f"TOP3_UP_TO_AVAILABLE violation: {query}/{doc}")
    if expected_ids is not None and ids != [str(item) for item in expected_ids]:
        raise RuntimeError(f"frozen chunk-selection mismatch: {query}/{doc}")
    if any(not str(item["raw_chunk_text"]).strip() or not str(item["chunk_id"]).startswith(doc + "_") for item in selected):
        raise RuntimeError(f"synthetic/empty/cross-document chunk: {query}/{doc}")
    return selected


def legacy_selected_chunks(
    chunks_root: Path,
    query: str,
    doc: str,
    question: str,
    expected_units: int,
    expected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """The current runner's uncached load/prepare/select path."""
    rows = load_jsonl_rows(chunks_root / f"{doc}.jsonl")
    selected = select_true_s2_prepared(question, prepare_document(rows), topk=3)
    return _validate_selected(query, doc, question, selected, expected_units, expected_ids)


def frozen_selected_chunks(
    chunks_root: Path, query: str, doc: str, frozen_ids: list[str]
) -> list[dict[str, Any]]:
    """Resolve frozen parity IDs without changing their order."""
    rows = load_jsonl_rows(chunks_root / f"{doc}.jsonl")
    by_id = {str(item.get("chunk_id", "")): item for item in rows}
    selected: list[dict[str, Any]] = []
    for chunk_id in frozen_ids:
        item = by_id.get(str(chunk_id))
        text = str(item.get("text", "")) if item is not None else ""
        if item is None or not str(chunk_id).startswith(doc + "_") or not text.strip():
            raise RuntimeError(f"unresolvable frozen parity chunk: {query}/{doc}/{chunk_id}")
        selected.append({"chunk_id": str(chunk_id), "raw_chunk_text": text})
    return selected


class DocumentPreparationCache:
    """One canonical read/preparation per document, with deterministic preload."""

    def __init__(self, chunks_root: Path) -> None:
        self.chunks_root = Path(chunks_root)
        self.prepared: dict[str, PreparedDocument] = {}
        self.document_path_resolution_seconds = 0.0
        self.document_pipeline_seconds = 0.0
        self.worker_load_seconds_sum = 0.0
        self.worker_prepare_seconds_sum = 0.0
        self.files_loaded = 0
        self.filesystem_scans = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prepare_document_calls = 0

    @property
    def document_index_seconds(self) -> float:
        """Compatibility name: direct path construction only, no filesystem scan."""
        return self.document_path_resolution_seconds

    @property
    def document_preparation_seconds(self) -> float:
        """Compatibility name: fused preload wall time."""
        return self.document_pipeline_seconds

    def preload(self, docs: Iterable[str], workers: int = 1) -> None:
        """Read and prepare requested documents once, preserving doc-key order."""
        requested = sorted({str(doc) for doc in docs})
        pending_docs: list[str] = []
        for doc in requested:
            if doc in self.prepared:
                self.cache_hits += 1
                continue
            pending_docs.append(doc)
        self.cache_misses += len(pending_docs)
        if not pending_docs:
            return
        workers = max(1, int(workers))
        path_start = time.perf_counter()
        pending = [(doc, self.chunks_root / f"{doc}.jsonl") for doc in pending_docs]
        self.document_path_resolution_seconds += time.perf_counter() - path_start
        pipeline_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_prepare_document_job, pending))
        self.document_pipeline_seconds += time.perf_counter() - pipeline_start
        self.worker_load_seconds_sum += sum(item[2] for item in results)
        self.worker_prepare_seconds_sum += sum(item[3] for item in results)
        self.files_loaded += len(results)
        self.prepare_document_calls += len(results)
        self.prepared.update({doc: prepared for doc, prepared, _, _ in results})

    def get(self, doc: str) -> PreparedDocument:
        key = str(doc)
        if key in self.prepared:
            self.cache_hits += 1
            return self.prepared[key]
        self.preload([key], workers=1)
        return self.prepared[key]


class OptimizedPreprocessor:
    """Canonical query preload + document preparation cache."""

    def __init__(self, questions: dict[str, str], chunks_root: Path) -> None:
        self.questions = {str(key): str(value) for key, value in questions.items()}
        self.documents = DocumentPreparationCache(chunks_root)
        self.selector_seconds = 0.0
        self.selected_units = 0

    def select(self, row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        query = str(row["query_id"])
        doc = str(row["document_id"])
        question = self.questions[query]
        expected_units = len(row["selected_chunk_ids"]) if "selected_chunk_ids" in row else int(row["expected_inference_units"])
        expected_ids = row.get("selected_chunk_ids")
        prepared = self.documents.get(doc)
        start = time.perf_counter()
        selected = select_true_s2_prepared(question, prepared, topk=3)
        self.selector_seconds += time.perf_counter() - start
        selected = _validate_selected(query, doc, question, selected, expected_units, expected_ids)
        self.selected_units += len(selected)
        return question, selected


def signature(query: str, question: str, doc: str, chunks: Iterable[dict[str, Any]]) -> tuple[Any, ...]:
    values = list(chunks)
    return (
        str(query),
        str(question),
        str(doc),
        tuple(str(item["chunk_id"]) for item in values),
        tuple(str(item["raw_chunk_text"]) for item in values),
        len(values),
    )
