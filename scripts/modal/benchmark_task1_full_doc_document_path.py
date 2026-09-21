"""Bounded CPU-only benchmark for the FULLDOC Qwen document path.

This local harness uses no model, labels, Modal, or Volume.  It compares the
old lazy document path with the new indexed/process-preloaded path on the exact
256-row production-probe prefix and checks all required structural inputs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.beam.task1_v2.evidence import prepare_document  # noqa: E402
from scripts.modal.task1_full_doc_top200_qwen_preprocessing import (  # noqa: E402
    OptimizedPreprocessor,
    frozen_selected_chunks,
    load_jsonl_rows,
    load_questions_once,
)

REPORT_ROOT = ROOT / "reports/task1/full_document_legal_field_retrieval"
WORKLIST_ROOT = REPORT_ROOT / "full_doc_top200_qwen_worklists"
CHUNKS = ROOT / "data/processed_v3/chunks"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
PROBE_LIMIT = 256
WORKERS = 8


def format_pair(query: str, document: str) -> list[dict]:
    """Exact text-only pair structure used by the locked scientific scorer."""
    return [
        {"role": "system", "content": [{"type": "text", "text": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<Instruct>: Given a Vietnamese legal question, determine whether the Document contains legal provisions relevant to answering the Query."},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "text", "text": document},
            ],
        },
    ]


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LegacyDocumentCache:
    """The pre-index/preload behavior used for the before measurement."""

    def __init__(self, chunks_root: Path) -> None:
        self.chunks_root = Path(chunks_root)
        self.prepared = {}
        self.document_index_seconds = 0.0
        self.document_preparation_seconds = 0.0
        self.files_loaded = 0
        self.filesystem_scans = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prepare_document_calls = 0

    def get(self, doc: str):
        if doc in self.prepared:
            self.cache_hits += 1
            return self.prepared[doc]
        self.cache_misses += 1
        start = time.perf_counter()
        loaded = load_jsonl_rows(self.chunks_root / f"{doc}.jsonl")
        self.document_index_seconds += time.perf_counter() - start
        self.files_loaded += 1
        start = time.perf_counter()
        prepared = prepare_document(loaded)
        self.document_preparation_seconds += time.perf_counter() - start
        self.prepare_document_calls += 1
        self.prepared[doc] = prepared
        return prepared


def selected_signature(row: dict, question: str, chunks: list[dict]) -> tuple:
    values = (
        str(row["query_id"]),
        str(question),
        str(row["document_id"]),
        tuple(str(item["chunk_id"]) for item in chunks),
        tuple(str(item["raw_chunk_text"]) for item in chunks),
        len(chunks),
    )
    formatted = tuple(
        json.dumps(format_pair(question, str(item["raw_chunk_text"])), ensure_ascii=False, separators=(",", ":"))
        for item in chunks
    )
    return values + (formatted,)


def old_outputs(work: list[dict], questions: dict[str, str], cache: LegacyDocumentCache | None = None) -> list[tuple]:
    output: list[tuple] = []
    for row in work:
        query = str(row["query_id"])
        doc = str(row["document_id"])
        question = questions[query]
        expected_ids = row.get("selected_chunk_ids")
        if expected_ids is not None:
            chunks = frozen_selected_chunks(CHUNKS, query, doc, list(expected_ids))
        else:
            if cache is None:
                raise RuntimeError("legacy cache is required for unfrozen rows")
            prepared = cache.get(doc)
            from scripts.beam.task1_v2.evidence import select_true_s2_prepared
            chunks = select_true_s2_prepared(question, prepared, topk=3)
        output.append(selected_signature(row, question, chunks))
    return output


def new_outputs(work: list[dict], questions: dict[str, str], workers: int) -> tuple[list[tuple], OptimizedPreprocessor]:
    pre = OptimizedPreprocessor(questions, CHUNKS)
    docs = {str(row["document_id"]) for row in work if "selected_chunk_ids" not in row}
    pre.documents.preload(docs, workers=workers)
    output: list[tuple] = []
    for row in work:
        query = str(row["query_id"])
        doc = str(row["document_id"])
        if "selected_chunk_ids" in row:
            question = questions[query]
            chunks = frozen_selected_chunks(CHUNKS, query, doc, list(row["selected_chunk_ids"]))
        else:
            question, chunks = pre.select(row)
        output.append(selected_signature(row, question, chunks))
    return output, pre


def main() -> None:
    historical = rows(REPORT_ROOT / "full_doc_top200_qwen_historical_parity_256.jsonl")
    current = rows(REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl")
    production = rows(WORKLIST_ROOT / "shard_00.jsonl")[:PROBE_LIMIT]
    if len(production) != PROBE_LIMIT:
        raise RuntimeError("production probe prefix is not 256 rows")
    if len({str(row["document_id"]) for row in production}) != 252:
        raise RuntimeError("production probe unique-document count is not 252")

    datasets = [("historical_parity", historical), ("current_canary", current), ("production_probe", production)]
    mismatch_by_dataset: dict[str, int] = {}
    for name, work in datasets:
        questions = load_questions_once(TRAIN, {str(row["query_id"]) for row in work})
        legacy_cache = LegacyDocumentCache(CHUNKS) if name != "historical_parity" else None
        old = old_outputs(work, questions, legacy_cache)
        new, pre = new_outputs(work, questions, WORKERS)
        mismatch_by_dataset[name] = sum(left != right for left, right in zip(old, new)) + abs(len(old) - len(new))
        print(json.dumps({"dataset": name, "rows": len(work), "mismatches": mismatch_by_dataset[name]}, sort_keys=True))
        if name == "production_probe":
            old_index = legacy_cache.document_index_seconds if legacy_cache else 0.0
            old_preparation = legacy_cache.document_preparation_seconds if legacy_cache else 0.0
            old_total = old_index + old_preparation
            new_index = pre.documents.document_index_seconds
            new_preparation = pre.documents.document_preparation_seconds
            new_total = new_index + new_preparation

    total_mismatches = sum(mismatch_by_dataset.values())
    result = {
        "rows": len(production),
        "unique_docs": len({str(row["document_id"]) for row in production}),
        "workers": WORKERS,
        "old_document_index_seconds": old_index,
        "new_document_index_seconds": new_index,
        "old_document_preparation_seconds": old_preparation,
        "new_document_preparation_seconds": new_preparation,
        "old_total_document_handling_seconds": old_total,
        "new_total_document_handling_seconds": new_total,
        "new_path_resolution_seconds": pre.documents.document_path_resolution_seconds,
        "new_fused_pipeline_wall_seconds": pre.documents.document_pipeline_seconds,
        "worker_load_seconds_sum": pre.documents.worker_load_seconds_sum,
        "worker_prepare_seconds_sum": pre.documents.worker_prepare_seconds_sum,
        "document_handling_speedup": old_total / new_total if new_total else None,
        "files_opened_before": legacy_cache.files_loaded if legacy_cache else 0,
        "files_opened_after": pre.documents.files_loaded,
        "filesystem_scans_before": legacy_cache.filesystem_scans if legacy_cache else 0,
        "filesystem_scans_after": pre.documents.filesystem_scans,
        "cache_hits_after": pre.documents.cache_hits,
        "cache_misses_after": pre.documents.cache_misses,
        "prepare_document_calls_after": pre.documents.prepare_document_calls,
        "mismatches": total_mismatches,
        "mismatches_by_dataset": mismatch_by_dataset,
        "exact_input_parity": total_mismatches == 0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    if total_mismatches:
        raise SystemExit("document pipeline structural parity failed")


if __name__ == "__main__":
    main()
