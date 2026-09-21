"""CPU-only equivalence and preprocessing benchmark for the optimized runner."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modal.task1_b2a_qwen3vl2b import format_pair
from scripts.modal.task1_full_doc_top200_qwen_preprocessing import (
    OptimizedPreprocessor,
    frozen_selected_chunks,
    legacy_selected_chunks,
    load_questions_once,
    signature,
)


REPORT_ROOT = ROOT / "reports/task1/full_document_legal_field_retrieval"
WORKLIST_ROOT = REPORT_ROOT / "full_doc_top200_qwen_worklists"
CHUNKS_ROOT = ROOT / "data/processed_v3/chunks"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_signature(row: dict, question: str, chunks: list[dict]) -> tuple:
    base = signature(row["query_id"], question, row["document_id"], chunks)
    formatted = tuple(
        json.dumps(format_pair(question, str(item["raw_chunk_text"])), ensure_ascii=False, separators=(",", ":"))
        for item in chunks
    )
    return base + (formatted,)


def old_outputs(work: list[dict], questions: dict[str, str], reload_question_each_row: bool = False) -> list[tuple]:
    output: list[tuple] = []
    for row in work:
        query = str(row["query_id"])
        question = load_questions_once(TRAIN, {query})[query] if reload_question_each_row else questions[query]
        expected_ids = row.get("selected_chunk_ids")
        if expected_ids is not None:
            chunks = frozen_selected_chunks(CHUNKS_ROOT, query, str(row["document_id"]), list(expected_ids))
        else:
            chunks = legacy_selected_chunks(
                CHUNKS_ROOT,
                query,
                str(row["document_id"]),
                question,
                int(row["expected_inference_units"]),
            )
        output.append(selected_signature(row, question, chunks))
    return output


def optimized_outputs(work: list[dict], questions: dict[str, str]) -> tuple[list[tuple], OptimizedPreprocessor]:
    pre = OptimizedPreprocessor(questions, CHUNKS_ROOT)
    output: list[tuple] = []
    for row in work:
        if "selected_chunk_ids" in row:
            query = str(row["query_id"])
            question = questions[query]
            chunks = frozen_selected_chunks(CHUNKS_ROOT, query, str(row["document_id"]), list(row["selected_chunk_ids"]))
        else:
            question, chunks = pre.select(row)
        output.append(selected_signature(row, question, chunks))
    return output, pre


def main() -> None:
    parity = rows(REPORT_ROOT / "full_doc_top200_qwen_historical_parity_256.jsonl")
    current = rows(REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl")
    all_production = rows(WORKLIST_ROOT / "shard_00.jsonl")
    # A deterministic spread across shard 00 avoids overrepresenting one
    # query while keeping the CPU-only audit bounded.  The unbounded 1,000-row
    # attempt was stopped after exceeding 80 minutes of CPU time.
    stride = max(1, len(all_production) // 100)
    production = [all_production[index] for index in range(0, len(all_production), stride)][:100]
    datasets = [("historical_parity", parity), ("current_canary", current), ("production_sample", production)]
    mismatch_count = 0
    checked: dict[str, int] = {}
    old_seconds = 0.0
    new_seconds = 0.0
    production_pre: OptimizedPreprocessor | None = None
    for name, work in datasets:
        query_ids = {str(row["query_id"]) for row in work}
        questions = load_questions_once(TRAIN, query_ids)
        old_start = time.perf_counter()
        old = old_outputs(work, questions, reload_question_each_row=(name == "production_sample"))
        old_elapsed = time.perf_counter() - old_start
        new_start = time.perf_counter()
        new, pre = optimized_outputs(work, questions)
        new_elapsed = time.perf_counter() - new_start
        if name == "production_sample":
            old_seconds = old_elapsed
            new_seconds = new_elapsed
            production_pre = pre
        mismatches = sum(left != right for left, right in zip(old, new)) + abs(len(old) - len(new))
        mismatch_count += mismatches
        checked[name] = len(work)
        print(json.dumps({"dataset": name, "rows": len(work), "mismatches": mismatches}, sort_keys=True))

    speedup = old_seconds / new_seconds if new_seconds else float("inf")
    result = {
        "mismatches": mismatch_count,
        "checked": checked,
        "old_cpu_preprocessing_seconds": old_seconds,
        "optimized_cpu_preprocessing_seconds": new_seconds,
        "cpu_speedup": speedup,
        "optimized_document_files_loaded": production_pre.documents.files_loaded if production_pre else 0,
        "optimized_prepared_documents": len(production_pre.documents.prepared) if production_pre else 0,
        "optimized_document_index_seconds": production_pre.documents.document_index_seconds if production_pre else 0.0,
        "optimized_document_preparation_seconds": production_pre.documents.document_preparation_seconds if production_pre else 0.0,
        "optimized_selector_seconds": production_pre.selector_seconds if production_pre else 0.0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    if mismatch_count:
        raise SystemExit("CPU equivalence failed")


if __name__ == "__main__":
    main()
