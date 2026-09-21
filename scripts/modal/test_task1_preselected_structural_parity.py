"""CPU-only structural check for the frozen preselected-input wire format.

This test exercises the live OptimizedPreprocessor selector on the three
already-frozen samples, round-trips the exact preselected JSON shape in
memory, and compares every scorer input component.  It never loads a model,
uses labels, calls Modal, or writes an artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modal.benchmark_task1_full_doc_document_path import format_pair  # noqa: E402
from scripts.modal.task1_full_doc_top200_qwen_preprocessing import (  # noqa: E402
    OptimizedPreprocessor,
    frozen_selected_chunks,
    load_questions_once,
)


REPORT_ROOT = ROOT / "reports/task1/full_document_legal_field_retrieval"
WORKLIST_ROOT = REPORT_ROOT / "full_doc_top200_qwen_worklists"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS = ROOT / "data/processed_v3/chunks"
PRODUCTION_PROBE_LIMIT = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_row(
    row: dict[str, Any], question: str, selected: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "query_id": str(row["query_id"]),
        "document_id": str(row["document_id"]),
        "question": str(question),
        "expected_inference_units": int(row.get("expected_inference_units", len(selected))),
        "selected_chunks": [
            {"chunk_id": str(item["chunk_id"]), "raw_chunk_text": str(item["raw_chunk_text"])}
            for item in selected
        ],
    }


def scorer_inputs(row: dict[str, Any]) -> list[str]:
    return [
        json.dumps(
            format_pair(str(row["question"]), str(chunk["raw_chunk_text"])),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for chunk in row["selected_chunks"]
    ]


def run_dataset(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_ids = {str(row["query_id"]) for row in rows}
    questions = load_questions_once(TRAIN, query_ids)
    preprocessor = OptimizedPreprocessor(questions, CHUNKS)
    preprocessor.documents.preload(
        {str(row["document_id"]) for row in rows}, workers=8
    )
    mismatches = 0
    for source in rows:
        query = str(source["query_id"])
        doc = str(source["document_id"])
        # The historical parity sample intentionally carries frozen chunk IDs
        # from the historical scorer; it is not a live selector worklist.
        # Current-canary and production rows exercise the live selector.
        if "selected_chunk_ids" in source:
            live_question = questions[query]
            live_chunks = frozen_selected_chunks(
                CHUNKS, query, doc, list(source["selected_chunk_ids"])
            )
        else:
            live_question, live_chunks = preprocessor.select(source)
        live = frozen_row(source, live_question, live_chunks)
        round_tripped = json.loads(
            json.dumps(live, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        live_inputs = scorer_inputs(live)
        serialized_inputs = scorer_inputs(round_tripped)
        if (
            (live["query_id"], live["document_id"])
            != (round_tripped["query_id"], round_tripped["document_id"])
            or live["question"] != round_tripped["question"]
            or live["expected_inference_units"] != round_tripped["expected_inference_units"]
            or live["selected_chunks"] != round_tripped["selected_chunks"]
            or live_inputs != serialized_inputs
            or len(live["selected_chunks"]) != live["expected_inference_units"]
        ):
            mismatches += 1
            raise RuntimeError(f"{name} preselected serialization mismatch: {query}/{doc}")
    return {
        "rows": len(rows),
        "mismatches": mismatches,
        "prepared_documents": len(preprocessor.documents.prepared),
        "selected_units": preprocessor.selected_units,
    }


def main() -> None:
    historical = read_jsonl(REPORT_ROOT / "full_doc_top200_qwen_historical_parity_256.jsonl")
    current = read_jsonl(REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl")
    production = read_jsonl(WORKLIST_ROOT / "shard_00.jsonl")[:PRODUCTION_PROBE_LIMIT]
    datasets = {
        "historical_parity": historical,
        "current_canary": current,
        "production_probe": production,
    }
    results = {name: run_dataset(name, rows) for name, rows in datasets.items()}
    output = {
        "status": "PASS",
        "cpu_structural_parity": "PASS",
        "historical_mismatches": results["historical_parity"]["mismatches"],
        "current_canary_mismatches": results["current_canary"]["mismatches"],
        "production_probe_mismatches": results["production_probe"]["mismatches"],
        "datasets": results,
        "gpu_runs": 0,
        "modal_runs": 0,
        "labels_used": False,
        "artifact_writes": 0,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
