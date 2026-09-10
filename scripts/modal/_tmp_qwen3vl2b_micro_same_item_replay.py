"""Bounded same-item diagnostic replay for the frozen B2a 24-q-doc manifest.

This is intentionally a launchable Modal diagnostic, not a canonical inference
runner.  It reads no labels, never opens a canonical checkpoint or prediction,
and refuses any workload other than the frozen 24 q-docs / 72 chunks.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import modal

# Reuse the frozen Qwen preparation, model loading, and scorer verbatim.  In
# particular, this is the same score_batch used by validation400 and run_remote.
from scripts.modal import task1_b2a_qwen3vl2b as canonical


APP_NAME = "task1-b2a-qwen3vl2b-micro-same-item-replay"
REMOTE_REPO_ROOT = Path("/root/udsc2026")
MANIFEST = REMOTE_REPO_ROOT / (
    "reports/task1/workflow_b/tv2/b2a/manifests/"
    "qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl"
)
QUERY_SOURCE = REMOTE_REPO_ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS = REMOTE_REPO_ROOT / "data/processed_v3/chunks"
MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODEL_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
INSTRUCTION = canonical.INSTRUCTION
INFERENCE_BATCH_SIZE = 1
EXPECTED_QDOCS = 24
EXPECTED_CHUNKS = 72
EXPECTED_STAGE_COUNTS = {"A": 8, "B": 8, "C": 8}
OFFICIAL_PARITY_QDOCS = 8
MATERIAL_SCORE_DELTA = 1e-5

# File-mode Modal places this script directly under /root.  Local repository
# discovery is therefore deliberately unavailable during remote module import.
if modal.is_local():
    LOCAL_REPO_ROOT: Path | None = Path(__file__).resolve().parents[2]
    LOCAL_MANIFEST: Path | None = LOCAL_REPO_ROOT / (
        "reports/task1/workflow_b/tv2/b2a/manifests/"
        "qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl"
    )
    LOCAL_QUERY_SOURCE: Path | None = LOCAL_REPO_ROOT / "data/raw/btc/LegalIR/train.json"
    LOCAL_WORKLIST: Path | None = LOCAL_REPO_ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"
else:
    LOCAL_REPO_ROOT = None
    LOCAL_MANIFEST = None
    LOCAL_QUERY_SOURCE = None
    LOCAL_WORKLIST = None


def local_manifest_doc_ids() -> tuple[str, ...]:
    """Read only the frozen local manifest while constructing the Modal image."""
    if not modal.is_local() or LOCAL_MANIFEST is None:
        raise RuntimeError("local manifest discovery is forbidden in the remote container")
    if not LOCAL_MANIFEST.is_file():
        raise RuntimeError(f"missing local frozen manifest: {LOCAL_MANIFEST}")
    rows = [json.loads(line) for line in LOCAL_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_ids = tuple(sorted({str(row["doc_id"]) for row in rows}, key=lambda value: (int(value) if value.isdigit() else value)))
    if len(rows) != EXPECTED_QDOCS or len(doc_ids) > EXPECTED_QDOCS:
        raise RuntimeError("local frozen-manifest packaging cardinality mismatch")
    return doc_ids


LOCAL_DOC_IDS = local_manifest_doc_ids() if modal.is_local() else ()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=4.57.0,<5.0",
        "accelerate>=1.1,<2.0",
        "huggingface-hub>=0.30",
        "qwen-vl-utils>=0.0.14",
        "safetensors>=0.5,<1.0",
    )
)
if modal.is_local():
    assert LOCAL_REPO_ROOT is not None and LOCAL_MANIFEST is not None and LOCAL_QUERY_SOURCE is not None
    image = image.add_local_python_source("scripts", copy=True)
    image = image.add_local_file(
        LOCAL_MANIFEST,
        (REMOTE_REPO_ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl").as_posix(),
        copy=True,
    )
    image = image.add_local_file(
        LOCAL_QUERY_SOURCE,
        (REMOTE_REPO_ROOT / "data/raw/btc/LegalIR/train.json").as_posix(),
        copy=True,
    )
    for _doc_id in LOCAL_DOC_IDS:
        image = image.add_local_file(
            LOCAL_REPO_ROOT / f"data/processed_v3/chunks/{_doc_id}.jsonl",
            (REMOTE_REPO_ROOT / f"data/processed_v3/chunks/{_doc_id}.jsonl").as_posix(),
            copy=True,
        )
app = modal.App(APP_NAME)


def natural_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load and hard-validate the immutable selection without changing it."""
    if not path.is_file():
        raise RuntimeError(f"missing frozen manifest: {path}")
    rows: list[dict[str, Any]] = []
    immutable = {
        "query_id",
        "doc_id",
        "stage",
        "selected_chunk_ids",
        "good_document_score",
        "bad_full_document_score",
        "selection_reason",
        "canonical_k77_rank",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = immutable - set(row)
            if missing:
                raise RuntimeError(f"manifest row {line_number} missing immutable fields: {sorted(missing)}")
            row["query_id"] = str(row["query_id"])
            row["doc_id"] = str(row["doc_id"])
            row["stage"] = str(row["stage"])
            row["selected_chunk_ids"] = [str(value) for value in row["selected_chunk_ids"]]
            rows.append(row)
    pairs = [(row["query_id"], row["doc_id"]) for row in rows]
    stage_counts = {stage: sum(row["stage"] == stage for row in rows) for stage in EXPECTED_STAGE_COUNTS}
    if len(rows) != EXPECTED_QDOCS or len(set(pairs)) != EXPECTED_QDOCS:
        raise RuntimeError(f"frozen q-doc count mismatch: rows={len(rows)} unique={len(set(pairs))}")
    if stage_counts != EXPECTED_STAGE_COUNTS:
        raise RuntimeError(f"frozen stage composition mismatch: {stage_counts}")
    if any(len(row["selected_chunk_ids"]) != 3 for row in rows):
        raise RuntimeError("every frozen q-doc must have exactly three selected chunk IDs")
    if sum(len(row["selected_chunk_ids"]) for row in rows) != EXPECTED_CHUNKS:
        raise RuntimeError("frozen selected-chunk count mismatch")
    return rows


def local_packaging_preflight() -> dict[str, Any]:
    """CPU-only verification of the exact sources staged into the remote image."""
    if not modal.is_local() or LOCAL_MANIFEST is None or LOCAL_QUERY_SOURCE is None or LOCAL_WORKLIST is None:
        raise RuntimeError("local packaging preflight is unavailable in the remote container")
    rows = manifest_rows(LOCAL_MANIFEST)
    if not LOCAL_QUERY_SOURCE.is_file() or not LOCAL_WORKLIST.is_file():
        raise RuntimeError("required local query source or canonical-worklist source is missing")
    required_files = [LOCAL_REPO_ROOT / f"data/processed_v3/chunks/{doc_id}.jsonl" for doc_id in LOCAL_DOC_IDS]
    if any(not path.is_file() for path in required_files):
        raise RuntimeError("one or more required local chunk files are missing")
    selected = {(row["query_id"], row["doc_id"], chunk_id) for row in rows for chunk_id in row["selected_chunk_ids"]}
    canonical_order: list[tuple[str, str, str]] = []
    with LOCAL_WORKLIST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                key = (str(item["query_id"]), str(item["doc_id"]), str(item["chunk_id"]))
                if key in selected:
                    canonical_order.append(key)
    if canonical_order != full_worklist_order(rows):
        raise RuntimeError("packaged PATH_B order does not match local canonical worklist order")
    return {
        "scripts_package_explicit": True,
        "canonical_import_local": canonical.__name__ == "scripts.modal.task1_b2a_qwen3vl2b",
        "manifest": str(LOCAL_MANIFEST), "query_source": str(LOCAL_QUERY_SOURCE),
        "unique_chunk_documents": len(LOCAL_DOC_IDS), "required_chunk_files": len(required_files),
        "q_docs": len(rows), "chunks": sum(len(row["selected_chunk_ids"]) for row in rows),
        "stage_counts": {stage: sum(row["stage"] == stage for row in rows) for stage in EXPECTED_STAGE_COUNTS},
        "remote_repo_root": str(REMOTE_REPO_ROOT),
    }


def resolve_frozen_work(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve only manifest-selected IDs from their documented chunk files."""
    work: list[dict[str, Any]] = []
    for row in rows:
        path = CHUNKS / f"{row['doc_id']}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing selected chunk artifact: {path}")
        available: dict[str, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    available[str(chunk["chunk_id"])] = str(chunk.get("text", chunk.get("chunk_text", "")))
        for chunk_id in row["selected_chunk_ids"]:
            text = available.get(chunk_id, "")
            if not text:
                raise RuntimeError(f"missing selected chunk: {row['query_id']}/{row['doc_id']}/{chunk_id}")
            work.append({
                "stage": row["stage"], "query_id": row["query_id"], "doc_id": row["doc_id"],
                "chunk_id": chunk_id, "chunk_text": text,
            })
    if len(work) != EXPECTED_CHUNKS:
        raise RuntimeError(f"resolved frozen chunk count mismatch: {len(work)}")
    return work


def full_worklist_order(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Recreate the selected canonical worklist order without shipping its 1.3M rows.

    The local packaging preflight verifies this order against the canonical
    worklist.  Runtime membership remains exclusively the manifest's 72 IDs.
    """
    ordered_rows = sorted(enumerate(rows), key=lambda item: (natural_key(item[1]["query_id"]), item[0]))
    found = [
        (row["query_id"], row["doc_id"], chunk_id)
        for _, row in ordered_rows
        for chunk_id in row["selected_chunk_ids"]
    ]
    if len(found) != EXPECTED_CHUNKS or len(set(found)) != EXPECTED_CHUNKS:
        raise RuntimeError("frozen full-path order cardinality mismatch")
    return found


def path_a_work(rows: list[dict[str, Any]], resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """validation400_batch1_remote-style: manifest group construction then Batch1."""
    output = sorted(
        resolved,
        key=lambda item: (natural_key(item["query_id"]), natural_key(item["doc_id"]), item["chunk_id"]),
    )
    if len(output) != EXPECTED_CHUNKS:
        raise RuntimeError("PATH_A work cardinality mismatch")
    return output


def path_b_work(rows: list[dict[str, Any]], resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """run_remote-style: canonical full-worklist order, restricted to frozen items."""
    order = full_worklist_order(rows)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in resolved:
        by_key[(item["query_id"], item["doc_id"], item["chunk_id"])] = item
    output = [by_key[key] for key in order]
    if len(output) != EXPECTED_CHUNKS:
        raise RuntimeError("PATH_B work cardinality mismatch")
    return output


def patched_full_path_query_text(questions: dict[str, str], query_id: str) -> str:
    """POSTFIX mirror of run_remote: query_text = questions[query_id]."""
    if query_id not in questions:
        raise RuntimeError(f"canonical question missing for query_id: {query_id}")
    query_text = questions[query_id]
    if not isinstance(query_text, str) or not query_text.strip() or query_text == str(query_id):
        raise RuntimeError(f"invalid canonical question text for query_id: {query_id}")
    return query_text


def pre_gpu_postfix_parity(path_a: list[dict[str, Any]], path_b: list[dict[str, Any]], questions: dict[str, str]) -> dict[str, Any]:
    """Abort before model loading unless patched PATH_B inputs equal PATH_A."""
    def trace(work: list[dict[str, Any]], postfix: bool) -> dict[tuple[str, str, str], tuple[str, str, str]]:
        result: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        for item in work:
            query_id = item["query_id"]
            query_text = patched_full_path_query_text(questions, query_id) if postfix else questions[query_id]
            key = (query_id, item["doc_id"], item["chunk_id"])
            result[key] = (query_text, item["chunk_text"], INSTRUCTION)
        return result

    left, right = trace(path_a, False), trace(path_b, True)
    if set(left) != set(right) or any(left[key] != right[key] for key in left):
        raise RuntimeError("POSTFIX PATH_A/PATH_B pre-tokenization input parity failure")
    qdoc_pairs = {(item["query_id"], item["doc_id"]) for item in path_a}
    if len(qdoc_pairs) != EXPECTED_QDOCS or len(left) != EXPECTED_CHUNKS:
        raise RuntimeError("POSTFIX micro input cardinality failure")
    expected_13844 = "\u004dang thai h\u1ed9 v\u00ec m\u1ee5c \u0111\u00edch nh\u00e2n \u0111\u1ea1o \u0111\u01b0\u1ee3c quy \u0111\u1ecbnh nh\u01b0 th\u1ebf n\u00e0o?"
    actual_13844 = patched_full_path_query_text(questions, "13844")
    if actual_13844 != expected_13844 or actual_13844 == "13844":
        raise RuntimeError("POSTFIX query_id 13844 assertion failed")
    return {"q_docs": len(qdoc_pairs), "chunks": len(left), "query_13844": actual_13844}


def rendered_input(processor: Any, query: str, chunk: str) -> str:
    return processor.apply_chat_template(
        canonical.format_pair(query, chunk), tokenize=False, add_generation_prompt=True
    )


def input_trace(processor: Any, work: list[dict[str, Any]], query_texts: dict[str, str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in work:
        key = (item["query_id"], item["doc_id"], item["chunk_id"])
        query_text = query_texts[item["query_id"]]
        prompt = rendered_input(processor, query_text, item["chunk_text"])
        encoded, lengths, truncated, _ = canonical.tokenized_pair_inputs(
            processor, [query_text], [item["chunk_text"]]
        )
        result[key] = {
            "query_text": query_text,
            "chunk_text": item["chunk_text"],
            "rendered_prompt": prompt,
            "input_ids": encoded["input_ids"].tolist()[0],
            "attention_mask": encoded["attention_mask"].tolist()[0],
            "sequence_length": int(encoded["attention_mask"].sum().item()),
            "truncated": bool(truncated),
            "untruncated_length": lengths[0],
        }
    if len(result) != EXPECTED_CHUNKS:
        raise RuntimeError("input trace cardinality mismatch")
    return result


def compare_input_traces(left: dict[tuple[str, str, str], dict[str, Any]], right: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    fields = ("query_text", "chunk_text", "rendered_prompt", "input_ids", "attention_mask", "sequence_length", "truncated")
    if set(left) != set(right):
        return {"exact_equality": False, "first_mismatch": {"kind": "key_set", "path_a_keys": len(left), "path_b_keys": len(right)}}
    for key in sorted(left, key=lambda value: tuple(natural_key(part) for part in value)):
        for field in fields:
            if left[key][field] != right[key][field]:
                return {
                    "exact_equality": False,
                    "first_mismatch": {"key": key, "field": field, "path_a": left[key][field], "path_b": right[key][field]},
                }
    return {"exact_equality": True, "first_mismatch": None}


def score_work(processor: Any, model: Any, torch: Any, work: list[dict[str, Any]], query_texts: dict[str, str], label: str, diagnostic_db: Path | None = None) -> list[dict[str, Any]]:
    """Score Batch1 exclusively through the canonical score_batch implementation."""
    expected_score_count = len(work)
    expected_keys = {(item["query_id"], item["doc_id"], item["chunk_id"]) for item in work}
    if not expected_score_count or len(expected_keys) != expected_score_count:
        raise RuntimeError(f"{label} submitted-work cardinality mismatch")
    connection = None
    if diagnostic_db is not None:
        connection = sqlite3.connect(diagnostic_db)
        connection.execute("CREATE TABLE scores (query_id TEXT, doc_id TEXT, chunk_id TEXT, score REAL, PRIMARY KEY(query_id, doc_id, chunk_id))")
    rows: list[dict[str, Any]] = []
    for item in work:
        values, _, _, _ = canonical.score_batch(
            processor, model, torch, [query_texts[item["query_id"]]], [item["chunk_text"]]
        )
        if len(values) != INFERENCE_BATCH_SIZE or not math.isfinite(values[0]):
            raise RuntimeError(f"{label} Batch1 scoring failure")
        record = {**item, "score": values[0]}
        rows.append(record)
        if connection is not None:
            connection.execute("INSERT INTO scores VALUES (?,?,?,?)", (item["query_id"], item["doc_id"], item["chunk_id"], values[0]))
    if connection is not None:
        connection.commit()
        count = int(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
        connection.close()
        if count != expected_score_count:
            raise RuntimeError("isolated PATH_B diagnostic checkpoint cardinality mismatch")
    if len(rows) != expected_score_count or {(row["query_id"], row["doc_id"], row["chunk_id"]) for row in rows} != expected_keys:
        raise RuntimeError(f"{label} score cardinality mismatch")
    return rows


def document_scores(chunk_rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in chunk_rows:
        key = (row["query_id"], row["doc_id"])
        result[key] = max(result.get(key, float("-inf")), float(row["score"]))
    if len(result) != EXPECTED_QDOCS:
        raise RuntimeError("MAX document aggregation cardinality mismatch")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("empty statistic")
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise RuntimeError("Pearson cardinality mismatch")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else None


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def comparisons(left: dict[Any, float], right: dict[Any, float]) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("comparison join mismatch")
    keys = sorted(left, key=lambda key: tuple(natural_key(str(part)) for part in (key if isinstance(key, tuple) else (key,))))
    left_values, right_values = [left[key] for key in keys], [right[key] for key in keys]
    deltas = [abs(x - y) for x, y in zip(left_values, right_values)]
    return {
        "joined": len(keys), "max_abs_delta": max(deltas), "mean_abs_delta": statistics.fmean(deltas),
        "median_abs_delta": statistics.median(deltas), "p95_abs_delta": percentile(deltas, 0.95),
        "pearson": pearson(left_values, right_values), "spearman": pearson(average_ranks(left_values), average_ranks(right_values)),
    }


def historical_scores(rows: list[dict[str, Any]], field: str, stage: str | None = None) -> dict[tuple[str, str], float]:
    filtered = [row for row in rows if stage is None or row["stage"] == stage]
    return {(row["query_id"], row["doc_id"]): float(row[field]) for row in filtered}


def stage_comparisons(rows: list[dict[str, Any]], fresh: dict[tuple[str, str], float], historical_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in EXPECTED_STAGE_COUNTS:
        keys = set(historical_scores(rows, historical_field, stage))
        result[stage] = comparisons({key: fresh[key] for key in keys}, historical_scores(rows, historical_field, stage))
    return result


def classify(input_parity: dict[str, Any], order: dict[str, Any], fresh_ab: dict[str, Any], a_good: dict[str, Any], b_good: dict[str, Any], a_bad: dict[str, Any], b_bad: dict[str, Any]) -> str:
    if not input_parity["exact_equality"]:
        return "INPUT_PREPROCESSING_DIVERGENCE"
    if order["max_abs_delta"] > MATERIAL_SCORE_DELTA:
        return "ORDER_OR_STATE_DEPENDENCE"
    agrees = fresh_ab["max_abs_delta"] <= MATERIAL_SCORE_DELTA
    good = max(a_good["mean_abs_delta"], b_good["mean_abs_delta"]) <= 1e-3
    bad = max(a_bad["mean_abs_delta"], b_bad["mean_abs_delta"]) <= 1e-3
    a_good_b_bad = a_good["mean_abs_delta"] <= 1e-3 and b_bad["mean_abs_delta"] <= 1e-3
    if a_good_b_bad:
        return "FRESH_FULL_PATH_REPRODUCES_BAD_REGIME"
    if agrees and good and not bad:
        return "FRESH_BOTH_PATHS_MATCH_GOOD_REGIME"
    if agrees and bad:
        return "FRESH_BOTH_PATHS_MATCH_BAD_REGIME"
    if not agrees:
        return "FRESH_PATHS_DIVERGE_OTHER"
    return "UNRESOLVED"


@app.function(image=image, gpu="A10", timeout=4 * 60 * 60, cpu=8, memory=32768)
def micro_replay_remote() -> dict[str, Any]:
    """Run the bounded 72-chunk same-item diagnostic on an A10, Batch1 only."""
    import torch
    import transformers

    if REMOTE_REPO_ROOT != Path("/root/udsc2026"):
        raise RuntimeError(f"unexpected remote repository root: {REMOTE_REPO_ROOT}")
    if not MANIFEST.is_file() or not QUERY_SOURCE.is_file():
        raise RuntimeError("required remote manifest or query source is not mounted")
    # canonical.load_questions reads canonical.TRAIN; point it at the explicit
    # image file rather than the canonical runner's Volume-only path.
    canonical.TRAIN = QUERY_SOURCE
    canonical.CHUNKS = CHUNKS
    rows = manifest_rows(MANIFEST)
    remote_doc_ids = {row["doc_id"] for row in rows}
    if len(remote_doc_ids) > EXPECTED_QDOCS or any(not (CHUNKS / f"{doc_id}.jsonl").is_file() for doc_id in remote_doc_ids):
        raise RuntimeError("one or more required remote chunk files are not mounted")
    resolved = resolve_frozen_work(rows)
    if len(rows) != EXPECTED_QDOCS or len(resolved) != EXPECTED_CHUNKS:
        raise RuntimeError("pre-inference frozen cardinality guard failed")
    if {stage: sum(row["stage"] == stage for row in rows) for stage in EXPECTED_STAGE_COUNTS} != EXPECTED_STAGE_COUNTS:
        raise RuntimeError("pre-inference stage guard failed")
    if INFERENCE_BATCH_SIZE != 1:
        raise RuntimeError("effective batch must be one")

    # Reads only question text; canonical.load_questions does not load answer labels.
    questions = canonical.load_questions({row["query_id"] for row in rows})
    path_a = path_a_work(rows, resolved)
    path_b = path_b_work(rows, resolved)
    # Must pass before any model snapshot download or model scoring.
    postfix_preflight = pre_gpu_postfix_parity(path_a, path_b, questions)
    full_path_questions = {
        query_id: patched_full_path_query_text(questions, query_id) for query_id in questions
    }
    model_path, resolved_revision, provenance, yes_token_id, no_token_id = canonical.download_locked_snapshot()
    if resolved_revision != MODEL_REVISION or (yes_token_id, no_token_id) != (9693, 2152):
        raise RuntimeError("locked model contract guard failed")
    processor, model, _, gpu_name, _, _ = canonical.load_model(
        torch, transformers, model_path, yes_token_id, no_token_id
    )
    if "A10" not in gpu_name.upper():
        raise RuntimeError(f"A10-only guard failed: {gpu_name}")

    parity = compare_input_traces(input_trace(processor, path_a, questions), input_trace(processor, path_b, full_path_questions))

    # All SQLite activity is a fresh temporary diagnostic database, never a canonical checkpoint.
    with tempfile.TemporaryDirectory(prefix="qwen3vl2b_micro_same_item_") as temp_dir:
        diagnostic_db = Path(temp_dir) / "path_b_diagnostic.sqlite3"
        scored_a = score_work(processor, model, torch, path_a, questions, "PATH_A")
        scored_b = score_work(processor, model, torch, path_b, full_path_questions, "PATH_B", diagnostic_db)
        # Keep query text fixed here: this is an order/state test, not a second
        # test of the already-recorded PATH_A/PATH_B query-construction split.
        order_1 = score_work(processor, model, torch, resolved, questions, "ORDER_1_MANIFEST")
        order_2 = score_work(processor, model, torch, path_b, questions, "ORDER_2_FULL_WORKLIST")

        # Official-contract replay uses the exact canonical score_batch with the exact rendered input.
        official_pairs = {(row["query_id"], row["doc_id"]) for row in rows[:OFFICIAL_PARITY_QDOCS]}
        official_work = [item for item in path_a if (item["query_id"], item["doc_id"]) in official_pairs]
        print(f"OFFICIAL_PARITY_QDOCS={len(official_pairs)}", flush=True)
        print(f"OFFICIAL_PARITY_CHUNKS={len(official_work)}", flush=True)
        print(f"OFFICIAL_EXPECTED_SCORES={len(official_work)}", flush=True)
        official_scored = score_work(processor, model, torch, official_work, questions, "OFFICIAL_CONTRACT")
        print(f"OFFICIAL_ACTUAL_SCORES={len(official_scored)}", flush=True)

    a_chunks = {(row["query_id"], row["doc_id"], row["chunk_id"]): row["score"] for row in scored_a}
    b_chunks = {(row["query_id"], row["doc_id"], row["chunk_id"]): row["score"] for row in scored_b}
    a_docs, b_docs = document_scores(scored_a), document_scores(scored_b)
    order_1_docs, order_2_docs = document_scores(order_1), document_scores(order_2)
    official_chunks = {(row["query_id"], row["doc_id"], row["chunk_id"]): row["score"] for row in official_scored}
    official_reference = {key: a_chunks[key] for key in official_chunks}

    a_good = comparisons(a_docs, historical_scores(rows, "good_document_score"))
    b_good = comparisons(b_docs, historical_scores(rows, "good_document_score"))
    a_bad = comparisons(a_docs, historical_scores(rows, "bad_full_document_score"))
    b_bad = comparisons(b_docs, historical_scores(rows, "bad_full_document_score"))
    chunk_comparison, document_comparison = comparisons(a_chunks, b_chunks), comparisons(a_docs, b_docs)
    order_comparison = comparisons(order_1_docs, order_2_docs)
    official_comparison = comparisons(official_reference, official_chunks)
    payload = {
        "status": "PASS", "model": MODEL_ID, "revision": resolved_revision, "gpu": gpu_name,
        "effective_batch": INFERENCE_BATCH_SIZE, "manifest": str(MANIFEST), "q_docs": len(rows), "chunks": len(resolved),
        "stage_counts": {stage: sum(row["stage"] == stage for row in rows) for stage in EXPECTED_STAGE_COUNTS},
        "path_a": "validation400_batch1_remote preparation -> canonical.score_batch",
        "path_b": "patched run_remote canonical query-text construction -> canonical.score_batch (temporary diagnostic SQLite only)",
        "input_parity": parity,
        "postfix_pre_gpu_input_parity": postfix_preflight,
        "fresh_chunk_scores": [{"stage": row["stage"], "query_id": row["query_id"], "doc_id": row["doc_id"], "chunk_id": row["chunk_id"], "fresh_A_score": a_chunks[(row["query_id"], row["doc_id"], row["chunk_id"])], "fresh_B_score": b_chunks[(row["query_id"], row["doc_id"], row["chunk_id"])]} for row in path_a],
        "fresh_document_scores": [{"stage": row["stage"], "query_id": row["query_id"], "doc_id": row["doc_id"], "fresh_A_document_score": a_docs[(row["query_id"], row["doc_id"])], "fresh_B_document_score": b_docs[(row["query_id"], row["doc_id"])]} for row in rows],
        "fresh_A_vs_fresh_B": {"chunk": chunk_comparison, "document": document_comparison},
        "historical_comparisons": {
            "fresh_A_vs_good": {"all": a_good, "by_stage": stage_comparisons(rows, a_docs, "good_document_score")},
            "fresh_B_vs_good": {"all": b_good, "by_stage": stage_comparisons(rows, b_docs, "good_document_score")},
            "fresh_A_vs_bad_full": {"all": a_bad, "by_stage": stage_comparisons(rows, a_docs, "bad_full_document_score")},
            "fresh_B_vs_bad_full": {"all": b_bad, "by_stage": stage_comparisons(rows, b_docs, "bad_full_document_score")},
        },
        "order_hidden_state": {"order_1": "manifest order", "order_2": "canonical full-worklist order", "document": order_comparison},
        "official_scorer_parity": {"q_docs": OFFICIAL_PARITY_QDOCS, "status": "PASS" if official_comparison["max_abs_delta"] == 0.0 else "FAIL", "max_absolute_score_delta": official_comparison["max_abs_delta"]},
        "root_cause_classification": classify(parity, order_comparison, document_comparison, a_good, b_good, a_bad, b_bad),
        "canonical_checkpoints_writable": False, "fold0_used": False, "public_labels_used": False,
        "temporary_diagnostic_sqlite_only": True, "snapshot_provenance_sha256": provenance,
    }
    print("B2A_QWEN3VL2B_MICRO_REPLAY_RESULT=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    return payload


@app.local_entrypoint()
def main() -> None:
    """Explicit launch entry point; no GPU is launched during source validation."""
    call = micro_replay_remote.spawn()
    print("MICRO_REPLAY_CALL_SPAWNED", flush=True)
    print(f"app_id={getattr(call, '_app_id', None) or getattr(app, '_app_id', None) or 'UNAVAILABLE'}", flush=True)
    print(f"call_id={call.object_id}", flush=True)
