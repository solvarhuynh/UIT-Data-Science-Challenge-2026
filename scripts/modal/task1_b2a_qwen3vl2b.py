"""Canonical Modal runner for the frozen B2a Qwen3-VL-Reranker-2B track.

This file intentionally owns both the deterministic smoke path and the full
inference path.  It never reads Task1 labels and it never recomputes True-S2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import modal


APP_NAME = "task1-b2a-qwen3-vl-reranker-2b"
PROFILE = "hanky77k1"
VOLUME_NAME = "udsc-p13"
MOUNT = Path("/workspace/p13")
RUNTIME = MOUNT / "runtime"
WORKLIST = RUNTIME / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"
CHUNKS = RUNTIME / "data/processed_v3/chunks"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
DEPLOYMENT_QUERY_SOURCE = RUNTIME / "data/raw/btc/LegalIR/public-official.json"
MODEL_REPO = "Qwen/Qwen3-VL-Reranker-2B"
# Query-text-fixed Batch1 output is a new immutable run namespace.  The prior
# "full-corrected" artifact used query IDs as model input and is historical only.
OUTPUT_ROOT = RUNTIME / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-full-querytext-fixed"
CHECKPOINT_DB = OUTPUT_ROOT / "chunk_scores.sqlite3"
PREDICTIONS = OUTPUT_ROOT / "predictions.jsonl"
PREDICTIONS_TMP = OUTPUT_ROOT / "predictions.jsonl.tmp"
ERRORS = OUTPUT_ROOT / "errors.jsonl"
RUN_STATE = OUTPUT_ROOT / "run_state.json"
SMOKE_DIAGNOSTIC = RUNTIME / "reports/task1/workflow_b/tv2/b2a/runtime/qwen3vl2b_smoke_diagnostic.json"
SMOKE_WRAPPER_ENTRY = RUNTIME / "reports/task1/workflow_b/tv2/b2a/runtime/qwen3vl2b_smoke_wrapper_entry.json"
BENCHMARK_DIAGNOSTIC = RUNTIME / "reports/task1/workflow_b/tv2/b2a/runtime/qwen3vl2b_batch_benchmark.json"

EXPECTED_WORKLIST_SHA256 = "a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a"
EXPECTED_CHUNK_FILES = 8532
EXPECTED_QUERIES = 5600
EXPECTED_PAIRS = 431200
EXPECTED_DOCS_PER_QUERY = 77
EXPECTED_SELECTED_CHUNKS = 1293198
EXPECTED_DEPLOYMENT_QUERIES = 1000
RUN_MODE = "SCIENTIFIC_F1_F4_ONLY"
FULL_RUN_MODE = "QUERYTEXT_FIXED_BATCH1"
MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODEL_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the Document "
    "contains legal provisions relevant to answering the Query."
)
MAX_LENGTH = 8192
SMOKE_BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 1
SMOKE_PAIRS = 64
BENCHMARK_PAIRS = 512
BENCHMARK_BATCH_SIZES = (16, 32, 64, 128)

QWEN_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


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
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def groups_for_pairs(targets: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for query, doc, rows in jsonl_groups(WORKLIST):
        pair = (query, doc)
        if pair in targets:
            found[pair] = rows
    if set(found) != targets:
        raise RuntimeError("deterministic smoke pair coverage mismatch")
    return found


def marker(name: str) -> None:
    print(name, flush=True)


def jsonl_groups(path: Path) -> Iterable[tuple[str, str, list[dict[str, Any]]]]:
    current: tuple[str, str] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (str(row["query_id"]), str(row["doc_id"]))
            if current is not None and pair != current:
                yield current[0], current[1], rows
                rows = []
            current = pair
            rows.append(row)
        if current is not None:
            yield current[0], current[1], rows


def load_questions(query_ids: set[str]) -> dict[str, str]:
    """Read only question text from the canonical train object."""

    decoder = json.JSONDecoder()
    result: dict[str, str] = {}
    with TRAIN.open("r", encoding="utf-8-sig") as handle:
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
            if query_id in query_ids:
                if not isinstance(record, dict) or "question" not in record:
                    raise RuntimeError(f"canonical question missing: {query_id}")
                result[query_id] = str(record["question"])
            trim()
            if buffer.startswith(","):
                buffer = buffer[1:]
    if set(result) != query_ids:
        raise RuntimeError("canonical question coverage mismatch")
    return result


def verify_remote_inputs() -> dict[str, Any]:
    if not WORKLIST.is_file():
        raise RuntimeError(f"missing canonical worklist: {WORKLIST}")
    worklist_hash = sha256_file(WORKLIST)
    if worklist_hash != EXPECTED_WORKLIST_SHA256:
        raise RuntimeError(f"worklist SHA256 mismatch: {worklist_hash}")
    if not CHUNKS.is_dir():
        raise RuntimeError(f"missing canonical chunk directory: {CHUNKS}")
    chunk_files = sorted(CHUNKS.glob("*.jsonl"))
    if len(chunk_files) != EXPECTED_CHUNK_FILES:
        raise RuntimeError(f"chunk file count mismatch: {len(chunk_files)}")

    pair_count = 0
    first_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    query_docs: dict[str, set[str]] = defaultdict(set)
    duplicate_groups = 0
    wrong_doc = 0
    selected_chunks = 0
    current: tuple[str, str] | None = None
    with WORKLIST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row["query_id"])
            doc = str(row["doc_id"])
            pair = (query, doc)
            if pair != current:
                if pair in seen:
                    duplicate_groups += 1
                else:
                    seen.add(pair)
                    pair_count += 1
                    if len(first_pairs) < 20:
                        first_pairs.append(pair)
                current = pair
            query_docs[query].add(doc)
            selected_chunks += 1
            expected_artifact = f"data/processed_v3/chunks/{doc}.jsonl"
            reference = str(row.get("raw_chunk_text_reference", ""))
            if row.get("source_chunk_artifact") != expected_artifact or not reference.startswith(expected_artifact + "#chunk_id="):
                wrong_doc += 1
    counts = sorted(len(value) for value in query_docs.values())
    median = counts[len(counts) // 2] if counts else 0
    if len(query_docs) != EXPECTED_QUERIES or pair_count != EXPECTED_PAIRS:
        raise RuntimeError(f"worklist cardinality mismatch: {len(query_docs)}/{pair_count}")
    if (min(counts), median, max(counts)) != (77, 77, 77):
        raise RuntimeError(f"docs/query mismatch: {min(counts)}/{median}/{max(counts)}")
    if selected_chunks != EXPECTED_SELECTED_CHUNKS or duplicate_groups or wrong_doc:
        raise RuntimeError(
            f"worklist structure mismatch: chunks={selected_chunks}, "
            f"duplicates={duplicate_groups}, wrong_doc={wrong_doc}"
        )

    references_checked = 0
    first_groups = groups_for_pairs(set(first_pairs))
    for query, doc in first_pairs:
        rows = first_groups[(query, doc)]
        raw_path = CHUNKS / f"{doc}.jsonl"
        raw_chunks = {}
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    raw_chunks[str(item["chunk_id"])] = str(item.get("text", item.get("chunk_text", "")))
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in raw_chunks or not raw_chunks[chunk_id]:
                raise RuntimeError(f"unresolved raw chunk reference: {query}/{doc}/{chunk_id}")
            references_checked += 1
    if references_checked < 20:
        raise RuntimeError(f"only {references_checked} chunk references resolved")
    return {
        "worklist_sha256": worklist_hash,
        "queries": len(query_docs),
        "pairs": pair_count,
        "docs_per_query": [min(counts), median, max(counts)],
        "duplicate_pairs": duplicate_groups,
        "wrong_doc_mappings": wrong_doc,
        "selected_chunks": selected_chunks,
        "chunk_files": len(chunk_files),
        "references_checked": references_checked,
        "first_pairs": first_pairs,
        "query_ids": set(query_docs),
    }


def format_pair(query: str, document: str) -> list[dict[str, Any]]:
    """Return the text-only Qwen3-VL reranker conversation from its snapshot script."""

    return [
        {"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<Instruct>: {INSTRUCTION}"},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "text", "text": document},
            ],
        },
    ]


def tokenized_pair_inputs(
    processor: Any,
    queries: list[str],
    documents: list[str],
) -> tuple[Any, list[int], int, int]:
    """Tokenize text-only pairs using Qwen3-VL-Reranker's snapshot contract."""
    if len(queries) != len(documents):
        raise RuntimeError("query/document batch length mismatch")
    all_ids: list[list[int]] = []
    untruncated_lengths: list[int] = []
    truncated = 0
    special_ids = set(processor.tokenizer.all_special_ids)
    for query, document in zip(queries, documents):
        text = processor.apply_chat_template(format_pair(query, document), tokenize=False, add_generation_prompt=True)
        input_ids = processor.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        full_length = len(input_ids)
        untruncated_lengths.append(full_length)
        if len(input_ids) > MAX_LENGTH:
            # Exact policy from the model-supplied qwen3_vl_reranker.py: retain
            # all special tokens and the first non-special tokens, then restore
            # the final five generation-template tokens.
            prefix = input_ids[:-5]
            non_special_budget = MAX_LENGTH - sum(token in special_ids for token in prefix)
            kept: list[int] = []
            non_special = 0
            for token in prefix:
                if token in special_ids:
                    kept.append(token)
                elif non_special < non_special_budget:
                    kept.append(token)
                    non_special += 1
            input_ids = kept + input_ids[-5:]
            truncated += 1
        all_ids.append(input_ids)
    processor.tokenizer.padding_side = "left"
    encoded = processor.tokenizer.pad({"input_ids": all_ids}, padding=True, return_tensors="pt")
    return encoded, untruncated_lengths, truncated, max(untruncated_lengths) if untruncated_lengths else 0


def tokenized_inputs(processor: Any, query: str, documents: list[str]) -> tuple[Any, list[int], int]:
    """Compatibility wrapper for the existing one-query scoring path."""
    encoded, lengths, truncated, _ = tokenized_pair_inputs(processor, [query] * len(documents), documents)
    return encoded, lengths, truncated


def score_batch(
    processor: Any,
    model: Any,
    torch: Any,
    queries: list[str],
    documents: list[str],
) -> tuple[list[float], list[int], int, int]:
    encoded, lengths, truncated, max_tokens = tokenized_pair_inputs(processor, queries, documents)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = model.model(**encoded).last_hidden_state[:, -1]
        scores = torch.sigmoid(hidden @ model._b2a_score_direction)
    return [float(value) for value in scores.detach().cpu().tolist()], lengths, truncated, max_tokens


def score_documents(processor: Any, model: Any, torch: Any, query: str, documents: list[str]) -> tuple[list[float], list[int], int, float]:
    values, lengths, truncated, _ = score_batch(processor, model, torch, [query] * len(documents), documents)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite Qwen score")
    return values, lengths, truncated, max(lengths) if lengths else 0


def load_chunks(doc: str, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    if doc not in cache:
        path = CHUNKS / f"{doc}.jsonl"
        values: dict[str, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    values[str(item["chunk_id"])] = str(item.get("text", item.get("chunk_text", "")))
        cache[doc] = values
        if len(cache) > 32:
            cache.pop(next(iter(cache)))
    return cache[doc]


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS chunk_scores (query_id TEXT, doc_id TEXT, chunk_id TEXT, score REAL NOT NULL, PRIMARY KEY (query_id, doc_id, chunk_id))"
    )
    connection.commit()
    return connection


def inspect_checkpoint() -> dict[str, int | str]:
    if not CHECKPOINT_DB.is_file():
        # A new, isolated 2B namespace must begin without inheriting any 0.6B
        # rows.  The first permitted 2B run creates this database itself.
        return {"integrity": "absent", "completed_chunks": 0, "completed_pairs": 0}
    connection = sqlite3.connect(f"file:{CHECKPOINT_DB}?mode=ro", uri=True)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        completed_chunks = int(connection.execute("SELECT COUNT(*) FROM chunk_scores").fetchone()[0])
        completed_pairs = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT query_id, doc_id FROM chunk_scores GROUP BY query_id, doc_id)"
            ).fetchone()[0]
        )
        return {
            "integrity": integrity,
            "completed_chunks": completed_chunks,
            "completed_pairs": completed_pairs,
        }
    finally:
        connection.close()


def collect_candidate_groups(pair_count: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    target_indices = {round(i * (pair_count - 1) / 127) for i in range(128)}
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    index = -1
    current: tuple[str, str] | None = None
    rows: list[dict[str, Any]] = []
    with WORKLIST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (str(row["query_id"]), str(row["doc_id"]))
            if current is None:
                current = pair
                index = 0
            elif pair != current:
                if index in target_indices:
                    found[current] = rows
                current = pair
                index += 1
                rows = []
            rows.append(row)
        if current is not None and index in target_indices:
            found[current] = rows
    if len(found) != len(target_indices):
        raise RuntimeError(f"deterministic smoke candidate coverage mismatch: {len(found)}")
    return found


def collect_benchmark_groups() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Take one deterministic, bounded prefix of the canonical worklist."""
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for query, doc, rows in jsonl_groups(WORKLIST):
        found[(query, doc)] = rows
        if len(found) >= BENCHMARK_PAIRS:
            break
    if len(found) != BENCHMARK_PAIRS:
        raise RuntimeError(f"benchmark sample coverage mismatch: {len(found)}")
    return found


def flatten_benchmark_groups(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    questions: dict[str, str],
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    keys: list[tuple[str, str, str]] = []
    queries: list[str] = []
    documents: list[str] = []
    cache: dict[str, dict[str, str]] = {}
    for query, doc in sorted(groups, key=lambda pair: (int(pair[0]) if pair[0].isdigit() else pair[0], int(pair[1]) if pair[1].isdigit() else pair[1])):
        chunks = load_chunks(doc, cache)
        for row in groups[(query, doc)]:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in chunks or not chunks[chunk_id]:
                raise RuntimeError(f"unresolved benchmark chunk: {query}/{doc}/{chunk_id}")
            keys.append((query, doc, chunk_id))
            queries.append(questions[query])
            documents.append(chunks[chunk_id])
    if not keys:
        raise RuntimeError("empty benchmark sample")
    return keys, queries, documents


def natural_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def selected_document_order(
    keys: list[tuple[str, str, str]],
    scores: list[float],
) -> list[tuple[str, str]]:
    by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for (query, doc, _), score in zip(keys, scores):
        by_query[query][doc] = max(by_query[query].get(doc, float("-inf")), score)
    ordered: list[tuple[str, str]] = []
    for query in sorted(by_query, key=natural_key):
        docs = sorted(by_query[query], key=lambda doc: (-by_query[query][doc], natural_key(doc)))
        ordered.extend((query, doc) for doc in docs)
    return ordered


def score_order(keys: list[tuple[str, str, str]], scores: list[float]) -> list[tuple[str, str, str]]:
    return [
        key
        for _, key in sorted(
            zip(scores, keys),
            key=lambda item: (-item[0], natural_key(item[1][0]), natural_key(item[1][1]), item[1][2]),
        )
    ]


def benchmark_batch(
    tokenizer: Any,
    model: Any,
    torch: Any,
    queries: list[str],
    documents: list[str],
    batch_size: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    scores: list[float] = []
    max_tokens = 0
    truncated = 0
    try:
        for offset in range(0, len(documents), batch_size):
            values, lengths, count, batch_max_tokens = score_batch(
                tokenizer,
                model,
                torch,
                queries[offset : offset + batch_size],
                documents[offset : offset + batch_size],
            )
            scores.extend(values)
            max_tokens = max(max_tokens, batch_max_tokens)
            truncated += count
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "chunks": 0,
            "runtime_seconds": None,
            "throughput": None,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "scores": [],
            "max_tokens": max_tokens,
            "truncated": truncated,
            "nan_inf": 0,
            "oom": True,
        }
    runtime = time.perf_counter() - start
    nan_inf = sum(not math.isfinite(value) for value in scores)
    return {
        "batch_size": batch_size,
        "chunks": len(scores),
        "runtime_seconds": runtime,
        "throughput": len(scores) / runtime if runtime > 0 else 0.0,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "scores": scores,
        "max_tokens": max_tokens,
        "truncated": truncated,
        "nan_inf": nan_inf,
        "oom": False,
    }


def resolve_candidate_documents(
    candidate_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], list[str]]:
    documents_by_pair: dict[tuple[str, str], list[str]] = {}
    cache: dict[str, dict[str, str]] = {}
    for query, doc in sorted(candidate_groups):
        chunks = load_chunks(doc, cache)
        documents_by_pair[(query, doc)] = [
            chunks[str(row["chunk_id"])] for row in candidate_groups[(query, doc)]
        ]
    return documents_by_pair


def deterministic_smoke_pairs(
    candidate_documents: dict[tuple[str, str], list[str]],
    questions: dict[str, str],
    tokenizer: Any,
) -> list[tuple[str, str]]:
    if not candidate_documents:
        raise RuntimeError("no canonical worklist pairs")
    candidates: list[tuple[int, tuple[str, str]]] = []
    for query, doc in sorted(candidate_documents):
        documents = candidate_documents[(query, doc)]
        _, lengths, _ = tokenized_inputs(tokenizer, questions[query], documents)
        candidates.append((max(lengths), (query, doc)))
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    selected: list[tuple[str, str]] = []
    for bucket in range(4):
        start = bucket * len(candidates) // 4
        end = (bucket + 1) * len(candidates) // 4
        bucket_values = candidates[start:end]
        for item in bucket_values[:: max(1, len(bucket_values) // (SMOKE_PAIRS // 4))][: SMOKE_PAIRS // 4]:
            selected.append(item[1])
    selected = sorted(set(selected), key=lambda pair: (int(pair[0]) if pair[0].isdigit() else pair[0], int(pair[1]) if pair[1].isdigit() else pair[1]))
    selected = selected[:SMOKE_PAIRS]
    return selected


def download_locked_snapshot() -> tuple[Path, str, dict[str, str], int, int]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION)
    resolved_revision = str(info.sha or "")
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Hugging Face revision mismatch: requested={MODEL_REVISION}, resolved={resolved_revision}"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir="/tmp/hf-cache",
            local_files_only=False,
        )
    )
    required = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "1_LogitScore/config.json",
    ]
    provenance: dict[str, str] = {}
    for relative in required:
        path = snapshot / relative
        if path.is_file():
            provenance[relative] = sha256_file(path)
    logit_config_path = snapshot / "1_LogitScore/config.json"
    if not logit_config_path.is_file():
        raise RuntimeError("locked snapshot lacks 1_LogitScore/config.json")
    logit_config = json.loads(logit_config_path.read_text(encoding="utf-8"))
    true_token_id = int(logit_config.get("true_token_id", -1))
    false_token_id = int(logit_config.get("false_token_id", -1))
    if (true_token_id, false_token_id) != (9693, 2152):
        raise RuntimeError(
            f"locked LogitScore token IDs mismatch: {true_token_id}/{false_token_id}"
        )
    return snapshot, resolved_revision, provenance, true_token_id, false_token_id


def load_model(
    torch: Any,
    transformers: Any,
    model_path: Path,
    true_token_id: int,
    false_token_id: int,
) -> tuple[Any, Any, float, str, int, int]:
    start = time.perf_counter()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    tokenizer = processor.tokenizer
    if tokenizer.convert_tokens_to_ids("yes") != true_token_id or tokenizer.convert_tokens_to_ids("no") != false_token_id:
        raise RuntimeError("tokenizer yes/no IDs do not match locked LogitScore configuration")
    if len(tokenizer.encode("yes", add_special_tokens=False)) != 1 or len(tokenizer.encode("no", add_special_tokens=False)) != 1:
        raise RuntimeError("Qwen3-VL yes/no scoring tokens are not single-token labels")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    # Qwen3-VL-Reranker scores the last hidden state with the difference
    # between the frozen yes/no LM-head vectors, followed by sigmoid.  It is
    # not the historical 0.6B final-token-logit softmax contract.
    with torch.inference_mode():
        model._b2a_score_direction = (
            model.lm_head.weight[true_token_id] - model.lm_head.weight[false_token_id]
        ).detach()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    gpu_name = torch.cuda.get_device_name(0)
    return processor, model, time.perf_counter() - start, gpu_name, true_token_id, false_token_id


def run_remote() -> dict[str, Any]:
    import torch
    import transformers

    remote = verify_remote_inputs()
    checkpoint = inspect_checkpoint()
    if int(checkpoint["completed_chunks"]) or int(checkpoint["completed_pairs"]):
        raise RuntimeError(
            f"fresh full run blocked: checkpoint already contains "
            f"chunks={checkpoint['completed_chunks']} q_docs={checkpoint['completed_pairs']} path={CHECKPOINT_DB}"
        )
    if PREDICTIONS.exists() or RUN_STATE.exists():
        raise RuntimeError(f"fresh full run blocked: existing output artifact in {OUTPUT_ROOT}")
    print(f"FULL_RUN_MODE={FULL_RUN_MODE}", flush=True)
    print(f"MODEL={MODEL_ID}", flush=True)
    print(f"REVISION={MODEL_REVISION}", flush=True)
    print(f"BATCH={INFERENCE_BATCH_SIZE}", flush=True)
    print(f"EXPECTED_QUERIES={EXPECTED_QUERIES}", flush=True)
    print(f"EXPECTED_QDOCS={EXPECTED_PAIRS}", flush=True)
    print(f"EXPECTED_CHUNKS={EXPECTED_SELECTED_CHUNKS}", flush=True)
    print("QUERY_TEXT_GUARD=ENABLED", flush=True)
    print("CHECKPOINT_MODE=FRESH", flush=True)
    print("INITIAL_CHECKPOINT_CHUNKS=0", flush=True)
    print("INITIAL_CHECKPOINT_QDOCS=0", flush=True)
    print("OLD_BAD_CHECKPOINT_REUSED=NO", flush=True)
    print("FOLD0_USED=NO", flush=True)
    print("PUBLIC_LABELS_USED=NO", flush=True)
    print(
        f"CHECKPOINT_VERIFIED path={CHECKPOINT_DB} integrity={checkpoint['integrity']} "
        f"completed_chunks={checkpoint['completed_chunks']} "
        f"completed_q_doc_pairs={checkpoint['completed_pairs']}",
        flush=True,
    )
    questions = load_questions(set(remote["query_ids"]))
    model_path, resolved_revision, provenance, true_token_id, false_token_id = download_locked_snapshot()
    processor, model, load_seconds, gpu_name, _, _ = load_model(
        torch, transformers, model_path, true_token_id, false_token_id
    )
    marker("MODEL_LOAD_DONE")
    print(f"FRESH_CHECKPOINT_READY\nremaining_chunks={remote['selected_chunks']}", flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    connection = init_db(CHECKPOINT_DB)
    errors = 0
    completed_chunks = int(connection.execute("SELECT COUNT(*) FROM chunk_scores").fetchone()[0])
    completed_pairs = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT query_id, doc_id FROM chunk_scores GROUP BY query_id, doc_id)"
        ).fetchone()[0]
    )
    initial_completed_chunks = completed_chunks
    initial_completed_pairs = completed_pairs
    if initial_completed_chunks != 0 or initial_completed_pairs != 0:
        raise RuntimeError("fresh checkpoint unexpectedly became non-empty before scoring")
    full_start = time.perf_counter()
    cache: dict[str, dict[str, str]] = {}
    # Model scoring stays strictly in worklist order at batch size one.  Only
    # durable SQLite writes are buffered, to avoid a synchronous transaction
    # and database-wide progress recount for every q-doc pair.
    pending_rows: list[tuple[str, str, str, float]] = []
    pending_pairs: list[tuple[str, str, bool]] = []
    rolling_commits: deque[tuple[float, int]] = deque()
    last_rolling_throughput = 0.0
    pairs_seen = 0

    def flush_pending() -> None:
        nonlocal completed_chunks, completed_pairs, errors, last_rolling_throughput
        if not pending_rows:
            return
        try:
            changes_before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO chunk_scores(query_id,doc_id,chunk_id,score) VALUES (?,?,?,?)",
                pending_rows,
            )
            connection.commit()
            inserted_rows = connection.total_changes - changes_before
            if inserted_rows != len(pending_rows):
                raise RuntimeError(
                    f"unexpected checkpoint insert conflict: inserted={inserted_rows}, pending={len(pending_rows)}"
                )
            completed_chunks += inserted_rows
            completed_pairs += sum(1 for _, _, pair_was_present in pending_pairs if not pair_was_present)
            now = time.perf_counter()
            rolling_commits.append((now, inserted_rows))
            window_start = max(full_start, now - 60.0)
            while rolling_commits and rolling_commits[0][0] < window_start:
                rolling_commits.popleft()
            rolling_rows = sum(rows for _, rows in rolling_commits)
            rolling_elapsed = now - window_start
            last_rolling_throughput = rolling_rows / rolling_elapsed if rolling_elapsed > 0 else 0.0
        except Exception as exc:
            connection.rollback()
            errors += 1
            with ERRORS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"error": repr(exc), "pending_pairs": len(pending_pairs)}) + "\n")
        pending_rows.clear()
        pending_pairs.clear()
        elapsed = time.perf_counter() - full_start
        resumed_completed = completed_chunks - initial_completed_chunks
        throughput = resumed_completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, remote["selected_chunks"] - completed_chunks)
        eta = remaining / throughput if throughput > 0 else float("inf")
        eta_text = "UNKNOWN" if not math.isfinite(eta) else f"{eta:.1f}s"
        print(
            f"FULL_PROGRESS chunks_completed={completed_chunks}/{remote['selected_chunks']} "
            f"q_doc_pairs_completed={completed_pairs}/{remote['pairs']} elapsed={elapsed:.1f}s "
            f"throughput={throughput:.3f} chunks/sec "
            f"cumulative_new_chunks_per_sec={throughput:.3f} "
            f"rolling_new_chunks_per_sec={last_rolling_throughput:.3f} "
            f"estimated_remaining={eta_text} "
            "checkpoint_written=SQLITE_COMMIT",
            flush=True,
        )

    for query, doc, rows in jsonl_groups(WORKLIST):
        pairs_seen += 1
        existing = {row[0] for row in connection.execute("SELECT chunk_id FROM chunk_scores WHERE query_id=? AND doc_id=?", (query, doc))}
        missing_ids = [str(row["chunk_id"]) for row in rows if str(row["chunk_id"]) not in existing]
        if missing_ids:
            if query not in questions:
                raise RuntimeError(f"canonical question missing for query_id: {query}")
            query_text = questions[query]
            if not isinstance(query_text, str) or not query_text.strip() or query_text == str(query):
                raise RuntimeError(f"invalid canonical question text for query_id: {query}")
            raw_chunks = load_chunks(doc, cache)
            for chunk_id in missing_ids:
                if chunk_id not in raw_chunks or not raw_chunks[chunk_id]:
                    raise RuntimeError(f"unresolved raw chunk: {query}/{doc}/{chunk_id}")
                values, _, _, _ = score_batch(processor, model, torch, [query_text], [raw_chunks[chunk_id]])
                if len(values) != 1 or not math.isfinite(values[0]):
                    raise RuntimeError("non-finite Qwen score")
                pending_rows.append((query, doc, chunk_id, values[0]))
            pending_pairs.append((query, doc, bool(existing)))
        if len(pending_pairs) >= 256 or len(pending_rows) >= 1024:
            flush_pending()
    flush_pending()
    full_seconds = time.perf_counter() - full_start
    connection.commit()
    scored = int(connection.execute("SELECT COUNT(*) FROM chunk_scores").fetchone()[0])
    if errors or scored != EXPECTED_SELECTED_CHUNKS or pairs_seen != EXPECTED_PAIRS:
        raise RuntimeError(f"full inference incomplete: scored={scored}, pairs={pairs_seen}, errors={errors}")
    resumed_scored = scored - initial_completed_chunks
    resumed_throughput = resumed_scored / full_seconds if full_seconds > 0 else 0.0

    by_query: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    rank_rows = {(query, doc): int(rank) for query, doc, rows in jsonl_groups(WORKLIST) for rank in [rows[0]["canonical_k77_rank"]]}
    for query, doc, score in connection.execute("SELECT query_id,doc_id,MAX(score) FROM chunk_scores GROUP BY query_id,doc_id"):
        by_query[str(query)].append((float(score), rank_rows[(str(query), str(doc))], str(doc)))
    if len(by_query) != EXPECTED_QUERIES or sum(len(value) for value in by_query.values()) != EXPECTED_PAIRS:
        raise RuntimeError("document-score aggregation cardinality mismatch")
    with PREDICTIONS_TMP.open("w", encoding="utf-8") as handle:
        for query in sorted(by_query, key=lambda value: int(value) if value.isdigit() else value):
            ordered = sorted(by_query[query], key=lambda item: (-item[0], item[1], int(item[2]) if item[2].isdigit() else item[2]))
            handle.write(json.dumps({"query_id": query, "predicted_doc_ids": [item[2] for item in ordered[:5]], "document_scores": [{"doc_id": item[2], "score": item[0], "canonical_k77_rank": item[1]} for item in ordered]}, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(PREDICTIONS_TMP, PREDICTIONS)
    prediction_hash = sha256_file(PREDICTIONS)
    state = {
        "status": "PASS_PREDICTIONS_FROZEN",
        "model": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": resolved_revision,
        "model_revision_verified": resolved_revision == MODEL_REVISION,
        "true_token_id": true_token_id,
        "false_token_id": false_token_id,
        "snapshot_provenance_sha256": provenance,
        "instruction": INSTRUCTION,
        "worklist_sha256": remote["worklist_sha256"],
        "smoke": "NOT_RUN_FULL_MODE",
        "full_q_doc_pairs": EXPECTED_PAIRS,
        "full_chunks_scored": scored,
        "checkpoint_integrity": checkpoint["integrity"],
        "checkpoint_chunks_recovered": initial_completed_chunks,
        "chunks_skipped_as_already_complete": initial_completed_chunks,
        "chunks_resumed_scored": resumed_scored,
        "full_runtime_seconds": full_seconds,
        "full_throughput_chunks_per_second": resumed_throughput,
        "errors": errors,
        "prediction_artifact": str(PREDICTIONS),
        "prediction_sha256": prediction_hash,
        "labels_loaded": False,
        "fold0_used": False,
        "metrics": "NOT_RUN",
    }
    RUN_STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print("B2A QWEN3-VL-RERANKER-2B FULL SCIENTIFIC INFERENCE", flush=True)
    print("\nStatus:\nPASS_PREDICTIONS_FROZEN", flush=True)
    print("\nRunner:\nscripts/modal/task1_b2a_qwen3vl2b.py", flush=True)
    print(f"\nGPU:\n{gpu_name}", flush=True)
    print(f"\nBatch size:\n{INFERENCE_BATCH_SIZE}", flush=True)
    print(f"\nQueries:\n{remote['queries']}", flush=True)
    print(f"\nQ-doc pairs:\n{remote['pairs']}", flush=True)
    print(f"\nSelected chunks expected:\n{EXPECTED_SELECTED_CHUNKS}", flush=True)
    print(f"\nSelected chunks scored:\n{scored}", flush=True)
    print(f"\nErrors:\n{errors}", flush=True)
    print(f"\nElapsed runtime:\n{full_seconds:.2f} seconds", flush=True)
    print(f"\nAverage resumed throughput:\n{resumed_throughput:.6f} chunks/sec", flush=True)
    print(f"\nCheckpoint integrity:\n{str(checkpoint['integrity']).upper()}", flush=True)
    print(f"\nCheckpoint chunks recovered:\n{initial_completed_chunks}", flush=True)
    print(f"\nChunks skipped as already complete:\n{initial_completed_chunks}", flush=True)
    print(f"\nChunks resumed/scored:\n{resumed_scored}", flush=True)
    print(f"\nPrediction artifact:\n{PREDICTIONS}", flush=True)
    print(f"\nPrediction SHA256:\n{prediction_hash}", flush=True)
    print("\nCheckpoint/resume used:\nYES", flush=True)
    print("\nLabels loaded:\nNO", flush=True)
    print("\nFold0 used:\nNO", flush=True)
    print("\nMetrics:\nNOT_RUN", flush=True)
    print("\nSTOP.", flush=True)
    return state


def replay100_batch1_remote() -> dict[str, Any]:
    """Isolated canonical batch-1 replay of the fixed 100-query recovery sample."""
    import torch
    import transformers

    replay_root = RUNTIME / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-100q-replay"
    replay_predictions = replay_root / "predictions.jsonl"
    if replay_predictions.exists():
        raise RuntimeError(f"replay namespace already contains predictions: {replay_predictions}")
    targets = {round(i * 5599 / 99) for i in range(100)}
    selected: list[tuple[str, str, list[dict[str, Any]]]] = []
    query_ids: list[str] = []
    previous: str | None = None
    index = -1
    include = False
    for query, doc, rows in jsonl_groups(WORKLIST):
        if query != previous:
            previous = query; index += 1; include = index in targets
            if include: query_ids.append(query)
        if include: selected.append((query, doc, rows))
    if len(query_ids) != 100 or len(selected) != 7700:
        raise RuntimeError(f"replay sample mismatch: {len(query_ids)} queries / {len(selected)} pairs")
    questions = load_questions(set(query_ids))
    model_path, revision, provenance, yes, no = download_locked_snapshot()
    processor, model, _, gpu_name, _, _ = load_model(torch, transformers, model_path, yes, no)
    scores: dict[tuple[str, str, str], float] = {}
    ranks: dict[tuple[str, str], int] = {}
    cache: dict[str, dict[str, str]] = {}
    work: list[tuple[str, str, str, str]] = []
    for query, doc, rows in selected:
        ranks[(query, doc)] = int(rows[0]["canonical_k77_rank"])
        raw = load_chunks(doc, cache)
        for row in rows:
            chunk = str(row["chunk_id"])
            if chunk not in raw or not raw[chunk]: raise RuntimeError(f"unresolved replay chunk {query}/{doc}/{chunk}")
            work.append((query, doc, chunk, raw[chunk]))
    work.sort(key=lambda item: (int(item[0]), int(item[1]), item[2]))
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for query, doc, chunk, text in work:
        values, _, _, _ = score_batch(processor, model, torch, [questions[query]], [text])
        key = (query, doc, chunk)
        if key in scores or len(values) != 1: raise RuntimeError("replay score-key mismatch")
        scores[key] = values[0]
    elapsed = time.perf_counter() - started
    if len(scores) != len(work): raise RuntimeError("replay incomplete")
    grouped: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    best: dict[tuple[str, str], float] = {}
    for (query, doc, _), score in scores.items(): best[(query, doc)] = max(best.get((query, doc), float("-inf")), score)
    for (query, doc), score in best.items(): grouped[query].append((doc, score, ranks[(query, doc)]))
    replay_root.mkdir(parents=True, exist_ok=False)
    with replay_predictions.open("w", encoding="utf-8") as handle:
        for query in query_ids:
            ordered = sorted(grouped[query], key=lambda item: (-item[1], item[2], int(item[0])))
            handle.write(json.dumps({"query_id": query, "predicted_doc_ids": [item[0] for item in ordered[:5]], "document_scores": [{"doc_id": item[0], "score": item[1], "canonical_k77_rank": item[2]} for item in ordered]}, separators=(",", ":")) + "\n")
    prediction_hash = sha256_file(replay_predictions)
    records = json.load(TRAIN.open(encoding="utf-8-sig"))
    recalls: list[float] = []; precisions: list[float] = []; auc_rows: list[tuple[float, bool]] = []
    for query in query_ids:
        gold = {str(doc) for doc in records[query]["answer"]}; ordered = sorted(grouped[query], key=lambda item: (-item[1], item[2], int(item[0]))); top = {doc for doc, _, _ in ordered[:5]}
        recalls.append(len(gold & top) / len(gold)); precisions.append(len(gold & top) / 5)
        auc_rows.extend((score, doc in gold) for doc, score, _ in ordered)
    ordered_auc = sorted(auc_rows); pos = sum(label for _, label in ordered_auc); neg = len(ordered_auc) - pos; rank = 1; positive_ranks = 0.0; start = 0
    while start < len(ordered_auc):
        end = start + 1
        while end < len(ordered_auc) and ordered_auc[end][0] == ordered_auc[start][0]: end += 1
        positive_ranks += ((rank + rank + end - start - 1) / 2) * sum(label for _, label in ordered_auc[start:end]); rank += end - start; start = end
    auc = (positive_ranks - pos * (pos + 1) / 2) / (pos * neg)
    payload = {"status": "PASS", "gpu": gpu_name, "queries": len(query_ids), "pairs": len(selected), "chunks": len(work), "missing_keys": 0, "duplicate_keys": 0, "checkpoint_errors": 0, "elapsed_seconds": elapsed, "throughput": len(work) / elapsed, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "recall": sum(recalls) / len(recalls), "precision": sum(precisions) / len(precisions), "auc": auc, "prediction_sha256": prediction_hash, "model_revision": revision, "provenance": provenance}
    (replay_root / "run_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("BATCH1_REPLAY_RESULT=" + json.dumps(payload, separators=(",", ":")), flush=True)
    return payload


def validation400_batch1_remote() -> dict[str, Any]:
    """Run the disjoint, fold-balanced 400-query batch-1 validation only."""
    import torch
    import transformers

    validation_root = RUNTIME / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-400q-disjoint-validation"
    validation_predictions = validation_root / "predictions.jsonl"
    validation_state = validation_root / "run_state.json"
    if validation_root.exists():
        raise RuntimeError(f"validation namespace already exists: {validation_root}")

    remote = verify_remote_inputs()
    replay_positions = {round(i * (EXPECTED_QUERIES - 1) / 99) for i in range(100)}
    replay_query_ids: set[str] = set()
    eligible_by_fold: dict[int, list[str]] = {fold: [] for fold in range(1, 5)}
    fold_by_query: dict[str, int] = {}
    previous: str | None = None
    query_index = -1
    for query, _, rows in jsonl_groups(WORKLIST):
        if query != previous:
            previous = query
            query_index += 1
            fold = int(rows[0]["fold"])
            if fold not in eligible_by_fold:
                raise RuntimeError(f"non-F1-F4 query encountered: {query} fold={fold}")
            fold_by_query[query] = fold
            if query_index in replay_positions:
                replay_query_ids.add(query)
            else:
                eligible_by_fold[fold].append(query)
    if query_index + 1 != EXPECTED_QUERIES or len(replay_query_ids) != 100:
        raise RuntimeError("canonical replay-query reconstruction mismatch")

    selected_by_fold: dict[int, list[str]] = {}
    for fold, candidates in eligible_by_fold.items():
        positions = [round(i * (len(candidates) - 1) / 99) for i in range(100)]
        selected = [candidates[position] for position in positions]
        if len(selected) != 100 or len(set(selected)) != 100:
            raise RuntimeError(f"deterministic F{fold} selection mismatch")
        selected_by_fold[fold] = selected
    selected_query_ids = {query for values in selected_by_fold.values() for query in values}
    overlap = selected_query_ids & replay_query_ids
    if len(selected_query_ids) != 400 or overlap:
        raise RuntimeError(f"disjoint validation selection mismatch: queries={len(selected_query_ids)} overlap={len(overlap)}")
    print("B1_400_SELECTION=" + json.dumps({
        "fold_counts": {f"F{fold}": len(selected_by_fold[fold]) for fold in range(1, 5)},
        "total_queries": len(selected_query_ids),
        "overlap_with_replay100": len(overlap),
    }, separators=(",", ":")), flush=True)

    selected: list[tuple[str, str, list[dict[str, Any]]]] = []
    pair_counts: dict[str, int] = defaultdict(int)
    for query, doc, rows in jsonl_groups(WORKLIST):
        if query in selected_query_ids:
            if int(rows[0]["fold"]) != fold_by_query[query]:
                raise RuntimeError(f"fold association changed for query {query}")
            selected.append((query, doc, rows))
            pair_counts[query] += 1
    if len(selected) != 400 * EXPECTED_DOCS_PER_QUERY or set(pair_counts) != selected_query_ids or any(count != EXPECTED_DOCS_PER_QUERY for count in pair_counts.values()):
        raise RuntimeError("400Q candidate-pool cardinality mismatch")

    questions = load_questions(selected_query_ids)
    model_path, revision, provenance, yes, no = download_locked_snapshot()
    processor, model, _, gpu_name, _, _ = load_model(torch, transformers, model_path, yes, no)
    scores: dict[tuple[str, str, str], float] = {}
    ranks: dict[tuple[str, str], int] = {}
    chunk_cache: dict[str, dict[str, str]] = {}
    work: list[tuple[str, str, str, str]] = []
    for query, doc, rows in selected:
        ranks[(query, doc)] = int(rows[0]["canonical_k77_rank"])
        chunks = load_chunks(doc, chunk_cache)
        for row in rows:
            chunk = str(row["chunk_id"])
            text = chunks.get(chunk)
            if not text:
                raise RuntimeError(f"unresolved validation chunk {query}/{doc}/{chunk}")
            work.append((query, doc, chunk, text))
    work.sort(key=lambda item: (int(item[0]), int(item[1]), item[2]))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for query, doc, chunk, text in work:
        values, _, _, _ = score_batch(processor, model, torch, [questions[query]], [text])
        key = (query, doc, chunk)
        if key in scores or len(values) != 1:
            raise RuntimeError("validation score-key mismatch")
        scores[key] = values[0]
    elapsed = time.perf_counter() - started
    if len(scores) != len(work):
        raise RuntimeError("validation scoring incomplete")

    grouped: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    best: dict[tuple[str, str], float] = {}
    for (query, doc, _), score in scores.items():
        best[(query, doc)] = max(best.get((query, doc), float("-inf")), score)
    for (query, doc), score in best.items():
        grouped[query].append((doc, score, ranks[(query, doc)]))
    if len(grouped) != 400 or any(len(grouped[query]) != EXPECTED_DOCS_PER_QUERY for query in selected_query_ids):
        raise RuntimeError("validation document-score cardinality mismatch")

    ordered_queries = [query for fold in range(1, 5) for query in selected_by_fold[fold]]
    final_top5: dict[str, list[str]] = {}
    validation_root.mkdir(parents=True, exist_ok=False)
    with validation_predictions.open("w", encoding="utf-8") as handle:
        for query in ordered_queries:
            ordered = sorted(grouped[query], key=lambda item: (-item[1], item[2], int(item[0])))
            final_top5[query] = [item[0] for item in ordered[:5]]
            handle.write(json.dumps({"query_id": query, "predicted_doc_ids": final_top5[query], "document_scores": [{"doc_id": item[0], "score": item[1], "canonical_k77_rank": item[2]} for item in ordered]}, separators=(",", ":")) + "\n")
    prediction_hash = sha256_file(validation_predictions)

    def auc_for(score_rows: list[tuple[float, bool]]) -> float:
        ordered_auc = sorted(score_rows)
        positive = sum(label for _, label in ordered_auc)
        negative = len(ordered_auc) - positive
        if not positive or not negative:
            raise RuntimeError("validation AUC requires positive and negative documents")
        rank = 1
        positive_ranks = 0.0
        start = 0
        while start < len(ordered_auc):
            end = start + 1
            while end < len(ordered_auc) and ordered_auc[end][0] == ordered_auc[start][0]:
                end += 1
            positive_ranks += ((rank + rank + end - start - 1) / 2) * sum(label for _, label in ordered_auc[start:end])
            rank += end - start
            start = end
        return (positive_ranks - positive * (positive + 1) / 2) / (positive * negative)

    records = json.load(TRAIN.open(encoding="utf-8-sig"))
    metrics: dict[int, dict[str, Any]] = {fold: {"recalls": [], "precisions": [], "auc_rows": []} for fold in range(1, 5)}
    pooled_recalls: list[float] = []
    pooled_precisions: list[float] = []
    pooled_auc_rows: list[tuple[float, bool]] = []
    top5_consistency = 0
    for query in ordered_queries:
        fold = fold_by_query[query]
        ordered = sorted(grouped[query], key=lambda item: (-item[1], item[2], int(item[0])))
        expected_top5 = [item[0] for item in ordered[:5]]
        if final_top5[query] == expected_top5:
            top5_consistency += 1
        gold = {str(doc) for doc in records[query]["answer"]}
        overlap_count = len(gold & set(final_top5[query]))
        recall = overlap_count / len(gold)
        precision = overlap_count / 5
        pooled_recalls.append(recall)
        pooled_precisions.append(precision)
        metrics[fold]["recalls"].append(recall)
        metrics[fold]["precisions"].append(precision)
        score_rows = [(score, doc in gold) for doc, score, _ in ordered]
        pooled_auc_rows.extend(score_rows)
        metrics[fold]["auc_rows"].extend(score_rows)
    fold_results = {str(fold): {"queries": len(metrics[fold]["recalls"]), "recall": sum(metrics[fold]["recalls"]) / len(metrics[fold]["recalls"]), "precision": sum(metrics[fold]["precisions"]) / len(metrics[fold]["precisions"]), "auc": auc_for(metrics[fold]["auc_rows"])} for fold in range(1, 5)}
    payload = {
        "status": "PASS" if top5_consistency == 400 else "FAIL",
        "model": MODEL_ID,
        "model_revision": revision,
        "gpu": gpu_name,
        "effective_batch": 1,
        "queries": len(ordered_queries),
        "pairs": len(selected),
        "chunks": len(work),
        "overlap_with_replay100": len(overlap),
        "missing_query_ids": len(selected_query_ids - set(grouped)),
        "duplicate_query_ids": len(ordered_queries) - len(set(ordered_queries)),
        "predicted_docs_per_query": 5,
        "top5_consistency": top5_consistency,
        "recall": sum(pooled_recalls) / len(pooled_recalls),
        "precision": sum(pooled_precisions) / len(pooled_precisions),
        "auc": auc_for(pooled_auc_rows),
        "folds": fold_results,
        "elapsed_seconds": elapsed,
        "throughput_chunks_per_second": len(work) / elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "prediction_sha256": prediction_hash,
        "worklist_sha256": remote["worklist_sha256"],
        "provenance": provenance,
        "fold0_used": False,
        "public_labels_used": False,
    }
    validation_state.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("B1_400_VALIDATION_RESULT=" + json.dumps(payload, separators=(",", ":")), flush=True)
    return payload


def smoke_remote() -> dict[str, Any]:
    """Run the bounded, non-persistent 64-pair compatibility path only."""

    import traceback

    stage = "SMOKE_START"

    def mark(name: str) -> None:
        nonlocal stage
        stage = name
        print(name, flush=True)

    def persist(payload: dict[str, Any]) -> None:
        SMOKE_DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_DIAGNOSTIC.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        volume.commit()

    try:
        mark("SMOKE_START")
        import torch
        import transformers

        mark("REMOTE_INPUT_VERIFY_START")
        remote = verify_remote_inputs()
        mark("REMOTE_INPUT_VERIFY_DONE")

        mark("CANDIDATE_GROUPS_START")
        candidate_groups = collect_candidate_groups(EXPECTED_PAIRS)
        mark("CANDIDATE_GROUPS_DONE")

        mark("QUESTION_LOAD_START")
        questions = load_questions({query for query, _ in candidate_groups})
        mark("QUESTION_LOAD_DONE")

        mark("MODEL_DOWNLOAD_START")
        model_path, resolved_revision, provenance, true_token_id, false_token_id = download_locked_snapshot()
        mark("MODEL_DOWNLOAD_DONE")

        mark("PROCESSOR_MODEL_LOAD_START")
        processor, model, load_seconds, gpu_name, _, _ = load_model(
            torch, transformers, model_path, true_token_id, false_token_id
        )
        mark("PROCESSOR_MODEL_LOAD_DONE")

        mark("CHUNK_RESOLUTION_START")
        candidate_documents = resolve_candidate_documents(candidate_groups)
        mark("CHUNK_RESOLUTION_DONE")

        mark("SMOKE_PAIR_SELECTION_START")
        selected_pairs = deterministic_smoke_pairs(candidate_documents, questions, processor)
        groups = {pair: candidate_groups[pair] for pair in selected_pairs}
        mark("SMOKE_PAIR_SELECTION_DONE")

        mark("FLATTEN_INPUTS_START")
        keys, queries, documents = flatten_benchmark_groups(groups, questions)
        mark("FLATTEN_INPUTS_DONE")

        mark("TOKENIZATION_FORWARD_START")
        result = benchmark_batch(processor, model, torch, queries, documents, SMOKE_BATCH_SIZE)
        mark("TOKENIZATION_FORWARD_DONE")
        if result["oom"] or result["nan_inf"] or result["chunks"] != len(keys):
            raise RuntimeError("2B smoke compatibility failure")

        mark("DOC_AGGREGATION_START")
        doc_scores: dict[tuple[str, str], float] = {}
        for (query, doc, _), score in zip(keys, result["scores"]):
            doc_scores[(query, doc)] = max(doc_scores.get((query, doc), float("-inf")), score)
        if len(doc_scores) != SMOKE_PAIRS or not all(math.isfinite(score) for score in doc_scores.values()):
            raise RuntimeError("2B smoke document-score aggregation failure")
        mark("DOC_AGGREGATION_DONE")

        values = list(doc_scores.values())
        result_payload = {
            "status": "PASS",
            "stage": "SMOKE_PASS",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": MODEL_ID,
            "revision": resolved_revision,
            "gpu": gpu_name,
            "gpu_requested": "A10",
            "batch_size": SMOKE_BATCH_SIZE,
            "smoke_q_doc_pairs": len(selected_pairs),
            "smoke_chunks": len(keys),
            "model_load_seconds": load_seconds,
            "runtime_seconds": result["runtime_seconds"],
            "throughput_chunks_per_second": result["throughput"],
            "peak_vram_gib": result["peak_vram_gib"],
            "max_untruncated_input_tokens": result["max_tokens"],
            "truncated_inputs": result["truncated"],
            "score_min": min(values),
            "score_median": sorted(values)[len(values) // 2],
            "score_max": max(values),
            "unique_document_scores": len(set(values)),
            "nan_inf": result["nan_inf"],
            "oom": False,
            "labels_loaded": False,
            "fold0_used": False,
            "full_inference_started": False,
        }
        persist(result_payload)
        mark("SMOKE_PASS")
        return {
            "mode": RUN_MODE,
            "deployment_query_source": str(DEPLOYMENT_QUERY_SOURCE),
            "deployment_queries_expected": EXPECTED_DEPLOYMENT_QUERIES,
            "model": MODEL_ID,
            "resolved_revision": resolved_revision,
            "snapshot_provenance_sha256": provenance,
            "model_load_seconds": load_seconds,
            "gpu": gpu_name,
            "smoke_q_doc_pairs": len(selected_pairs),
            "smoke_chunks": len(keys),
            "scores": result,
            "labels_loaded": False,
            "fold0_used": False,
            "metrics": "NOT_RUN",
            "persistent_checkpoint_written": False,
        }
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "stage": stage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "gpu_requested": "A10",
            "batch_size": SMOKE_BATCH_SIZE,
            "labels_loaded": False,
            "fold0_used": False,
            "full_inference_started": False,
        }
        persist(failure)
        print(f"SMOKE_DIAGNOSTIC_PERSISTED={SMOKE_DIAGNOSTIC}", flush=True)
        raise


def benchmark_remote() -> dict[str, Any]:
    """Benchmark the existing deterministic True-S2 sample without inference outputs."""

    import traceback

    stage = "BENCHMARK_START"

    def mark(name: str) -> None:
        nonlocal stage
        stage = name
        print(name, flush=True)

    def persist(payload: dict[str, Any]) -> None:
        BENCHMARK_DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_DIAGNOSTIC.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        volume.commit()

    def summarize(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key != "scores"}

    try:
        mark("BENCHMARK_START")
        import torch
        import transformers

        mark("REMOTE_INPUT_VERIFY_START")
        verify_remote_inputs()
        mark("REMOTE_INPUT_VERIFY_DONE")

        mark("BENCHMARK_GROUPS_START")
        groups = collect_benchmark_groups()
        questions = load_questions({query for query, _ in groups})
        keys, queries, documents = flatten_benchmark_groups(groups, questions)
        if len(groups) != BENCHMARK_PAIRS:
            raise RuntimeError(f"benchmark q-doc coverage mismatch: {len(groups)}")
        mark("BENCHMARK_GROUPS_DONE")

        mark("MODEL_DOWNLOAD_START")
        model_path, resolved_revision, provenance, true_token_id, false_token_id = download_locked_snapshot()
        mark("MODEL_DOWNLOAD_DONE")

        mark("PROCESSOR_MODEL_LOAD_START")
        processor, model, load_seconds, gpu_name, _, _ = load_model(
            torch, transformers, model_path, true_token_id, false_token_id
        )
        mark("PROCESSOR_MODEL_LOAD_DONE")

        results: dict[int, dict[str, Any]] = {}
        for batch_size in (16, 32, 64):
            mark(f"BATCH_{batch_size}_START")
            results[batch_size] = benchmark_batch(processor, model, torch, queries, documents, batch_size)
            mark(f"BATCH_{batch_size}_DONE")

        reference = results[16]
        if reference["oom"] or reference["nan_inf"] or reference["chunks"] != len(keys):
            raise RuntimeError("batch 16 reference benchmark failed")
        reference_scores = reference["scores"]
        reference_chunk_order = score_order(keys, reference_scores)
        reference_document_order = selected_document_order(keys, reference_scores)

        reports: dict[str, dict[str, Any]] = {}
        for batch_size, result in results.items():
            report = summarize(result)
            scores = result["scores"]
            finite_documents = 0
            if not result["oom"] and len(scores) == len(keys):
                document_values: dict[tuple[str, str], float] = {}
                for (query, doc, _), score in zip(keys, scores):
                    document_values[(query, doc)] = max(document_values.get((query, doc), float("-inf")), score)
                finite_documents = sum(math.isfinite(value) for value in document_values.values())
            report["finite_q_doc_scores"] = finite_documents
            if batch_size == 16:
                report["reference"] = True
                report["max_abs_score_delta_vs_batch16"] = 0.0
                report["chunk_ordering_parity"] = True
                report["document_ordering_parity"] = True
            elif result["oom"] or result["nan_inf"] or len(scores) != len(reference_scores):
                report["max_abs_score_delta_vs_batch16"] = None
                report["chunk_ordering_parity"] = False
                report["document_ordering_parity"] = False
            else:
                report["max_abs_score_delta_vs_batch16"] = max(
                    abs(value - baseline) for value, baseline in zip(scores, reference_scores)
                )
                report["chunk_ordering_parity"] = score_order(keys, scores) == reference_chunk_order
                report["document_ordering_parity"] = (
                    selected_document_order(keys, scores) == reference_document_order
                )
            report["eligible_for_full_inference"] = bool(
                not report["oom"]
                and report["nan_inf"] == 0
                and report["chunks"] == len(keys)
                and report["finite_q_doc_scores"] == BENCHMARK_PAIRS
                and report["peak_vram_gib"] <= 18.0
                and report["max_abs_score_delta_vs_batch16"] is not None
                and report["max_abs_score_delta_vs_batch16"] <= 1e-4
                and report["chunk_ordering_parity"]
                and report["document_ordering_parity"]
            )
            reports[str(batch_size)] = report

        eligible = [
            (batch_size, report)
            for batch_size, report in ((16, reports["16"]), (32, reports["32"]), (64, reports["64"]))
            if report["eligible_for_full_inference"]
        ]
        selected_batch = max(eligible, key=lambda item: (item[1]["throughput"], item[0]))[0] if eligible else None
        status = "PASS_BATCH_SELECTED" if selected_batch is not None else "BLOCKED"
        payload = {
            "status": status,
            "stage": "BENCHMARK_PASS" if selected_batch is not None else "BENCHMARK_NO_ELIGIBLE_BATCH",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": MODEL_ID,
            "revision": resolved_revision,
            "snapshot_provenance_sha256": provenance,
            "gpu": gpu_name,
            "gpu_requested": "A10",
            "benchmark_q_doc_pairs": len(groups),
            "benchmark_chunks": len(keys),
            "model_load_seconds": load_seconds,
            "batches": reports,
            "selected_full_inference_batch": selected_batch,
            "labels_loaded": False,
            "fold0_used": False,
            "full_inference_started": False,
        }
        persist(payload)
        mark(payload["stage"])
        return payload
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "stage": stage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "gpu_requested": "A10",
            "labels_loaded": False,
            "fold0_used": False,
            "full_inference_started": False,
        }
        persist(failure)
        print(f"BENCHMARK_DIAGNOSTIC_PERSISTED={BENCHMARK_DIAGNOSTIC}", flush=True)
        raise


@app.function(
    image=image,
    gpu="A10",
    volumes={str(MOUNT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=32768,
)
def run() -> dict[str, Any]:
    return run_remote()


@app.function(
    image=image,
    gpu="A10",
    volumes={str(MOUNT): volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
)
def replay100() -> dict[str, Any]:
    return replay100_batch1_remote()


@app.function(
    image=image,
    gpu="A10",
    volumes={str(MOUNT): volume},
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
)
def validation400() -> dict[str, Any]:
    return validation400_batch1_remote()


@app.function(
    image=image,
    gpu="A10",
    volumes={str(MOUNT): volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
)
def smoke() -> dict[str, Any]:
    print("CANONICAL_SMOKE_WRAPPER_ENTERED", flush=True)
    SMOKE_WRAPPER_ENTRY.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_WRAPPER_ENTRY.write_text(
        json.dumps({"wrapper_entered": True, "stage": "BEFORE_SMOKE_REMOTE_CALL"}, indent=2) + "\n",
        encoding="utf-8",
    )
    volume.commit()
    return smoke_remote()


@app.function(
    image=image,
    gpu="A10",
    volumes={str(MOUNT): volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
)
def benchmark() -> dict[str, Any]:
    return benchmark_remote()


def full_call_is_already_active() -> bool:
    """Refuse a duplicate full submission without touching an existing call."""

    stats = run.get_current_stats()
    if not stats.num_running_inputs and not stats.backlog:
        return False
    print("FULL_CALL_ALREADY_ACTIVE", flush=True)
    print(f"backlog={stats.backlog}", flush=True)
    print(f"running_inputs={stats.num_running_inputs}", flush=True)
    return True


@app.local_entrypoint()
def main(mode: str = "smoke", call_id: str = "") -> None:
    if mode == "smoke":
        call = smoke.spawn()
        app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
        print("SMOKE_CALL_SPAWNED", flush=True)
        print(f"app_id={app_id}", flush=True)
        print(f"call_id={call.object_id}", flush=True)
    elif mode == "benchmark":
        call = benchmark.spawn()
    elif mode == "reattach":
        if not call_id.startswith("fc-"):
            raise ValueError("call_id must be an existing fc-* ID")
        result = modal.FunctionCall.from_id(call_id).get()
        print(result, flush=True)
        return
    elif mode == "full":
        if full_call_is_already_active():
            return
        call = run.spawn()
    elif mode == "replay100":
        call = replay100.spawn()
    elif mode == "validation400":
        call = validation400.spawn()
    else:
        raise ValueError("mode must be 'smoke', 'benchmark', 'full', 'replay100', or 'validation400'")
    app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
    print(f"{mode.upper()}_CALL_SPAWNED", flush=True)
    print(f"app_id={app_id}", flush=True)
    print(f"call_id={call.object_id}", flush=True)
