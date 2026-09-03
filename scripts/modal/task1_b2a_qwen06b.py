"""Canonical Modal runner for the frozen B2a Qwen3-Reranker-0.6B track.

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import modal


APP_NAME = "task1-b2a-qwen06b"
PROFILE = "hanky77k1"
VOLUME_NAME = "udsc-p13"
MOUNT = Path("/workspace/p13")
RUNTIME = MOUNT / "runtime"
WORKLIST = RUNTIME / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"
CHUNKS = RUNTIME / "data/processed_v3/chunks"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
MODEL_REPO = "Qwen/Qwen3-Reranker-0.6B"
OUTPUT_ROOT = RUNTIME / "artifacts/task1/workflow_b/tv2/b2a/qwen06b"
CHECKPOINT_DB = OUTPUT_ROOT / "chunk_scores.sqlite3"
PREDICTIONS = OUTPUT_ROOT / "predictions.jsonl"
PREDICTIONS_TMP = OUTPUT_ROOT / "predictions.jsonl.tmp"
ERRORS = OUTPUT_ROOT / "errors.jsonl"
RUN_STATE = OUTPUT_ROOT / "run_state.json"

EXPECTED_WORKLIST_SHA256 = "a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a"
EXPECTED_CHUNK_FILES = 8532
EXPECTED_QUERIES = 5600
EXPECTED_PAIRS = 431200
EXPECTED_DOCS_PER_QUERY = 77
EXPECTED_SELECTED_CHUNKS = 1293198
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the Document "
    "contains legal provisions relevant to answering the Query."
)
MAX_LENGTH = 32768
INFERENCE_BATCH_SIZE = 16
SMOKE_PAIRS = 64
BENCHMARK_PAIRS = 512
BENCHMARK_BATCH_SIZES = (16, 32, 64, 128)

QWEN_SYSTEM = (
    "Judge whether the Document meets the requirement based on the Query. "
    'Note that the answer can only be "yes" or "no".'
)
QWEN_PREFIX = (
    "<|im_start|>system\n"
    + QWEN_SYSTEM
    + "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.0.0",
        "accelerate>=1.1,<2.0",
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


def format_parts(query: str, document: str) -> tuple[str, str, str]:
    return QWEN_PREFIX, f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: ", document + QWEN_SUFFIX


def tokenized_pair_inputs(
    tokenizer: Any,
    queries: list[str],
    documents: list[str],
) -> tuple[Any, list[int], int, int]:
    """Tokenize query-document pairs with the frozen truncation contract."""
    if len(queries) != len(documents):
        raise RuntimeError("query/document batch length mismatch")
    all_ids: list[list[int]] = []
    untruncated_lengths: list[int] = []
    truncated = 0
    for query, document in zip(queries, documents):
        prefix, middle, suffix = format_parts(query, document)
        prefix_ids = tokenizer.encode(prefix + middle, add_special_tokens=False)
        document_ids = tokenizer.encode(document, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        full_length = len(prefix_ids) + len(document_ids) + len(suffix_ids)
        untruncated_lengths.append(full_length)
        room = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)
        if room < 1:
            raise RuntimeError("formatted instruction leaves no document token room")
        if len(document_ids) > room:
            document_ids = document_ids[:room]
            truncated += 1
        all_ids.append(prefix_ids + document_ids + suffix_ids)
    tokenizer.padding_side = "left"
    encoded = tokenizer.pad({"input_ids": all_ids}, padding=True, return_tensors="pt")
    return encoded, untruncated_lengths, truncated, max(untruncated_lengths) if untruncated_lengths else 0


def tokenized_inputs(tokenizer: Any, query: str, documents: list[str]) -> tuple[Any, list[int], int]:
    """Compatibility wrapper for the existing one-query scoring path."""
    encoded, lengths, truncated, _ = tokenized_pair_inputs(tokenizer, [query] * len(documents), documents)
    return encoded, lengths, truncated


def score_batch(
    tokenizer: Any,
    model: Any,
    torch: Any,
    queries: list[str],
    documents: list[str],
) -> tuple[list[float], list[int], int, int]:
    encoded, lengths, truncated, max_tokens = tokenized_pair_inputs(tokenizer, queries, documents)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
        selected = logits[:, [model._b2a_no_id, model._b2a_yes_id]]
        scores = torch.softmax(selected, dim=-1)[:, 1]
    return [float(value) for value in scores.detach().cpu().tolist()], lengths, truncated, max_tokens


def score_documents(tokenizer: Any, model: Any, torch: Any, query: str, documents: list[str]) -> tuple[list[float], list[int], int, float]:
    values, lengths, truncated, _ = score_batch(tokenizer, model, torch, [query] * len(documents), documents)
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
        raise RuntimeError(f"missing canonical SQLite checkpoint: {CHECKPOINT_DB}")
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
        if completed_chunks < 202392:
            raise RuntimeError(f"checkpoint is materially behind the recorded commit: {completed_chunks}")
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.convert_tokens_to_ids("yes") != true_token_id or tokenizer.convert_tokens_to_ids("no") != false_token_id:
        raise RuntimeError("tokenizer yes/no IDs do not match locked LogitScore configuration")
    if len(tokenizer.encode("yes", add_special_tokens=False)) != 1 or len(tokenizer.encode("no", add_special_tokens=False)) != 1:
        raise RuntimeError("Qwen yes/no are not single final-token labels")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model._b2a_yes_id = int(true_token_id)
    model._b2a_no_id = int(false_token_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    gpu_name = torch.cuda.get_device_name(0)
    return tokenizer, model, time.perf_counter() - start, gpu_name, true_token_id, false_token_id


def run_remote() -> dict[str, Any]:
    import torch
    import transformers

    remote = verify_remote_inputs()
    checkpoint = inspect_checkpoint()
    print(
        f"CHECKPOINT_VERIFIED path={CHECKPOINT_DB} integrity={checkpoint['integrity']} "
        f"completed_chunks={checkpoint['completed_chunks']} "
        f"completed_q_doc_pairs={checkpoint['completed_pairs']}",
        flush=True,
    )
    questions = load_questions(set(remote["query_ids"]))
    model_path, resolved_revision, provenance, true_token_id, false_token_id = download_locked_snapshot()
    tokenizer, model, load_seconds, gpu_name, _, _ = load_model(
        torch, transformers, model_path, true_token_id, false_token_id
    )
    marker("MODEL_LOAD_DONE")
    print(
        f"RESUME_CHECKPOINT_LOADED\ncompleted_chunks={checkpoint['completed_chunks']}\n"
        f"remaining_chunks={remote['selected_chunks'] - int(checkpoint['completed_chunks'])}",
        flush=True,
    )

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
    full_start = time.perf_counter()
    cache: dict[str, dict[str, str]] = {}
    pending: list[tuple[str, str, list[dict[str, Any]], list[str]]] = []
    pairs_seen = 0

    def flush_pending() -> None:
        nonlocal completed_chunks, completed_pairs, errors
        if not pending:
            return
        work: list[tuple[str, str, str, str]] = []
        try:
            for query, doc, _, missing_ids in pending:
                raw_chunks = load_chunks(doc, cache)
                for chunk_id in missing_ids:
                    if chunk_id not in raw_chunks or not raw_chunks[chunk_id]:
                        raise RuntimeError(f"unresolved raw chunk: {query}/{doc}/{chunk_id}")
                    work.append((query, doc, chunk_id, raw_chunks[chunk_id]))
            for offset in range(0, len(work), INFERENCE_BATCH_SIZE):
                batch = work[offset : offset + INFERENCE_BATCH_SIZE]
                values, _, _, _ = score_batch(
                    tokenizer,
                    model,
                    torch,
                    [item[0] for item in batch],
                    [item[3] for item in batch],
                )
                if len(values) != len(batch) or not all(math.isfinite(value) for value in values):
                    raise RuntimeError("non-finite Qwen score")
                connection.executemany(
                    "INSERT OR IGNORE INTO chunk_scores(query_id,doc_id,chunk_id,score) VALUES (?,?,?,?)",
                    [(item[0], item[1], item[2], value) for item, value in zip(batch, values)],
                )
            connection.commit()
            completed_chunks = int(connection.execute("SELECT COUNT(*) FROM chunk_scores").fetchone()[0])
            completed_pairs = int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT query_id, doc_id FROM chunk_scores GROUP BY query_id, doc_id)"
                ).fetchone()[0]
            )
        except Exception as exc:
            errors += 1
            with ERRORS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"error": repr(exc), "pending_pairs": len(pending)}) + "\n")
        pending.clear()
        elapsed = time.perf_counter() - full_start
        resumed_completed = completed_chunks - initial_completed_chunks
        throughput = resumed_completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, remote["selected_chunks"] - completed_chunks)
        eta = remaining / throughput if throughput > 0 else float("inf")
        eta_text = "UNKNOWN" if not math.isfinite(eta) else f"{eta:.1f}s"
        print(
            f"FULL_PROGRESS chunks_completed={completed_chunks}/{remote['selected_chunks']} "
            f"q_doc_pairs_completed={completed_pairs}/{remote['pairs']} elapsed={elapsed:.1f}s "
            f"throughput={throughput:.3f} chunks/sec estimated_remaining={eta_text} "
            "checkpoint_written=SQLITE_COMMIT",
            flush=True,
        )

    for query, doc, rows in jsonl_groups(WORKLIST):
        pairs_seen += 1
        existing = {row[0] for row in connection.execute("SELECT chunk_id FROM chunk_scores WHERE query_id=? AND doc_id=?", (query, doc))}
        missing_ids = [str(row["chunk_id"]) for row in rows if str(row["chunk_id"]) not in existing]
        if missing_ids:
            pending.append((query, doc, rows, missing_ids))
        if len(pending) >= INFERENCE_BATCH_SIZE:
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
    print("B2A QWEN 0.6B FULL FROZEN INFERENCE", flush=True)
    print("\nStatus:\nPASS_PREDICTIONS_FROZEN", flush=True)
    print("\nRunner:\nscripts/modal/task1_b2a_qwen06b.py", flush=True)
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


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={str(MOUNT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=32768,
)
def run() -> dict[str, Any]:
    return run_remote()


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
def main() -> None:
    if full_call_is_already_active():
        return
    call = run.spawn()
    app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
    print("FULL_CALL_SPAWNED", flush=True)
    print(f"app_id={app_id}", flush=True)
    print(f"call_id={call.object_id}", flush=True)
