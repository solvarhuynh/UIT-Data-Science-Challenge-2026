"""Complete only the missing B2a True-S2 pairs from the derived canonical K77.

The K77 population is deliberately not persisted as a second artifact.  It is
derived from the historical full candidate pool by applying the same target
fold and union-rank rules recorded by the existing Task1 scripts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
FULL_POOL = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS = ROOT / "data/processed_v3/chunks"
WORKLIST = ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"
REPORT = ROOT / "reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen_true_s2_top3_worklist_report.json"
TEMP = WORKLIST.with_name(WORKLIST.name + ".completion.tmp")
EXPECTED_POOL_SHA256 = "e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6"
TARGET_FOLDS = {1, 2, 3, 4}
EXPECTED_QUERIES = 5600
EXPECTED_K77_PAIRS = 431200
EXPECTED_EXISTING_PAIRS = 423835
EXPECTED_MISSING_PAIRS = 7365
SOURCE_PAYLOAD_PATH = "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
_SELECT_TRUE_S2 = None
QUERY_ID_FIELD = re.compile(r'"query_id"\s*:\s*"([^"]+)"')
FOLD_FIELD = re.compile(r'"fold"\s*:\s*(\d+)')
DOC_ID_FIELD = re.compile(r'"doc_id"\s*:\s*"([^"]+)"')
UNION_RANK_FIELD = re.compile(r'"union_rank"\s*:\s*(\d+)')


def fail(message: str) -> None:
    raise RuntimeError(message)


def sort_key(pair: tuple[str, str]) -> tuple[int, int | str]:
    def value(text: str) -> int | str:
        return int(text) if text.isdigit() else text

    return value(pair[0]), value(pair[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_prefix(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                fail("worklist prefix ended before the recorded pre-run size")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def worklist_fields(raw: bytes) -> tuple[str, str, str, int, int, str, str]:
    def string_value(key: bytes) -> str:
        marker = b'"' + key + b'"'
        position = raw.find(marker)
        if position < 0:
            fail(f"existing worklist row is missing {key.decode()}")
        start = raw.find(b'"', raw.find(b":", position) + 1)
        end = raw.find(b'"', start + 1)
        if start < 0 or end < 0:
            fail("existing worklist row has an invalid string field")
        return raw[start + 1:end].decode("utf-8")

    def integer_value(key: bytes) -> int:
        marker = b'"' + key + b'"'
        position = raw.find(marker)
        if position < 0:
            fail(f"existing worklist row is missing {key.decode()}")
        start = raw.find(b":", position) + 1
        while start < len(raw) and raw[start] in b" \t":
            start += 1
        end = start
        while end < len(raw) and raw[end] in b"-0123456789":
            end += 1
        return int(raw[start:end])

    required = ("query_id", "doc_id", "chunk_id", "source_chunk_artifact", "raw_chunk_text_reference")
    return (
        string_value(b"query_id"), string_value(b"doc_id"), string_value(b"chunk_id"),
        integer_value(b"bm25_within_doc_rank"), integer_value(b"canonical_k77_rank"),
        string_value(b"source_chunk_artifact"), string_value(b"raw_chunk_text_reference"),
    )


def target_fold_map() -> dict[str, int]:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for item in payload["folds"]:
        fold = int(item["fold"])
        if fold in TARGET_FOLDS:
            for query_id in item["validation_ids"]:
                query = str(query_id)
                if query in result:
                    fail(f"duplicate target fold query: {query}")
                result[query] = fold
    if len(result) != EXPECTED_QUERIES or set(result.values()) != TARGET_FOLDS:
        fail("target fold map does not contain exactly the approved folds 1-4")
    return result


def whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def skip_json_value(text: str, index: int) -> int:
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text, whitespace(text, index))
    return end


def load_questions(targets: set[str]) -> dict[str, str]:
    """Extract only target question text from the canonical train object.

    The answer field is never read or used; this is query-text loading only.
    The structural walk follows the existing Task1 reader rather than relying
    on an alternate query source.
    """

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
                    fail("unexpected EOF in canonical train source")
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
            fail("canonical train source is not a JSON object")
        buffer = buffer[1:]
        while True:
            trim()
            if buffer.startswith("}"):
                break
            query_id = str(decode())
            trim()
            if not buffer.startswith(":"):
                fail("invalid canonical train object")
            buffer = buffer[1:]
            record = decode()
            if query_id in targets:
                if not isinstance(record, dict) or "question" not in record:
                    fail(f"canonical question missing: {query_id}")
                result[query_id] = str(record["question"])
            trim()
            if buffer.startswith(","):
                buffer = buffer[1:]
    if set(result) != targets:
        fail("canonical question coverage does not match target K77 queries")
    return result


def derive_k77(folds: dict[str, int]) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Build the K77 projection in memory without creating a K77 artifact."""

    records: dict[tuple[str, str], int] = {}
    query_counts: dict[str, int] = {}
    source_queries: set[str] = set()
    with FULL_POOL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_match = QUERY_ID_FIELD.search(line)
            fold_match = FOLD_FIELD.search(line)
            if query_match is None or fold_match is None:
                fail("full candidate pool row lacks canonical query/fold fields")
            query = query_match.group(1)
            if query not in folds:
                continue
            if int(fold_match.group(1)) != folds[query]:
                fail(f"candidate fold mismatch: {query}")
            if query in source_queries:
                fail(f"duplicate candidate query record: {query}")
            source_queries.add(query)
            local: set[str] = set()
            kept = 0
            docs = DOC_ID_FIELD.findall(line)
            ranks = UNION_RANK_FIELD.findall(line)
            if len(docs) != 200 or len(ranks) != 200:
                fail(f"full candidate pool row is not the canonical 200-candidate schema: {query}")
            for doc_value, rank_value in zip(docs, ranks):
                doc = str(doc_value)
                if doc in local:
                    fail(f"duplicate candidate document: {query}/{doc}")
                local.add(doc)
                union_rank = int(rank_value)
                if union_rank <= 77:
                    key = (query, doc)
                    if key in records:
                        fail(f"duplicate derived K77 pair: {query}/{doc}")
                    records[key] = union_rank
                    kept += 1
            query_counts[query] = kept
    if source_queries != set(folds):
        fail("full candidate pool does not cover exactly the target fold queries")
    counts = list(query_counts.values())
    median = sorted(counts)[len(counts) // 2] if counts else 0
    if (
        len(query_counts) != EXPECTED_QUERIES
        or len(records) != EXPECTED_K77_PAIRS
        or (min(counts), median, max(counts)) != (77, 77, 77)
    ):
        fail("derived K77 projection failed the 5600 x 77 contract")
    return records, query_counts


def read_existing_worklist() -> tuple[set[tuple[str, str]], int, str, int, int, int]:
    pairs: set[tuple[str, str]] = set()
    current_pair: tuple[str, str] | None = None
    current_rows: set[tuple[str, str, int, int]] = set()
    wrong_doc = 0
    duplicate_rows = 0
    digest = hashlib.sha256()
    size = 0
    with WORKLIST.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            query = str(row["query_id"])
            doc = str(row["doc_id"])
            pair = (query, doc)
            if pair != current_pair:
                current_pair = pair
                current_rows = set()
                if pair in pairs:
                    duplicate_rows += 1
            pairs.add(pair)
            identity = (query, doc, int(row["bm25_within_doc_rank"]), int(row["canonical_k77_rank"]))
            if identity in current_rows:
                duplicate_rows += 1
            current_rows.add(identity)
            expected_artifact = f"data/processed_v3/chunks/{doc}.jsonl"
            if row.get("source_chunk_artifact") != expected_artifact:
                wrong_doc += 1
            reference = str(row.get("raw_chunk_text_reference", ""))
            if not reference.startswith(expected_artifact + "#chunk_id="):
                wrong_doc += 1
    return pairs, size, digest.hexdigest(), duplicate_rows, wrong_doc, len(pairs)


def load_chunks(doc: str) -> tuple[dict[str, Any], ...]:
    path = CHUNKS / f"{doc}.jsonl"
    if not path.is_file():
        fail(f"missing canonical chunk file: {path}")
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        fail(f"empty canonical chunk file: {path}")
    return tuple(chunks)


def historical_selector():
    global _SELECT_TRUE_S2
    if _SELECT_TRUE_S2 is None:
        path = ROOT / "scripts/beam/task1_v2/evidence.py"
        spec = importlib.util.spec_from_file_location("b2a_historical_evidence", path)
        if spec is None or spec.loader is None:
            fail(f"cannot load historical evidence module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SELECT_TRUE_S2 = module.select_true_s2
    return _SELECT_TRUE_S2


def selector_rows(query: str, doc: str, fold: int, canonical_rank: int, question: str) -> list[dict[str, Any]]:
    selected = historical_selector()(question, load_chunks(doc), topk=3)
    ranks = [int(item["bm25_rank"]) for item in selected]
    if len(selected) > 3 or ranks != list(range(1, len(selected) + 1)):
        fail(f"invalid True-S2 selection ranks: {query}/{doc}")
    output: list[dict[str, Any]] = []
    source_artifact = f"data/processed_v3/chunks/{doc}.jsonl"
    for item in selected:
        chunk_id = str(item["chunk_id"])
        output.append(
            {
                "bm25_within_doc_rank": int(item["bm25_rank"]),
                "bm25_within_doc_score": float(item["bm25_score"]),
                "canonical_k77_rank": int(canonical_rank),
                "chunk_id": chunk_id,
                "doc_id": doc,
                "fold": int(fold),
                "query_id": query,
                "raw_chunk_text_reference": f"{source_artifact}#chunk_id={chunk_id}",
                "selector_name": str(item["selector_name"]),
                "source_chunk_artifact": source_artifact,
                "source_payload_path": SOURCE_PAYLOAD_PATH,
                "union_rank": int(canonical_rank),
            }
        )
    return output


def pair_signature(rows: Iterable[dict[str, Any]]) -> list[tuple[int, str]]:
    return sorted((int(row["bm25_within_doc_rank"]), str(row["chunk_id"])) for row in rows)


def validate_worklist(
    path: Path,
    k77: dict[tuple[str, str], dict[str, Any]],
    existing_size: int,
    existing_sha256: str,
    sample_pairs: list[tuple[str, str]],
) -> tuple[int, int, int, int, dict[tuple[str, str], list[dict[str, Any]]]]:
    pairs: set[tuple[str, str]] = set()
    qdocs: dict[str, set[str]] = defaultdict(set)
    current_pair: tuple[str, str] | None = None
    current_ranks: list[int] = []
    current_chunks: set[str] = set()
    current_ids: set[tuple[str, str, int, str]] = set()
    invalid_groups = 0
    duplicate_rows = 0
    sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    wrong_doc = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row["query_id"])
            doc = str(row["doc_id"])
            pair = (query, doc)
            if pair != current_pair:
                if current_pair is not None and (len(current_ranks) > 3 or sorted(current_ranks) != list(range(1, len(current_ranks) + 1)) or len(current_ranks) != len(current_chunks)):
                    invalid_groups += 1
                current_pair = pair
                current_ranks = []
                current_chunks = set()
                current_ids = set()
                if pair in pairs:
                    duplicate_rows += 1
            pairs.add(pair)
            qdocs[query].add(doc)
            rank = int(row["bm25_within_doc_rank"])
            chunk_id = str(row["chunk_id"])
            identity = (query, doc, rank, chunk_id)
            if identity in current_ids:
                duplicate_rows += 1
            current_ids.add(identity)
            current_ranks.append(rank)
            current_chunks.add(chunk_id)
            expected_artifact = f"data/processed_v3/chunks/{doc}.jsonl"
            if row.get("source_chunk_artifact") != expected_artifact or not str(row.get("raw_chunk_text_reference", "")).startswith(expected_artifact + "#chunk_id="):
                wrong_doc += 1
            if pair in sample_pairs:
                sample[pair].append(json.loads(line))
    if current_pair is not None and (len(current_ranks) > 3 or sorted(current_ranks) != list(range(1, len(current_ranks) + 1)) or len(current_ranks) != len(current_chunks)):
        invalid_groups += 1
    missing = len(set(k77) - pairs)
    outside = len(pairs - set(k77))
    if sha256_prefix(path, existing_size) != existing_sha256:
        fail("existing worklist content changed before the appended region")
    if missing or outside or wrong_doc or duplicate_rows or invalid_groups:
        fail(f"final worklist validation failed: missing={missing}, outside={outside}, wrong_doc={wrong_doc}, duplicates={duplicate_rows}, invalid_groups={invalid_groups}")
    counts = sorted(len(value) for value in qdocs.values())
    median = counts[len(counts) // 2] if counts else 0
    if len(qdocs) != EXPECTED_QUERIES or (min(counts), median, max(counts)) != (77, 77, 77):
        fail("final worklist query/doc cardinality failed")
    return len(qdocs), len(pairs), min(counts), max(counts), sample


def reproduce(
    sample_pairs: list[tuple[str, str]],
    sample_rows: dict[tuple[str, str], list[dict[str, Any]]],
    questions: dict[str, str],
) -> int:
    for query, doc in sample_pairs:
        expected = pair_signature(sample_rows.get((query, doc), []))
        actual = pair_signature(
            {
                "bm25_within_doc_rank": item["bm25_rank"],
                "chunk_id": item["chunk_id"],
            }
            for item in historical_selector()(questions[query], load_chunks(doc), topk=3)
        )
        if actual != expected:
            fail(f"100-pair reproduction mismatch: {query}/{doc}")
    return len(sample_pairs)


def update_existing_report(values: dict[str, Any]) -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report.update(values)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if sha256_file(FULL_POOL) != EXPECTED_POOL_SHA256:
        fail("full candidate pool SHA256 mismatch")
    folds = target_fold_map()
    k77, query_counts = derive_k77(folds)
    existing, old_size, old_sha256, old_duplicates, old_wrong_doc, old_row_keys = read_existing_worklist()
    outside = existing - set(k77)
    missing = set(k77) - existing
    if len(existing) != EXPECTED_EXISTING_PAIRS or len(outside) != 0 or len(missing) != EXPECTED_MISSING_PAIRS or old_duplicates != 0 or old_wrong_doc != 0:
        fail(f"preflight failed: existing={len(existing)}, outside={len(outside)}, missing={len(missing)}, duplicates={old_duplicates}, wrong_doc={old_wrong_doc}")
    questions = load_questions(set(folds))
    if TEMP.exists():
        fail(f"temporary completion file already exists: {TEMP}")
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        with WORKLIST.open("rb") as source, TEMP.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
            for query, doc in sorted(missing, key=sort_key):
                canonical_rank = k77[(query, doc)]
                rows = selector_rows(query, doc, folds[query], canonical_rank, questions[query])
                for row in rows:
                    target.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                processed += 1
        if processed != EXPECTED_MISSING_PAIRS:
            fail(f"processed pair count mismatch: {processed}")
        old_sample = sorted(existing, key=sort_key)[:50]
        new_sample = sorted(missing, key=sort_key)[:50]
        sample_pairs = old_sample + new_sample
        qcount, final_pairs, min_docs, max_docs, sample_rows = validate_worklist(TEMP, k77, old_size, old_sha256, sample_pairs)
        reproduction = reproduce(sample_pairs, sample_rows, questions)
        if reproduction != 100:
            fail(f"reproduction sample size mismatch: {reproduction}")
        os.replace(TEMP, WORKLIST)
        update_existing_report(
            {
                "worklist_build_status": "COMPLETE",
                "current_processed_pair_count": final_pairs,
                "missing_canonical_pairs": 0,
                "missing_canonical_queries": 0,
                "current_complete_queries_at_k77": qcount,
                "current_partial_query_docs": 0,
                "recovered_exact_worklist_producer": True,
                "canonical_k77_source": "derived from candidate_refs_full.jsonl; target folds 1-4; union_rank <= 77",
                "canonical_k77_source_sha256": EXPECTED_POOL_SHA256,
                "missing_pairs_processed": processed,
                "existing_pairs_changed": 0,
                "post_completion_reproduction": "100/100",
                "qwen_status": "NOT_RUN",
                "gpu_status": "NOT_RUN",
                "scientific_metrics_computed": False,
            }
        )
        print("B2A TRUE-S2 DERIVED-K77 COMPLETION")
        print("\nStatus:\nPASS_WORKLIST_COMPLETE")
        print("\nFull pool SHA256 verified:\nYES")
        print(f"\nDerived K77 queries:\n{len(query_counts)}")
        print(f"\nDerived K77 pairs:\n{len(k77)}")
        print(f"\nDerived docs/query:\n77/77/77")
        print(f"\nExisting pairs before:\n{len(existing)}")
        print(f"\nExisting pairs outside derived K77:\n{len(outside)}")
        print(f"\nMissing pairs before:\n{len(missing)}")
        print(f"\nMissing pairs processed:\n{processed}")
        print("\nExisting pairs changed:\n0")
        print(f"\nFinal pairs:\n{final_pairs}")
        print(f"\nQueries with exactly 77 docs:\n{qcount}")
        print("\nDuplicate pairs:\n0")
        print("\nMissing canonical pairs:\n0")
        print("\nWrong-doc mappings:\n0")
        print(f"\nPost-completion reproduction:\n{reproduction}/100")
        print("\nBuilder:\nscripts/beam/task1_v2/build_b2a_true_s2_missing_pairs.py")
        print("\nStandalone K77 file created:\nNO")
        print("\nQwen:\nNOT_RUN")
        print("\nGPU:\nNOT_RUN")
        print("\nMetrics:\nNOT_RUN")
        print("\nSTOP.")
    except Exception:
        if TEMP.exists():
            print(f"Temporary safe-write file retained for inspection: {TEMP}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
