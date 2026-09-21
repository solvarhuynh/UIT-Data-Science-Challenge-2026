"""CPU-only BGE scorer for bounded Task1 (query_id, document_id) worklists.

The scorer deliberately reuses the active Sentence-Transformers CrossEncoder
contract and the canonical Task1 true-S2 top-three chunk selector.  It never
uses the teacher-Qwen fields for scoring or ranking decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.beam.task1_v2.evidence import select_true_s2  # noqa: E402


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAX_LENGTH = 512
AGGREGATION = "MAX"
SELECTOR = "true_s2_bm25_within_document_v2"
QUESTIONS_PATH = PROJECT_ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS_ROOT = PROJECT_ROOT / "data/processed_v3/chunks"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_worklist(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"worklist not found: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
            doc = str(row.get("document_id", ""))
            if not qid or not doc:
                raise ValueError(f"worklist row {line_number} needs query_id/document_id")
            identity = (qid, doc)
            if identity in seen:
                raise ValueError(f"duplicate worklist identity: {qid}/{doc}")
            seen.add(identity)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("worklist selection is empty")
    return rows


def load_questions(path: Path, query_ids: set[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical question source not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result: dict[str, str] = {}
    for query_id in query_ids:
        record = payload.get(query_id)
        if not isinstance(record, dict) or "question" not in record:
            raise KeyError(f"canonical question missing: {query_id}")
        question = str(record["question"])
        if not question.strip() or question == query_id:
            raise ValueError(f"invalid canonical question text: {query_id}")
        result[query_id] = question
    return result


def load_document_chunks(doc_id: str, cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if doc_id in cache:
        return cache[doc_id]
    path = CHUNKS_ROOT / f"{doc_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"canonical document chunks not found: {path}")
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id", ""))
            text = str(item.get("text", item.get("raw_chunk_text", "")))
            item["chunk_id"] = chunk_id
            item["text"] = text
            if not chunk_id.startswith(doc_id + "_") or not text.strip():
                raise ValueError(f"invalid canonical chunk {path}:{line_number}")
            chunks.append(item)
    if not chunks:
        raise ValueError(f"canonical document has no chunks: {path}")
    if len({str(item["chunk_id"]) for item in chunks}) != len(chunks):
        raise ValueError(f"duplicate canonical chunk IDs: {path}")
    cache[doc_id] = chunks
    return chunks


def resolve_pairs(
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    document_cache: dict[str, list[dict[str, Any]]],
) -> tuple[list[tuple[str, str]], list[dict[str, Any]], float, int]:
    pairs: list[tuple[str, str]] = []
    resolved: list[dict[str, Any]] = []
    start = time.perf_counter()
    for row in rows:
        qid = str(row["query_id"])
        doc = str(row["document_id"])
        question = questions[qid]
        chunks = load_document_chunks(doc, document_cache)
        selected = select_true_s2(question, chunks, topk=3)
        if not (1 <= len(selected) <= 3):
            raise ValueError(f"TOP3_UP_TO_AVAILABLE violation: {qid}/{doc}")
        selected_ids = [str(item["chunk_id"]) for item in selected]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"duplicate selected chunk IDs: {qid}/{doc}")
        if any(
            not str(item.get("raw_chunk_text", "")).strip()
            or not chunk_id.startswith(doc + "_")
            for item, chunk_id in zip(selected, selected_ids)
        ):
            raise ValueError(f"invalid selected chunk text/identity: {qid}/{doc}")
        start_index = len(pairs)
        for item in selected:
            pairs.append((question, str(item["raw_chunk_text"])))
        resolved.append(
            {
                "query_id": qid,
                "document_id": doc,
                "selected_chunk_ids": selected_ids,
                "pair_start": start_index,
                "pair_count": len(selected),
            }
        )
    return pairs, resolved, time.perf_counter() - start, len(document_cache)


def current_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def validate_model_config(model_path: Path) -> dict[str, Any]:
    if "qwen" in str(model_path).casefold():
        raise RuntimeError("UNEXPECTED_QWEN_RUNTIME_PATH")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"BGE config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_labels = int(config.get("num_labels", len(config.get("id2label", {})) or -1))
    if config.get("model_type") != "xlm-roberta" or num_labels != 1:
        raise RuntimeError("UNEXPECTED_NON_BGE_MODEL_CONFIG")
    config["_resolved_num_labels"] = num_labels
    return config


def score_rows(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    batch_size: int,
    output_path: Path,
    metrics_path: Path | None,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    config = validate_model_config(model_path)
    questions = load_questions(QUESTIONS_PATH, {str(row["query_id"]) for row in rows})
    document_cache: dict[str, list[dict[str, Any]]] = {}
    pairs, resolved, preparation_seconds, unique_docs = resolve_pairs(rows, questions, document_cache)
    rss_values = [value for value in (current_rss_mb(),) if value is not None]

    load_start = time.perf_counter()
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        str(model_path),
        device="cpu",
        max_length=MAX_LENGTH,
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - load_start
    rss_values.append(current_rss_mb())

    inference_start = time.perf_counter()
    raw_scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    inference_seconds = time.perf_counter() - inference_start
    rss_values.append(current_rss_mb())
    scores = [float(value) for value in raw_scores]
    if len(scores) != len(pairs):
        raise RuntimeError(f"score count mismatch: {len(scores)}/{len(pairs)}")
    if any(not math.isfinite(value) for value in scores):
        raise RuntimeError("non-finite BGE score")

    output_rows: list[dict[str, Any]] = []
    for item in resolved:
        start = int(item["pair_start"])
        count = int(item["pair_count"])
        chunk_scores = scores[start : start + count]
        output_rows.append(
            {
                "query_id": item["query_id"],
                "document_id": item["document_id"],
                "bge_score": max(chunk_scores),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "aggregation": AGGREGATION,
                "selector": SELECTOR,
                "selected_chunk_ids": item["selected_chunk_ids"],
                "chunk_scores": chunk_scores,
            }
        )
    identities = [(row["query_id"], row["document_id"]) for row in output_rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate output identities")
    if any(not math.isfinite(float(row["bge_score"])) for row in output_rows):
        raise RuntimeError("non-finite document score")

    output_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows)
    atomic_write_text(output_path, output_text)
    wall_seconds = time.perf_counter() - wall_start
    rss_values = [value for value in rss_values if value is not None]
    metrics: dict[str, Any] = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_branch": "sentence-transformers.CrossEncoder",
        "model_config": {
            "model_type": config.get("model_type"),
            "num_labels": config.get("_resolved_num_labels"),
            "torch_dtype": config.get("torch_dtype"),
        },
        "device": "cpu",
        "gpu_used": False,
        "modal_used": False,
        "max_length": MAX_LENGTH,
        "truncation": True,
        "batch_size": batch_size,
        "activation": "CrossEncoder default activation_fn/model.activation_fn; single-label default Sigmoid; no manual sigmoid",
        "dtype": "local model config / CPU execution; use_fp16=False",
        "selector": SELECTOR,
        "aggregation": AGGREGATION,
        "query_source": str(QUESTIONS_PATH),
        "chunk_source": str(CHUNKS_ROOT),
        "worklist": None,
        "output": str(output_path),
        "worklist_pairs": len(rows),
        "output_rows": len(output_rows),
        "chunk_pairs_scored": len(pairs),
        "unique_documents": unique_docs,
        "all_identities_resolved": True,
        "all_scores_finite": True,
        "duplicate_outputs": 0,
        "document_preparation_seconds": preparation_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "total_wall_seconds": wall_seconds,
        "worklist_pairs_per_second": len(rows) / wall_seconds if wall_seconds else 0.0,
        "chunk_pairs_per_second": len(pairs) / inference_seconds if inference_seconds else 0.0,
        "seconds_per_worklist_pair": wall_seconds / len(rows) if rows else 0.0,
        "peak_rss_mb_observed": max(rss_values) if rss_values else None,
        "output_sha256": sha256_file(output_path),
    }
    if metrics_path is not None:
        atomic_write_json(metrics_path, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded CPU-only BGE scorer for Task1 q-doc worklists")
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/reranker"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metrics-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    rows = load_worklist(args.worklist, args.limit)
    metrics = score_rows(
        rows,
        model_path=args.model,
        batch_size=args.batch_size,
        output_path=args.output,
        metrics_path=args.metrics_output,
    )
    metrics["worklist"] = str(args.worklist)
    if args.metrics_output is not None:
        atomic_write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
