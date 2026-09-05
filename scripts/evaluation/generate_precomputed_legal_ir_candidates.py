"""Build full LegalIR document candidates from cached query embeddings and FAISS.

This is the CPU-side bridge between the immutable DEk21 chunk index and the
leakage-safe Task 1 reranker workflow.  It performs one batched FAISS search,
collapses chunk hits to documents in memory, and writes a compact candidate
cache without loading the embedding model again.
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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.infrastructure.vector_db.faiss_adapter import (  # noqa: E402
    FaissAdapter,
)

DEFAULT_TRAIN = Path("data/raw/btc/LegalIR/train.json")
DEFAULT_EMBEDDINGS = Path("artifacts/task1/train_question_embeddings.npy")
DEFAULT_EMBEDDING_IDS = Path("artifacts/task1/train_question_ids.json")
DEFAULT_FAISS_ROOT = Path("data/vector_store/faiss/legal_chunks_dek21_v2_768")
DEFAULT_OUTPUT = Path("artifacts/task1/training/full_dense200_candidates.jsonl")
DEFAULT_MANIFEST = Path(
    "artifacts/task1/training/full_dense200_candidates_manifest.json"
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Accept answer:null public questions and omit recall metrics.",
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-ids", type=Path, default=DEFAULT_EMBEDDING_IDS)
    parser.add_argument("--faiss-root", type=Path, default=DEFAULT_FAISS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--chunk-k",
        type=_positive_int,
        default=1000,
        help="Nearest chunks retrieved before document collapse.",
    )
    parser.add_argument(
        "--document-depth",
        type=_positive_int,
        default=200,
        help="Maximum unique documents retained per query.",
    )
    parser.add_argument(
        "--evidence-limit",
        type=int,
        choices=(1, 2),
        default=2,
        help="Highest-ranked distinct child chunks retained per document.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=max(1, os.cpu_count() or 1),
        help="FAISS OpenMP threads.",
    )
    parser.add_argument(
        "--max-questions",
        type=_positive_int,
        help="Deterministic first-N smoke run; omit for all 7,000 questions.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_questions(
    path: Path, *, allow_unlabeled: bool = False
) -> tuple[list[str], dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--train must be a non-empty LegalIR object mapping")
    ids: list[str] = []
    gold: dict[str, list[str]] = {}
    for raw_id, raw in payload.items():
        question_id = str(raw_id).strip()
        if not question_id or not isinstance(raw, dict):
            raise ValueError("--train contains an invalid question record")
        question = str(raw.get("question", "")).strip()
        answers = raw.get("answer")
        if not question:
            raise ValueError(f"invalid labeled LegalIR question: {question_id}")
        documents = (
            [str(item).strip() for item in answers if str(item).strip()]
            if isinstance(answers, list)
            else []
        )
        if not documents and not allow_unlabeled:
            raise ValueError(f"LegalIR question has no usable gold IDs: {question_id}")
        ids.append(question_id)
        gold[question_id] = documents
    if len(ids) != len(set(ids)):
        raise ValueError("--train question IDs must be unique")
    return ids, gold


def _load_embedding_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--embedding-ids must contain a non-empty JSON array")
    ids = [str(item).strip() for item in payload]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("--embedding-ids contains blank or duplicate IDs")
    return ids


def _validated_batch(
    embeddings: np.ndarray[Any, Any], start: int, end: int
) -> NDArray[np.float32]:
    batch = np.asarray(embeddings[start:end], dtype=np.float32)
    if batch.ndim != 2 or not np.isfinite(batch).all():
        raise ValueError(f"embedding batch {start}:{end} is invalid")
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"embedding batch {start}:{end} has a zero/invalid norm")
    return cast(
        NDArray[np.float32], np.ascontiguousarray(batch / norms, dtype=np.float32)
    )


def collapse_faiss_row(
    item_ids: Sequence[int],
    scores: Sequence[float],
    payloads: dict[int, dict[str, Any]],
    *,
    chunk_k: int,
    document_depth: int,
    evidence_limit: int,
) -> dict[str, Any]:
    """Collapse one ranked chunk row to deterministic document candidates."""

    documents: list[dict[str, Any]] = []
    by_doc: dict[str, dict[str, Any]] = {}
    unique_docs: set[str] = set()
    valid_hits = 0
    for raw_rank, (raw_id, raw_score) in enumerate(zip(item_ids, scores), start=1):
        item_id = int(raw_id)
        if item_id < 0:
            continue
        payload = payloads.get(item_id)
        if not isinstance(payload, dict):
            raise ValueError(f"FAISS returned missing payload ID {item_id}")
        doc_id = str(payload.get("doc_id", "")).strip()
        chunk_id = str(payload.get("chunk_id", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not doc_id or not chunk_id or not text:
            raise ValueError(f"FAISS payload {item_id} lacks doc/chunk/text")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"FAISS returned a non-finite score for payload {item_id}")
        valid_hits += 1
        unique_docs.add(doc_id)
        document = by_doc.get(doc_id)
        if document is None:
            if len(documents) >= document_depth:
                continue
            metadata = payload.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            document = {
                "doc_id": doc_id,
                "rank": len(documents) + 1,
                "best_chunk_rank": raw_rank,
                "dense_score": score,
                "law_name": str(
                    payload.get("law_name") or metadata.get("law_name") or ""
                ).strip(),
                "metadata": {
                    key: metadata[key]
                    for key in (
                        "source_family",
                        "structure_status",
                        "structure_type",
                        "structure_warnings",
                        "review_reasons",
                    )
                    if key in metadata
                },
                "evidence": [],
            }
            by_doc[doc_id] = document
            documents.append(document)
        evidence = document["evidence"]
        if len(evidence) < evidence_limit and all(
            row["evidence_id"] != chunk_id for row in evidence
        ):
            evidence.append(
                {
                    "evidence_id": chunk_id,
                    "text": text,
                    "score": score,
                    "chunk_rank": raw_rank,
                }
            )
    return {
        "raw_chunk_k": min(chunk_k, valid_hits),
        "unique_document_count": len(unique_docs),
        "document_depth": document_depth,
        "documents": documents,
        "collapse_version": "precomputed-faiss-document-collapse-v2",
        "evidence_limit": evidence_limit,
    }


def _open_temporary_output(path: Path) -> tuple[Path, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    return Path(temporary_name), os.fdopen(
        descriptor, "w", encoding="utf-8", newline="\n"
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    for path, label in (
        (args.train, "--train"),
        (args.embeddings, "--embeddings"),
        (args.embedding_ids, "--embedding-ids"),
        (args.faiss_root / "index.faiss", "FAISS index"),
        (args.faiss_root / "payloads.json", "FAISS payloads"),
        (args.faiss_root / "manifest.json", "FAISS manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.document_depth > args.chunk_k:
        raise ValueError("--document-depth cannot exceed --chunk-k")
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("--output and --manifest must be different files")

    train_ids, gold = _load_questions(args.train, allow_unlabeled=args.allow_unlabeled)
    embedding_ids = _load_embedding_ids(args.embedding_ids)
    if embedding_ids != train_ids:
        raise ValueError(
            "embedding ID order does not exactly match LegalIR train order"
        )
    selected_count = len(train_ids)
    if args.max_questions is not None:
        selected_count = min(selected_count, args.max_questions)
        train_ids = train_ids[:selected_count]

    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(embedding_ids):
        raise ValueError(
            "embedding matrix rows must exactly match --embedding-ids "
            f"({embeddings.shape!r} vs {len(embedding_ids)})"
        )
    if embeddings.dtype.kind != "f":
        raise ValueError("embedding matrix must use a floating-point dtype")

    index_manifest = json.loads(
        (args.faiss_root / "manifest.json").read_text(encoding="utf-8-sig")
    )
    if not isinstance(index_manifest, dict):
        raise ValueError("FAISS manifest must contain an object")
    adapter = FaissAdapter(
        index_path=str(args.faiss_root.parent),
        collection_name=args.faiss_root.name,
    )
    if adapter.index is None or not adapter.payloads:
        raise ValueError("FAISS index is empty")
    if adapter.index.d != embeddings.shape[1]:
        raise ValueError(
            f"embedding dimension {embeddings.shape[1]} != index {adapter.index.d}"
        )
    if adapter.distance != "cosine":
        raise ValueError("cached DEk21 workflow requires a cosine FAISS index")

    import faiss  # Imported after adapter validation for an actionable error.

    faiss.omp_set_num_threads(args.threads)
    started = time.perf_counter()
    recall_sum = 0.0
    labeled_query_count = 0
    full_gold_count = 0
    missing_gold_occurrences = 0
    recall_depths = sorted(
        {
            depth
            for depth in (5, 10, 20, 50, 100, 200, 300, 500, args.document_depth)
            if depth <= args.document_depth
        }
    )
    recall_at_depth = {depth: 0.0 for depth in recall_depths}
    full_gold_at_depth = {depth: 0 for depth in recall_depths}
    candidate_counts: list[int] = []
    unique_counts: list[int] = []
    temporary, stream = _open_temporary_output(args.output)
    completed = False
    try:
        with stream:
            for start in range(0, selected_count, args.batch_size):
                end = min(start + args.batch_size, selected_count)
                batch_started = time.perf_counter()
                batch = _validated_batch(embeddings, start, end)
                batch_scores, batch_ids = adapter.index.search(batch, args.chunk_k)
                for offset, question_id in enumerate(train_ids[start:end]):
                    collapsed = collapse_faiss_row(
                        batch_ids[offset],
                        batch_scores[offset],
                        adapter.payloads,
                        chunk_k=args.chunk_k,
                        document_depth=args.document_depth,
                        evidence_limit=args.evidence_limit,
                    )
                    document_ids = [row["doc_id"] for row in collapsed["documents"]]
                    expected = set(gold[question_id])
                    query_recall: float | None = None
                    if expected:
                        labeled_query_count += 1
                        matched = expected.intersection(document_ids)
                        query_recall = len(matched) / len(expected)
                        recall_sum += query_recall
                        full_gold_count += matched == expected
                        missing_gold_occurrences += len(expected - matched)
                        for depth in recall_depths:
                            depth_matched = expected.intersection(document_ids[:depth])
                            recall_at_depth[depth] += len(depth_matched) / len(expected)
                            full_gold_at_depth[depth] += depth_matched == expected
                    candidate_counts.append(len(document_ids))
                    unique_counts.append(int(collapsed["unique_document_count"]))
                    record = {"question_id": question_id, **collapsed}
                    if query_recall is not None:
                        record["gold_candidate_recall"] = query_recall
                    stream.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                stream.flush()
                elapsed = time.perf_counter() - batch_started
                total_elapsed = time.perf_counter() - started
                rate = end / total_elapsed if total_elapsed else 0.0
                eta = (selected_count - end) / rate if rate else 0.0
                recall_status = (
                    f"macro-candidate-recall {recall_sum / labeled_query_count:.6f}"
                    if labeled_query_count
                    else "unlabeled"
                )
                print(
                    f"candidates {end}/{selected_count} | batch {elapsed:.1f}s | "
                    f"{recall_status} | ETA {eta / 60:.1f} min",
                    flush=True,
                )
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)

    output_sha = _sha256(args.output)
    elapsed_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": "legal-ir-precomputed-document-candidates-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train": str(args.train),
            "train_sha256": _sha256(args.train),
            "embeddings": str(args.embeddings),
            "embeddings_sha256": _sha256(args.embeddings),
            "embedding_ids": str(args.embedding_ids),
            "embedding_ids_sha256": _sha256(args.embedding_ids),
            "faiss_root": str(args.faiss_root),
            "faiss_manifest_sha256": _sha256(args.faiss_root / "manifest.json"),
            "faiss_manifest": index_manifest,
        },
        "settings": {
            "chunk_k": args.chunk_k,
            "document_depth": args.document_depth,
            "evidence_limit": args.evidence_limit,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "max_questions": args.max_questions,
            "allow_unlabeled": args.allow_unlabeled,
        },
        "results": {
            "query_count": selected_count,
            "labeled_query_count": labeled_query_count,
            "macro_gold_candidate_recall": (
                recall_sum / labeled_query_count if labeled_query_count else None
            ),
            "full_gold_query_count": full_gold_count,
            "full_gold_query_ratio": (
                full_gold_count / labeled_query_count if labeled_query_count else None
            ),
            "missing_gold_occurrences": missing_gold_occurrences,
            "candidate_document_count_min": min(candidate_counts),
            "candidate_document_count_max": max(candidate_counts),
            "candidate_document_count_mean": sum(candidate_counts)
            / len(candidate_counts),
            "retrieved_unique_document_count_min": min(unique_counts),
            "retrieved_unique_document_count_max": max(unique_counts),
            "retrieved_unique_document_count_mean": sum(unique_counts)
            / len(unique_counts),
            "candidate_recall_curve": {
                str(depth): {
                    "macro_gold_candidate_recall": recall_at_depth[depth]
                    / labeled_query_count
                    if labeled_query_count
                    else None,
                    "full_gold_query_count": full_gold_at_depth[depth],
                    "full_gold_query_ratio": full_gold_at_depth[depth]
                    / labeled_query_count
                    if labeled_query_count
                    else None,
                }
                for depth in recall_depths
            },
            "elapsed_seconds": elapsed_seconds,
        },
        "output": str(args.output),
        "output_sha256": output_sha,
    }
    _write_json_atomic(args.manifest, manifest)
    return args.output, args.manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"precomputed candidate error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
