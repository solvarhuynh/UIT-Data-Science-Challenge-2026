"""Generate TV2 dense candidates in the strict PredictionSample format for TV5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    load_benchmark,
    load_warmup,
    write_run_manifest,
)
from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.contracts.retrieval import RetrievalHit  # noqa: E402
from udsc2026.infrastructure.config import load_config  # noqa: E402
from udsc2026.infrastructure.embedding.bkai_client import (  # noqa: E402
    EmbeddingClient,
)
from udsc2026.infrastructure.vector_db.factory import (  # noqa: E402
    get_vector_db_adapter,
    resolve_vector_db_config,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    question_group = parser.add_mutually_exclusive_group()
    question_group.add_argument(
        "--benchmark",
        type=Path,
        help="TV4 benchmark source; use this for synthetic/internal retrieval smoke.",
    )
    question_group.add_argument(
        "--legal-ir-train",
        type=Path,
        help="Organizer LegalIR train.json; use this for full 7,000-query candidates.",
    )
    question_group.add_argument(
        "--public-official",
        type=Path,
        help="Label-free public-official.json; use this for production public candidates.",
    )
    question_group.add_argument(
        "--private-official",
        type=Path,
        help="Label-free private-official.json; use this for private retrieval only.",
    )
    parser.add_argument("--config-env", default="gpu")
    parser.add_argument("--candidate-k", type=_positive_int, default=50)
    parser.add_argument("--query-batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--max-questions",
        "--limit",
        dest="limit",
        type=_positive_int,
        help="Use only the first N questions for a smoke run.",
    )
    parser.add_argument(
        "--benchmark-subset",
        type=Path,
        default=Path("artifacts/tv2/research/smoke/smoke_benchmark.jsonl"),
        help="Filtered benchmark written when --limit is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tv2/production/dense_predictions.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/tv2/production/dense_run_manifest.json"),
    )
    parser.add_argument(
        "--pv1-index",
        type=Path,
        help="Explicit PV1 IndexFlatIP file; enables mapping-backed PV1 retrieval.",
    )
    parser.add_argument(
        "--pv1-mapping",
        type=Path,
        help="Explicit PV1 row-to-chunk JSONL mapping.",
    )
    parser.add_argument(
        "--pv1-chunks-dir",
        type=Path,
        help="Canonical PV1 chunks directory used to materialize selected hits.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_completed_question_ids(path: Path) -> set[str]:
    """Read completed JSONL IDs without retaining prediction objects in RAM."""

    if not path.exists():
        return set()
    if path.suffix.casefold() != ".jsonl":
        raise ValueError("resume output must use .jsonl")

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                question_id = payload["question_id"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prediction JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(
                    f"prediction question_id must be a non-empty string at "
                    f"{path}:{line_number}"
                )
            if question_id in completed:
                raise ValueError(f"duplicate question_id in existing output: {question_id}")
            completed.add(question_id)
    return completed


def _write_prediction_batch(stream: Any, predictions: list[PredictionSample]) -> None:
    """Write one batch with the same JSONL serialization as write_predictions."""

    for prediction in predictions:
        payload = prediction.model_dump(mode="json", exclude_none=True)
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _run_pv1_mapping_backend(
    args: argparse.Namespace,
    samples: list[tuple[str, str]],
) -> list[Path]:
    """Run the same dense search contract against an explicit PV1 index/mapping."""
    if not (args.pv1_index and args.pv1_mapping and args.pv1_chunks_dir):
        raise ValueError("--pv1-index, --pv1-mapping and --pv1-chunks-dir are required together")
    if len(samples) != 2080 or len({question_id for question_id, _ in samples}) != 2080:
        raise ValueError("PV1 Private retrieval requires exactly 2080 unique queries")
    if any(not question.strip() for _, question in samples):
        raise ValueError("Private input contains a missing question")

    import faiss
    import numpy as np

    pv1_manifest_path = Path("data/processed_pv1/metadata/pv1_dense_faiss_manifest.json")
    pv1_manifest = json.loads(pv1_manifest_path.read_text(encoding="utf-8"))
    corpus_manifest = json.loads(Path("data/processed_pv1/metadata/pv1_corpus_manifest.json").read_text(encoding="utf-8"))
    if pv1_manifest.get("dense_faiss_gate") != "PASS":
        raise ValueError("PV1 dense/FAISS manifest is not PASS")
    index = faiss.read_index(str(args.pv1_index))
    if type(index).__name__ != "IndexFlatIP" or index.d != 768 or index.ntotal != 1_272_971:
        raise ValueError(
            f"unexpected PV1 FAISS contract: type={type(index).__name__} d={index.d} ntotal={index.ntotal}"
        )

    row_to_item: list[tuple[str, str]] = []
    chunk_ids: set[str] = set()
    with args.pv1_mapping.open(encoding="utf-8") as stream:
        for expected_row, line in enumerate(stream):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["row"]) != expected_row:
                raise ValueError("PV1 mapping row sequence is not contiguous")
            chunk_id = str(item["chunk_id"])
            if chunk_id in chunk_ids:
                raise ValueError(f"duplicate PV1 mapping chunk_id: {chunk_id}")
            chunk_ids.add(chunk_id)
            row_to_item.append((chunk_id, str(item["document_id"])))
    if len(row_to_item) != index.ntotal or len(chunk_ids) != index.ntotal:
        raise ValueError("PV1 mapping does not exactly cover FAISS rows")

    config = load_config(args.config_env)
    embedding = dict(config.get("embedding", {}))
    model_path = str(embedding.get("model_path", "./models/dek21-v2"))
    if embedding.get("model_id") != "huyydangg/DEk21_hcmute_embedding_v2":
        raise ValueError("PV1 query embedding model_id mismatch")
    if int(embedding.get("max_length", 256)) != 256 or int(embedding.get("output_dimension", 768)) != 768:
        raise ValueError("PV1 query embedding dimension/max_length mismatch")
    if not bool(embedding.get("normalize_embeddings", True)):
        raise ValueError("PV1 query embedding normalization must be enabled")
    if str(embedding.get("device", "cpu")).casefold() != "cpu":
        raise ValueError("PV1 retrieval is CPU-only")
    embedder = EmbeddingClient(
        model_path=model_path,
        device="cpu",
        batch_size=int(embedding.get("batch_size", 32)),
        max_length=256,
        normalize_embeddings=True,
        output_dimension=768,
        window_long_texts=False,
    )

    started = time.perf_counter()
    search_rows: list[list[tuple[int, float]]] = []
    for offset in range(0, len(samples), args.query_batch_size):
        batch = samples[offset : offset + args.query_batch_size]
        vectors = np.asarray(
            embedder.embed_documents([question for _, question in batch], batch_size=args.query_batch_size),
            dtype="float32",
        )
        scores, rows = index.search(vectors, args.candidate_k)
        for query_scores, query_rows in zip(scores, rows):
            selected: list[tuple[int, float]] = []
            seen_docs: set[str] = set()
            for score, row in zip(query_scores, query_rows):
                if int(row) < 0:
                    continue
                document_id = row_to_item[int(row)][1]
                if document_id in seen_docs:
                    continue
                seen_docs.add(document_id)
                selected.append((int(row), float(score)))
            search_rows.append(selected)
        print(f"PV1_DENSE_PROGRESS completed={len(search_rows)}/{len(samples)}", flush=True)

    selected_chunk_ids = {row_to_item[row][0] for hits in search_rows for row, _ in hits}
    selected_records: dict[str, dict[str, Any]] = {}
    for path in sorted(args.pv1_chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_id = str(record["chunk_id"])
                if chunk_id in selected_chunk_ids:
                    selected_records[chunk_id] = record
    if set(selected_records) != selected_chunk_ids:
        raise ValueError("PV1 selected chunk records are incomplete")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for (question_id, _), hits in zip(samples, search_rows):
            materialized: list[RetrievalHit] = []
            for rank, (row, score) in enumerate(hits, start=1):
                record = selected_records[row_to_item[row][0]]
                materialized.append(
                    RetrievalHit(
                        chunk_id=str(record["chunk_id"]),
                        parent_id=record.get("parent_id"),
                        doc_id=str(record["doc_id"]),
                        text=str(record["text"]),
                        score=score,
                        dense_score=score,
                        final_score=score,
                        source=record.get("source"),
                        law_name=record.get("law_name"),
                        article=record.get("article"),
                        clause=record.get("clause"),
                        metadata=dict(record.get("metadata") or {}),
                        rank=rank,
                    )
                )
            _write_prediction_batch(stream, [PredictionSample(question_id=question_id, hits=materialized)])

    output_sha = _sha256(args.output)
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": "CPU",
        "question_source": str(args.private_official),
        "question_source_sha256": _sha256(args.private_official),
        "question_source_type": "private_official",
        "question_count": len(samples),
        "candidate_k": args.candidate_k,
        "output": str(args.output),
        "output_sha256": output_sha,
        "elapsed_seconds": time.perf_counter() - started,
        "query_embeddings": "RECOMPUTED_CPU",
        "query_embedding_contract": {
            "model_id": embedding.get("model_id"),
            "model_path": model_path,
            "pooling": "mean",
            "normalization": True,
            "max_length": 256,
            "dimension": 768,
        },
        "pv1_corpus_fingerprint": pv1_manifest["pv1_corpus_fingerprint"],
        "faiss_sha256": _sha256(args.pv1_index),
        "mapping_sha256": _sha256(args.pv1_mapping),
        "faiss_index_type": "IndexFlatIP",
        "faiss_dimension": index.d,
        "faiss_ntotal": index.ntotal,
        "mapping_rows": len(row_to_item),
        "mapping_unique_chunk_ids": len(chunk_ids),
        "output_queries": len(search_rows),
        "zero_candidate_queries": sum(not hits for hits in search_rows),
        "duplicate_documents": 0,
        "invalid_documents": 0,
        "repaired_documents_seen": len(
            {row_to_item[row][1] for hits in search_rows for row, _ in hits}
            & set(corpus_manifest.get("changed_document_ids", []))
        ),
        "complete": len(search_rows) == len(samples),
    }
    write_run_manifest(manifest, args.manifest)
    return [args.output, args.manifest]


def run(args: argparse.Namespace) -> list[Path]:
    question_source = (
        args.legal_ir_train
        or args.benchmark
        or args.public_official
        or args.private_official
    )
    if question_source is None:
        question_source = Path("data/processed_v3/benchmarks/synthetic_qa.jsonl")
    if args.legal_ir_train is not None:
        samples = [
            (sample.id, sample.question) for sample in load_warmup(question_source)
        ]
    elif args.public_official is not None or args.private_official is not None:
        official_path = args.public_official or args.private_official
        payload = json.loads(official_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("official question input must be an object")
        samples = []
        for question_id, record in payload.items():
            if not isinstance(record, dict) or not isinstance(record.get("question"), str):
                raise ValueError(f"invalid official question {question_id!r}")
            # Official answers are intentionally never accessed.
            samples.append((str(question_id), str(record["question"])))
    else:
        samples = [
            (sample.question_id, sample.question)
            for sample in load_benchmark(question_source)
        ]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("benchmark selection must not be empty")
    if any(value is not None for value in (args.pv1_index, args.pv1_mapping, args.pv1_chunks_dir)):
        return _run_pv1_mapping_backend(args, samples)
    completed_ids = _read_completed_question_ids(args.output)
    selected_ids = {question_id for question_id, _ in samples}
    unexpected_ids = completed_ids - selected_ids
    if unexpected_ids:
        raise ValueError(
            "existing output contains question IDs outside the selected input: "
            + ", ".join(sorted(unexpected_ids)[:5])
        )
    outputs = [args.output, args.manifest]
    if args.limit is not None:
        args.benchmark_subset.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_subset.write_text(
            "".join(
                json.dumps(
                    {"question_id": question_id, "question": question},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for question_id, question in samples
            ),
            encoding="utf-8",
        )
        outputs.append(args.benchmark_subset)

    started = time.perf_counter()
    pending = [sample for sample in samples if sample[0] not in completed_ids]
    if not pending:
        print(f"complete: {args.output} already contains {len(samples)} queries")
        config = load_config(args.config_env)
        embedding = dict(config.get("embedding", {}))
        model_path = str(embedding.get("model_path", "./models/dek21-v2"))
    else:
        config = load_config(args.config_env)
        embedding = dict(config.get("embedding", {}))
        model_path = str(embedding.get("model_path", "./models/dek21-v2"))
        embedder = EmbeddingClient(
            model_path=model_path,
            device=str(embedding.get("device", "cpu")),
            batch_size=int(embedding.get("batch_size", 32)),
            max_length=int(embedding.get("max_length", 256)),
            normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
            output_dimension=embedding.get("output_dimension"),
            window_long_texts=False,
        )
        vector_db = get_vector_db_adapter(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            progress = tqdm(
                total=len(samples),
                initial=len(completed_ids),
                unit="q",
                bar_format="{n_fmt}/{total_fmt} [{percentage:3.0f}%] | "
                "{rate_fmt} | elapsed {elapsed} | ETA {remaining}",
            )
            try:
                for offset in range(0, len(pending), args.query_batch_size):
                    batch = pending[offset : offset + args.query_batch_size]
                    batch_started = time.perf_counter()
                    vectors = embedder.embed_documents(
                        [question for _, question in batch],
                        batch_size=args.query_batch_size,
                    )
                    embedding_elapsed_ms = (time.perf_counter() - batch_started) * 1000
                    per_query_embedding_ms = embedding_elapsed_ms / len(batch)
                    batch_predictions: list[PredictionSample] = []
                    for (question_id, question), vector in zip(batch, vectors):
                        search_started = time.perf_counter()
                        hits = vector_db.search(vector, args.candidate_k)
                        search_elapsed_ms = (time.perf_counter() - search_started) * 1000
                        ranked_hits = [
                            hit.model_copy(
                                update={
                                    "rank": rank,
                                    "dense_score": (
                                        hit.dense_score
                                        if hit.dense_score is not None
                                        else hit.score
                                    ),
                                    "final_score": (
                                        hit.dense_score
                                        if hit.dense_score is not None
                                        else hit.score
                                    ),
                                }
                            )
                            for rank, hit in enumerate(hits, start=1)
                        ]
                        batch_predictions.append(
                            PredictionSample(
                                question_id=question_id,
                                hits=ranked_hits,
                                latency_ms=per_query_embedding_ms + search_elapsed_ms,
                            )
                        )
                    _write_prediction_batch(stream, batch_predictions)
                    stream.flush()
                    progress.update(len(batch_predictions))
                    del batch_predictions, ranked_hits, hits, vectors, batch
            finally:
                progress.close()
    vector_settings = resolve_vector_db_config(config)
    index_manifest = (
        Path(str(vector_settings["faiss_index_path"]))
        / str(vector_settings["collection_name"])
        / "manifest.json"
    )
    index_metadata: dict[str, Any] | None = None
    if index_manifest.is_file():
        index_metadata = json.loads(index_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_source": str(question_source),
        "question_source_sha256": _sha256(question_source),
        "question_source_type": (
            "legal_ir_train"
            if args.legal_ir_train is not None
            else (
                "public_official"
                if args.public_official is not None
                else (
                    "private_official"
                    if args.private_official is not None
                    else "benchmark"
                )
            )
        ),
        "question_count": len(samples),
        "candidate_k": args.candidate_k,
        "model_path": model_path,
        "model_id": embedding.get("model_id"),
        "output_dimension": embedding.get("output_dimension"),
        "index_manifest": index_metadata,
        "output": str(args.output),
        "elapsed_seconds": time.perf_counter() - started,
        "complete": len(completed_ids) + len(pending) == len(samples),
    }
    write_run_manifest(manifest, args.manifest)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"dense candidate error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
