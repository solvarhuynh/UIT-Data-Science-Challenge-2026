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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    load_benchmark,
    load_warmup,
    write_predictions,
    write_run_manifest,
)
from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.infrastructure.config import load_config  # noqa: E402
from udsc2026.infrastructure.embedding.bkai_client import (  # noqa: E402
    EmbeddingClient,
)
from udsc2026.infrastructure.vector_db.factory import (  # noqa: E402
    get_vector_db_adapter,
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
        default=Path("artifacts/tv2/smoke_benchmark.jsonl"),
        help="Filtered benchmark written when --limit is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tv2/dense_predictions.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/tv2/dense_run_manifest.json"),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> list[Path]:
    question_source = args.legal_ir_train or args.benchmark
    if question_source is None:
        question_source = Path("data/processed_v3/benchmarks/synthetic_qa.jsonl")
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
    if args.legal_ir_train is not None:
        samples = [
            (sample.id, sample.question) for sample in load_warmup(question_source)
        ]
    else:
        samples = [
            (sample.question_id, sample.question)
            for sample in load_benchmark(question_source)
        ]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("benchmark selection must not be empty")
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

    predictions: list[PredictionSample] = []
    started = time.perf_counter()
    for offset in range(0, len(samples), args.query_batch_size):
        batch = samples[offset : offset + args.query_batch_size]
        batch_started = time.perf_counter()
        vectors = embedder.embed_documents(
            [question for _, question in batch],
            batch_size=args.query_batch_size,
        )
        embedding_elapsed_ms = (time.perf_counter() - batch_started) * 1000
        per_query_embedding_ms = embedding_elapsed_ms / len(batch)
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
            predictions.append(
                PredictionSample(
                    question_id=question_id,
                    hits=ranked_hits,
                    latency_ms=per_query_embedding_ms + search_elapsed_ms,
                )
            )

    write_predictions(predictions, args.output)
    vector_settings = dict(config.get("vector_db", {}))
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
            "legal_ir_train" if args.legal_ir_train is not None else "benchmark"
        ),
        "question_count": len(predictions),
        "candidate_k": args.candidate_k,
        "model_path": model_path,
        "model_id": embedding.get("model_id"),
        "output_dimension": embedding.get("output_dimension"),
        "index_manifest": index_metadata,
        "output": str(args.output),
        "elapsed_seconds": time.perf_counter() - started,
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
