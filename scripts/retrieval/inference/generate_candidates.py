"""Generate dense retrieval candidates from an ID-to-question mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    load_benchmark,
    load_warmup,
    write_run_manifest,
)
from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.infrastructure.config import load_config  # noqa: E402
from udsc2026.infrastructure.modal_volume import (  # noqa: E402
    commit_mounted_volume_sync,
)
from udsc2026.infrastructure.embedding.client import (  # noqa: E402
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
        help="A prepared benchmark question source for retrieval evaluation.",
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
    question_group.add_argument(
        "--question-mapping",
        type=Path,
        help=(
            "Organizer ID-to-question JSON mapping for another task, such as "
            "LegalQA; answer/reference fields are ignored."
        ),
    )
    parser.add_argument("--config-env", default="gpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the existing output manifest matches this run.",
    )
    parser.add_argument("--candidate-k", type=_positive_int, default=50)
    parser.add_argument("--query-batch-size", type=_positive_int, default=64)
    parser.add_argument("--save-every", type=_positive_int, default=500)
    parser.add_argument("--commit-modal-volume", action="store_true")
    parser.add_argument("--modal-volume-name", default="udsc")
    parser.add_argument(
        "--max-questions",
        "--limit",
        dest="limit",
        type=_positive_int,
        help="Bound the number of selected questions for a partial run.",
    )
    parser.add_argument(
        "--benchmark-subset",
        type=Path,
        default=Path("data/predictions/retrieval/subsets/questions.jsonl"),
        help="Filtered benchmark written when --limit is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/predictions/retrieval/dense_predictions.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/predictions/retrieval/dense_run_manifest.json"),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(environment: str) -> str:
    config_dir = Path(os.environ.get("UDSC2026_CONFIG_DIR", "configs"))
    paths = [config_dir / "base.yaml"]
    if environment != "base":
        paths.append(config_dir / f"{environment}.yaml")
    return _combined_file_sha256(paths)


def _combined_file_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
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


def _load_index_identity(
    config: dict[str, Any], embedding: dict[str, Any], config_env: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a complete FAISS index built from the selected embedding/corpus."""

    vector_settings = resolve_vector_db_config(config)
    if vector_settings.get("type") != "faiss":
        raise ValueError("the production retrieval path requires a FAISS index")
    manifest_path = (
        Path(str(vector_settings["faiss_index_path"]))
        / str(vector_settings["collection_name"])
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"verified FAISS index manifest is missing: {manifest_path}")
    try:
        index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FAISS index manifest: {manifest_path}") from exc
    if not isinstance(index_manifest, dict):
        raise ValueError("FAISS index manifest must contain a JSON object")
    index_directory = manifest_path.parent
    index_path = index_directory / "index.faiss"
    payload_path = index_directory / "payloads.json"
    if index_manifest.get("complete") is not True:
        raise ValueError("FAISS index manifest does not mark a completed build")
    if (
        not index_path.is_file()
        or index_manifest.get("index_sha256") != _sha256(index_path)
    ):
        raise ValueError("FAISS index file does not match its verified manifest")
    if (
        not payload_path.is_file()
        or index_manifest.get("payload_sha256") != _sha256(payload_path)
    ):
        raise ValueError("FAISS payload file does not match its verified manifest")

    processing_manifest_path = Path("data/processed/metadata/processing_manifest.json")
    try:
        processing_manifest = json.loads(
            processing_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid processed-corpus manifest: {processing_manifest_path}"
        ) from exc
    counts = processing_manifest.get("counts")
    expected_chunks = counts.get("chunks") if isinstance(counts, dict) else None
    expected_model = embedding.get("model_id")
    expected_dimension = embedding.get("output_dimension")
    checks = {
        "model_id": expected_model,
        "chunk_count": expected_chunks,
        "embedding_dimension": expected_dimension,
        "collection_name": vector_settings.get("collection_name"),
        "preprocessor": embedding.get("preprocessor", "pyvi"),
        "query_max_length": embedding.get(
            "query_max_length", embedding.get("max_length", 256)
        ),
        "document_max_length": embedding.get(
            "document_max_length", embedding.get("max_length", 256)
        ),
        "embedding_implementation_sha256": _sha256(
            PROJECT_ROOT / "src/udsc2026/infrastructure/embedding/client.py"
        ),
        "config_sha256": _config_sha256(config_env),
        "indexing_implementation_sha256": _combined_file_sha256(
            [
                PROJECT_ROOT / "scripts/data_prep/index_chunks.py",
                PROJECT_ROOT
                / "src/udsc2026/infrastructure/vector_db/faiss_adapter.py",
            ]
        ),
    }
    mismatches = {
        name: {"expected": expected, "actual": index_manifest.get(name)}
        for name, expected in checks.items()
        if expected is None or index_manifest.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"FAISS index is not current for this corpus/model: {mismatches}")
    if processing_manifest.get("integrity_gate_passed") is not True:
        raise ValueError("processed corpus did not pass its integrity gate")
    archive_path = Path("data/processed/chunks_aiteam.tar.gz")
    if (
        index_manifest.get("source_archive") != archive_path.as_posix()
        or index_manifest.get("source_archive_sha256") != _sha256(archive_path)
    ):
        raise ValueError("FAISS index does not match the uploaded chunk archive")
    return vector_settings, index_manifest


def _write_prediction_batch(stream: Any, predictions: list[PredictionSample]) -> None:
    """Write one batch of prediction rows as JSONL."""

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


def run(args: argparse.Namespace) -> list[Path]:
    question_source = (
        args.legal_ir_train
        or args.benchmark
        or args.public_official
        or args.private_official
        or args.question_mapping
    )
    if question_source is None:
        raise ValueError(
            "provide --legal-ir-train, --public-official, --private-official, "
            "--question-mapping, or an explicit --benchmark question source"
        )
    if args.legal_ir_train is not None:
        samples = [
            (sample.id, sample.question) for sample in load_warmup(question_source)
        ]
    elif any(
        value is not None
        for value in (
            args.public_official,
            args.private_official,
            args.question_mapping,
        )
    ):
        questions_path = (
            args.public_official
            or args.private_official
            or args.question_mapping
        )
        payload = json.loads(questions_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("question mapping must be an object")
        samples = []
        for question_id, record in payload.items():
            if not isinstance(record, dict) or not isinstance(record.get("question"), str):
                raise ValueError(f"invalid official question {question_id!r}")
            # Only the question field is read; answer/reference fields are ignored.
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

    config = load_config(args.config_env)
    embedding = dict(config.get("embedding", {}))
    vector_settings, index_metadata = _load_index_identity(
        config, embedding, args.config_env
    )
    model_path = str(
        embedding.get("model_path", "./models/aiteam-vietnamese-embedding-v2")
    )
    question_source_type = (
        "legal_ir_train"
        if args.legal_ir_train is not None
        else (
            "public_official"
            if args.public_official is not None
            else (
                "private_official"
                if args.private_official is not None
                else "question_mapping" if args.question_mapping is not None else "benchmark"
            )
        )
    )
    run_identity: dict[str, Any] = {
        "question_source_sha256": _sha256(question_source),
        "question_source_type": question_source_type,
        "question_count": len(samples),
        "candidate_k": args.candidate_k,
        "query_batch_size": args.query_batch_size,
        "checkpoint_every": args.save_every,
        "config_sha256": _config_sha256(args.config_env),
        "model_id": embedding.get("model_id"),
        "output_dimension": embedding.get("output_dimension"),
        "query_max_length": embedding.get(
            "query_max_length", embedding.get("max_length", 256)
        ),
        "document_max_length": embedding.get(
            "document_max_length", embedding.get("max_length", 256)
        ),
        "preprocessor": embedding.get("preprocessor", "pyvi"),
        "embedding_implementation_sha256": _sha256(
            PROJECT_ROOT / "src/udsc2026/infrastructure/embedding/client.py"
        ),
        "collection_name": vector_settings["collection_name"],
        "index_model_hash": index_metadata.get("model_hash"),
        "index_corpus_hash": index_metadata.get("corpus_hash"),
        "index_chunk_count": index_metadata.get("chunk_count"),
    }
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("prediction output and run manifest must be different files")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume only for a matching run: {args.output}")
    if args.manifest.exists() and not args.resume:
        raise FileExistsError(f"manifest exists; use a new output path: {args.manifest}")

    completed_ids = _read_completed_question_ids(args.output)
    last_committed_count = len(completed_ids)
    if args.resume and args.output.exists():
        if not args.manifest.is_file():
            raise ValueError("cannot resume without the original run manifest")
        try:
            previous_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot resume from an invalid run manifest") from exc
        if not isinstance(previous_manifest, dict):
            raise ValueError("run manifest must contain a JSON object")
        mismatches = {
            key: {"expected": value, "actual": previous_manifest.get(key)}
            for key, value in run_identity.items()
            if previous_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"run inputs changed; refusing to resume: {mismatches}")
        if previous_manifest.get("completed_count") != len(completed_ids):
            raise ValueError("prediction rows do not match the saved run checkpoint")
    elif args.manifest.exists():
        raise FileExistsError(f"manifest exists without a resumable output: {args.manifest}")

    selected_ids = {question_id for question_id, _ in samples}
    unexpected_ids = completed_ids - selected_ids
    if unexpected_ids:
        raise ValueError(
            "existing output contains question IDs outside the selected input: "
            + ", ".join(sorted(unexpected_ids)[:5])
        )
    if (
        args.resume
        and args.output.is_file()
        and previous_manifest.get("complete") is True
    ):
        if previous_manifest.get("output_sha256") != _sha256(args.output):
            raise ValueError("completed dense output does not match its manifest")
        print(f"Dense candidates are already complete: {args.output}", flush=True)
        return [args.output, args.manifest]
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
    if pending:
        embedder = EmbeddingClient(
            model_path=model_path,
            device=str(embedding.get("device", "cpu")),
            batch_size=int(embedding.get("batch_size", 32)),
            max_length=int(embedding.get("max_length", 256)),
            query_max_length=int(
                embedding.get(
                    "query_max_length", embedding.get("max_length", 256)
                )
            ),
            document_max_length=int(
                embedding.get(
                    "document_max_length", embedding.get("max_length", 256)
                )
            ),
            normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
            output_dimension=embedding.get("output_dimension"),
            preprocessor=str(embedding.get("preprocessor", "pyvi")),
            window_long_texts=False,
        )
        vector_db = get_vector_db_adapter(config)
        index = getattr(vector_db, "index", None)
        if index is None or index.ntotal != int(index_metadata["chunk_count"]):
            raise ValueError("loaded FAISS vector count does not match its verified manifest")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_run_manifest(
            {
                "schema_version": 2,
                **run_identity,
                "output": str(args.output),
                "completed_count": len(completed_ids),
                "complete": False,
            },
            args.manifest,
        )
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
                    vectors = embedder.embed_queries(
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
                    os.fsync(stream.fileno())
                    completed_ids.update(
                        prediction.question_id for prediction in batch_predictions
                    )
                    write_run_manifest(
                        {
                            "schema_version": 2,
                            **run_identity,
                            "output": str(args.output),
                            "completed_count": len(completed_ids),
                            "complete": False,
                        },
                        args.manifest,
                    )
                    if (
                        args.commit_modal_volume
                        and len(completed_ids) - last_committed_count >= args.save_every
                    ):
                        commit_mounted_volume_sync(args.modal_volume_name)
                        last_committed_count = len(completed_ids)
                    progress.update(len(batch_predictions))
                    del batch_predictions, ranked_hits, hits, vectors, batch
            finally:
                progress.close()
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_source": str(question_source),
        **run_identity,
        "model_path": model_path,
        "completed_count": len(completed_ids),
        "complete": len(completed_ids) == len(samples),
        "output_sha256": _sha256(args.output),
        "output": str(args.output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_run_manifest(manifest, args.manifest)
    if args.commit_modal_volume:
        commit_mounted_volume_sync(args.modal_volume_name)
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

