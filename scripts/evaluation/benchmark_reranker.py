"""Run TV5 cross-encoder reranking and write reproducible evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    compare_reports,
    evaluate_predictions,
    load_benchmark,
    load_predictions,
    rerank_prediction_samples,
    validate_rerank_candidate_pools,
    write_predictions,
    write_report_bundle,
    write_run_manifest,
)
from udsc2026.infrastructure.reranker import (  # noqa: E402
    CrossEncoderClient,
    RerankerClient,
)
from udsc2026.retrieval.reranking import CrossEncoderReranker  # noqa: E402

ClientFactory = Callable[..., RerankerClient]


def _positive_int(value: str) -> int:
    """Parse one strictly positive integer for argparse."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the TV5 batch-reranking CLI without loading model weights."""

    parser = argparse.ArgumentParser(
        description=(
            "Rerank TV2 candidate predictions with a cross-encoder, then write "
            "before/after metrics and a reproducibility manifest."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="TV4-compatible benchmark .json or .jsonl file.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="TV2 baseline predictions .json or .jsonl file.",
    )
    parser.add_argument(
        "--model",
        default="models/reranker",
        help="Local model directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Model device such as cuda:0 or cpu (default: automatic).",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument(
        "--candidate-k",
        type=_positive_int,
        default=50,
        help="Maximum number of TV2 candidates scored per query.",
    )
    parser.add_argument(
        "--top-n",
        type=_positive_int,
        default=5,
        help="Number of reranked hits retained per query.",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        nargs="+",
        default=[1, 3, 5],
        metavar="K",
        help="Unique Recall@K cutoffs (default: 1 3 5).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load model weights as FP16; use only with a CUDA device.",
    )
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow sentence-transformers to download a missing Hub model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tv5/bge-reranker-v2-m3"),
    )
    parser.add_argument("--before-name", default="before_rerank")
    parser.add_argument("--after-name", default="after_rerank")
    return parser


def _sha256(path: Path) -> str:
    """Return the SHA-256 of one input file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(directory: Path) -> dict[str, Path]:
    """Return the complete stable artifact layout for one run."""

    evaluation = directory / "evaluation"
    return {
        "predictions_before": directory / "predictions_before.json",
        "predictions_after": directory / "predictions_after.jsonl",
        "manifest": directory / "run_manifest.json",
        "before_json": evaluation / "before.json",
        "before_md": evaluation / "before.md",
        "after_json": evaluation / "after.json",
        "after_md": evaluation / "after.md",
        "comparison_json": evaluation / "comparison.json",
        "comparison_md": evaluation / "comparison.md",
    }


def _reject_input_output_collisions(
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
) -> None:
    """Prevent a rerank run from overwriting either input artifact."""

    inputs = {path.resolve(): path for path in input_paths}
    for output in output_paths:
        source = inputs.get(output.resolve())
        if source is not None:
            raise ValueError(
                "output must not overwrite input file: "
                f"{output} conflicts with {source}"
            )


def run(
    args: argparse.Namespace,
    *,
    client_factory: ClientFactory = CrossEncoderClient,
) -> list[Path]:
    """Execute one complete reranking experiment and return written artifacts."""

    if len(args.k) != len(set(args.k)):
        raise ValueError("--k values must be unique")
    if args.top_n > args.candidate_k:
        raise ValueError("--top-n must not exceed --candidate-k")
    if (
        args.fp16
        and args.device is not None
        and not args.device.strip().casefold().startswith("cuda")
    ):
        raise ValueError("--fp16 requires --device cuda or automatic selection")

    outputs = _output_paths(args.output_dir)
    _reject_input_output_collisions(
        [args.benchmark, args.candidates],
        list(outputs.values()),
    )
    benchmark_hash = _sha256(args.benchmark)
    candidates_hash = _sha256(args.candidates)
    benchmark = load_benchmark(args.benchmark)
    predictions = load_predictions(args.candidates)

    client = client_factory(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=not args.allow_remote_model,
        use_fp16=args.fp16,
    )
    reranker = CrossEncoderReranker(client)
    batch = rerank_prediction_samples(
        benchmark,
        predictions,
        reranker,
        candidate_k=args.candidate_k,
        top_n=args.top_n,
    )
    validate_rerank_candidate_pools(batch.before, batch.after)

    before = evaluate_predictions(
        benchmark,
        batch.before,
        args.k,
        name=args.before_name,
    )
    after = evaluate_predictions(
        benchmark,
        batch.after,
        args.k,
        name=args.after_name,
    )
    comparison = compare_reports(before, after)

    written = list(outputs.values())
    write_predictions(batch.before, outputs["predictions_before"])
    write_predictions(batch.after, outputs["predictions_after"])
    write_report_bundle(
        [
            (before, outputs["before_json"], outputs["before_md"]),
            (after, outputs["after_json"], outputs["after_md"]),
            (
                comparison,
                outputs["comparison_json"],
                outputs["comparison_md"],
            ),
        ]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "benchmark": str(args.benchmark),
            "benchmark_sha256": benchmark_hash,
            "candidates": str(args.candidates),
            "candidates_sha256": candidates_hash,
            "dataset_fingerprint": before.dataset_fingerprint,
        },
        "model": {
            "name_or_path": args.model,
            "device": args.device or "auto",
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "local_files_only": not args.allow_remote_model,
            "use_fp16": args.fp16,
        },
        "evaluation": {
            "sample_count": len(batch.after),
            "candidate_pair_count": batch.candidate_pair_count,
            "candidate_k": args.candidate_k,
            "top_n": args.top_n,
            "k_values": sorted(args.k),
            "total_rerank_ms": batch.total_rerank_ms,
            "mean_rerank_ms": batch.total_rerank_ms / len(batch.after),
            "delta": comparison.delta.model_dump(mode="json"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_run_manifest(manifest, outputs["manifest"])
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning 2 for invalid inputs or runtime failures."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        written = run(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"reranker error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
