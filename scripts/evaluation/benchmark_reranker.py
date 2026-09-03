"""Run TV5 cross-encoder reranking and write reproducible evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    LegalIRReference,
    compare_legal_ir_document_diagnostics,
    compare_reports,
    evaluate_legal_ir_document_diagnostics,
    evaluate_predictions,
    load_benchmark,
    load_predictions,
    load_warmup,
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
        default="models/qwen3-vl-reranker-2b",
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
        default=Path("artifacts/tv5/qwen3-vl-reranker-2b"),
    )
    parser.add_argument("--before-name", default="before_rerank")
    parser.add_argument("--after-name", default="after_rerank")
    parser.add_argument(
        "--legal-ir-references",
        type=Path,
        help=(
            "Optional organizer LegalIR mapping (for example train.json) used "
            "only for document-level OOF diagnostics. It must have exactly the "
            "same question IDs as --benchmark."
        ),
    )
    parser.add_argument(
        "--legal-ir-final-documents",
        type=_positive_int,
        default=5,
        help="Unique documents retained for official LegalIR scoring (1..5).",
    )
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
        "label_status_json": evaluation / "label_status.json",
        "legal_ir_before_json": evaluation / "legal_ir_before.json",
        "legal_ir_after_json": evaluation / "legal_ir_after.json",
        "legal_ir_comparison_json": evaluation / "legal_ir_comparison.json",
    }


def _benchmark_label_status(benchmark: Sequence[Any]) -> str:
    """Classify whether a transport benchmark has real retrieval labels."""

    unlabeled = [
        bool(getattr(sample, "metadata", {}).get("unlabeled_public_question"))
        for sample in benchmark
    ]
    if any(unlabeled) and not all(unlabeled):
        raise ValueError(
            "benchmark mixes unlabeled public transport rows with labeled rows"
        )
    return "unlabeled_transport" if unlabeled and all(unlabeled) else "labeled"


def _load_legal_ir_references(path: Path) -> list[LegalIRReference]:
    """Load organizer document labels without treating chunk labels as documents."""

    return [
        LegalIRReference(id=sample.id, gold_documents=list(sample.gold_documents))
        for sample in load_warmup(path)
    ]


def _write_json(path: Path, payload: Any) -> None:
    """Write one deterministic UTF-8 JSON diagnostic artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
    if args.legal_ir_final_documents > 5:
        raise ValueError("--legal-ir-final-documents must not exceed 5")
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
    label_status = _benchmark_label_status(benchmark)
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

    before = after = comparison = None
    if label_status == "labeled":
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

    legal_ir_before = legal_ir_after = legal_ir_comparison = None
    if args.legal_ir_references is not None:
        references = _load_legal_ir_references(args.legal_ir_references)
        legal_ir_before = evaluate_legal_ir_document_diagnostics(
            references,
            batch.before,
            candidate_predictions=batch.before,
            final_document_limit=args.legal_ir_final_documents,
        )
        legal_ir_after = evaluate_legal_ir_document_diagnostics(
            references,
            batch.after,
            candidate_predictions=batch.before,
            final_document_limit=args.legal_ir_final_documents,
        )
        legal_ir_comparison = compare_legal_ir_document_diagnostics(
            legal_ir_before,
            legal_ir_after,
        )

    written = [outputs["predictions_before"], outputs["predictions_after"]]
    write_predictions(batch.before, outputs["predictions_before"])
    write_predictions(batch.after, outputs["predictions_after"])
    label_status_payload = {
        "schema_version": "reranker-label-status-v1",
        "status": label_status,
        "generic_chunk_metrics": "available" if before is not None else "skipped",
        "reason": (
            None
            if before is not None
            else (
                "benchmark rows are an explicitly unlabeled public transport "
                "artifact; placeholder chunk IDs are not evaluation labels"
            )
        ),
        "legal_ir_document_metrics": (
            "available" if legal_ir_before is not None else "not_requested"
        ),
    }
    _write_json(outputs["label_status_json"], label_status_payload)
    written.append(outputs["label_status_json"])
    if before is not None and after is not None and comparison is not None:
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
        written.extend(
            [
                outputs["before_json"],
                outputs["before_md"],
                outputs["after_json"],
                outputs["after_md"],
                outputs["comparison_json"],
                outputs["comparison_md"],
            ]
        )
    if (
        legal_ir_before is not None
        and legal_ir_after is not None
        and legal_ir_comparison is not None
    ):
        _write_json(outputs["legal_ir_before_json"], legal_ir_before)
        _write_json(outputs["legal_ir_after_json"], legal_ir_after)
        _write_json(outputs["legal_ir_comparison_json"], legal_ir_comparison)
        written.extend(
            [
                outputs["legal_ir_before_json"],
                outputs["legal_ir_after_json"],
                outputs["legal_ir_comparison_json"],
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
            "dataset_fingerprint": (
                before.dataset_fingerprint if before is not None else None
            ),
            "legal_ir_references": (
                str(args.legal_ir_references)
                if args.legal_ir_references is not None
                else None
            ),
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
            "label_status": label_status,
            "delta": (
                comparison.delta.model_dump(mode="json")
                if comparison is not None
                else None
            ),
            "legal_ir": (
                {
                    "before": {
                        "official_recall": legal_ir_before["official_recall"],
                        "official_precision": legal_ir_before["official_precision"],
                    },
                    "after": {
                        "official_recall": legal_ir_after["official_recall"],
                        "official_precision": legal_ir_after["official_precision"],
                    },
                    "delta": legal_ir_comparison["delta"],
                }
                if legal_ir_before is not None
                and legal_ir_after is not None
                and legal_ir_comparison is not None
                else None
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_run_manifest(manifest, outputs["manifest"])
    written.append(outputs["manifest"])
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
