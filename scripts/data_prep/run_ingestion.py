"""Run the full TV4 ingestion pipeline with safe local-machine defaults."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.ingestion import run_ingestion_pipeline  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw/btc"))
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed_candidate")
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=192)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--progress-every", type=_positive_int, default=100)
    parser.add_argument("--benchmark-count", type=_positive_int, default=100)
    parser.add_argument("--benchmark-seed", type=int, default=2026)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Do not generate the post-ingestion synthetic benchmark.",
    )
    parser.add_argument(
        "--allow-missing-benchmark-types",
        action="store_true",
        help="Allow small exploratory corpora that cannot cover all question types.",
    )
    return parser


def _ensure_new_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. Choose a new path so existing "
            "processed data is never mixed or overwritten."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        print(
            "ingestion error: chunk-overlap must be non-negative and less than "
            "chunk-size",
            file=sys.stderr,
        )
        return 2
    if not 100 <= args.benchmark_count <= 200:
        print(
            "ingestion error: benchmark-count must be between 100 and 200",
            file=sys.stderr,
        )
        return 2
    if not args.raw_directory.is_dir():
        print(
            f"ingestion error: raw directory not found: {args.raw_directory}",
            file=sys.stderr,
        )
        return 2
    try:
        _ensure_new_output_root(args.processed_root)
    except OSError as exc:
        print(f"ingestion error: {exc}", file=sys.stderr)
        return 2

    started_at = time.monotonic()

    def report_progress(count: int, doc_id: str) -> None:
        if count == 1 or count % args.progress_every == 0:
            elapsed_minutes = (time.monotonic() - started_at) / 60
            print(
                f"chunked={count} latest_doc={doc_id} "
                f"elapsed_min={elapsed_minutes:.1f}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "raw_directory": str(args.raw_directory),
                "processed_root": str(args.processed_root),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "streaming": True,
                "synthetic_benchmark_count": (
                    None if args.skip_benchmark else args.benchmark_count
                ),
                "synthetic_benchmark_seed": args.benchmark_seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    try:
        result = run_ingestion_pipeline(
            raw_directory=args.raw_directory,
            processed_root=args.processed_root,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            progress_callback=report_progress,
            synthetic_benchmark_count=(
                None if args.skip_benchmark else args.benchmark_count
            ),
            synthetic_benchmark_seed=args.benchmark_seed,
            synthetic_benchmark_require_all_question_types=(
                not args.allow_missing_benchmark_types
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ingestion error: {exc}", file=sys.stderr)
        return 1

    payload = result.model_dump(mode="json")
    payload["elapsed_minutes"] = round((time.monotonic() - started_at) / 60, 2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not result.integrity_gate_passed:
        print("ingestion integrity gate failed; inspect disk audit", file=sys.stderr)
        return 1
    if not result.semantic_completeness_gate_passed:
        print(
            "ingestion completed with upstream semantic gaps; inspect disk audit",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
