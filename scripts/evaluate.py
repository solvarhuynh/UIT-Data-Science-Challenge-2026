"""Evaluate file-based retrieval/QA predictions without loading any ML model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    compare_reports,
    evaluate_predictions,
    load_benchmark,
    load_predictions,
    validate_rerank_candidate_pools,
    write_report_bundle,
)


def _positive_int(value: str) -> int:
    """Argparse converter for retrieval cutoffs."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without performing side effects."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JSON/JSONL predictions against a TV4-compatible benchmark. "
            "Supplying --after also creates a before/after comparison."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Benchmark .json or .jsonl file.",
    )
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Baseline prediction .json or .jsonl file.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        help="Optional post-rerank prediction .json or .jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Report directory (default: artifacts/evaluation).",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        nargs="+",
        default=[1, 3, 5],
        metavar="K",
        help="Unique positive Recall@K cutoffs (default: 1 3 5).",
    )
    parser.add_argument(
        "--before-name",
        default="before_rerank",
        help="Human-readable baseline label.",
    )
    parser.add_argument(
        "--after-name",
        default="after_rerank",
        help="Human-readable post-rerank label.",
    )
    return parser


def _output_paths(directory: Path, *, comparison: bool) -> list[Path]:
    """Return the complete ordered output set for the selected CLI mode."""

    stems = ("before", "after", "comparison") if comparison else ("report",)
    return [
        directory / f"{stem}.{suffix}" for stem in stems for suffix in ("json", "md")
    ]


def _reject_input_output_collisions(
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
) -> None:
    """Prevent report generation from replacing a benchmark or prediction file."""

    inputs_by_resolved_path = {path.resolve(): path for path in input_paths}
    for output_path in output_paths:
        source_path = inputs_by_resolved_path.get(output_path.resolve())
        if source_path is not None:
            raise ValueError(
                "report output must not overwrite an input file: "
                f"{output_path} conflicts with {source_path}"
            )


def run(args: argparse.Namespace) -> list[Path]:
    """Execute evaluation for already parsed arguments and return output files."""

    if len(args.k) != len(set(args.k)):
        raise ValueError("--k values must be unique")
    written = _output_paths(args.output_dir, comparison=args.after is not None)
    input_paths = [args.benchmark, args.before]
    if args.after is not None:
        input_paths.append(args.after)
    _reject_input_output_collisions(input_paths, written)

    benchmark = load_benchmark(args.benchmark)
    baseline_predictions = load_predictions(args.before)
    before = evaluate_predictions(
        benchmark,
        baseline_predictions,
        args.k,
        name=args.before_name,
    )

    if args.after is None:
        write_report_bundle([(before, written[0], written[1])])
    else:
        after_predictions = load_predictions(args.after)
        validate_rerank_candidate_pools(
            baseline_predictions,
            after_predictions,
        )
        after = evaluate_predictions(
            benchmark,
            after_predictions,
            args.k,
            name=args.after_name,
        )
        comparison = compare_reports(before, after)
        write_report_bundle(
            [
                (before, written[0], written[1]),
                (after, written[2], written[3]),
                (comparison, written[4], written[5]),
            ]
        )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: return 0 on success and 2 for invalid input/runtime I/O."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        written = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
