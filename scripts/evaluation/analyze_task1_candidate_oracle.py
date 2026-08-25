"""Build the full 7,000-query Task1 raw K=200 versus K=500 oracle report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from scripts.evaluation.evaluate_task1_recovery import (
    _load_gold,
    _load_rankings,
    _oracle_for_file,
    _source_arguments,
    _write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--raw-k200", type=Path, required=True)
    parser.add_argument("--raw-k500", type=Path, required=True)
    parser.add_argument(
        "--recovery-source",
        action="append",
        default=[],
        help="Optional repeatable NAME=PATH for parent/BM25/KNN recovery checks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    gold, questions = _load_gold(args.gold)
    source_paths = _source_arguments(args.recovery_source)
    sources = {name: _load_rankings(path) for name, path in source_paths.items()}
    rows = []
    for path, depth in ((args.raw_k200, 200), (args.raw_k500, 500)):
        report, _ = _oracle_for_file(
            path,
            depth=depth,
            gold=gold,
            questions=questions,
            recovery_sources=sources,
        )
        rows.append(report)
    recall200 = float(rows[0]["macro_gold_coverage"])
    recall500 = float(rows[1]["macro_gold_coverage"])
    delta = recall500 - recall200
    report = {
        "schema_version": "task1-candidate-oracle-k200-k500-v1",
        "status": "complete",
        "query_count": len(gold),
        "diagnostic_only": True,
        "oracles": rows,
        "recommendation": {
            "coverage_delta": delta,
            "recommend_k500_reranking": delta >= 0.01 and recall200 < 0.98,
            "rule": "recommend only when delta>=0.01 and K=200 oracle<0.98",
            "note": (
                "Oracle coverage is a ceiling diagnostic, not OOF ranking performance."
            ),
        },
    }
    _write_json(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task1 oracle error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
