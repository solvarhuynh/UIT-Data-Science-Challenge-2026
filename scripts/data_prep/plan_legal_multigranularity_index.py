"""Plan a separate P4 LegalIR multi-granularity dense index.

This CPU-safe command validates canonical JSONL input and writes only an
artifact plan.  On Kaggle pass ``--execute`` to embed/index the selected
representation into its versioned collection; it never targets legacy
``legal_chunks``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from udsc2026.contracts import LegalChunk, LegalParent  # noqa: E402
from udsc2026.retrieval.multigranularity import (  # noqa: E402
    CHILD_META_V1,
    CHILD_RAW_V1,
    PARENT_META_V1,
    RepresentationVersion,
    build_child_retrieval_text,
    build_parent_retrieval_text,
    retrieval_corpus_hash,
    separate_collection_name,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=(CHILD_RAW_V1, CHILD_META_V1, PARENT_META_V1),
        required=True,
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed_v3")
    )
    parser.add_argument("--max-records", type=_positive, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task1/evaluation/p4_multigranularity"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for Kaggle full embedding/index execution.",
    )
    return parser


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _records(
    root: Path, version: RepresentationVersion, limit: int | None
) -> Iterator[tuple[str, str, str]]:
    directory = root / ("parents" if version == PARENT_META_V1 else "chunks")
    model = LegalParent if version == PARENT_META_V1 else LegalChunk
    yielded = 0
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = model.model_validate_json(line)
            text = (
                build_parent_retrieval_text(record)
                if isinstance(record, LegalParent)
                else build_child_retrieval_text(record, version)
            )
            record_id = (
                record.parent_id if isinstance(record, LegalParent) else record.chunk_id
            )
            yield record_id, record.doc_id, text
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def run(args: argparse.Namespace) -> Path:
    version = args.representation
    records = list(_records(args.processed_root, version, args.max_records))
    if not records:
        raise ValueError("selected canonical corpus contains no records")
    if args.execute:
        raise RuntimeError(
            "P4 execution is Kaggle-only. Invoke the approved embedding/index job "
            "with this plan's versioned collection name; local execution is disabled."
        )
    output = args.output_dir / f"{version}_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "legal-ir-p4-index-plan-v1",
                "representation_version": version,
                "collection_name": separate_collection_name(version),
                "processed_root": str(args.processed_root),
                "record_count": len(records),
                "corpus_hash": retrieval_corpus_hash(
                    (record_id, text) for record_id, _, text in records
                ),
                "git_commit": _git_commit(),
                "max_records": args.max_records,
                "execute": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
