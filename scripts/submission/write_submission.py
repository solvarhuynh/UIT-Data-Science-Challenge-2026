"""Create a schema-configured CSV from shared QAResponse JSON/JSONL records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    SubmissionSchema,
    load_qa_responses,
    write_submission,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the submission-writer CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Write QAResponse objects to CSV. The default is the generic "
            "qa_response_v1 layout, not an assumed CodaLab schema."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="QAResponse .json or .jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("SUBMISSION_PATH", "artifacts/submission.csv")),
        help=(
            "Destination .csv file (default: SUBMISSION_PATH or "
            "artifacts/submission.csv)."
        ),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Optional strict SubmissionSchema JSON file.",
    )
    return parser


def _load_schema(path: Path | None) -> SubmissionSchema:
    """Load a strict schema config or return the documented generic layout."""

    if path is None:
        return SubmissionSchema()
    if not path.is_file():
        raise FileNotFoundError(f"schema file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid schema JSON in {path}: {exc.msg}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("columns"), list):
        # JSON arrays are the wire representation of the immutable tuple used
        # internally; normalize only that structural difference before strict
        # scalar/nested validation.
        payload = {**payload, "columns": tuple(payload["columns"])}
    try:
        return SubmissionSchema.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(f"invalid submission schema in {path}: {exc}") from exc


def _reject_input_output_collisions(
    input_path: Path,
    schema_path: Path | None,
    output_path: Path,
) -> None:
    """Prevent a symlinked CSV target from replacing an input or schema file."""

    protected_paths = [input_path]
    if schema_path is not None:
        protected_paths.append(schema_path)
    output_resolved = output_path.resolve()
    for protected_path in protected_paths:
        if protected_path.resolve() == output_resolved:
            raise ValueError(
                "submission output must not overwrite an input file: "
                f"{output_path} conflicts with {protected_path}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: return 0 on success and 2 for invalid input/runtime I/O."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _reject_input_output_collisions(args.input, args.schema, args.output)
        results = load_qa_responses(args.input)
        schema = _load_schema(args.schema)
        write_submission(results, args.output, schema)
    except (OSError, TypeError, ValueError) as exc:
        print(f"submission error: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
