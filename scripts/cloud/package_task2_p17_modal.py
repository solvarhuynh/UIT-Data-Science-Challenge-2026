"""Package the compact, reference-safe inputs for the P17 Modal selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P14_RESULT = PROJECT_ROOT / "artifacts/task2/modal_download/task2_p14_modal_result.zip"
P14_ZIP_RUN = "results/runs/16d81ab92bd57c46050dc876"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-bank",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task2/evaluation/p15_answer_selector_meteor_fast"
            / "candidate_bank.jsonl"
        ),
    )
    parser.add_argument(
        "--question-ids",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/training/parent_ce_eval_ids.json",
    )
    parser.add_argument("--p14-result", type=Path, default=P14_RESULT)
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/nltk_data",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task2/evaluation/p15_answer_selector_meteor_fast"
            / "raw_prefix_352_predictions.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/task2_p17_modal_input.zip",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(args: argparse.Namespace) -> tuple[Path, Path]:
    static_members = [
        (args.candidate_bank, "heldout/candidate_bank.jsonl"),
        (args.question_ids, "heldout/question_ids.json"),
        (args.baseline_predictions, "heldout/baseline_predictions.json"),
        (
            args.nltk_data / "corpora/wordnet.zip",
            "nltk_data/corpora/wordnet.zip",
        ),
        (
            args.nltk_data / "corpora/omw-1.4.zip",
            "nltk_data/corpora/omw-1.4.zip",
        ),
    ]
    missing = [str(path) for path, _ in static_members if not path.is_file()]
    if not args.p14_result.is_file():
        missing.append(str(args.p14_result))
    if missing:
        raise FileNotFoundError("P17 inputs are missing: " + ", ".join(missing))

    public_members = {
        f"{P14_ZIP_RUN}/public_qwen_predictions.json": "public/qwen.json",
        f"{P14_ZIP_RUN}/public_extractive_predictions.json": "public/extractive.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, archive_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    archive = Path(archive_name)
    member_meta: list[dict[str, Any]] = []
    try:
        with (
            zipfile.ZipFile(args.p14_result) as p14,
            zipfile.ZipFile(
                archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as output,
        ):
            missing_public = set(public_members) - set(p14.namelist())
            if missing_public:
                raise FileNotFoundError(
                    "P14 result lacks: " + ", ".join(sorted(missing_public))
                )
            for source, name in static_members:
                output.write(source, name)
                member_meta.append(
                    {
                        "archive_path": name,
                        "bytes": source.stat().st_size,
                        "sha256": _sha256(source),
                    }
                )
            for source_name, output_name in public_members.items():
                content = p14.read(source_name)
                output.writestr(output_name, content)
                member_meta.append(
                    {
                        "archive_path": output_name,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        with zipfile.ZipFile(archive) as output:
            if output.testzip() is not None:
                raise ValueError("P17 archive failed its integrity check")
            if set(output.namelist()) != {
                str(row["archive_path"]) for row in member_meta
            }:
                raise ValueError("P17 archive member contract changed")
        os.replace(archive, args.output)
    finally:
        archive.unlink(missing_ok=True)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "task2-p17-modal-input-v1",
        "archive": str(args.output),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": _sha256(args.output),
        "members": member_meta,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return args.output, manifest_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        output, manifest = package(args)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Task 2 P17 packaging error: {exc}")
        return 2
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
