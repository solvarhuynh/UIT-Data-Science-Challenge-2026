"""Package compact P16 ranking/evaluation inputs for the Modal RAG sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P14_RUN = (
    PROJECT_ROOT
    / "artifacts/task2/modal_download/task2_p14_modal_result/results/runs"
    / "16d81ab92bd57c46050dc876"
)
P14_RESULT = PROJECT_ROOT / "artifacts/task2/modal_download/task2_p14_modal_result.zip"
P14_ZIP_RUN = "results/runs/16d81ab92bd57c46050dc876"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout-rankings",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task2/evaluation/parent_ce_hn16_v2_strict1000"
            / "ranked_parents.jsonl"
        ),
    )
    parser.add_argument("--p14-result", type=Path, default=P14_RESULT)
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task2/training/p15_adaptation/adaptation_dev_ids.json"
        ),
    )
    parser.add_argument(
        "--heldout-ids",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/training/parent_ce_eval_ids.json",
    )
    parser.add_argument("--rank-limit", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/task2_p16_modal_input.zip",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_row(
    row: dict[str, Any], *, rank_limit: int, candidate_limit: int
) -> dict[str, Any]:
    question_id = str(row.get("question_id", "")).strip()
    parents = row.get("parents")
    if not question_id or not isinstance(parents, list) or not parents:
        raise ValueError(f"invalid ranking row {question_id!r}")
    selected = [
        parent
        for parent in parents
        if isinstance(parent, dict)
        and (
            int(parent.get("rank", 10**9)) <= rank_limit
            or int(parent.get("candidate_rank", 10**9)) <= candidate_limit
        )
    ]
    if not selected:
        raise ValueError(f"compaction removed every parent for {question_id!r}")
    selected.sort(
        key=lambda parent: (
            int(parent.get("rank", 10**9)),
            int(parent.get("candidate_rank", 10**9)),
            str(parent.get("doc_id", "")),
            str(parent.get("parent_id", "")),
        )
    )
    return {"question_id": question_id, "parents": selected}


def _compact_stream(
    source: TextIO,
    destination: TextIO,
    *,
    rank_limit: int,
    candidate_limit: int,
) -> dict[str, int]:
    questions = 0
    parents = 0
    seen: set[str] = set()
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"ranking line {line_number} is not an object")
        compact = compact_row(
            payload,
            rank_limit=rank_limit,
            candidate_limit=candidate_limit,
        )
        question_id = compact["question_id"]
        if question_id in seen:
            raise ValueError(f"duplicate ranking question {question_id!r}")
        seen.add(question_id)
        destination.write(json.dumps(compact, ensure_ascii=False, allow_nan=False))
        destination.write("\n")
        questions += 1
        parents += len(compact["parents"])
    if not questions:
        raise ValueError("ranking source is empty")
    return {"questions": questions, "parents": parents}


def _zip_member_to_file(bundle: zipfile.ZipFile, member: str, output: Path) -> None:
    with bundle.open(member) as source, output.open("wb") as destination:
        while block := source.read(4 * 1024 * 1024):
            destination.write(block)


def _compact_binary_stream(
    source: BinaryIO,
    output: Path,
    *,
    rank_limit: int,
    candidate_limit: int,
) -> dict[str, int]:
    import io

    with io.TextIOWrapper(source, encoding="utf-8") as text_source, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        return _compact_stream(
            text_source,
            destination,
            rank_limit=rank_limit,
            candidate_limit=candidate_limit,
        )


def package(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.rank_limit < 1 or args.candidate_limit < 1:
        raise ValueError("ranking limits must be positive")
    required = (
        args.heldout_rankings,
        args.p14_result,
        args.dev_ids,
        args.heldout_ids,
        P14_RUN / "strict_predictions.json",
        P14_RUN / "extractive_predictions.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("P16 inputs are missing: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task2-p16-") as temporary_name:
        temporary = Path(temporary_name)
        heldout_compact = temporary / "heldout_rankings.jsonl"
        public_compact = temporary / "public_rankings.jsonl"
        with (
            args.heldout_rankings.open(encoding="utf-8") as source,
            heldout_compact.open(
                "w", encoding="utf-8", newline="\n"
            ) as destination,
        ):
            heldout_stats = _compact_stream(
                source,
                destination,
                rank_limit=args.rank_limit,
                candidate_limit=args.candidate_limit,
            )
        public_member = f"{P14_ZIP_RUN}/public_rankings.jsonl"
        public_qwen_member = f"{P14_ZIP_RUN}/public_qwen_predictions.json"
        public_extractive_member = (
            f"{P14_ZIP_RUN}/public_extractive_predictions.json"
        )
        public_qwen = temporary / "public_qwen_predictions.json"
        public_extractive = temporary / "public_extractive_predictions.json"
        with zipfile.ZipFile(args.p14_result) as p14:
            required_members = {
                public_member,
                public_qwen_member,
                public_extractive_member,
            }
            if missing_members := required_members - set(p14.namelist()):
                raise FileNotFoundError(
                    "P14 result members are missing: "
                    + ", ".join(sorted(missing_members))
                )
            with p14.open(public_member) as source:
                public_stats = _compact_binary_stream(
                    source,
                    public_compact,
                    rank_limit=args.rank_limit,
                    candidate_limit=args.candidate_limit,
                )
            _zip_member_to_file(p14, public_qwen_member, public_qwen)
            _zip_member_to_file(p14, public_extractive_member, public_extractive)

        members = [
            (heldout_compact, "rankings/heldout_rankings.jsonl"),
            (public_compact, "rankings/public_rankings.jsonl"),
            (args.dev_ids, "training/dev_ids.json"),
            (args.heldout_ids, "training/heldout_ids.json"),
            (P14_RUN / "strict_predictions.json", "predictions/heldout_qwen.json"),
            (
                P14_RUN / "extractive_predictions.json",
                "predictions/heldout_extractive.json",
            ),
            (public_qwen, "predictions/public_qwen.json"),
            (public_extractive, "predictions/public_extractive.json"),
        ]
        descriptor, archive_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        archive = Path(archive_name)
        try:
            with zipfile.ZipFile(
                archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as output:
                for source, name in members:
                    output.write(source, name)
            with zipfile.ZipFile(archive) as output:
                if output.testzip() is not None:
                    raise ValueError("P16 bundle failed its integrity check")
                if set(output.namelist()) != {name for _, name in members}:
                    raise ValueError("P16 bundle member contract changed")
            os.replace(archive, args.output)
        finally:
            archive.unlink(missing_ok=True)

        manifest_path = args.output.with_suffix(".manifest.json")
        payload = {
            "schema_version": "task2-p16-modal-input-v1",
            "archive": str(args.output),
            "archive_bytes": args.output.stat().st_size,
            "archive_sha256": _sha256(args.output),
            "ranking_compaction": {
                "rank_limit": args.rank_limit,
                "candidate_limit": args.candidate_limit,
                "heldout": heldout_stats,
                "public": public_stats,
            },
            "members": [
                {
                    "archive_path": name,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
                for source, name in members
            ],
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return args.output, manifest_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        output, manifest = package(args)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Task 2 P16 packaging error: {exc}")
        return 2
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
