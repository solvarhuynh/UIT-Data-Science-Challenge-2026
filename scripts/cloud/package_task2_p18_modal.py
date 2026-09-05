"""Package the compact P18 preference data for verified Modal upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFERENCE_ROOT = PROJECT_ROOT / "artifacts/task2/training/p15_adaptation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preference-train",
        type=Path,
        default=PREFERENCE_ROOT / "preference_train.jsonl",
    )
    parser.add_argument(
        "--preference-all",
        type=Path,
        default=PREFERENCE_ROOT / "preference_all.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/task2_p18_modal_input.zip",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_count(path: Path) -> int:
    count = 0
    seen: set[str] = set()
    required = {"id", "question", "contexts", "chosen", "rejected"}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required <= set(row):
                raise ValueError(f"{path}:{line_number} is not a preference row")
            question_id = str(row["id"])
            if question_id in seen:
                raise ValueError(f"duplicate preference ID {question_id!r}")
            seen.add(question_id)
            if not isinstance(row["contexts"], list) or not row["contexts"]:
                raise ValueError(f"blank contexts for preference {question_id!r}")
            for field in ("question", "chosen", "rejected"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"blank {field} for preference {question_id!r}")
            if row["chosen"].strip() == row["rejected"].strip():
                continue
            count += 1
    if count == 0:
        raise ValueError(f"preference data are empty: {path}")
    return count


def package(args: argparse.Namespace) -> tuple[Path, Path]:
    members = (
        (args.preference_train, "preferences/train.jsonl"),
        (args.preference_all, "preferences/all.jsonl"),
    )
    missing = [str(path) for path, _ in members if not path.is_file()]
    if missing:
        raise FileNotFoundError("P18 inputs are missing: " + ", ".join(missing))
    counts = {name: _validated_count(path) for path, name in members}
    train_ids = {
        str(json.loads(line)["id"])
        for line in args.preference_train.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    all_ids = {
        str(json.loads(line)["id"])
        for line in args.preference_all.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not train_ids < all_ids:
        raise ValueError("P18 train IDs must be a strict subset of all IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    metadata = [
        {
            "archive_path": name,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "records": counts[name],
        }
        for source, name in members
    ]
    if args.output.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        reusable = (
            existing.get("schema_version") == "task2-p18-modal-input-v1"
            and existing.get("members") == metadata
            and int(existing.get("archive_bytes", -1)) == args.output.stat().st_size
            and str(existing.get("archive_sha256", "")) == _sha256(args.output)
        )
        if reusable:
            with zipfile.ZipFile(args.output) as bundle:
                if bundle.testzip() is None:
                    return args.output, manifest_path

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as bundle:
            for source, name in members:
                bundle.write(source, name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P18 archive failed its integrity check")
            if set(bundle.namelist()) != {name for _, name in members}:
                raise ValueError("P18 archive member contract changed")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": "task2-p18-modal-input-v1",
        "archive": str(args.output),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": _sha256(args.output),
        "members": metadata,
        "train_ids_are_strict_subset": True,
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
        print(f"Task 2 P18 packaging error: {exc}")
        return 2
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
