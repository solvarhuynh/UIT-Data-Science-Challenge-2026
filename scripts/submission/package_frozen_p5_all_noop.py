"""Byte-preserving CodaBench package for the frozen P5 all-NO_OP submission."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/task1/submission_p5_all_noop/public_submission.json"
OUTPUT = ROOT / "artifacts/task1/submission_p5_all_noop/submission.zip"
QUESTION_IDS = ROOT / "artifacts/task1/recovery_096/public_anchor_093/public_question_ids.json"
CORPUS = ROOT / "artifacts/task1/corpus_document_ids.json"
VALIDATOR = ROOT / "scripts/submission/validate_legal_ir_submission.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    source_bytes = SOURCE.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    # Write only the required CodaBench member and preserve source bytes exactly.
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", source_bytes)
    with ZipFile(OUTPUT) as archive:
        if archive.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP must contain exactly one member: submission.json")
        extracted = archive.read("submission.json")
    if extracted != source_bytes or sha256_bytes(extracted) != source_hash:
        raise RuntimeError("extracted submission.json does not match source bytes")
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--input", str(OUTPUT),
            "--questions", str(QUESTION_IDS),
            "--corpus-manifest", str(CORPUS),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stdout.strip())
    print(f"source_sha256={source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
