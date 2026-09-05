"""Build the minimal local-only Task 2 P14 archive for a Beam Volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/task2/task2_p14_beam_input.tar.zst"
INCLUDE_PATHS = (
    Path("artifacts/task2/models/dek21-parent-crossencoder-hn16-v2/checkpoint-best"),
    Path("artifacts/task2/training/parent_ce_hn16_train.jsonl"),
    Path("artifacts/task2/training/parent_ce_hn16_eval.jsonl"),
    Path("artifacts/task2/training/parent_ce_train_ids.json"),
    Path("artifacts/task2/training/parent_ce_eval_ids.json"),
    Path("artifacts/task2/training/task2_qwen_sft_train.jsonl"),
    Path("artifacts/task2/training/task2_qwen_sft_train.manifest.json"),
    Path("artifacts/task2/train500-bge-reranker-all/predictions_after.jsonl"),
    Path("artifacts/task2/train500b-bge-reranker-all/predictions_after.jsonl"),
    Path("artifacts/task2/public-bge-reranker-all/predictions_after.jsonl"),
    Path("data/processed_v3/parents"),
    Path("data/raw/btc/LegalQA/train.json"),
    Path("data/raw/btc/LegalQA/public-official.json"),
)
EXCLUDED_NAMES = {"training_state.pt"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def iter_input_files() -> Iterable[tuple[Path, Path]]:
    """Yield absolute sources and repository-relative archive names."""

    for relative in INCLUDE_PATHS:
        source = PROJECT_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"required Beam input is missing: {relative}")
        if source.is_file():
            if source.name not in EXCLUDED_NAMES:
                yield source, relative
            continue
        for child in sorted(source.rglob("*")):
            if child.is_file() and child.name not in EXCLUDED_NAMES:
                yield child, child.relative_to(PROJECT_ROOT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_archive(output: Path, *, compression_level: int, force: bool) -> Path:
    """Write an atomic tar.zst plus a hash/size sidecar manifest."""

    if not 1 <= compression_level <= 19:
        raise ValueError("compression-level must be in [1, 19]")
    if output.suffixes[-2:] != [".tar", ".zst"]:
        raise ValueError("output must end in .tar.zst")
    if output.exists() and not force:
        raise FileExistsError(f"output exists; pass --force: {output}")
    files = list(iter_input_files())
    if not files:
        raise ValueError("Beam input plan is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        try:
            import zstandard
        except ImportError:
            zstandard = None

        def write_tar(destination: Any) -> None:
            with tarfile.open(fileobj=destination, mode="w|") as archive:
                for index, (source, relative) in enumerate(files, 1):
                    archive.add(
                        source,
                        arcname=relative.as_posix(),
                        recursive=False,
                    )
                    if index % 500 == 0:
                        print(f"packed={index}/{len(files)}", flush=True)

        if zstandard is not None:
            with temporary.open("wb") as destination:
                compressor = zstandard.ZstdCompressor(
                    level=compression_level,
                    threads=-1,
                )
                with compressor.stream_writer(
                    destination,
                    closefd=False,
                ) as compressed:
                    write_tar(compressed)
                destination.flush()
                os.fsync(destination.fileno())
        else:
            executable = shutil.which("zstd")
            if executable is None:
                raise ImportError(
                    "install Python zstandard or add zstd.exe to PATH"
                )
            process = subprocess.Popen(
                [
                    executable,
                    f"-{compression_level}",
                    "-T0",
                    "-q",
                    "-f",
                    "-o",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
            )
            if process.stdin is None:
                raise RuntimeError("zstd subprocess exposed no stdin")
            try:
                write_tar(process.stdin)
                process.stdin.close()
                return_code = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            if return_code != 0:
                raise RuntimeError(f"zstd exited with code {return_code}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = output.with_suffix(".manifest.json")
    _write_json_atomic(
        manifest,
        {
            "schema_version": "task2-p14-beam-input-v1",
            "archive": output.name,
            "archive_bytes": output.stat().st_size,
            "archive_sha256": _sha256(output),
            "file_count": len(files),
            "uncompressed_bytes": sum(source.stat().st_size for source, _ in files),
            "excluded_names": sorted(EXCLUDED_NAMES),
            "qwen_policy": (
                "not uploaded; downloaded server-side from "
                "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2"
            ),
            "files": [relative.as_posix() for _, relative in files],
        },
    )
    print(f"archive={output}", flush=True)
    print(f"manifest={manifest}", flush=True)
    print(f"bytes={output.stat().st_size}", flush=True)
    print(f"sha256={_sha256(output)}", flush=True)
    return output


def main() -> int:
    args = build_parser().parse_args()
    try:
        build_archive(
            args.output.resolve(),
            compression_level=args.compression_level,
            force=args.force,
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"Task 2 Beam packaging error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
