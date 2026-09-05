"""Build the small source bundle used by the Task 2 P14 Kaggle runner."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/task2/kaggle/task2_p14_code_bundle.zip"
DIRECTORIES = ("src", "scripts", "configs")
DOCUMENTS = (
    "pyproject.toml",
    "docs/members/tv5/tv5_task2_p14.md",
    "docs/members/tv5/tv5_kaggle_task2_p14.md",
)
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "data",
    "models",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def collect_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in DIRECTORIES:
        directory = repo_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(
                f"required source directory is missing: {directory}"
            )
        files.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix != ".pyc"
            and not any(
                part in EXCLUDED_PARTS for part in path.relative_to(repo_root).parts
            )
        )
    for relative in DOCUMENTS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Task 2 bundle file is missing: {path}")
        files.add(path)
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def create_bundle(repo_root: Path, output: Path) -> list[str]:
    files = collect_files(repo_root)
    output = output if output.is_absolute() else repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [path.relative_to(repo_root).as_posix() for path in files]
    manifest = {
        "schema_version": "task2-p14-kaggle-code-v1",
        "includes": names,
        "input_archive_included": False,
        "qwen_weights_included": False,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, name in zip(files, names):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo(
            "task2_p14_code_manifest.json", date_time=(1980, 1, 1, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        bundle.writestr(
            info,
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    return [*names, "task2_p14_code_manifest.json"]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = (
            args.output
            if args.output.is_absolute()
            else args.repo_root / args.output
        )
        names = create_bundle(args.repo_root.resolve(), output.resolve())
    except (OSError, ValueError) as exc:
        print(f"Task 2 Kaggle packaging error: {exc}")
        return 1
    print(f"TASK2_KAGGLE_CODE_BUNDLE={output.resolve()}")
    print(f"files={len(names)} bytes={output.resolve().stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
