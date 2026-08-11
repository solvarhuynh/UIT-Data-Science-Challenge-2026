"""Build a deterministic, weight-free source bundle for the Task1 Kaggle runbook."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("artifacts/task1/kaggle/task1_code_bundle.zip")

DIRECTORIES = ("src", "scripts", "configs")
DOCUMENTS = (
    "README.md",
    "pyproject.toml",
    "requirements_runtime.txt",
    "requirements_dev.txt",
    "download_models.py",
    "docs/members/tv2/tv2_task1_kaggle.md",
    "docs/members/tv2/tv2_setup.md",
    "docs/members/tv2/tv2_progress_report.md",
    "docs/models/model_registry.md",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Bundle path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root used for collecting source files.",
    )
    return parser


def _is_allowed(path: Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.parts)


def collect_files(repo_root: Path = ROOT) -> list[Path]:
    """Collect only source/config/document files, never runtime data or weights."""

    files: set[Path] = set()
    for relative_directory in DIRECTORIES:
        directory = repo_root / relative_directory
        if directory.is_dir():
            files.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix != ".pyc"
                and _is_allowed(path.relative_to(repo_root))
            )
    for relative_file in DOCUMENTS:
        path = repo_root / relative_file
        if not path.is_file():
            raise FileNotFoundError(f"required Task1 bundle file is missing: {path}")
        files.add(path)
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_bundle(repo_root: Path, output: Path) -> list[str]:
    """Write a deterministic ZIP and return its source member names."""

    output = output if output.is_absolute() else repo_root / output
    files = collect_files(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [path.relative_to(repo_root).as_posix() for path in files]
    manifest = {
        "schema_version": "task1-kaggle-code-bundle-v1",
        "git_commit": _git_commit(repo_root),
        "includes": names,
        "weights_included": False,
        "runtime_data_included": False,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in zip(files, names):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        manifest_info = zipfile.ZipInfo(
            "task1_bundle_manifest.json", date_time=(1980, 1, 1, 0, 0, 0)
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(
            manifest_info,
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    return [*names, "task1_bundle_manifest.json"]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = (
            args.output if args.output.is_absolute() else args.repo_root / args.output
        )
        names = create_bundle(args.repo_root, output)
    except (OSError, ValueError) as exc:
        print(f"Task1 Kaggle packaging error: {exc}")
        return 2
    print(f"Created {output} with {len(names)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
