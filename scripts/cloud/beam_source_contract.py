"""Canonical source-tree fingerprint shared by Beam runtime and download gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_DIRECTORIES = ("configs", "scripts", "src")
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def iter_source_files(root: Path) -> list[Path]:
    project_file = root / "pyproject.toml"
    if not project_file.is_file():
        raise FileNotFoundError(f"Beam source file is missing: {project_file}")
    files = [project_file]
    for relative in SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Beam source directory is missing: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in IGNORED_SUFFIXES
        )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def compute_source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()
