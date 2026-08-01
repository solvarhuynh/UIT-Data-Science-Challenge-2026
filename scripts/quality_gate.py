"""Run deterministic Python quality checks with the project virtual environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _project_python(project_root: Path) -> Path:
    """Prefer the repository virtual environment and otherwise reuse this Python."""

    candidates = (
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _run(python: Path, arguments: Sequence[str], project_root: Path) -> None:
    """Run one Python module and stop immediately if it fails."""

    command = [str(python), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)  # noqa: S603


def main() -> int:
    """Run formatting, linting, typing, docstring, and security checks."""

    project_root = Path(__file__).resolve().parents[1]
    python = _project_python(project_root)
    commands = (
        ("-m", "ruff", "format", "--check", "."),
        ("-m", "ruff", "check", "."),
        ("-m", "mypy", "src", "--no-warn-unused-configs"),
        ("-m", "pydocstyle", "src"),
        ("-m", "bandit", "-q", "-r", "src"),
    )
    for command in commands:
        _run(python, command, project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
