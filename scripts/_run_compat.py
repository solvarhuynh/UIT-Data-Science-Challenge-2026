"""Run a relocated command while preserving its original CLI behaviour."""

from __future__ import annotations

import runpy
from pathlib import Path


def run_relocated(folder: str, script_name: str) -> None:
    """Execute a command from its canonical subdirectory as ``__main__``."""
    script_path = Path(__file__).resolve().parent / folder / script_name
    runpy.run_path(str(script_path), run_name="__main__")
