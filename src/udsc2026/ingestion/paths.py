"""Canonical paths for organizer data kept outside the source package."""

from __future__ import annotations

from pathlib import Path


CANONICAL_LEGAL_IR_DIRNAME = "Legall-IR"


def legal_ir_dir(raw_root: str | Path = Path("data/raw/btc")) -> Path:
    """Return the one supported LegalIR directory.

    The organizer directory is spelled ``Legall-IR``.  Resolving it in one
    place prevents callers from silently creating a second ``LegalIR`` tree.
    """

    root = Path(raw_root)
    path = root / CANONICAL_LEGAL_IR_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"canonical LegalIR directory is missing: {path}; "
            "do not create a duplicate LegalIR directory"
        )
    legacy = root / "LegalIR"
    if legacy.exists():
        raise ValueError(
            f"duplicate LegalIR directories found: {path} and {legacy}; "
            "remove the legacy duplicate before running the pipeline"
        )
    return path


__all__ = ["CANONICAL_LEGAL_IR_DIRNAME", "legal_ir_dir"]
