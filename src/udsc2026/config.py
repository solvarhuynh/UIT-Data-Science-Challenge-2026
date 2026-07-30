"""Single source of truth for the project YAML configuration path."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def resolve_config_path(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve an explicit path or ``UDSC2026_CONFIG_PATH`` consistently."""

    environment = os.environ if environ is None else environ
    raw_path: str | Path = (
        environment.get("UDSC2026_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
        if path is None
        else path
    )
    if isinstance(raw_path, str) and not raw_path.strip():
        raise ValueError("UDSC2026_CONFIG_PATH must not be blank")
    return Path(raw_path)


def load_project_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load the configured YAML root and require a mapping."""

    config_path = resolve_config_path(path, environ=environ)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config: object = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {config_path}") from exc
    if raw_config is None:
        return {}
    if not isinstance(raw_config, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return dict(raw_config)
