"""Shared environment-aware configuration loader."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(env: str = "development") -> dict[str, Any]:
    """Load base config and deeply merge the selected environment overrides."""
    config_dir = Path("configs")
    with (config_dir / "base.yaml").open("r", encoding="utf-8") as config_file:
        base = yaml.safe_load(config_file) or {}
    if env == "base":
        return base
    environment_path = config_dir / f"{env}.yaml"
    if not environment_path.exists():
        raise FileNotFoundError(f"Environment config not found: {environment_path}")
    with environment_path.open("r", encoding="utf-8") as config_file:
        override = yaml.safe_load(config_file) or {}
    return _deep_merge(base, override)
