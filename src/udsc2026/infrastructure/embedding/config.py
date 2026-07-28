"""Configuration loading for the embedding infrastructure."""

from pathlib import Path
from typing import Any

import yaml


def load_embedding_config(path: str = "configs/base.yaml") -> dict[str, Any]:
    """Load the embedding section from the shared YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config.get("embedding", {})
