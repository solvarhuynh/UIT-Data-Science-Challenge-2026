import os
import sys
from typing import Generator
from unittest.mock import patch

import pytest

# Add src package to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    with patch("qdrant_client.QdrantClient") as mock:
        yield mock


@pytest.fixture
def mock_database():
    """Mock database connection"""
    with patch("sqlalchemy.create_engine") as mock:
        yield mock


@pytest.fixture
def test_config():
    return {
        "QDRANT_URL": "http://localhost:6333",
        "MODEL_EMBEDDER_PATH": "./models/bkai-bi-encoder",
        "MODEL_LLM_PATH": "./models/qwen3-legal",
        "DEBUG": True,
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ.update(
        {
            "QDRANT_URL": "http://localhost:6333",
            "MODEL_EMBEDDER_PATH": "./models/bkai-bi-encoder",
            "MODEL_LLM_PATH": "./models/qwen3-legal",
            "DEBUG": "true",
        }
    )
