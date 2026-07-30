"""Tests for lightweight liveness and dependency-aware readiness endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient, Response

import udsc2026.api.app as api_module
from udsc2026.contracts import LegalChunk
from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

_READINESS_ENVIRONMENT = {
    "BM25_INDEX_PATH",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DEVICE",
    "EMBEDDING_MAX_LENGTH",
    "EMBEDDING_NORMALIZE_EMBEDDINGS",
    "FAISS_COLLECTION_NAME",
    "FAISS_INDEX_PATH",
    "LLM_BACKEND",
    "LLM_DEVICE",
    "LLM_DTYPE",
    "LLM_MAX_NEW_TOKENS",
    "LLM_REPETITION_PENALTY",
    "LLM_STREAM",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "LLM_TOP_P",
    "MODEL_EMBEDDER_PATH",
    "MODEL_LLM_PATH",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_URL",
    "REDIS_URL",
    "RERANKER_ENABLED",
    "RERANKER_BATCH_SIZE",
    "RERANKER_DEVICE",
    "RERANKER_LOCAL_FILES_ONLY",
    "RERANKER_MAX_LENGTH",
    "RERANKER_MODEL_PATH",
    "RERANKER_TOP_N",
    "UDSC2026_CONFIG_PATH",
    "VECTOR_DB_TYPE",
}


class _FakeHttpResponse:
    def __init__(self, status_code: int) -> None:
        self.status = status_code

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeRedisConnection:
    def __init__(self, response_chunks: list[bytes]) -> None:
        self.response_chunks = response_chunks
        self.timeout: float | None = None
        self.sent = bytearray()
        self.closed = False

    def __enter__(self) -> "_FakeRedisConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def sendall(self, payload: bytes) -> None:
        self.sent.extend(payload)

    def recv(self, _size: int) -> bytes:
        if not self.response_chunks:
            return b""
        return self.response_chunks.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_readiness_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in _READINESS_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)


async def _get(path: str) -> Response:
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


def _configure_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("top_k: 10\n", encoding="utf-8")
    monkeypatch.setenv("UDSC2026_CONFIG_PATH", str(config_path))
    return config_path


def _create_model_directory(root: Path, name: str) -> Path:
    model_path = root / name
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    return model_path


@pytest.mark.asyncio
async def test_health_does_not_run_readiness_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> dict[str, bool]:
        raise AssertionError("liveness must not probe dependencies")

    monkeypatch.setattr(api_module, "_readiness_checks", fail_if_called)

    response = await _get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "udsc2026-backend"


@pytest.mark.asyncio
async def test_ready_with_only_an_existing_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"config": True},
        "missing": [],
    }


@pytest.mark.asyncio
async def test_ready_uses_yaml_runtime_dependencies_without_compose_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "embedding:\n"
        f"  embedder_model_path: {tmp_path / 'missing-embedding'}\n"
        "llm:\n"
        f"  model_path: {tmp_path / 'missing-llm'}\n"
        "reranker:\n"
        "  enabled: true\n"
        f"  model_name_or_path: {tmp_path / 'missing-reranker'}\n"
        "vector_db:\n"
        "  type: faiss\n"
        f"  faiss_index_path: {tmp_path / 'missing-faiss'}\n"
        "  collection_name: legal_chunks\n"
        "hybrid:\n"
        f"  bm25_index_path: {tmp_path / 'missing-bm25.json'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("UDSC2026_CONFIG_PATH", str(config_path))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "config": True,
        "embedding_model": False,
        "llm_model": False,
        "reranker_model": False,
        "faiss": False,
        "bm25": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variable", "value", "expected_check"),
    [
        ("EMBEDDING_DEVICE", "cpu", "embedding_model"),
        ("LLM_BACKEND", "transformers", "llm_model"),
    ],
)
async def test_any_model_runtime_override_activates_dependency_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str,
    value: str,
    expected_check: str,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv(variable, value)

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"][expected_check] is False
    assert expected_check in response.json()["missing"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hybrid_yaml",
    [
        "[]",
        "{dense_weight: true, sparse_weight: 0.0}",
        "{dense_weight: 0.8, sparse_weight: 0.8}",
        "{candidate_k: 0}",
    ],
)
async def test_ready_rejects_invalid_hybrid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hybrid_yaml: str,
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        f"hybrid: {hybrid_yaml}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("UDSC2026_CONFIG_PATH", str(config_path))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "config": True,
        "hybrid_config": False,
    }
    assert response.json()["missing"] == ["hybrid_config"]


@pytest.mark.asyncio
async def test_ready_rejects_reranker_top_n_above_hybrid_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "reranker:\n"
        "  enabled: true\n"
        f"  model_name_or_path: {tmp_path / 'missing-reranker'}\n"
        "  top_n: 11\n"
        "hybrid:\n"
        "  candidate_k: 10\n"
        f"  bm25_index_path: {tmp_path / 'missing-bm25.json'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("UDSC2026_CONFIG_PATH", str(config_path))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["reranker_pipeline_config"] is False
    assert "reranker_pipeline_config" in response.json()["missing"]


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", ["", "[not, a, mapping]\n", "invalid: [\n"])
async def test_ready_rejects_empty_non_mapping_or_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str,
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("UDSC2026_CONFIG_PATH", str(config_path))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["config"] is False
    assert response.json()["missing"] == ["config"]


@pytest.mark.asyncio
async def test_ready_requires_configured_models_to_be_nonempty_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    embedding_path = _create_model_directory(tmp_path, "embedding")
    llm_path = _create_model_directory(tmp_path, "llm")
    reranker_path = _create_model_directory(tmp_path, "reranker")
    monkeypatch.setenv("MODEL_EMBEDDER_PATH", str(embedding_path))
    monkeypatch.setenv("MODEL_LLM_PATH", str(llm_path))
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(reranker_path))

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "config": True,
        "embedding_model": True,
        "llm_model": True,
        "reranker_model": True,
    }


@pytest.mark.asyncio
async def test_ready_rejects_empty_file_and_missing_model_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    empty_directory = tmp_path / "empty-model"
    empty_directory.mkdir()
    regular_file = tmp_path / "not-a-model-directory"
    regular_file.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("MODEL_EMBEDDER_PATH", str(empty_directory))
    monkeypatch.setenv("MODEL_LLM_PATH", str(regular_file))
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path / "missing-model"))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "config": True,
        "embedding_model": False,
        "llm_model": False,
        "reranker_model": False,
    }
    assert response.json()["missing"] == [
        "embedding_model",
        "llm_model",
        "reranker_model",
    ]


@pytest.mark.asyncio
async def test_disabled_reranker_does_not_require_its_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("RERANKER_ENABLED", " FALSE ")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path / "missing-reranker"))

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {"config": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_value", ["", "0", "1", "yes", "disabled"])
async def test_invalid_reranker_boolean_makes_readiness_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_value: str,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("RERANKER_ENABLED", invalid_value)

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["reranker_enabled"] is False
    assert response.json()["missing"] == ["reranker_enabled"]


@pytest.mark.asyncio
async def test_qdrant_readiness_probes_healthz(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", " QDRANT ")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333/api?ignored=true")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeHttpResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return _FakeHttpResponse(200)

    monkeypatch.setattr(api_module, "urlopen", fake_urlopen)

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["qdrant"] is True
    assert captured == {
        "url": "http://qdrant.internal:6333/healthz",
        "method": "GET",
        "timeout": api_module._READINESS_TIMEOUT_SECONDS,
    }


@pytest.mark.asyncio
async def test_qdrant_readiness_sends_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "  secret-key  ")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeHttpResponse:
        captured["api_key"] = request.get_header("Api-key")
        captured["timeout"] = timeout
        return _FakeHttpResponse(200)

    monkeypatch.setattr(api_module, "urlopen", fake_urlopen)

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["qdrant"] is True
    assert captured == {
        "api_key": "secret-key",
        "timeout": api_module._READINESS_TIMEOUT_SECONDS,
    }


@pytest.mark.asyncio
async def test_qdrant_readiness_requires_configured_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "legal_chunks")
    captured_urls: list[str] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeHttpResponse:
        assert timeout == api_module._READINESS_TIMEOUT_SECONDS
        captured_urls.append(request.full_url)
        return _FakeHttpResponse(200 if request.full_url.endswith("/healthz") else 404)

    monkeypatch.setattr(api_module, "urlopen", fake_urlopen)

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"] is False
    assert captured_urls == [
        "http://qdrant:6333/healthz",
        "http://qdrant:6333/collections/legal_chunks",
    ]


@pytest.mark.asyncio
async def test_qdrant_readiness_handles_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    def failing_urlopen(*_args: object, **_kwargs: object) -> _FakeHttpResponse:
        raise OSError("connection refused")

    monkeypatch.setattr(api_module, "urlopen", failing_urlopen)

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"] is False
    assert response.json()["missing"] == ["qdrant"]


@pytest.mark.asyncio
async def test_qdrant_requires_a_valid_http_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "redis://not-qdrant:6379")

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"] is False


@pytest.mark.asyncio
async def test_redis_readiness_uses_resp_ping_pong(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6380/2")
    connection = _FakeRedisConnection([b"+OK\r\n", b"+PO", b"NG\r\n"])
    captured: dict[str, Any] = {}

    def fake_create_connection(
        address: tuple[str, int],
        *,
        timeout: float,
    ) -> _FakeRedisConnection:
        captured["address"] = address
        captured["timeout"] = timeout
        return connection

    monkeypatch.setattr(api_module.socket, "create_connection", fake_create_connection)

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["redis"] is True
    assert captured == {
        "address": ("cache.internal", 6380),
        "timeout": api_module._READINESS_TIMEOUT_SECONDS,
    }
    assert connection.timeout == api_module._READINESS_TIMEOUT_SECONDS
    assert bytes(connection.sent) == (
        b"*2\r\n$6\r\nSELECT\r\n$1\r\n2\r\n*1\r\n$4\r\nPING\r\n"
    )


@pytest.mark.asyncio
async def test_redis_readiness_authenticates_before_ping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "REDIS_URL",
        "redis://legal-user:p%40ss@cache.internal:6379/0",
    )
    connection = _FakeRedisConnection([b"+OK\r\n", b"+PONG\r\n"])
    monkeypatch.setattr(
        api_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["redis"] is True
    assert bytes(connection.sent) == (
        b"*3\r\n$4\r\nAUTH\r\n$10\r\nlegal-user\r\n$4\r\np@ss\r\n*1\r\n$4\r\nPING\r\n"
    )


@pytest.mark.asyncio
async def test_redis_readiness_supports_legacy_password_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("REDIS_URL", "redis://:secret@cache.internal:6379/0")
    connection = _FakeRedisConnection([b"+OK\r\n", b"+PONG\r\n"])
    monkeypatch.setattr(
        api_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )

    response = await _get("/ready")

    assert response.status_code == 200
    assert bytes(connection.sent) == (
        b"*2\r\n$4\r\nAUTH\r\n$6\r\nsecret\r\n*1\r\n$4\r\nPING\r\n"
    )


@pytest.mark.asyncio
async def test_redis_readiness_rejects_non_pong_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    connection = _FakeRedisConnection([b"-NOAUTH Authentication required.\r\n"])
    monkeypatch.setattr(
        api_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["redis"] is False
    assert response.json()["missing"] == ["redis"]


@pytest.mark.asyncio
async def test_blank_redis_url_disables_optional_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("REDIS_URL", "")

    response = await _get("/ready")

    assert response.status_code == 200
    assert "redis" not in response.json()["checks"]


@pytest.mark.asyncio
async def test_rediss_handshake_failure_closes_raw_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("REDIS_URL", "rediss://cache.internal:6380/0")
    connection = _FakeRedisConnection([])

    class FailingTlsContext:
        def wrap_socket(self, *_args: object, **_kwargs: object) -> object:
            raise api_module.ssl.SSLError("TLS handshake failed")

    monkeypatch.setattr(
        api_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        api_module.ssl,
        "create_default_context",
        lambda: FailingTlsContext(),
    )

    response = await _get("/ready")

    assert response.status_code == 503
    assert connection.closed is True


@pytest.mark.asyncio
async def test_faiss_readiness_requires_matching_nonempty_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    index_root = tmp_path / "faiss"
    adapter = FaissAdapter(str(index_root), "legal-collection")
    adapter.create_collection("legal-collection", vector_size=2)
    adapter.upsert(
        [LegalChunk(chunk_id="chunk-1", doc_id="law", text="Điều 1.")],
        [[1.0, 0.0]],
    )
    monkeypatch.setenv("VECTOR_DB_TYPE", "faiss")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(index_root))
    monkeypatch.setenv("FAISS_COLLECTION_NAME", "legal-collection")

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["faiss"] is True


@pytest.mark.asyncio
async def test_faiss_readiness_rejects_empty_payload_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    collection_path = tmp_path / "faiss" / "legal_chunks"
    collection_path.mkdir(parents=True)
    (collection_path / "index.faiss").write_bytes(b"faiss-index")
    (collection_path / "payloads.json").touch()
    monkeypatch.setenv("VECTOR_DB_TYPE", "faiss")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "faiss"))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["faiss"] is False
    assert response.json()["missing"] == ["faiss"]


@pytest.mark.asyncio
async def test_faiss_readiness_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    index_root = tmp_path / "faiss"
    adapter = FaissAdapter(str(index_root), "legal_chunks")
    adapter.create_collection("legal_chunks", vector_size=2)
    adapter.upsert(
        [LegalChunk(chunk_id="chunk-1", doc_id="law", text="Điều 1.")],
        [[1.0, 0.0]],
    )
    with (index_root / "legal_chunks" / "index.faiss").open("ab") as index_file:
        index_file.write(b"tampered")
    monkeypatch.setenv("VECTOR_DB_TYPE", "faiss")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(index_root))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["faiss"] is False


@pytest.mark.asyncio
async def test_faiss_readiness_rejects_path_traversal_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "faiss")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "faiss"))
    monkeypatch.setenv("FAISS_COLLECTION_NAME", "../outside")

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["faiss"] is False


@pytest.mark.asyncio
async def test_unknown_vector_backend_makes_readiness_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTOR_DB_TYPE", "unsupported")

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["vector_db_type"] is False
    assert response.json()["missing"] == ["vector_db_type"]


@pytest.mark.asyncio
async def test_bm25_readiness_validates_schema_and_unique_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    index_path = tmp_path / "bm25.json"
    index_path.write_text(
        '{"schema_version":1,"chunks":['
        '{"chunk_id":"c1","doc_id":"law","text":"Điều 1"}'
        "]}",
        encoding="utf-8",
    )
    monkeypatch.setenv("BM25_INDEX_PATH", str(index_path))

    response = await _get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["bm25"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents",
    [
        '{"schema_version":1,"chunks":[]}',
        '{"schema_version":1,"chunks":['
        '{"chunk_id":"c1","doc_id":"law","text":"Điều 1"},'
        '{"chunk_id":"c1","doc_id":"law","text":"Điều 2"}]}',
        '{"schema_version":1,"chunks":['
        '{"chunk_id":"c1","doc_id":"law","text":"Điều 1",'
        '"metadata":{"score":NaN}}]}',
    ],
)
async def test_bm25_readiness_rejects_invalid_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str,
) -> None:
    _configure_existing_file(monkeypatch, tmp_path)
    index_path = tmp_path / "bm25.json"
    index_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("BM25_INDEX_PATH", str(index_path))

    response = await _get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["bm25"] is False
    assert response.json()["missing"] == ["bm25"]
