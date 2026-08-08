"""Minimal FastAPI application shell used by local and container health checks.

TV1 can attach query/orchestration routers later without changing the container
entrypoint. This module deliberately avoids importing retrieval or model
clients during startup, keeping liveness checks lightweight.
"""

from __future__ import annotations

import logging
import os
import socket
import ssl
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Response, status

from udsc2026.api.cache import FallbackCache, InMemoryCache, RedisCache
from udsc2026.api.orchestrator import RAGOrchestrator
from udsc2026.config import load_project_config, resolve_config_path
from udsc2026.contracts.api import QueryRequest, QueryResponse
from udsc2026.contracts.health import HealthResponse, ReadinessResponse
from udsc2026.infrastructure.embedding.config import load_embedding_config
from udsc2026.infrastructure.llm.config import load_llm_config
from udsc2026.infrastructure.reranker.config import (
    RerankerSettings,
    load_reranker_settings,
)
from udsc2026.infrastructure.vector_db.base import validate_collection_name
from udsc2026.infrastructure.vector_db.factory import resolve_vector_db_config

if TYPE_CHECKING:
    from udsc2026.retrieval.hybrid.config import HybridSettings

_READINESS_TIMEOUT_SECONDS = 0.5
_REDIS_PING = b"*1\r\n$4\r\nPING\r\n"
_REDIS_PONG = b"+PONG\r\n"
_REDIS_OK = b"+OK\r\n"
_MAX_REDIS_RESPONSE_BYTES = 128
_EMBEDDING_ENVIRONMENT_FIELDS = (
    "MODEL_EMBEDDER_PATH",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_LENGTH",
    "EMBEDDING_NORMALIZE_EMBEDDINGS",
)
_EMBEDDING_MODEL_ENVIRONMENT_FIELDS = ("MODEL_EMBEDDER_PATH",)
_LLM_ENVIRONMENT_FIELDS = (
    "MODEL_LLM_PATH",
    "LLM_BACKEND",
    "LLM_DEVICE",
    "LLM_DTYPE",
    "LLM_MAX_NEW_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "LLM_REPETITION_PENALTY",
    "LLM_STREAM",
    "LLM_TIMEOUT_SECONDS",
)
_LLM_MODEL_ENVIRONMENT_FIELDS = ("MODEL_LLM_PATH",)
LOGGER = logging.getLogger(__name__)


def _build_orchestrator() -> RAGOrchestrator:
    """Lazily build heavyweight pipeline dependencies on the first query."""
    from udsc2026.infrastructure.embedding.client import EmbeddingClient
    from udsc2026.infrastructure.embedding.config import load_embedding_config
    from udsc2026.infrastructure.llm.client import LLMClient
    from udsc2026.infrastructure.reranker.config import load_reranker_settings
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
    from udsc2026.qa.qa_engine import QAEngine
    from udsc2026.retrieval.dense.dense_retriever import DenseRetriever
    from udsc2026.retrieval.hybrid.config import load_hybrid_settings
    from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever
    from udsc2026.retrieval.reranking.cross_encoder import CrossEncoderReranker
    from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

    config = load_project_config()
    embedding_config = load_embedding_config()
    embedding_client = EmbeddingClient(
        model_path=embedding_config["embedder_model_path"],
        device=embedding_config.get("device", "cpu"),
        batch_size=embedding_config.get("batch_size", 32),
        max_length=embedding_config.get("max_length", 256),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )
    dense = DenseRetriever(embedding_client, get_vector_db_adapter(config))
    hybrid_settings = load_hybrid_settings()
    sparse = BM25Retriever(hybrid_settings.bm25_index_path)
    sparse.load()
    retriever = HybridRetriever(
        dense,
        sparse,
        dense_weight=hybrid_settings.dense_weight,
        sparse_weight=hybrid_settings.sparse_weight,
        candidate_k=hybrid_settings.candidate_k,
        min_score=hybrid_settings.min_score,
    )
    reranker_settings = load_reranker_settings()
    reranker = None
    reranker_version = "disabled"
    if reranker_settings.enabled:
        reranker = CrossEncoderReranker(reranker_settings.create_client())
        reranker_version = reranker_settings.model_name_or_path
    local_cache = InMemoryCache()
    redis_url = os.getenv("REDIS_URL", "").strip()
    cache = (
        FallbackCache(RedisCache(redis_url), local_cache) if redis_url else local_cache
    )
    return RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        qa_engine=QAEngine(LLMClient(load_llm_config())),
        cache=cache,
        retriever_version="hybrid_v1",
        reranker_version=reranker_version,
    )


def _package_version() -> str:
    try:
        return version("udsc2026")
    except PackageNotFoundError:
        return "0.1.0"


def _parse_optional_boolean(name: str) -> bool | None:
    """Parse an optional environment boolean without accepting ambiguous aliases."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be either 'true' or 'false'")


def _is_nonempty_directory(raw_path: str | None) -> bool:
    """Return whether a configured model path is an accessible non-empty directory."""
    if not raw_path:
        return False
    path = Path(raw_path)
    try:
        return path.is_dir() and next(path.iterdir(), None) is not None
    except OSError:
        return False


def _probe_qdrant(
    raw_url: str | None,
    raw_api_key: str | None = None,
    raw_collection_name: str | None = None,
) -> bool:
    """Probe Qdrant health and, when configured, the actual collection."""
    if not raw_url:
        return False
    try:
        parsed = urlsplit(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        urls = [f"{base_url}/healthz"]
        if raw_collection_name is not None:
            collection_name = validate_collection_name(raw_collection_name)
            urls.append(f"{base_url}/collections/{collection_name}")
        headers: dict[str, str] = {}
        if raw_api_key and raw_api_key.strip():
            headers["api-key"] = raw_api_key.strip()
        for url in urls:
            request = Request(url, headers=headers, method="GET")
            # URLs are reconstructed only after the HTTP(S)-scheme allowlist.
            with urlopen(  # nosec B310
                request,
                timeout=_READINESS_TIMEOUT_SECONDS,
            ) as response:
                status_code = getattr(response, "status", None)
                if not isinstance(status_code, int) or not 200 <= status_code < 300:
                    return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def _receive_redis_line(connection: socket.socket) -> bytes:
    response = bytearray()
    while len(response) < _MAX_REDIS_RESPONSE_BYTES:
        chunk = connection.recv(_MAX_REDIS_RESPONSE_BYTES - len(response))
        if not chunk:
            break
        response.extend(chunk)
        if response.endswith(b"\r\n"):
            break
    return bytes(response)


def _encode_redis_command(*parts: str) -> bytes:
    encoded_parts = [part.encode("utf-8") for part in parts]
    command = bytearray(f"*{len(encoded_parts)}\r\n".encode())
    for part in encoded_parts:
        command.extend(f"${len(part)}\r\n".encode())
        command.extend(part)
        command.extend(b"\r\n")
    return bytes(command)


def _send_redis_command(
    connection: socket.socket,
    expected_response: bytes,
    *parts: str,
) -> bool:
    connection.sendall(_encode_redis_command(*parts))
    return _receive_redis_line(connection) == expected_response


def _probe_redis(raw_url: str | None) -> bool:
    """Authenticate/select the configured Redis DB, then require PONG."""
    if not raw_url:
        return False
    raw_connection: socket.socket | None = None
    connection: socket.socket | None = None
    try:
        parsed = urlsplit(raw_url)
        if parsed.scheme not in {"redis", "rediss"} or parsed.hostname is None:
            return False
        port = parsed.port or 6379
        database_text = parsed.path.lstrip("/")
        if "/" in database_text or (database_text and not database_text.isdecimal()):
            return False
        database = int(database_text or "0")
        raw_username = parsed.username
        username = unquote(raw_username) if raw_username else None
        password = unquote(parsed.password) if parsed.password is not None else None
        if username is not None and password is None:
            return False
        raw_connection = socket.create_connection(
            (parsed.hostname, port),
            timeout=_READINESS_TIMEOUT_SECONDS,
        )
        connection = raw_connection
        if parsed.scheme == "rediss":
            connection = ssl.create_default_context().wrap_socket(
                raw_connection,
                server_hostname=parsed.hostname,
            )
        with connection:
            connection.settimeout(_READINESS_TIMEOUT_SECONDS)
            if password is not None:
                auth_parts = (
                    ("AUTH", username, password)
                    if username is not None
                    else ("AUTH", password)
                )
                if not _send_redis_command(connection, _REDIS_OK, *auth_parts):
                    return False
            if database and not _send_redis_command(
                connection,
                _REDIS_OK,
                "SELECT",
                str(database),
            ):
                return False
            return _send_redis_command(connection, _REDIS_PONG, "PING")
    except (OSError, ValueError, ssl.SSLError):
        return False
    finally:
        if raw_connection is not None:
            raw_connection.close()


def _check_faiss_artifacts(
    configured_root: str | None = None,
    configured_collection_name: str | None = None,
) -> bool:
    """Validate a FAISS generation once per observed artifact snapshot."""
    raw_root = configured_root or os.getenv("FAISS_INDEX_PATH")
    if not raw_root:
        return False
    collection_name = (
        configured_collection_name
        or os.getenv("FAISS_COLLECTION_NAME")
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "legal_chunks"
    )
    try:
        validated_name = validate_collection_name(collection_name)
        collection_path = Path(raw_root) / validated_name
        index_stat = (collection_path / "index.faiss").stat()
        payload_stat = (collection_path / "payloads.json").stat()
    except (OSError, ValueError):
        return False
    return _validate_faiss_snapshot(
        raw_root,
        validated_name,
        index_stat.st_mtime_ns,
        index_stat.st_size,
        payload_stat.st_mtime_ns,
        payload_stat.st_size,
    )


@lru_cache(maxsize=16)
def _validate_faiss_snapshot(
    root: str,
    collection_name: str,
    _index_mtime_ns: int,
    _index_size: int,
    _payload_mtime_ns: int,
    _payload_size: int,
) -> bool:
    try:
        from udsc2026.infrastructure.vector_db.faiss_adapter import FaissAdapter

        adapter = FaissAdapter(root, collection_name)
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    return (
        adapter.index is not None
        and adapter.index.ntotal > 0
        and len(adapter.payloads) == adapter.index.ntotal
    )


def _check_bm25_artifact(raw_path: str | None) -> bool:
    """Validate BM25 once per observed artifact snapshot."""
    if not raw_path:
        return False
    try:
        path = Path(raw_path).resolve()
        artifact_stat = path.stat()
    except OSError:
        return False
    return _validate_bm25_snapshot(
        str(path),
        artifact_stat.st_mtime_ns,
        artifact_stat.st_size,
    )


@lru_cache(maxsize=16)
def _validate_bm25_snapshot(
    path: str,
    _mtime_ns: int,
    _size: int,
) -> bool:
    try:
        from udsc2026.retrieval.sparse import BM25Retriever

        retriever = BM25Retriever(path)
        retriever.load()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    return retriever.indexed_chunk_count > 0


def _readiness_checks() -> dict[str, bool]:
    config_path = Path()
    reranker_settings: RerankerSettings | None = None
    hybrid_settings: HybridSettings | None = None
    try:
        config_path = resolve_config_path()
        project_config = load_project_config(config_path)
        if not project_config:
            raise ValueError("Configuration must not be empty")
    except (OSError, UnicodeError, ValueError):
        checks = {"config": False}
        project_config = {}
    else:
        checks = {"config": True}

    embedding_environment_present = any(
        os.getenv(name) is not None for name in _EMBEDDING_ENVIRONMENT_FIELDS
    )
    if "embedding" in project_config or embedding_environment_present:
        try:
            embedding = load_embedding_config(config_path)
            raw_embedding_path = embedding.get("embedder_model_path")
            embedding_path = (
                raw_embedding_path if isinstance(raw_embedding_path, str) else None
            )
        except (OSError, TypeError, ValueError):
            checks["embedding_config"] = False
        else:
            if "embedding" in project_config or any(
                os.getenv(name) is not None
                for name in _EMBEDDING_MODEL_ENVIRONMENT_FIELDS
            ):
                checks["embedding_model"] = _is_nonempty_directory(embedding_path)
            else:
                checks["embedding_model"] = False

    llm_environment_present = any(
        os.getenv(name) is not None for name in _LLM_ENVIRONMENT_FIELDS
    )
    if "llm" in project_config or llm_environment_present:
        try:
            llm = load_llm_config(config_path)
        except (OSError, TypeError, ValueError):
            checks["llm_config"] = False
        else:
            if "llm" in project_config or any(
                os.getenv(name) is not None for name in _LLM_MODEL_ENVIRONMENT_FIELDS
            ):
                checks["llm_model"] = _is_nonempty_directory(llm.model_path)
            else:
                checks["llm_model"] = False

    reranker_environment_present = any(
        os.getenv(name) is not None
        for name in (
            "RERANKER_ENABLED",
            "RERANKER_MODEL_PATH",
            "RERANKER_DEVICE",
            "RERANKER_BATCH_SIZE",
            "RERANKER_MAX_LENGTH",
            "RERANKER_TOP_N",
            "RERANKER_LOCAL_FILES_ONLY",
        )
    )
    if "reranker" in project_config or reranker_environment_present:
        try:
            _parse_optional_boolean("RERANKER_ENABLED")
        except ValueError:
            checks["reranker_enabled"] = False
        else:
            try:
                reranker_settings = load_reranker_settings(config_path)
            except (OSError, TypeError, ValueError):
                checks["reranker_config"] = False
            else:
                if reranker_settings.enabled:
                    checks["reranker_model"] = _is_nonempty_directory(
                        reranker_settings.model_name_or_path
                    )

    vector_environment_present = any(
        os.getenv(name) is not None
        for name in (
            "VECTOR_DB_TYPE",
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "QDRANT_COLLECTION_NAME",
            "FAISS_INDEX_PATH",
            "FAISS_COLLECTION_NAME",
        )
    )
    if "vector_db" in project_config or vector_environment_present:
        try:
            vector_config = resolve_vector_db_config(project_config)
        except (TypeError, ValueError):
            raw_vector_config = project_config.get("vector_db")
            configured_backend = (
                raw_vector_config.get("type")
                if isinstance(raw_vector_config, Mapping)
                else None
            )
            selected_backend = os.getenv("VECTOR_DB_TYPE", configured_backend)
            if isinstance(selected_backend, str):
                normalized_backend = selected_backend.strip().casefold()
            else:
                normalized_backend = ""
            checks[
                (
                    normalized_backend
                    if normalized_backend in {"faiss", "qdrant"}
                    else "vector_db_config"
                )
            ] = False
        else:
            vector_backend = vector_config["type"]
            if vector_backend == "qdrant":
                raw_url = vector_config.get("qdrant_url")
                raw_api_key = vector_config.get("api_key")
                raw_collection = vector_config.get("collection_name")
                checks["qdrant"] = _probe_qdrant(
                    raw_url if isinstance(raw_url, str) else None,
                    raw_api_key if isinstance(raw_api_key, str) else None,
                    raw_collection if isinstance(raw_collection, str) else None,
                )
            elif vector_backend == "faiss":
                raw_root = vector_config.get("faiss_index_path")
                raw_collection = vector_config.get("collection_name")
                checks["faiss"] = _check_faiss_artifacts(
                    raw_root if isinstance(raw_root, str) else None,
                    raw_collection if isinstance(raw_collection, str) else None,
                )
            else:
                checks["vector_db_type"] = False

    if "hybrid" in project_config or os.getenv("BM25_INDEX_PATH") is not None:
        try:
            from udsc2026.retrieval.hybrid.config import load_hybrid_settings

            hybrid_settings = load_hybrid_settings(config_path)
        except (ImportError, OSError, TypeError, ValueError):
            checks["hybrid_config"] = False
        else:
            checks["bm25"] = _check_bm25_artifact(hybrid_settings.bm25_index_path)

    if (
        reranker_settings is not None
        and reranker_settings.enabled
        and hybrid_settings is not None
        and reranker_settings.top_n > hybrid_settings.candidate_k
    ):
        checks["reranker_pipeline_config"] = False

    raw_redis_url = os.getenv("REDIS_URL")
    if raw_redis_url:
        checks["redis"] = _probe_redis(raw_redis_url)
    return checks


def create_app() -> FastAPI:
    """Create API routes without loading models or external services at startup."""
    application = FastAPI(
        title="UDSC2026 Legal RAG",
        version=_package_version(),
        docs_url="/docs",
        redoc_url=None,
    )
    application.state.orchestrator = None
    application.state.orchestrator_factory = _build_orchestrator

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(version=_package_version())

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
        tags=["system"],
    )
    def ready(response: Response) -> ReadinessResponse:
        checks = _readiness_checks()
        missing = [name for name, available in checks.items() if not available]
        if missing:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(
                status="not_ready",
                checks=checks,
                missing=missing,
            )
        return ReadinessResponse(status="ready", checks=checks)

    @application.post(
        "/api/v1/query",
        response_model=QueryResponse,
        tags=["query"],
    )
    @application.post("/query", response_model=QueryResponse, include_in_schema=False)
    async def query(request: QueryRequest) -> QueryResponse:
        """Run the RAG pipeline with lazy, injectable dependencies."""
        orchestrator = cast(RAGOrchestrator | None, application.state.orchestrator)
        if orchestrator is None:
            try:
                factory = cast(
                    Callable[[], RAGOrchestrator],
                    application.state.orchestrator_factory,
                )
                orchestrator = factory()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG pipeline dependencies are unavailable",
                ) from exc
            application.state.orchestrator = orchestrator
        try:
            return await orchestrator.answer(request)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            LOGGER.exception("RAG query failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline could not process the query",
            ) from exc

    return application


app = create_app()


