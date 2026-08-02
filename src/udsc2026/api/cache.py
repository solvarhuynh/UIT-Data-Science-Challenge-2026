"""Small async cache adapters used by the API orchestration layer."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from time import monotonic
from urllib.parse import unquote, urlsplit

LOGGER = logging.getLogger(__name__)


class CacheClient:
    """Protocol-like base class for caches returning JSON-compatible values."""

    async def get(self, key: str) -> dict[str, object] | None:
        """Return a cached payload or ``None`` when it is unavailable."""
        raise NotImplementedError

    async def set(
        self, key: str, value: Mapping[str, object], ttl_seconds: int
    ) -> None:
        """Store a payload for a positive time-to-live period."""
        raise NotImplementedError


class InMemoryCache(CacheClient):
    """Concurrency-safe process-local fallback for development and tests."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._items: dict[str, tuple[float, dict[str, object]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> dict[str, object] | None:
        """Return an unexpired cached payload."""
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= monotonic():
                del self._items[key]
                return None
            return dict(value)

    async def set(
        self, key: str, value: Mapping[str, object], ttl_seconds: int
    ) -> None:
        """Store a copy of a payload with a positive TTL."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        async with self._lock:
            self._items[key] = (monotonic() + ttl_seconds, dict(value))


class RedisCache(CacheClient):
    """Minimal Redis RESP cache adapter with no additional runtime dependency."""

    def __init__(self, url: str, timeout_seconds: float = 1.0) -> None:
        """Validate the Redis URL used for each short-lived cache operation."""
        parsed = urlsplit(url)
        if parsed.scheme != "redis" or not parsed.hostname:
            raise ValueError("Redis cache requires a redis:// URL")
        self._host = parsed.hostname
        self._port = parsed.port or 6379
        self._password = unquote(parsed.password) if parsed.password else None
        self._database = parsed.path.lstrip("/") or "0"
        if not self._database.isdecimal():
            raise ValueError("Redis database must be a non-negative integer")
        self._timeout_seconds = timeout_seconds

    async def get(self, key: str) -> dict[str, object] | None:
        """Fetch and decode a JSON cache value."""
        reply = await self._command("GET", key)
        if reply is None:
            return None
        if not isinstance(reply, bytes):
            raise RuntimeError("Unexpected Redis GET response")
        value = json.loads(reply.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("Cached Redis value must be an object")
        return value

    async def set(
        self, key: str, value: Mapping[str, object], ttl_seconds: int
    ) -> None:
        """Store a JSON payload with Redis EX expiry."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        reply = await self._command("SET", key, payload, "EX", str(ttl_seconds))
        if reply != b"OK":
            raise RuntimeError("Redis SET did not return OK")

    async def _command(self, *parts: str) -> bytes | None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), self._timeout_seconds
        )
        try:
            if self._password is not None:
                await self._write_command(writer, "AUTH", self._password)
                if await self._read_reply(reader) != b"OK":
                    raise RuntimeError("Redis authentication failed")
            if self._database != "0":
                await self._write_command(writer, "SELECT", self._database)
                if await self._read_reply(reader) != b"OK":
                    raise RuntimeError("Redis database selection failed")
            await self._write_command(writer, *parts)
            return await self._read_reply(reader)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _write_command(self, writer: asyncio.StreamWriter, *parts: str) -> None:
        encoded = [part.encode("utf-8") for part in parts]
        command = [f"*{len(encoded)}\r\n".encode("ascii")]
        for part in encoded:
            command.extend((f"${len(part)}\r\n".encode("ascii"), part, b"\r\n"))
        writer.write(b"".join(command))
        await writer.drain()

    async def _read_reply(self, reader: asyncio.StreamReader) -> bytes | None:
        line = await asyncio.wait_for(reader.readline(), self._timeout_seconds)
        if not line:
            raise RuntimeError("Redis closed the connection")
        if line.startswith(b"+"):
            return line[1:-2]
        if line.startswith(b"$"):
            size = int(line[1:-2])
            if size == -1:
                return None
            data = await reader.readexactly(size + 2)
            return data[:-2]
        if line.startswith(b"-"):
            raise RuntimeError(line[1:-2].decode("utf-8", errors="replace"))
        raise RuntimeError("Unsupported Redis response")


class FallbackCache(CacheClient):
    """Prefer a remote cache while preserving service availability on failures."""

    def __init__(self, primary: CacheClient, fallback: CacheClient) -> None:
        """Initialize primary and local fallback cache adapters."""
        self._primary = primary
        self._fallback = fallback

    async def get(self, key: str) -> dict[str, object] | None:
        """Read from primary, then local fallback if the primary is unavailable."""
        try:
            value = await self._primary.get(key)
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            LOGGER.warning("Remote cache get failed; using in-memory fallback: %s", exc)
            return await self._fallback.get(key)
        return value if value is not None else await self._fallback.get(key)

    async def set(
        self, key: str, value: Mapping[str, object], ttl_seconds: int
    ) -> None:
        """Write local cache regardless of the primary cache's availability."""
        await self._fallback.set(key, value, ttl_seconds)
        try:
            await self._primary.set(key, value, ttl_seconds)
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            LOGGER.warning("Remote cache set failed; retained in-memory entry: %s", exc)
