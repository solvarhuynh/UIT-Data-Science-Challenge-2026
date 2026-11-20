"""Optional Modal Volume commits used by checkpointed production jobs."""

from __future__ import annotations

import asyncio


async def commit_mounted_volume(name: str) -> None:
    """Commit the mounted named Volume from an asynchronous job."""

    if not name.strip():
        raise ValueError("volume name must not be blank")
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("Modal is required to commit its mounted Volume") from exc
    await modal.Volume.from_name(name).commit.aio()


def commit_mounted_volume_sync(name: str) -> None:
    """Commit the mounted named Volume from a synchronous CLI process."""

    asyncio.run(commit_mounted_volume(name))
