"""Non-sensitive two-invocation Modal Volume durability probe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

MOUNT = Path("/workspace/p13")
PATH = MOUNT / "runtime/full_doc_top200_qwen/volume_test/payload.json"
volume = modal.Volume.from_name("udsc-p13", create_if_missing=False)
app = modal.App("task1-full-doc-top200-volume-test")
image = modal.Image.debian_slim(python_version="3.12")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, retries=0)
def write() -> dict[str, str]:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = PATH.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": "volume_test_v1", "content": "durable-cross-container-test"}, handle, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, PATH); value = digest(PATH); volume.commit()
    return {"status": "WRITTEN_AND_COMMITTED", "sha256": value}


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, retries=0)
def verify(expected_sha256: str) -> dict[str, str]:
    volume.reload()
    actual = digest(PATH) if PATH.is_file() else "MISSING"
    if actual != expected_sha256:
        raise RuntimeError(f"Volume durability mismatch: {actual}")
    return {"status": "PASS", "sha256": actual}


@app.local_entrypoint()
def main(mode: str, expected_sha256: str = "") -> None:
    result = write.remote() if mode == "write" else verify.remote(expected_sha256) if mode == "verify" else None
    if result is None:
        raise ValueError("mode must be write or verify")
    print("VOLUME_TEST_RESULT=" + json.dumps(result, sort_keys=True))
