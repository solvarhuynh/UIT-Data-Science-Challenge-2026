"""CPU-only fresh-container test of the production durability module."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import modal

from task1_full_doc_top200_durability import DurableShardStore, sha256_file


MOUNT = Path("/workspace/p13")
ROOT = MOUNT / "runtime/full_doc_top200_qwen/modal_resume_test"
VOLUME = modal.Volume.from_name("udsc-p13", create_if_missing=False)
APP = modal.App("task1-full-doc-top200-modal-resume-test")
LOCAL_DURABILITY = Path(__file__).with_name("task1_full_doc_top200_durability.py")
IMAGE = modal.Image.debian_slim(python_version="3.12").add_local_file(
    str(LOCAL_DURABILITY), remote_path="/root/task1_full_doc_top200_durability.py"
)
IDENTITY = {
    "contract_sha256": "modal-resume-test-contract-v1",
    "universe_sha256": "modal-resume-test-universe-v1",
    "shard_sha256": "modal-resume-test-shard-v1",
    "shard_id": "00",
    "run_kind": "modal_resume_test",
}


def marker() -> dict[str, str | int]:
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def test_root(test_id: str) -> Path:
    if not test_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in test_id):
        raise ValueError("invalid test_id")
    return ROOT / test_id


@APP.function(image=IMAGE, volumes={"/workspace/p13": VOLUME}, timeout=600, retries=0)
def phase_a(test_id: str) -> dict:
    root = test_root(test_id); root.mkdir(parents=True, exist_ok=False)
    store = DurableShardStore(root / "checkpoint.sqlite3", IDENTITY)
    store.commit_qdocs([("q1", "d1", [("c1", 0.25), ("c2", 0.75)])], VOLUME.commit)
    # This newer row lacks a durable-generation marker and must be discarded by B.
    store.db.execute("INSERT INTO chunk_scores VALUES ('q2','d2','c3',0.5,2)")
    store.db.commit(); store.close()
    payload = {"test_id": test_id, "schema": "full_doc_qwen_checkpoint_v1", "phase": "A", "marker": marker()}
    path = root / "phase_a.json"; path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return {"test_id": test_id, "path": str(root), "phase_a_sha256": sha256_file(path), "marker": marker()}


@APP.function(image=IMAGE, volumes={"/workspace/p13": VOLUME}, timeout=600, retries=0)
def phase_b(test_id: str) -> dict:
    VOLUME.reload(); root = test_root(test_id)
    phase_a_path = root / "phase_a.json"
    if not phase_a_path.is_file():
        raise RuntimeError("phase A payload absent after fresh-container reload")
    store = DurableShardStore(root / "checkpoint.sqlite3", IDENTITY)
    skipped = store.completed_identities()
    if skipped != {("q1", "d1")}:
        raise RuntimeError(f"durable completed identity mismatch: {skipped}")
    store.commit_qdocs([("q2", "d2", [("c3", 0.5)])], VOLUME.commit)
    rows = store.validate_complete({("q1", "d1"): 2, ("q2", "d2"): 1})
    store.close()
    duplicates = len(rows) - len({(row["query_id"], row["document_id"]) for row in rows})
    payload = {"test_id": test_id, "schema": "full_doc_qwen_checkpoint_v1", "phase": "B", "marker": marker(), "rows": rows}
    path = root / "phase_b.json"; path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return {"test_id": test_id, "phase_b_sha256": sha256_file(path), "completed_skipped": len(skipped), "incomplete_resumed": 1, "duplicates": duplicates, "missing": 0, "marker": marker()}


@APP.local_entrypoint()
def main(mode: str, test_id: str) -> None:
    if mode == "a": result = phase_a.remote(test_id)
    elif mode == "b": result = phase_b.remote(test_id)
    else: raise ValueError("mode must be a or b")
    print("MODAL_RESUME_TEST_RESULT=" + json.dumps(result, sort_keys=True))
