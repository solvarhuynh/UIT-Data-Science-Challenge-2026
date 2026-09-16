"""Pure-Python durable checkpoint protocol for FULLDOC TOP200 Qwen shards."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "full_doc_qwen_checkpoint_v1"


class CheckpointError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DurableShardStore:
    """Generation-based store; only rows at/below durable_generation count."""

    def __init__(self, path: Path, identity: dict[str, str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(self.path)
            integrity = str(self.db.execute("PRAGMA integrity_check").fetchone()[0])
        except sqlite3.DatabaseError as exc:
            if hasattr(self, "db"):
                self.db.close()
            raise CheckpointError("corrupted checkpoint") from exc
        if integrity != "ok":
            raise CheckpointError(f"corrupted checkpoint: {integrity}")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunk_scores(
              query_id TEXT NOT NULL, document_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
              score REAL NOT NULL, generation INTEGER NOT NULL,
              PRIMARY KEY(query_id,document_id,chunk_id));
            CREATE TABLE IF NOT EXISTS document_results(
              query_id TEXT NOT NULL, document_id TEXT NOT NULL, score REAL NOT NULL,
              chunk_count INTEGER NOT NULL, generation INTEGER NOT NULL,
              PRIMARY KEY(query_id,document_id));
            """
        )
        current = dict(self.db.execute("SELECT key,value FROM metadata"))
        required = {"schema_version": SCHEMA_VERSION, **identity}
        if current:
            for key, value in required.items():
                if current.get(key) != value:
                    self.db.close()
                    raise CheckpointError(f"checkpoint identity mismatch: {key}")
        else:
            self.db.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", list(required.items()) + [("durable_generation", "0")])
            self.db.commit()
        self.durable_generation = int(dict(self.db.execute("SELECT key,value FROM metadata"))["durable_generation"])
        # Rows beyond the last two-phase durable boundary are untrusted.
        self.db.execute("DELETE FROM chunk_scores WHERE generation > ?", (self.durable_generation,))
        self.db.execute("DELETE FROM document_results WHERE generation > ?", (self.durable_generation,))
        self.db.commit()

    def completed_identities(self) -> set[tuple[str, str]]:
        return {
            (str(query), str(doc))
            for query, doc in self.db.execute(
                "SELECT query_id,document_id FROM document_results WHERE generation <= ?",
                (self.durable_generation,),
            )
        }

    def commit_qdocs(
        self,
        qdocs: list[tuple[str, str, list[tuple[str, float]]]],
        volume_commit: Callable[[], None],
    ) -> int:
        if not qdocs:
            return self.durable_generation
        generation = self.durable_generation + 1
        try:
            for query, doc, chunks in qdocs:
                if not chunks or len({chunk for chunk, _ in chunks}) != len(chunks):
                    raise CheckpointError(f"invalid chunk identities: {query}/{doc}")
                if not all(math.isfinite(float(score)) for _, score in chunks):
                    raise CheckpointError(f"non-finite score: {query}/{doc}")
                self.db.executemany(
                    "INSERT INTO chunk_scores VALUES (?,?,?,?,?) ON CONFLICT(query_id,document_id,chunk_id) DO UPDATE SET score=excluded.score,generation=excluded.generation",
                    [(query, doc, chunk, float(score), generation) for chunk, score in chunks],
                )
                self.db.execute(
                    "INSERT INTO document_results VALUES (?,?,?,?,?) ON CONFLICT(query_id,document_id) DO UPDATE SET score=excluded.score,chunk_count=excluded.chunk_count,generation=excluded.generation",
                    (query, doc, max(float(score) for _, score in chunks), len(chunks), generation),
                )
            self.db.commit()
            volume_commit()  # result rows survive a Volume boundary first
            self.db.execute("UPDATE metadata SET value=? WHERE key='durable_generation'", (str(generation),))
            self.db.commit()
            volume_commit()  # durability marker itself must also survive
        except Exception:
            self.db.rollback()
            raise
        self.durable_generation = generation
        return generation

    def validate_complete(self, expected: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
        rows = list(self.db.execute(
            """SELECT d.query_id,d.document_id,d.score,d.chunk_count,
                      COUNT(c.chunk_id),MAX(c.score)
                 FROM document_results d
                 LEFT JOIN chunk_scores c
                   ON c.query_id=d.query_id AND c.document_id=d.document_id
                  AND c.generation <= ?
                WHERE d.generation <= ?
                GROUP BY d.query_id,d.document_id,d.score,d.chunk_count
                ORDER BY d.query_id,d.document_id""",
            (self.durable_generation, self.durable_generation),
        ))
        actual = {(str(q), str(d)): int(count) for q, d, _, count, _, _ in rows}
        if actual != expected:
            raise CheckpointError("final q-doc coverage/count mismatch")
        if not all(
            math.isfinite(float(score)) and int(count) == int(persisted_count)
            and persisted_max is not None and float(score) == float(persisted_max)
            for _, _, score, count, persisted_count, persisted_max in rows
        ):
            raise CheckpointError("chunk accounting/MAX aggregation mismatch")
        return [{"query_id": str(q), "document_id": str(d), "score": float(score), "chunk_count": int(count)} for q, d, score, count, _, _ in rows]

    def close(self) -> None:
        self.db.close()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return sha256_file(path)


def run_self_tests() -> dict[str, Any]:
    commits = 0
    def commit() -> None:
        nonlocal commits
        commits += 1
    identity = {"contract_sha256": "contract", "universe_sha256": "universe", "shard_sha256": "shard", "shard_id": "00"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); db = root / "checkpoint.sqlite3"
        first = DurableShardStore(db, identity)
        first.commit_qdocs([("q1", "d1", [("c1", .1), ("c2", .2)])], commit)
        # Simulate a transaction that exists locally but never crossed the durable marker.
        first.db.execute("INSERT INTO chunk_scores VALUES ('q2','d2','c3',.3,2)")
        first.db.commit(); first.close()
        resumed = DurableShardStore(db, identity)
        skipped = resumed.completed_identities()
        if skipped != {("q1", "d1")}:
            raise AssertionError(f"resume skip mismatch: {skipped}")
        resumed.commit_qdocs([("q2", "d2", [("c3", .3)])], commit)
        rows = resumed.validate_complete({("q1", "d1"): 2, ("q2", "d2"): 1})
        duplicate_count = len(rows) - len({(row["query_id"], row["document_id"]) for row in rows})
        resumed.close()
        rejected_contract = rejected_universe = rejected_corrupt = False
        try:
            DurableShardStore(db, {**identity, "contract_sha256": "wrong"})
        except CheckpointError:
            rejected_contract = True
        try:
            DurableShardStore(db, {**identity, "universe_sha256": "wrong"})
        except CheckpointError:
            rejected_universe = True
        corrupt = root / "corrupt.sqlite3"; corrupt.write_bytes(b"not sqlite")
        try:
            DurableShardStore(corrupt, identity)
        except CheckpointError:
            rejected_corrupt = True
        if duplicate_count or not (rejected_contract and rejected_universe and rejected_corrupt) or commits != 4:
            raise AssertionError("durability self-test failed")
        return {"status": "PASS", "durable_commits": commits, "resumed_skipped": 1, "incomplete_resumed": 1, "duplicates": duplicate_count, "missing": 0, "wrong_contract_rejected": rejected_contract, "wrong_universe_rejected": rejected_universe, "corrupt_rejected": rejected_corrupt}


if __name__ == "__main__":
    print(json.dumps(run_self_tests(), indent=2))
