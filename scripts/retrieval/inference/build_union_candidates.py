"""Build a lexical--dense candidate union without changing the FAISS index.

The command reads an existing dense candidate file and scans the canonical
chunk archive (or its extracted directory) to build a query-vocabulary BM25
index.  Candidate generation is label-free.  Gold data is deliberately not a
command input.  The resulting manifest records both source hashes so a later
reranker cannot accidentally consume a different pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import tarfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
import sys

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.models import PredictionSample  # noqa: E402

WORD_RE = re.compile(r"(?u)\b\w+\b")
BM25_K1 = 1.2
BM25_B = 0.75
POSTING_POOL_MULTIPLIER = 10
STOPWORDS = {
    "bị", "bằng", "các", "cho", "có", "của", "do", "đã", "được", "để",
    "đến", "hay", "khi", "là", "làm", "này", "nào", "như", "nếu", "người",
    "những", "sau", "sẽ", "tại", "theo", "thì", "trên", "trong", "và", "vào",
    "về", "việc", "với", "từ", "một", "mỗi", "cần", "phải", "quy", "định",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _terms(tokens: list[str], vocabulary: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for length in (1, 2, 3):
        for start in range(len(tokens) - length + 1):
            term = " ".join(tokens[start : start + length])
            if term in vocabulary and (length > 1 or term not in STOPWORDS):
                counts[term] += 1
    return counts


def _question_terms(questions: dict[str, str]) -> tuple[dict[str, Counter[str]], set[str]]:
    by_id: dict[str, Counter[str]] = {}
    vocabulary: set[str] = set()
    for question in questions.values():
        # Build the complete vocabulary first, then count against it.  This
        # avoids retaining n-grams that cannot occur in a query.
        tokens = _tokens(question)
        for length in (1, 2, 3):
            for start in range(len(tokens) - length + 1):
                term = " ".join(tokens[start : start + length])
                if length > 1 or term not in STOPWORDS:
                    vocabulary.add(term)
    for question_id, question in questions.items():
        by_id[question_id] = _terms(_tokens(question), vocabulary)
    return by_id, vocabulary


def _chunk_members(archive: Path | None, root: Path | None) -> Iterator[dict[str, Any]]:
    if root is not None and root.is_dir():
        for path in sorted(root.glob("*.jsonl"), key=lambda item: item.name):
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line.strip():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"invalid chunk JSON at {path}:{line_number}") from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"chunk row must be an object at {path}:{line_number}")
                        yield row
        return
    if archive is None or not archive.is_file():
        raise FileNotFoundError("provide an extracted chunks directory or chunks archive")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = sorted(
            (member for member in tar.getmembers() if member.isfile() and member.name.endswith(".jsonl")),
            key=lambda member: member.name,
        )
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive member {member.name}")
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8")
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid chunk JSON at {archive}:{member.name}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"chunk row must be an object in {member.name}")
                    yield row


def _chunk_member_streams(
    archive: Path | None, root: Path | None
) -> Iterator[tuple[str, Iterator[dict[str, Any]]]]:
    """Yield archive members separately so completed members can be resumed."""
    if root is not None and root.is_dir():
        for path in sorted(root.glob("*.jsonl"), key=lambda item: item.name):
            def rows(path: Path = path) -> Iterator[dict[str, Any]]:
                with path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"invalid chunk JSON at {path}:{line_number}") from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"chunk row must be an object at {path}:{line_number}")
                        yield row
            yield path.name, rows()
        return
    if archive is None or not archive.is_file():
        raise FileNotFoundError("provide an extracted chunks directory or chunks archive")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = sorted(
            (member for member in tar.getmembers() if member.isfile() and member.name.endswith(".jsonl")),
            key=lambda member: member.name,
        )
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive member {member.name}")

            def rows(handle: Any = handle, member_name: str = member.name) -> Iterator[dict[str, Any]]:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.decode("utf-8")
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid chunk JSON at archive member {member_name}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"chunk row must be an object in {member_name}")
                    yield row
            yield member.name, rows()


def _checkpoint_provenance(
    questions: Path,
    dense: Path,
    dense_manifest: Path,
    source: Path,
    vocabulary: set[str],
    dense_k: int,
    lexical_k: int,
    candidate_k: int,
    rrf_k: int,
) -> dict[str, str]:
    source_fingerprint = _sha256(source) if source.is_file() else str(source)
    return {
        "schema": "legalir-union-checkpoint-v2",
        "questions_sha256": _sha256(questions),
        "dense_sha256": _sha256(dense),
        "dense_manifest_sha256": _sha256(dense_manifest),
        "source_fingerprint": source_fingerprint,
        "vocabulary_sha256": hashlib.sha256("\n".join(sorted(vocabulary)).encode("utf-8")).hexdigest(),
        "dense_k": str(dense_k),
        "lexical_k": str(lexical_k),
        "candidate_k": str(candidate_k),
        "rrf_k": str(rrf_k),
    }


def _open_checkpoint(database: Path, provenance: dict[str, str]) -> sqlite3.Connection:
    """Open a compatible checkpoint, discarding stale or incomplete schemas."""
    if database.exists():
        try:
            probe = sqlite3.connect(database)
            existing = dict(probe.execute("SELECT key, value FROM checkpoint_meta"))
            probe.close()
        except sqlite3.DatabaseError:
            database.unlink(missing_ok=True)
        else:
            # The same table also stores resumable phase flags (for example
            # lexical_complete and payload_target_hash).  Compare only the
            # immutable provenance keys; phase flags must survive a restart.
            if any(existing.get(key) != value for key, value in provenance.items()):
                database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS checkpoint_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    existing = dict(connection.execute("SELECT key, value FROM checkpoint_meta"))
    if existing and any(existing.get(key) != value for key, value in provenance.items()):
        connection.close()
        database.unlink(missing_ok=True)
        return _open_checkpoint(database, provenance)
    connection.executemany(
        "INSERT OR REPLACE INTO checkpoint_meta(key, value) VALUES (?, ?)", provenance.items()
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS postings (term TEXT NOT NULL, chunk_id TEXT NOT NULL, tf INTEGER NOT NULL, length INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS indexed_members (name TEXT PRIMARY KEY, chunk_count INTEGER NOT NULL, total_length INTEGER NOT NULL)"
    )
    connection.commit()
    return connection


def _chunk_text(row: dict[str, Any]) -> str:
    parts = [row.get("text"), row.get("law_name")]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(
            metadata.get(field)
            for field in (
                "law_name",
                "chapter",
                "chapter_heading",
                "section",
                "section_heading",
                "article",
                "clause",
                "point",
            )
        )
    return " ".join(str(part) for part in parts if isinstance(part, str) and part.strip())


def _validate_questions(path: Path) -> dict[str, str]:
    payload = _json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty mapping")
    result: dict[str, str] = {}
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not isinstance(record, dict):
            raise ValueError("questions must map string IDs to objects")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has invalid text")
        result[question_id] = question.strip()
    return result


def _load_dense(path: Path, questions: dict[str, str], dense_k: int) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = PredictionSample.model_validate_json(line, strict=True)
            except Exception as exc:
                raise ValueError(f"invalid dense prediction at line {line_number}") from exc
            if sample.question_id not in questions or sample.question_id in rows:
                raise ValueError(f"unexpected or duplicate dense question {sample.question_id!r}")
            if len(sample.hits) < dense_k:
                raise ValueError(f"dense question {sample.question_id!r} has fewer than {dense_k} hits")
            rows[sample.question_id] = [
                hit.model_dump(mode="json", exclude_none=True)
                for hit in sample.hits[:dense_k]
            ]
    if set(rows) != set(questions):
        missing = sorted(set(questions) - set(rows))[:5]
        raise ValueError(f"dense question coverage mismatch, missing={missing}")
    return rows


def _build_lexical_index(
    member_factory: Callable[[], Iterable[tuple[str, Iterator[dict[str, Any]]]]],
    vocabulary: set[str],
    database: Path,
    provenance: dict[str, str],
) -> tuple[dict[str, int], int, int]:
    """Write compact postings and commit each archive member for preemption resume."""
    connection = _open_checkpoint(database, provenance)
    connection.execute("CREATE TABLE IF NOT EXISTS stats (term TEXT PRIMARY KEY, df INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS corpus_stats (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    indexed = {row[0] for row in connection.execute("SELECT name FROM indexed_members")}
    completed = connection.execute(
        "SELECT value FROM checkpoint_meta WHERE key = 'lexical_complete'"
    ).fetchone()
    if completed and completed[0] == "1":
        # This covering index makes bounded per-term top-posting reads seekable
        # when a resumed job skips the corpus build phase.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS postings_term_tf ON postings(term, tf DESC, length, chunk_id)"
        )
        connection.commit()
        stats = dict(connection.execute("SELECT term, df FROM stats"))
        counts = dict(connection.execute("SELECT name, value FROM corpus_stats"))
        connection.close()
        return stats, int(counts["chunk_count"]), int(counts["total_length"])

    connection.execute("DELETE FROM checkpoint_meta WHERE key = 'lexical_complete'")
    connection.commit()
    total_length = sum(int(row[0]) for row in connection.execute("SELECT total_length FROM indexed_members"))
    chunk_count = sum(int(row[0]) for row in connection.execute("SELECT chunk_count FROM indexed_members"))
    for member_name, members in member_factory():
        if member_name in indexed:
            continue
        postings_batch: list[tuple[str, str, int, int]] = []
        document_frequency: Counter[str] = Counter()
        member_chunks = 0
        member_length = 0
        connection.execute("BEGIN")
        for row in members:
            chunk_id = row.get("chunk_id")
            text = row.get("text")
            if not isinstance(chunk_id, str) or not chunk_id.strip() or not isinstance(text, str) or not text.strip():
                connection.rollback()
                raise ValueError("chunk rows require non-empty chunk_id and text")
            token_list = _tokens(_chunk_text(row))
            length = max(1, len(token_list))
            counts = _terms(token_list, vocabulary)
            member_chunks += 1
            member_length += length
            for term, frequency in counts.items():
                postings_batch.append((term, chunk_id, frequency, length))
                document_frequency[term] += 1
            if len(postings_batch) >= 100_000:
                connection.executemany(
                    "INSERT INTO postings(term, chunk_id, tf, length) VALUES (?, ?, ?, ?)",
                    postings_batch,
                )
                postings_batch.clear()
        if postings_batch:
            connection.executemany(
                "INSERT INTO postings(term, chunk_id, tf, length) VALUES (?, ?, ?, ?)",
                postings_batch,
            )
        connection.executemany(
            "INSERT INTO stats(term, df) VALUES (?, ?) ON CONFLICT(term) DO UPDATE SET df = df + excluded.df",
            document_frequency.items(),
        )
        connection.execute(
            "INSERT INTO indexed_members(name, chunk_count, total_length) VALUES (?, ?, ?)",
            (member_name, member_chunks, member_length),
        )
        connection.commit()
        indexed.add(member_name)
        chunk_count += member_chunks
        total_length += member_length
        print(json.dumps({"phase": "lexical_index", "member": member_name, "chunks": chunk_count}, ensure_ascii=False), flush=True)
    if not chunk_count:
        connection.close()
        raise ValueError("chunk corpus is empty")
    connection.execute("CREATE INDEX IF NOT EXISTS postings_term ON postings(term)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS postings_term_tf ON postings(term, tf DESC, length, chunk_id)"
    )
    connection.execute("INSERT OR REPLACE INTO corpus_stats(name, value) VALUES ('chunk_count', ?)", (chunk_count,))
    connection.execute("INSERT OR REPLACE INTO corpus_stats(name, value) VALUES ('total_length', ?)", (total_length,))
    connection.execute("INSERT OR REPLACE INTO checkpoint_meta(key, value) VALUES ('lexical_complete', '1')")
    connection.commit()
    stats = dict(connection.execute("SELECT term, df FROM stats"))
    connection.close()
    return stats, chunk_count, total_length


def _lexical_rankings(
    questions: dict[str, str],
    query_terms: dict[str, Counter[str]],
    database: Path,
    document_frequency: dict[str, int],
    chunk_count: int,
    average_length: float,
    lexical_k: int,
) -> dict[str, list[tuple[str, float, int]]]:
    rankings: dict[str, list[tuple[str, float, int]]] = {}
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS lexical_rankings (question_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    stored = {
        question_id: payload
        for question_id, payload in connection.execute(
            "SELECT question_id, payload FROM lexical_rankings"
        )
    }
    for index, question_id in enumerate(questions, 1):
        if question_id in stored:
            try:
                decoded = json.loads(stored[question_id])
                if isinstance(decoded, list):
                    rankings[question_id] = [
                        (str(item[0]), float(item[1]), int(item[2]))
                        for item in decoded
                        if isinstance(item, list) and len(item) == 3
                    ]
                    if len(rankings[question_id]) <= lexical_k:
                        continue
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        terms = query_terms[question_id]
        scores: dict[str, float] = defaultdict(float)
        posting_limit = max(lexical_k * POSTING_POOL_MULTIPLIER, 2_000)
        for term, qtf in terms.items():
            frequency = document_frequency.get(term)
            if frequency is None:
                continue
            idf = math.log1p((chunk_count - frequency + 0.5) / (frequency + 0.5))
            rows = connection.execute(
                """
                SELECT chunk_id, tf, length
                FROM postings INDEXED BY postings_term_tf
                WHERE term = ?
                ORDER BY tf DESC, length ASC, chunk_id
                LIMIT ?
                """,
                (term, posting_limit),
            )
            for chunk_id, tf, length in rows:
                norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length)
                scores[chunk_id] += float(qtf) * len(term.split()) * idf * (
                    tf * (BM25_K1 + 1.0) / (tf + norm)
                )
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:lexical_k]
        rankings[question_id] = [(chunk_id, score, rank) for rank, (chunk_id, score) in enumerate(ordered, 1)]
        connection.execute(
            "INSERT OR REPLACE INTO lexical_rankings(question_id, payload) VALUES (?, ?)",
            (question_id, json.dumps(rankings[question_id], ensure_ascii=False, separators=(",", ":"))),
        )
        connection.commit()
        if index % 100 == 0 or index == len(questions):
            print(json.dumps({"phase": "lexical_rankings", "completed": index, "total": len(questions)}, ensure_ascii=False), flush=True)
    connection.close()
    return rankings


def _store_chunk_payloads(
    member_factory: Callable[[], Iterable[tuple[str, Iterator[dict[str, Any]]]]],
    wanted: set[str],
    database: Path,
    provenance: dict[str, str],
) -> None:
    connection = _open_checkpoint(database, provenance)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS payloads (chunk_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS payload_members (name TEXT PRIMARY KEY)"
    )
    target_hash = hashlib.sha256("\n".join(sorted(wanted)).encode("utf-8")).hexdigest()
    previous = connection.execute(
        "SELECT value FROM checkpoint_meta WHERE key = 'payload_target_hash'"
    ).fetchone()
    if previous and previous[0] != target_hash:
        connection.execute("DELETE FROM payloads")
        connection.execute("DELETE FROM payload_members")
    connection.execute(
        "INSERT OR REPLACE INTO checkpoint_meta(key, value) VALUES ('payload_target_hash', ?)",
        (target_hash,),
    )
    connection.commit()
    done_members = {row[0] for row in connection.execute("SELECT name FROM payload_members")}
    found_count = int(connection.execute("SELECT COUNT(*) FROM payloads").fetchone()[0])
    for member_name, members in member_factory():
        if member_name in done_members:
            continue
        batch: list[tuple[str, str]] = []
        for row in members:
            chunk_id = row.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id in wanted:
                batch.append((chunk_id, json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO payloads(chunk_id, payload) VALUES (?, ?)", batch
                    )
                    found_count = int(connection.execute("SELECT COUNT(*) FROM payloads").fetchone()[0])
                    connection.commit()
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO payloads(chunk_id, payload) VALUES (?, ?)", batch
            )
        connection.execute("INSERT OR IGNORE INTO payload_members(name) VALUES (?)", (member_name,))
        connection.commit()
        done_members.add(member_name)
        print(json.dumps({"phase": "payloads", "member": member_name, "found": found_count}, ensure_ascii=False), flush=True)
        found_count = int(connection.execute("SELECT COUNT(*) FROM payloads").fetchone()[0])
        if found_count == len(wanted):
            break
    found = {row[0] for row in connection.execute("SELECT chunk_id FROM payloads")}
    connection.commit()
    connection.close()
    missing = wanted - found
    if missing:
        raise ValueError(f"lexical chunk payloads missing: {sorted(missing)[:5]}")


def _hit_from_lexical(row: dict[str, Any], sparse_score: float, rank: int, hybrid_score: float) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "chunk_id": row["chunk_id"],
        "parent_id": row.get("parent_id"),
        "doc_id": str(row["doc_id"]),
        "text": row["text"],
        "source": row.get("source"),
        "law_name": row.get("law_name"),
        "article": row.get("article"),
        "clause": row.get("clause"),
        "metadata": metadata,
        "sparse_score": sparse_score,
        "hybrid_score": hybrid_score,
        "final_score": hybrid_score,
        "rank": rank,
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--chunks-root", type=Path)
    parser.add_argument("--chunks-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-k", type=int, default=500)
    parser.add_argument("--lexical-k", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=700)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


def run(args: argparse.Namespace) -> None:
    if args.dense_k <= 0 or args.lexical_k <= 0 or args.candidate_k < args.dense_k:
        raise ValueError("candidate depths are invalid")
    questions = _validate_questions(args.questions)
    dense_manifest = _json(args.dense_manifest)
    if not isinstance(dense_manifest, dict) or dense_manifest.get("complete") is not True:
        raise ValueError("dense manifest is missing or incomplete")
    if dense_manifest.get("output_sha256") != _sha256(args.dense):
        raise ValueError("dense output does not match its manifest")
    if dense_manifest.get("question_source_sha256") != _sha256(args.questions):
        raise ValueError("dense output and question mapping differ")
    dense_k = min(args.dense_k, int(dense_manifest.get("candidate_k", args.dense_k)))
    dense = _load_dense(args.dense, questions, dense_k)
    query_terms, vocabulary = _question_terms(questions)
    database = args.output.with_suffix(".lexical.sqlite")
    source = args.chunks_archive or args.chunks_root
    if source is None:
        raise ValueError("one of --chunks-archive or --chunks-root is required")
    provenance = _checkpoint_provenance(
        args.questions,
        args.dense,
        args.dense_manifest,
        source,
        vocabulary,
        dense_k,
        args.lexical_k,
        args.candidate_k,
        args.rrf_k,
    )
    print(json.dumps({"phase": "startup", "checkpoint": str(database), "resume": database.exists()}), flush=True)
    try:
        document_frequency, chunk_count, total_length = _build_lexical_index(
            lambda: _chunk_member_streams(args.chunks_archive, args.chunks_root),
            vocabulary,
            database,
            provenance,
        )
        lexical = _lexical_rankings(
            questions,
            query_terms,
            database,
            document_frequency,
            chunk_count,
            total_length / chunk_count,
            args.lexical_k,
        )
        wanted = {chunk_id for ranking in lexical.values() for chunk_id, _, _ in ranking}
        _store_chunk_payloads(
            lambda: _chunk_member_streams(args.chunks_archive, args.chunks_root),
            wanted,
            database,
            provenance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        payload_connection = sqlite3.connect(database)
        try:
            completed_ids: set[str] = set()
            if partial.exists():
                valid_lines: list[str] = []
                with partial.open("r", encoding="utf-8") as existing:
                    for line in existing:
                        if not line.strip():
                            continue
                        try:
                            sample = PredictionSample.model_validate_json(line, strict=True)
                        except Exception:
                            break
                        if sample.question_id in questions and sample.question_id not in completed_ids:
                            completed_ids.add(sample.question_id)
                            valid_lines.append(sample.model_dump_json(exclude_none=True, by_alias=False) + "\n")
                partial.write_text("".join(valid_lines), encoding="utf-8")
            with partial.open("a", encoding="utf-8") as stream:
                for index, question_id in enumerate(questions, 1):
                    if question_id in completed_ids:
                        continue
                    dense_hits = dense[question_id]
                    dense_rank = {hit["chunk_id"]: rank for rank, hit in enumerate(dense_hits, 1)}
                    lexical_items = lexical[question_id]
                    lexical_rank = {chunk_id: rank for chunk_id, _, rank in lexical_items}
                    lexical_score = {chunk_id: score for chunk_id, score, _ in lexical_items}
                    candidate_ids = set(dense_rank) | set(lexical_rank)
                    combined: list[tuple[str, float]] = []
                    for chunk_id in candidate_ids:
                        score = (1.0 / (args.rrf_k + dense_rank[chunk_id]) if chunk_id in dense_rank else 0.0)
                        score += (1.0 / (args.rrf_k + lexical_rank[chunk_id]) if chunk_id in lexical_rank else 0.0)
                        combined.append((chunk_id, score))
                    combined.sort(key=lambda item: (-item[1], item[0]))
                    selected = combined[:args.candidate_k]
                    dense_by_id = {hit["chunk_id"]: hit for hit in dense_hits}
                    lexical_only = [chunk_id for chunk_id, _ in selected if chunk_id not in dense_by_id]
                    payloads: dict[str, dict[str, Any]] = {}
                    if lexical_only:
                        placeholders = ",".join("?" for _ in lexical_only)
                        rows = payload_connection.execute(
                            f"SELECT chunk_id, payload FROM payloads WHERE chunk_id IN ({placeholders})",
                            tuple(lexical_only),
                        )
                        payloads = {
                            chunk_id: json.loads(payload)
                            for chunk_id, payload in rows
                        }
                    hits: list[dict[str, Any]] = []
                    for rank, (chunk_id, hybrid_score) in enumerate(selected, 1):
                        if chunk_id in dense_by_id:
                            hit = dict(dense_by_id[chunk_id])
                            hit["hybrid_score"] = hybrid_score
                            hit["final_score"] = hybrid_score
                            hit["rank"] = rank
                            if chunk_id in lexical_score:
                                hit["sparse_score"] = lexical_score[chunk_id]
                        else:
                            row = payloads.get(chunk_id)
                            if row is None:
                                raise ValueError(f"lexical chunk payload missing: {chunk_id}")
                            hit = _hit_from_lexical(row, lexical_score[chunk_id], rank, hybrid_score)
                        hits.append(hit)
                    sample = PredictionSample(question_id=question_id, hits=hits)
                    stream.write(sample.model_dump_json(exclude_none=True, by_alias=False) + "\n")
                    if index % 100 == 0:
                        stream.flush()
                        print(json.dumps({"phase": "output", "completed": len(completed_ids) + 1, "total": len(questions)}, ensure_ascii=False), flush=True)
                    completed_ids.add(question_id)
            payload_connection.close()
            if len(completed_ids) != len(questions):
                raise ValueError(f"partial output coverage mismatch: {len(completed_ids)}/{len(questions)}")
            os.replace(partial, args.output)
        finally:
            payload_connection.close()
        _write_manifest(
            args.manifest,
            {
                "schema_version": "legalir-union-candidate-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete": True,
                "output": str(args.output),
                "output_sha256": _sha256(args.output),
                "question_source": str(args.questions),
                "question_source_sha256": _sha256(args.questions),
                "question_count": len(questions),
                "model_id": "AITeamVN/Vietnamese_Embedding_v2",
                "dense_input": str(args.dense),
                "dense_input_sha256": _sha256(args.dense),
                "dense_manifest_sha256": _sha256(args.dense_manifest),
                "dense_candidate_k": dense_k,
                "lexical_candidate_k": args.lexical_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "lexical_method": "query-vocabulary-bm25-bounded-postings",
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "posting_pool_k": max(args.lexical_k * POSTING_POOL_MULTIPLIER, 2_000),
                "chunk_count": chunk_count,
                "indexed_query_term_count": len(vocabulary),
                "indexed_posting_term_count": len(document_frequency),
                "source_archive": str(args.chunks_archive) if args.chunks_archive else None,
                "source_archive_sha256": _sha256(args.chunks_archive) if args.chunks_archive else None,
            },
        )
        print(json.dumps({"questions": len(questions), "chunks": chunk_count, "output": str(args.output)}))
        database.unlink(missing_ok=True)
        for sidecar in (database.with_name(database.name + "-wal"), database.with_name(database.name + "-shm")):
            sidecar.unlink(missing_ok=True)
    finally:
        # Keep the SQLite checkpoint and .partial output on interruption so a
        # Modal preemption resumes safely.  Successful runs clean them above.
        pass


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except (OSError, TypeError, ValueError, tarfile.TarError) as exc:
        raise SystemExit(f"union candidate error: {exc}")



