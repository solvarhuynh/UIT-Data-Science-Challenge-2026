"""Deterministic, citation-grounded synthetic benchmark generation for Legal RAG."""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from tempfile import mkstemp
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from udsc2026.contracts import LegalChunk

QUESTION_TYPES = (
    "definition",
    "condition",
    "rights_obligations",
    "penalty",
    "procedure",
    "comparison",
    "multi_clause",
)
_SINGLE_TYPES = QUESTION_TYPES[:5]
_TYPE_PATTERNS = {
    "definition": re.compile(r"\b(là|được hiểu|khái niệm|định nghĩa)\b", re.I),
    "condition": re.compile(r"\b(khi|nếu|trường hợp|điều kiện)\b", re.I),
    "rights_obligations": re.compile(
        r"\b(quyền|nghĩa vụ|trách nhiệm|phải|có trách nhiệm)\b", re.I
    ),
    "penalty": re.compile(r"\b(phạt|mức phạt|xử phạt|chế tài|tiền phạt)\b", re.I),
    "procedure": re.compile(r"\b(thủ tục|trình tự|hồ sơ|nộp|thực hiện)\b", re.I),
}
_SINGLE_QUESTIONS = {
    "definition": "Khái niệm hoặc nội dung được định nghĩa tại {citation} là gì?",
    "condition": "Điều kiện hoặc trường hợp áp dụng nêu tại {citation} là gì?",
    "rights_obligations": "Quyền, nghĩa vụ hoặc trách nhiệm tại {citation} là gì?",
    "penalty": "Mức phạt hoặc chế tài được nêu tại {citation} là gì?",
    "procedure": "Thủ tục hoặc yêu cầu thực hiện tại {citation} là gì?",
}
_PROMPT_SUFFIXES = (
    "Trả lời ngắn gọn theo nội dung gốc.",
    "Chỉ nêu thông tin có trong căn cứ được dẫn.",
    "Tóm tắt đúng trọng tâm quy định.",
    "Không suy diễn ngoài văn bản nguồn.",
)


class SyntheticQA(BaseModel):
    """One retrieval-evaluation question grounded in one or more LegalChunks."""

    question_id: str
    question: str
    answer: str
    gold_chunk_ids: List[str] = Field(min_length=1)
    gold_citations: List[str] = Field(min_length=1)
    law_name: str
    article: str
    difficulty: str
    question_type: str
    metadata: Dict[str, object] = Field(default_factory=dict)


class SyntheticBenchmarkArtifact(BaseModel):
    """Auditable output produced by one bounded-memory benchmark build."""

    output_path: str
    record_count: int
    benchmark_sha256: str
    source_chunk_count: int
    source_chunk_corpus_sha256: str


class _CandidateReservoir:
    """Keep a deterministic bounded sample for each benchmark question type."""

    def __init__(self, capacity: int, randomizer: random.Random) -> None:
        self._capacity = capacity
        self._randomizer = randomizer
        self.values: Dict[str, List[Tuple[LegalChunk, ...]]] = {
            kind: [] for kind in QUESTION_TYPES
        }
        self._seen = {kind: 0 for kind in QUESTION_TYPES}

    def offer(self, kind: str, candidate: Tuple[LegalChunk, ...]) -> None:
        self._seen[kind] += 1
        values = self.values[kind]
        if len(values) < self._capacity:
            values.append(_compact_candidate(candidate))
            return
        replacement = self._randomizer.randrange(self._seen[kind])
        if replacement < self._capacity:
            values[replacement] = _compact_candidate(candidate)


class _ChunkCorpusFingerprint:
    """Strong ordered hash over a deterministic JSONL chunk stream."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0

    def update(self, chunk: LegalChunk) -> None:
        payload = _chunk_identity_payload(chunk)
        self._digest.update(len(payload).to_bytes(8, byteorder="big"))
        self._digest.update(payload)
        self.count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def generate_synthetic_benchmark(
    chunks: Iterable[LegalChunk],
    target_count: int = 100,
    seed: int = 2026,
    require_all_question_types: bool = True,
) -> List[SyntheticQA]:
    """Generate 100--200 deterministic Q&A records from citation-ready chunks.

    Each answer is an extractive, normalised source excerpt. No LLM is used and
    no fact is introduced beyond its gold chunk(s), making the benchmark safe
    for retrieval, reranking, and citation evaluation.
    """
    if not 100 <= target_count <= 200:
        raise ValueError("target_count must be between 100 and 200")
    # Determinism is required for benchmark reproducibility, not for security.
    randomizer = random.Random(seed)  # nosec B311
    reservoir = _CandidateReservoir(target_count, randomizer)
    fingerprint = _ChunkCorpusFingerprint()
    _collect_candidates(chunks, reservoir, fingerprint)
    candidates = reservoir.values
    if require_all_question_types:
        missing = [kind for kind in QUESTION_TYPES if not candidates[kind]]
        if missing:
            raise ValueError(
                "Corpus lacks grounded chunks for question types: {0}".format(
                    ", ".join(missing)
                )
            )
    available_types = [kind for kind in QUESTION_TYPES if candidates[kind]]
    if not available_types:
        raise ValueError("No chunks with text, law_name, and article are available")

    for kind in available_types:
        randomizer.shuffle(candidates[kind])

    records: List[SyntheticQA] = []
    positions = {kind: 0 for kind in available_types}
    for index in range(target_count):
        kind = available_types[index % len(available_types)]
        values = candidates[kind]
        candidate = values[positions[kind] % len(values)]
        positions[kind] += 1
        records.append(
            _to_record(
                index + 1,
                kind,
                candidate,
                positions[kind],
                source_chunk_count=fingerprint.count,
                source_chunk_corpus_sha256=fingerprint.hexdigest(),
            )
        )
    return records


def iter_legal_chunks(path: Union[str, Path]) -> Iterator[LegalChunk]:
    """Yield validated chunks from a JSONL file or directory in stable order."""

    source = Path(path)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
    if not files:
        raise ValueError("No LegalChunk JSONL files found at {0}".format(source))
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    yield LegalChunk.model_validate_json(line)
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "Invalid LegalChunk JSONL at {0}:{1}".format(
                            file_path, line_number
                        )
                    ) from error


def load_legal_chunks(path: Union[str, Path]) -> List[LegalChunk]:
    """Compatibility loader for tests and deliberately small corpora."""

    return list(iter_legal_chunks(path))


def write_synthetic_benchmark(
    records: Iterable[SyntheticQA], output_path: Union[str, Path]
) -> Path:
    """Atomically write one benchmark record per UTF-8 JSONL line."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(
        prefix=".{0}.".format(target.name), suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record.model_dump(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _is_usable(chunk: LegalChunk) -> bool:
    return bool(chunk.text.strip() and chunk.law_name and chunk.article)


def _collect_candidates(
    chunks: Iterable[LegalChunk],
    reservoir: _CandidateReservoir,
    fingerprint: _ChunkCorpusFingerprint,
) -> None:
    """Scan once while retaining at most ``target_count`` candidates per type."""

    current_group: Tuple[str, ...] | None = None
    distinct_clauses: Dict[str, LegalChunk] = {}
    for chunk in chunks:
        fingerprint.update(chunk)
        if not _is_usable(chunk):
            _offer_article_pair(distinct_clauses, reservoir)
            current_group = None
            distinct_clauses = {}
            continue

        for kind in _SINGLE_TYPES:
            if _TYPE_PATTERNS[kind].search(chunk.text):
                reservoir.offer(kind, (chunk,))

        group = _article_group(chunk)
        if current_group is not None and group != current_group:
            _offer_article_pair(distinct_clauses, reservoir)
            distinct_clauses = {}
        current_group = group
        clause_key = chunk.clause or chunk.chunk_id
        distinct_clauses.setdefault(clause_key, _compact_chunk(chunk))

    _offer_article_pair(distinct_clauses, reservoir)


def _article_group(chunk: LegalChunk) -> Tuple[str, ...]:
    if chunk.parent_id:
        return ("parent", chunk.parent_id)
    return ("article", chunk.law_name or "", chunk.article or "")


def _offer_article_pair(
    distinct_clauses: Dict[str, LegalChunk], reservoir: _CandidateReservoir
) -> None:
    values = list(distinct_clauses.values())
    if len(values) < 2:
        return
    pair = (values[0], values[1])
    reservoir.offer("comparison", pair)
    reservoir.offer("multi_clause", pair)


def _compact_chunk(chunk: LegalChunk) -> LegalChunk:
    """Never retain the legacy repeated parent body in a candidate reservoir."""

    return (
        chunk
        if chunk.parent_text is None
        else chunk.model_copy(update={"parent_text": None})
    )


def _compact_candidate(
    candidate: Tuple[LegalChunk, ...],
) -> Tuple[LegalChunk, ...]:
    return tuple(_compact_chunk(chunk) for chunk in candidate)


def _to_record(
    index: int,
    question_type: str,
    chunks: Tuple[LegalChunk, ...],
    variation: int,
    *,
    source_chunk_count: int,
    source_chunk_corpus_sha256: str,
) -> SyntheticQA:
    first = chunks[0]
    citations = [_citation(chunk) for chunk in chunks]
    suffix = _PROMPT_SUFFIXES[(variation - 1) % len(_PROMPT_SUFFIXES)]
    if question_type == "comparison":
        question = "So sánh nội dung giữa {0} và {1}. {2}".format(
            citations[0], citations[1], suffix
        )
        difficulty = "hard"
    elif question_type == "multi_clause":
        question = "Tổng hợp các quy định tại {0} và {1}. {2}".format(
            citations[0], citations[1], suffix
        )
        difficulty = "hard"
    else:
        question = "{0} {1}".format(
            _SINGLE_QUESTIONS[question_type].format(citation=citations[0]), suffix
        )
        difficulty = "easy" if question_type == "definition" else "medium"
    answer = _format_answer(question_type, citations, chunks)
    return SyntheticQA(
        question_id="syn_{0:04d}".format(index),
        question=question,
        answer=answer,
        gold_chunk_ids=[chunk.chunk_id for chunk in chunks],
        gold_citations=citations,
        law_name=first.law_name or "",
        article=first.article or "",
        difficulty=difficulty,
        question_type=question_type,
        metadata={
            "schema_version": "synthetic-qa-v2",
            "gold_chunk_sha256": [chunk_content_sha256(chunk) for chunk in chunks],
            "source_chunk_count": source_chunk_count,
            "source_chunk_corpus_sha256": source_chunk_corpus_sha256,
        },
    )


def _chunk_identity_payload(chunk: LegalChunk) -> bytes:
    return json.dumps(
        {
            "schema_version": "benchmark-chunk-identity-v1",
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "law_name": chunk.law_name,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "article": chunk.article,
            "clause": chunk.clause,
            "point": chunk.point,
            "effective_date": chunk.effective_date,
            "source": chunk.source,
            "metadata": chunk.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def chunk_content_sha256(chunk: LegalChunk) -> str:
    """Bind a gold ID to exact text, citation fields, source, and metadata."""

    return hashlib.sha256(_chunk_identity_payload(chunk)).hexdigest()


def _citation(chunk: LegalChunk) -> str:
    values = [chunk.law_name, chunk.article, chunk.clause, chunk.point]
    return ", ".join(value for value in values if value)


def _format_answer(
    question_type: str,
    citations: Sequence[str],
    chunks: Sequence[LegalChunk],
) -> str:
    """Render a concise prose answer that stays stable across benchmark phases."""
    body = " ".join(_normalise_text(chunk.text) for chunk in chunks)
    if question_type in {"comparison", "multi_clause"}:
        lead = "Căn cứ các quy định sau: {0}.".format("; ".join(citations))
    else:
        lead = "Căn cứ {0} quy định như sau:".format(citations[0])

    return "{0} {1}".format(lead, body).strip()


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def validate_synthetic_benchmark_sources(
    records: Sequence[SyntheticQA], chunks_path: Union[str, Path]
) -> Tuple[int, str]:
    """Re-scan the corpus and reject missing, duplicated, or changed gold chunks."""

    if not records:
        raise ValueError("Synthetic benchmark must contain at least one record")
    expected_hashes: Dict[str, str] = {}
    declared_counts: set[int] = set()
    declared_corpus_hashes: set[str] = set()
    for record in records:
        raw_hashes = record.metadata.get("gold_chunk_sha256")
        if not isinstance(raw_hashes, list) or len(raw_hashes) != len(
            record.gold_chunk_ids
        ):
            raise ValueError(
                "{0} has invalid gold_chunk_sha256 metadata".format(record.question_id)
            )
        for chunk_id, raw_hash in zip(record.gold_chunk_ids, raw_hashes):
            if not isinstance(raw_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", raw_hash
            ):
                raise ValueError(
                    "{0} has an invalid SHA-256 for {1}".format(
                        record.question_id, chunk_id
                    )
                )
            previous = expected_hashes.setdefault(chunk_id, raw_hash)
            if previous != raw_hash:
                raise ValueError(
                    "Conflicting gold hashes declared for chunk_id {0}".format(chunk_id)
                )

        raw_count = record.metadata.get("source_chunk_count")
        raw_corpus_hash = record.metadata.get("source_chunk_corpus_sha256")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError(
                "{0} has invalid source_chunk_count metadata".format(record.question_id)
            )
        if not isinstance(raw_corpus_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", raw_corpus_hash
        ):
            raise ValueError(
                "{0} has invalid source corpus SHA-256 metadata".format(
                    record.question_id
                )
            )
        declared_counts.add(raw_count)
        declared_corpus_hashes.add(raw_corpus_hash)

    if len(declared_counts) != 1 or len(declared_corpus_hashes) != 1:
        raise ValueError("Synthetic benchmark records disagree on source corpus")

    found: Dict[str, str] = {}
    fingerprint = _ChunkCorpusFingerprint()
    for chunk in iter_legal_chunks(chunks_path):
        fingerprint.update(chunk)
        if chunk.chunk_id not in expected_hashes:
            continue
        if chunk.chunk_id in found:
            raise ValueError(
                "Gold chunk_id is duplicated in source corpus: {0}".format(
                    chunk.chunk_id
                )
            )
        actual_hash = chunk_content_sha256(chunk)
        found[chunk.chunk_id] = actual_hash
        if actual_hash != expected_hashes[chunk.chunk_id]:
            raise ValueError(
                "Gold chunk identity changed for chunk_id {0}".format(chunk.chunk_id)
            )

    missing = sorted(set(expected_hashes).difference(found))
    if missing:
        raise ValueError("Gold chunk_ids are missing: {0}".format(", ".join(missing)))
    declared_count = next(iter(declared_counts))
    declared_hash = next(iter(declared_corpus_hashes))
    if fingerprint.count != declared_count:
        raise ValueError(
            "Source chunk count changed: expected {0}, found {1}".format(
                declared_count, fingerprint.count
            )
        )
    actual_corpus_hash = fingerprint.hexdigest()
    if actual_corpus_hash != declared_hash:
        raise ValueError(
            "Source chunk corpus hash changed: expected {0}, found {1}".format(
                declared_hash, actual_corpus_hash
            )
        )
    return fingerprint.count, actual_corpus_hash


def regenerate_synthetic_benchmark(
    chunks_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    target_count: int = 100,
    seed: int = 2026,
    require_all_question_types: bool = True,
) -> SyntheticBenchmarkArtifact:
    """Build, verify, and atomically replace a benchmark using bounded memory."""

    records = generate_synthetic_benchmark(
        iter_legal_chunks(chunks_path),
        target_count=target_count,
        seed=seed,
        require_all_question_types=require_all_question_types,
    )
    source_count, source_hash = validate_synthetic_benchmark_sources(
        records, chunks_path
    )
    target = write_synthetic_benchmark(records, output_path)
    return SyntheticBenchmarkArtifact(
        output_path=str(target),
        record_count=len(records),
        benchmark_sha256=_file_sha256(target),
        source_chunk_count=source_count,
        source_chunk_corpus_sha256=source_hash,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Generate a deterministic synthetic benchmark from processed chunks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed_v3/chunks"),
        help="LegalChunk JSONL file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed_v3/benchmarks/synthetic_qa.jsonl"),
        help="Benchmark JSONL output path.",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of Q&A pairs.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic seed.")
    parser.add_argument(
        "--allow-missing-types",
        action="store_true",
        help="Generate from available types when a corpus lacks one required group.",
    )
    args = parser.parse_args()
    artifact = regenerate_synthetic_benchmark(
        args.chunks_dir,
        args.output,
        target_count=args.count,
        seed=args.seed,
        require_all_question_types=not args.allow_missing_types,
    )
    print(json.dumps(artifact.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
