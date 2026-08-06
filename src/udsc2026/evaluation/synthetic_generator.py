"""Deterministic, citation-grounded synthetic benchmark generation for Legal RAG."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple, Union

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


def generate_synthetic_benchmark(
    chunks: Sequence[LegalChunk],
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
    usable_chunks = [chunk for chunk in chunks if _is_usable(chunk)]
    if not usable_chunks:
        raise ValueError("No chunks with text, law_name, and article are available")

    candidates = _build_candidates(usable_chunks)
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
        raise ValueError("No benchmark candidates could be built")

    # Determinism is required for benchmark reproducibility, not for security.
    randomizer = random.Random(seed)  # nosec B311
    for kind in available_types:
        randomizer.shuffle(candidates[kind])

    records: List[SyntheticQA] = []
    positions = {kind: 0 for kind in available_types}
    for index in range(target_count):
        kind = available_types[index % len(available_types)]
        values = candidates[kind]
        candidate = values[positions[kind] % len(values)]
        positions[kind] += 1
        records.append(_to_record(index + 1, kind, candidate, positions[kind]))
    return records


def load_legal_chunks(path: Union[str, Path]) -> List[LegalChunk]:
    """Load LegalChunk records from one JSONL file or a chunks directory."""
    source = Path(path)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
    chunks: List[LegalChunk] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(LegalChunk.model_validate_json(line))
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "Invalid LegalChunk JSONL at {0}:{1}".format(
                            file_path, line_number
                        )
                    ) from error
    return chunks


def write_synthetic_benchmark(
    records: Iterable[SyntheticQA], output_path: Union[str, Path]
) -> Path:
    """Write one benchmark record per UTF-8 JSONL line."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def _is_usable(chunk: LegalChunk) -> bool:
    return bool(chunk.text.strip() and chunk.law_name and chunk.article)


def _build_candidates(
    chunks: Sequence[LegalChunk],
) -> Dict[str, List[Tuple[LegalChunk, ...]]]:
    candidates: Dict[str, List[Tuple[LegalChunk, ...]]] = {
        kind: [] for kind in QUESTION_TYPES
    }
    for chunk in chunks:
        for kind in _SINGLE_TYPES:
            if _TYPE_PATTERNS[kind].search(chunk.text):
                candidates[kind].append((chunk,))

    by_article: DefaultDict[Tuple[str, str], List[LegalChunk]] = defaultdict(list)
    for chunk in chunks:
        by_article[(chunk.law_name or "", chunk.article or "")].append(chunk)
    for article_chunks in by_article.values():
        distinct_clauses = _distinct_clause_chunks(article_chunks)
        if len(distinct_clauses) >= 2:
            pair = (distinct_clauses[0], distinct_clauses[1])
            candidates["comparison"].append(pair)
            candidates["multi_clause"].append(pair)
    return candidates


def _distinct_clause_chunks(chunks: Sequence[LegalChunk]) -> List[LegalChunk]:
    chosen: Dict[str, LegalChunk] = {}
    for chunk in chunks:
        key = chunk.clause or chunk.chunk_id
        chosen.setdefault(key, chunk)
    return list(chosen.values())


def _to_record(
    index: int,
    question_type: str,
    chunks: Tuple[LegalChunk, ...],
    variation: int,
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
    )


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


def main() -> None:
    """Generate a deterministic synthetic benchmark from processed chunks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed/chunks"),
        help="LegalChunk JSONL file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/benchmarks/synthetic_qa.jsonl"),
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
    records = generate_synthetic_benchmark(
        load_legal_chunks(args.chunks_dir),
        target_count=args.count,
        seed=args.seed,
        require_all_question_types=not args.allow_missing_types,
    )
    write_synthetic_benchmark(records, args.output)
    print("Wrote {0} synthetic Q&A records to {1}".format(len(records), args.output))


if __name__ == "__main__":
    main()
