"""Canonical query-aware BM25-within-document evidence selector.

The implementation mirrors the reconciled B4 S2 selector.  V2 training and
inference both call :func:`select_true_s2`; neither path may use ``avail[:3]``
or reuse previously scored chunks.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

_PROTECTED = re.compile(
    r"(?:\b(?:(?:Điều|Khoản)\s+\d+[a-zđ]?|Điểm\s+[a-zđ])\b|"
    r"\b\d{1,4}/\d{4}/[a-zđ]{1,12}\d{0,4}\b)",
    re.IGNORECASE,
)


def tokenize_vi(text: str) -> list[str]:
    """Use the B4 PyVi tokenizer, retaining B4's protected legal tokens.

    The small fallback is only for local CPU smoke checks when PyVi is not
    installed.  Beam's V2 image installs PyVi, so production selection uses
    the exact B4 tokenizer.
    """
    if text is None or not str(text).strip():
        return []
    try:
        from pyvi import ViTokenizer  # type: ignore
    except ImportError:
        return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
    output: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(str(text)):
        output.extend(
            token.lower()
            for token in ViTokenizer.tokenize(str(text)[cursor : match.start()]).split()
            if token
        )
        output.append(match.group(0))
        cursor = match.end()
    output.extend(
        token.lower()
        for token in ViTokenizer.tokenize(str(text)[cursor:]).split()
        if token
    )
    return output


def bm25_scores(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """B4's BM25 scoring over the chunks of one document."""
    lengths = [len(document) for document in corpus]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if average <= 0:
        return [0.0] * len(corpus)
    frequencies = Counter(token for document in corpus for token in set(document))
    idf = {
        token: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
        for token, count in frequencies.items()
    }
    result: list[float] = []
    for document, length in zip(corpus, lengths):
        tf = Counter(document)
        norm = k1 * (1 - b + b * length / average)
        score = 0.0
        for token in query:
            count = tf.get(token, 0)
            if count:
                score += idf.get(token, 0.0) * count * (k1 + 1) / (count + norm)
        result.append(score)
    return result


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("raw_chunk_text", chunk.get("text", "")))


@dataclass(frozen=True)
class PreparedDocument:
    """Tokenized corpus for one document, reusable across query occurrences."""

    chunks: tuple[dict[str, Any], ...]
    corpus: tuple[tuple[str, ...], ...]


def prepare_document(chunks: Iterable[dict[str, Any]]) -> PreparedDocument:
    """Tokenize one document's chunk corpus exactly once."""
    materialized = tuple(dict(chunk) for chunk in chunks)
    corpus = tuple(tuple(tokenize_vi(_chunk_text(chunk))) for chunk in materialized)
    return PreparedDocument(materialized, corpus)


def select_true_s2_prepared(
    question: str, prepared: PreparedDocument, topk: int = 3
) -> list[dict[str, Any]]:
    """Select S2 from a prepared document without re-tokenizing its chunks."""
    query = tokenize_vi(str(question))
    scores = bm25_scores([list(tokens) for tokens in prepared.corpus], query) if query else [0.0] * len(prepared.chunks)
    ordered = sorted(
        range(len(prepared.chunks)),
        key=lambda index: (-scores[index], str(prepared.chunks[index].get("chunk_id", ""))),
    )
    selected: list[dict[str, Any]] = []
    for rank, index in enumerate(ordered[: max(0, int(topk))], 1):
        text = _chunk_text(prepared.chunks[index])
        if not text.strip():
            continue
        selected.append(
            {
                "chunk_id": str(prepared.chunks[index].get("chunk_id", "")),
                "raw_chunk_text": text,
                "bm25_score": float(scores[index]),
                "bm25_rank": rank,
                "selector_name": "true_s2_bm25_within_document_v2",
            }
        )
    return selected


def select_true_s2(
    question: str, chunks: Iterable[dict[str, Any]], topk: int = 3
) -> list[dict[str, Any]]:
    """Select the top ``topk`` chunks with true query-aware BM25 within a doc.

    Sort order is exactly BM25 descending, then ``chunk_id`` ascending.  The
    returned records contain raw text, chunk id, score, and selector rank.
    """
    return select_true_s2_prepared(question, prepare_document(chunks), topk)


def iter_payloads(path: Path) -> Iterator[dict[str, Any]]:
    """Stream values from the existing top-level ``payloads`` JSON object."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            block = handle.read(1 << 20)
            if block:
                buffer += block
            else:
                eof = True

        while '"payloads"' not in buffer and not eof:
            refill()
        marker = buffer.find('"payloads"')
        if marker < 0:
            raise ValueError("payloads.json has no top-level 'payloads' field")
        buffer = buffer[marker + len('"payloads"') :]
        while "{" not in buffer and not eof:
            refill()
        opening = buffer.find("{")
        if opening < 0:
            raise ValueError("payloads field is not an object")
        buffer = buffer[opening + 1 :]
        while True:
            while True:
                buffer = buffer.lstrip()
                if buffer or eof:
                    break
                refill()
            if not buffer:
                raise ValueError("unexpected EOF in payloads object")
            if buffer[0] == "}":
                return
            while True:
                try:
                    _, position = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            buffer = buffer[position:].lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if not buffer.startswith(":"):
                raise ValueError("invalid payload key/value separator")
            buffer = buffer[1:].lstrip()
            while True:
                try:
                    value, position = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(value, dict):
                raise ValueError("payload value is not an object")
            yield value
            buffer = buffer[position:].lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]