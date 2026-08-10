"""Train a leakage-safe third-parent selector for DSC2026 LegalQA.

This experiment keeps the first two *distinct parents* from the established
dense/BGE RRF ranking.  Every remaining distinct parent is a supervised
candidate.  On labeled BTC questions its target is the marginal exact-token
METEOR proxy of ``base answer + candidate`` relative to the two-parent base.

Only organizer-provided questions, retrieval hits, metadata, and parent text
are consumed.  Produced answers contain contiguous source spans only: there is
no prompt text, answer rewriting, label overlay, external corpus, or synthetic
question augmentation.

Two modes are provided:

* ``evaluate`` requires an explicit exhaustive, disjoint train/eval ID split.
  TF-IDF/IDF, optional supervised priors, model fitting, and threshold choice
  use only train IDs.  Eval labels are read only after prediction for scoring.
* ``fit-public`` fits on all labeled train questions and writes internal public
  ``[{id, answer}]`` predictions.  It never writes an official ZIP.

The optional supervised prior uses one best positive parent per training
question.  Training rows receive out-of-fold prior features; eval/public rows
use a prior refit on all permitted training IDs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NoReturn, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_NAMES = (
    "dense_rank",
    "dense_reciprocal_rank",
    "bge_rank",
    "bge_reciprocal_rank",
    "old_parent_rank",
    "old_parent_reciprocal_rank",
    "dense_score",
    "bge_score",
    "dense_bge_rank_gap",
    "lexical_score",
    "lexical_rank",
    "lexical_reciprocal_rank",
    "bm25_score",
    "bm25_rank",
    "bm25_reciprocal_rank",
    "query_parent_unigram_coverage",
    "query_parent_bigram_coverage",
    "query_number_coverage",
    "log_parent_tokens",
    "log_anchor_tokens",
    "same_doc_as_base",
    "same_law_as_base",
    "same_article_as_base",
    "has_law_name",
    "has_article",
    "has_clause",
    "has_section_heading",
    "prior_same_doc_similarity",
    "prior_same_article_similarity",
)

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_ROUGE_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ROUGE_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class Hit:
    """Compact merged dense/BGE hit used by the selector."""

    chunk_id: str
    doc_id: str
    parent_id: str | None
    text: str
    article: str
    clause: str
    law_name: str
    section_heading: str
    dense_rank: int
    bge_rank: int
    dense_score: float
    bge_score: float


@dataclass(frozen=True)
class QueryRanking:
    """One question and its common dense/BGE candidate pool."""

    question_id: str
    hits: tuple[Hit, ...]


@dataclass(frozen=True)
class ParentCandidate:
    """One distinct parent represented by its strongest old-RRF child."""

    key: tuple[str, str]
    hit: Hit
    old_parent_rank: int


@dataclass(frozen=True)
class CandidateMeta:
    """Identifiers needed by the optional supervised prior."""

    question_id: str
    doc_id: str
    article: str


@dataclass
class CandidateDataset:
    """Flat candidate rows plus per-question ranges and base answers."""

    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    metas: list[CandidateMeta]
    groups: dict[str, tuple[int, int]]
    base_answers: dict[str, str]
    candidate_spans: list[str] | None


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"input does not exist: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"input must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _opaque_id(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{label} must be a string or integer")
    result = str(value)
    if not result or result != result.strip() or _CONTROL_RE.search(result):
        raise ValueError(f"{label} must be a clean non-empty ID")
    return result


def normalize_question(text: str) -> str:
    """Normalize text only for leakage checks, never for answer generation."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def word_tokens(text: str) -> list[str]:
    """Return the Unicode lexical tokens used by selector features."""

    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _load_question_mapping(
    path: Path, *, require_answers: bool
) -> dict[str, dict[str, str | None]]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"question mapping must be a non-empty object: {path}")
    output: dict[str, dict[str, str | None]] = {}
    for raw_id, raw_record in payload.items():
        question_id = _opaque_id(raw_id, label="question ID")
        if not isinstance(raw_record, dict):
            raise TypeError(f"question {question_id!r} must be an object")
        question = raw_record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no text")
        answer = raw_record.get("answer")
        if require_answers and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError(f"question {question_id!r} has no labeled answer")
        if answer is not None and not isinstance(answer, str):
            raise TypeError(f"answer for {question_id!r} must be string or null")
        output[question_id] = {"question": question, "answer": answer}
    return output


def _load_id_array(path: Path, *, label: str) -> list[str]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} must be a non-empty JSON string array")
    values = [_opaque_id(value, label=label) for value in payload]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return values


def validate_strict_split(
    questions: dict[str, dict[str, str | None]],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Validate exhaustive inputs, then remove train duplicates of eval text."""

    train_set = set(train_ids)
    eval_set = set(eval_ids)
    if len(train_set) != len(train_ids) or len(eval_set) != len(eval_ids):
        raise ValueError("train/eval ID sequences must not contain duplicates")
    if train_set & eval_set:
        raise ValueError("train/eval ID sets overlap")
    expected = set(questions)
    if train_set | eval_set != expected:
        missing = sorted(expected - (train_set | eval_set))[:5]
        extra = sorted((train_set | eval_set) - expected)[:5]
        raise ValueError(
            f"train/eval split must exactly cover questions: "
            f"missing={missing}, extra={extra}"
        )
    eval_texts = {
        normalize_question(str(questions[item]["question"])) for item in eval_ids
    }
    removed_train_ids = [
        item
        for item in train_ids
        if normalize_question(str(questions[item]["question"])) in eval_texts
    ]
    removed_set = set(removed_train_ids)
    filtered_train_ids = [item for item in train_ids if item not in removed_set]
    if not filtered_train_ids:
        raise ValueError("all training IDs duplicate normalized eval questions")
    train_texts = {
        normalize_question(str(questions[item]["question"]))
        for item in filtered_train_ids
    }
    overlap = train_texts & eval_texts
    if overlap:
        raise AssertionError("normalized split de-duplication failed")
    return filtered_train_ids, removed_train_ids


def _numeric_score(hit: dict[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = hit.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    return 0.0


def _resolved_parent_id(hit: dict[str, Any]) -> str | None:
    direct = hit.get("parent_id")
    direct_id = direct.strip() if isinstance(direct, str) and direct.strip() else None
    metadata = hit.get("metadata")
    metadata_value = metadata.get("parent_id") if isinstance(metadata, dict) else None
    metadata_id = (
        metadata_value.strip()
        if isinstance(metadata_value, str) and metadata_value.strip()
        else None
    )
    if direct_id and metadata_id and direct_id != metadata_id:
        return None
    return direct_id or metadata_id


def _iter_prediction_jsonl(
    path: Path,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield validated rows without retaining the raw 500 MB JSON tree."""

    if not path.is_file():
        raise FileNotFoundError(f"prediction file does not exist: {path}")
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            question_id = _opaque_id(
                row.get("question_id"), label=f"question_id at {path}:{line_number}"
            )
            if question_id in seen:
                raise ValueError(f"duplicate question ID {question_id!r} in {path}")
            seen.add(question_id)
            hits = row.get("hits")
            if not isinstance(hits, list) or not hits:
                raise ValueError(f"question {question_id!r} has no hits in {path}")
            for hit in hits:
                if not isinstance(hit, dict):
                    raise TypeError(f"hits for {question_id!r} must be objects")
            yield question_id, hits
    if not seen:
        raise ValueError(f"prediction file is empty: {path}")


def load_rankings(
    dense_path: Path,
    reranked_path: Path,
    expected_question_ids: Sequence[str],
) -> dict[str, QueryRanking]:
    """Stream and strictly merge common dense/BGE chunk pools."""

    expected = set(expected_question_ids)
    dense_rows: dict[str, dict[str, tuple[int, str, float]]] = {}
    for question_id, raw_hits in _iter_prediction_jsonl(dense_path):
        if question_id not in expected:
            raise ValueError(
                f"dense question coverage mismatch: extra={[question_id]}"
            )
        dense_by_chunk: dict[str, tuple[int, str, float]] = {}
        for rank, raw_hit in enumerate(raw_hits, 1):
            chunk_id = _opaque_id(raw_hit.get("chunk_id"), label="chunk_id")
            if chunk_id in dense_by_chunk:
                raise ValueError(f"dense pool repeats chunk {chunk_id!r}")
            dense_by_chunk[chunk_id] = (
                rank,
                _opaque_id(raw_hit.get("doc_id"), label="doc_id"),
                _numeric_score(raw_hit, ("dense_score", "score", "final_score")),
            )
        dense_rows[question_id] = dense_by_chunk
    if set(dense_rows) != expected:
        missing = sorted(expected - set(dense_rows))[:5]
        raise ValueError(f"dense question coverage mismatch: missing={missing}")

    bge_seen: set[str] = set()
    output: dict[str, QueryRanking] = {}
    for question_id, bge_raw in _iter_prediction_jsonl(reranked_path):
        if question_id not in expected:
            raise ValueError(f"BGE question coverage mismatch: extra={[question_id]}")
        bge_seen.add(question_id)
        dense_by_chunk = dense_rows.pop(question_id)
        bge_by_chunk: dict[str, tuple[int, dict[str, Any]]] = {}
        for rank, raw_hit in enumerate(bge_raw, 1):
            chunk_id = _opaque_id(raw_hit.get("chunk_id"), label="chunk_id")
            if chunk_id in bge_by_chunk:
                raise ValueError(f"BGE pool repeats chunk {chunk_id!r}")
            bge_by_chunk[chunk_id] = (rank, raw_hit)
        if set(dense_by_chunk) != set(bge_by_chunk):
            raise ValueError(f"dense/BGE chunk pool mismatch at {question_id!r}")

        hits: list[Hit] = []
        for chunk_id, (bge_rank, raw_hit) in bge_by_chunk.items():
            dense_rank, dense_doc_id, dense_score = dense_by_chunk[chunk_id]
            doc_id = _opaque_id(raw_hit.get("doc_id"), label="doc_id")
            if dense_doc_id != doc_id:
                raise ValueError(f"dense/BGE doc mismatch for chunk {chunk_id!r}")
            text = raw_hit.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"chunk {chunk_id!r} has no source text")
            metadata = raw_hit.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}

            def field(name: str) -> str:
                value = raw_hit.get(name) or metadata.get(name) or ""
                return str(value).strip()

            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    parent_id=_resolved_parent_id(raw_hit),
                    text=text.strip(),
                    article=field("article"),
                    clause=field("clause"),
                    law_name=field("law_name"),
                    section_heading=field("section_heading"),
                    dense_rank=dense_rank,
                    bge_rank=bge_rank,
                    dense_score=dense_score,
                    bge_score=_numeric_score(
                        raw_hit, ("rerank_score", "final_score", "score")
                    ),
                )
            )
        output[question_id] = QueryRanking(question_id, tuple(hits))
    if bge_seen != expected:
        missing = sorted(expected - bge_seen)[:5]
        raise ValueError(f"BGE question coverage mismatch: missing={missing}")
    return {question_id: output[question_id] for question_id in expected_question_ids}


def parent_key(hit: Hit) -> tuple[str, str]:
    """Return a collision-safe source parent key, falling back to the child."""

    return (hit.doc_id, hit.parent_id or f"__child__:{hit.chunk_id}")


def enumerate_parent_candidates(
    ranking: QueryRanking,
    *,
    dense_weight: float = 0.2,
    bge_weight: float = 0.8,
    rrf_k: int = 60,
    candidate_depth: int = 0,
) -> tuple[list[ParentCandidate], list[ParentCandidate]]:
    """Return old-RRF top two distinct parents and all remaining parents."""

    ordered_hits = sorted(
        ranking.hits,
        key=lambda hit: (
            -(
                dense_weight / (rrf_k + hit.dense_rank)
                + bge_weight / (rrf_k + hit.bge_rank)
            ),
            hit.dense_rank,
            hit.bge_rank,
            hit.chunk_id,
        ),
    )
    distinct: list[ParentCandidate] = []
    seen: set[tuple[str, str]] = set()
    for hit in ordered_hits:
        key = parent_key(hit)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(ParentCandidate(key, hit, len(distinct) + 1))
    if len(distinct) < 2:
        raise ValueError(
            f"question {ranking.question_id!r} has fewer than two distinct parents"
        )
    remaining = distinct[2:]
    if candidate_depth > 0:
        remaining = remaining[:candidate_depth]
    return distinct[:2], remaining


class ParentTextStore:
    """Bounded raw JSONL parent cache; no generated or external text."""

    def __init__(self, root: Path, *, max_cached_documents: int = 2048) -> None:
        if max_cached_documents < 1:
            raise ValueError("parent cache size must be positive")
        self.root = root.resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"parents directory does not exist: {root}")
        self.max_cached_documents = max_cached_documents
        self.cache: OrderedDict[str, dict[str, tuple[str, str]]] = OrderedDict()

    def get(self, candidate: ParentCandidate) -> str:
        hit = candidate.hit
        if hit.parent_id is None:
            return hit.text
        parents = self.cache.pop(hit.doc_id, None)
        if parents is None:
            path = (self.root / f"{hit.doc_id}.jsonl").resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("parent path escaped root") from exc
            if not path.is_file():
                return hit.text
            parents = {}
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid parent JSON at {path}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise TypeError(f"parent at {path}:{line_number} is not object")
                    parent_id = row.get("parent_id")
                    text = row.get("text")
                    if not isinstance(parent_id, str) or not isinstance(text, str):
                        raise ValueError(f"invalid parent at {path}:{line_number}")
                    if parent_id in parents:
                        raise ValueError(f"duplicate parent {parent_id!r} in {path}")
                    article = row.get("article")
                    parents[parent_id] = (
                        text,
                        article.strip() if isinstance(article, str) else "",
                    )
        self.cache[hit.doc_id] = parents
        while len(self.cache) > self.max_cached_documents:
            self.cache.popitem(last=False)
        record = parents.get(hit.parent_id)
        if record is None:
            return hit.text
        text, article = record
        if hit.article and article and hit.article != article:
            return hit.text
        return text


def _find_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> int | None:
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return None
    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix[index] = matched
    matched = 0
    for index, token in enumerate(haystack):
        while matched and token != needle[matched]:
            matched = prefix[matched - 1]
        if token == needle[matched]:
            matched += 1
            if matched == len(needle):
                return index - len(needle) + 1
    return None


def contiguous_parent_span(parent_text: str, anchor: str, limit: int) -> str:
    """Return one contiguous source-only token window centered on its anchor."""

    if limit < 1:
        raise ValueError("span token limit must be positive")
    tokens = parent_text.split()
    if not tokens:
        return ""
    if len(tokens) <= limit:
        return parent_text.strip()
    anchor_tokens = anchor.split()
    start = _find_subsequence(
        [token.casefold() for token in tokens],
        [token.casefold() for token in anchor_tokens],
    )
    if start is None:
        source = anchor_tokens[:limit]
        return " ".join(source)
    end = start + len(anchor_tokens)
    extra = max(0, limit - (end - start))
    left = max(0, start - extra // 2)
    right = min(len(tokens), end + extra - extra // 2)
    if right - left < limit:
        left = max(0, right - limit)
    return " ".join(tokens[left:right])


def fast_exact_meteor(reference: str, prediction: str) -> float:
    """Exact-only NLTK-METEOR proxy using official whitespace tokenization."""

    reference_tokens = [token.lower() for token in reference.split()]
    prediction_tokens = [token.lower() for token in prediction.split()]
    if not reference_tokens or not prediction_tokens:
        return 0.0
    positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(reference_tokens):
        positions[token].append(index)
    matches: list[tuple[int, int]] = []
    for prediction_index in range(len(prediction_tokens) - 1, -1, -1):
        available = positions.get(prediction_tokens[prediction_index])
        if available:
            matches.append((prediction_index, available.pop()))
    matches.sort()
    count = len(matches)
    if count == 0:
        return 0.0
    precision = count / len(prediction_tokens)
    recall = count / len(reference_tokens)
    fmean = precision * recall / (0.9 * precision + 0.1 * recall)
    chunks = 1
    for previous, current in zip(matches, matches[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / count) ** 3
    return (1.0 - penalty) * fmean


@dataclass(frozen=True)
class FeatureContext:
    """Lexical statistics fitted exclusively on permitted training IDs."""

    idf: dict[str, float]
    unseen_idf: float

    @classmethod
    def fit(
        cls,
        rankings: dict[str, QueryRanking],
        train_ids: Sequence[str],
    ) -> "FeatureContext":
        document_frequency: Counter[str] = Counter()
        count = 0
        for question_id in train_ids:
            for hit in rankings[question_id].hits:
                tokens = word_tokens(_metadata_text(hit))
                document_frequency.update(set(tokens))
                count += 1
        if count == 0:
            raise ValueError("training split has no retrieval candidates")
        return cls(
            idf={
                token: math.log((count + 1) / (frequency + 1)) + 1.0
                for token, frequency in document_frequency.items()
            },
            unseen_idf=math.log(count + 1) + 1.0,
        )

    def token_idf(self, token: str) -> float:
        return self.idf.get(token, self.unseen_idf)


def _metadata_text(hit: Hit) -> str:
    return " ".join(
        value
        for value in (
            hit.text,
            hit.article,
            hit.clause,
            hit.section_heading,
            hit.law_name,
        )
        if value
    )


def _rank_descending(values: Sequence[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0] * len(values)
    for rank, index in enumerate(ordered, 1):
        ranks[index] = rank
    return ranks


def _feature_rows_for_question(
    question: str,
    base: Sequence[ParentCandidate],
    candidates: Sequence[ParentCandidate],
    store: ParentTextStore,
    context: FeatureContext,
) -> tuple[list[list[float]], list[str]]:
    query_tokens = word_tokens(question)
    query_unique = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    query_numbers = {token for token in query_unique if any(c.isdigit() for c in token)}
    denominator = sum(context.token_idf(token) for token in query_unique) or 1.0
    base_docs = {item.hit.doc_id for item in base}
    base_laws = {item.hit.law_name for item in base if item.hit.law_name}
    base_articles = {item.hit.article for item in base if item.hit.article}

    raw: list[dict[str, float]] = []
    parent_texts: list[str] = []
    parent_token_lists: list[list[str]] = []
    for candidate in candidates:
        text = store.get(candidate)
        parent_texts.append(text)
        tokens = word_tokens(text)
        parent_token_lists.append(tokens)
        token_set = set(tokens)
        bigrams = set(zip(tokens, tokens[1:]))
        unigram_coverage = sum(
            context.token_idf(token) for token in query_unique & token_set
        ) / denominator
        bigram_coverage = (
            len(query_bigrams & bigrams) / len(query_bigrams)
            if query_bigrams
            else 0.0
        )
        lexical_score = 0.7 * unigram_coverage + 0.3 * bigram_coverage
        raw.append(
            {
                "unigram": unigram_coverage,
                "bigram": bigram_coverage,
                "number": (
                    len(query_numbers & token_set) / len(query_numbers)
                    if query_numbers
                    else 0.0
                ),
                "lexical": lexical_score,
                "parent_tokens": float(len(text.split())),
            }
        )
    average_parent_length = sum(
        max(1, len(tokens)) for tokens in parent_token_lists
    ) / len(parent_token_lists)
    for values, tokens in zip(raw, parent_token_lists):
        counts = Counter(tokens)
        length_norm = 0.7 * (
            1.0 - 0.3 + 0.3 * max(1, len(tokens)) / average_parent_length
        )
        bm25_score = sum(
            context.token_idf(token)
            * (counts[token] * 1.7)
            / (counts[token] + length_norm)
            for token in query_unique
            if token in counts
        ) + 2.0 * len(query_bigrams & set(zip(tokens, tokens[1:])))
        values["bm25"] = bm25_score
    lexical_ranks = _rank_descending([item["lexical"] for item in raw])
    bm25_ranks = _rank_descending([item["bm25"] for item in raw])
    rows: list[list[float]] = []
    for index, (candidate, values) in enumerate(zip(candidates, raw)):
        hit = candidate.hit
        lexical_rank = lexical_ranks[index]
        bm25_rank = bm25_ranks[index]
        rows.append(
            [
                float(hit.dense_rank),
                1.0 / hit.dense_rank,
                float(hit.bge_rank),
                1.0 / hit.bge_rank,
                float(candidate.old_parent_rank),
                1.0 / candidate.old_parent_rank,
                hit.dense_score,
                hit.bge_score,
                float(abs(hit.dense_rank - hit.bge_rank)),
                values["lexical"],
                float(lexical_rank),
                1.0 / lexical_rank,
                values["bm25"],
                float(bm25_rank),
                1.0 / bm25_rank,
                values["unigram"],
                values["bigram"],
                values["number"],
                math.log1p(values["parent_tokens"]),
                math.log1p(len(hit.text.split())),
                float(hit.doc_id in base_docs),
                float(bool(hit.law_name) and hit.law_name in base_laws),
                float(bool(hit.article) and hit.article in base_articles),
                float(bool(hit.law_name)),
                float(bool(hit.article)),
                float(bool(hit.clause)),
                float(bool(hit.section_heading)),
                0.0,
                0.0,
            ]
        )
    return rows, parent_texts


def build_candidate_dataset(
    questions: dict[str, dict[str, str | None]],
    rankings: dict[str, QueryRanking],
    question_ids: Sequence[str],
    store: ParentTextStore,
    context: FeatureContext,
    args: argparse.Namespace,
    *,
    labeled: bool,
    retain_answer_spans: bool,
) -> CandidateDataset:
    """Build flat candidate rows without retaining unrelated corpus text."""

    feature_rows: list[list[float]] = []
    labels: list[float] = []
    sample_weights: list[float] = []
    metas: list[CandidateMeta] = []
    groups: dict[str, tuple[int, int]] = {}
    base_answers: dict[str, str] = {}
    candidate_spans: list[str] | None = [] if retain_answer_spans else None
    for question_id in question_ids:
        base, candidates = enumerate_parent_candidates(
            rankings[question_id],
            dense_weight=args.base_dense_weight,
            bge_weight=args.base_bge_weight,
            rrf_k=args.base_rrf_k,
            candidate_depth=args.candidate_depth,
        )
        if not candidates:
            groups[question_id] = (len(feature_rows), len(feature_rows))
            base_answer = "\n".join(
                contiguous_parent_span(
                    store.get(item), item.hit.text, args.base_parent_tokens
                )
                for item in base
            )
            if retain_answer_spans:
                base_answers[question_id] = base_answer
            continue
        base_answer = "\n".join(
            contiguous_parent_span(
                store.get(item), item.hit.text, args.base_parent_tokens
            )
            for item in base
        ).strip()
        if not base_answer:
            raise ValueError(f"question {question_id!r} produced empty base answer")
        if retain_answer_spans:
            base_answers[question_id] = base_answer
        start = len(feature_rows)
        rows, parent_texts = _feature_rows_for_question(
            str(questions[question_id]["question"]),
            base,
            candidates,
            store,
            context,
        )
        reference = questions[question_id]["answer"]
        base_score = (
            fast_exact_meteor(str(reference), base_answer) if labeled else 0.0
        )
        weight = 1.0 / len(candidates)
        for candidate, row, parent_text in zip(candidates, rows, parent_texts):
            third_span = contiguous_parent_span(
                parent_text,
                candidate.hit.text,
                args.third_parent_tokens,
            )
            answer = f"{base_answer}\n{third_span}".strip()
            feature_rows.append(row)
            if candidate_spans is not None:
                candidate_spans.append(third_span)
            labels.append(
                fast_exact_meteor(str(reference), answer) - base_score
                if labeled
                else 0.0
            )
            sample_weights.append(weight)
            metas.append(
                CandidateMeta(
                    question_id,
                    candidate.hit.doc_id,
                    candidate.hit.article,
                )
            )
        groups[question_id] = (start, len(feature_rows))
    if not feature_rows:
        raise ValueError("candidate dataset contains no third-parent candidates")
    return CandidateDataset(
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        sample_weights=np.asarray(sample_weights, dtype=np.float32),
        metas=metas,
        groups=groups,
        base_answers=base_answers,
        candidate_spans=candidate_spans,
    )


class SupervisedParentPrior:
    """TF-IDF similarity to best-positive train questions by doc/article."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None
        self.doc_rows: dict[str, list[int]] = {}
        self.article_rows: dict[tuple[str, str], list[int]] = {}

    def fit(
        self,
        dataset: CandidateDataset,
        questions: dict[str, dict[str, str | None]],
        allowed_question_ids: set[str],
    ) -> None:
        positive_indices: list[int] = []
        for question_id in sorted(allowed_question_ids):
            start, end = dataset.groups[question_id]
            if start == end:
                continue
            best = max(range(start, end), key=lambda index: dataset.labels[index])
            if dataset.labels[best] > 0.0:
                positive_indices.append(best)
        if not positive_indices:
            return
        texts = [
            str(questions[dataset.metas[index].question_id]["question"])
            for index in positive_indices
        ]
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError:
            return
        self.vectorizer = vectorizer
        self.matrix = matrix
        for row, index in enumerate(positive_indices):
            meta = dataset.metas[index]
            self.doc_rows.setdefault(meta.doc_id, []).append(row)
            if meta.article:
                self.article_rows.setdefault(
                    (meta.doc_id, meta.article), []
                ).append(row)

    def features_for_group(
        self,
        question: str,
        metas: Sequence[CandidateMeta],
    ) -> list[tuple[float, float]]:
        if self.vectorizer is None or self.matrix is None:
            return [(0.0, 0.0)] * len(metas)
        query = self.vectorizer.transform([question])
        similarities = np.asarray((self.matrix @ query.T).todense()).ravel()
        output: list[tuple[float, float]] = []
        for meta in metas:
            doc_indices = self.doc_rows.get(meta.doc_id, [])
            article_indices = self.article_rows.get((meta.doc_id, meta.article), [])
            output.append(
                (
                    max(
                        (float(similarities[index]) for index in doc_indices),
                        default=0.0,
                    ),
                    max(
                        (float(similarities[index]) for index in article_indices),
                        default=0.0,
                    ),
                )
            )
        return output


def add_oof_prior_features(
    dataset: CandidateDataset,
    questions: dict[str, dict[str, str | None]],
    train_ids: Sequence[str],
    folds: int,
) -> SupervisedParentPrior:
    """Fill leakage-safe OOF train priors, then return an all-train prior."""

    if folds < 2:
        raise ValueError("supervised-prior folds must be at least two")
    question_key_by_id = {
        item: normalize_question(str(questions[item]["question"]))
        for item in train_ids
    }
    ordered_keys = sorted(set(question_key_by_id.values()))
    fold_by_key = {key: index % folds for index, key in enumerate(ordered_keys)}
    fold_by_id = {
        item: fold_by_key[question_key_by_id[item]] for item in train_ids
    }
    all_ids = set(train_ids)
    for fold in range(folds):
        held_out = {item for item in train_ids if fold_by_id[item] == fold}
        prior = SupervisedParentPrior()
        prior.fit(dataset, questions, all_ids - held_out)
        for question_id in held_out:
            start, end = dataset.groups[question_id]
            if start == end:
                continue
            features = prior.features_for_group(
                str(questions[question_id]["question"]), dataset.metas[start:end]
            )
            dataset.features[start:end, -2:] = np.asarray(features, dtype=np.float32)
    final_prior = SupervisedParentPrior()
    final_prior.fit(dataset, questions, all_ids)
    return final_prior


def apply_prior_features(
    dataset: CandidateDataset,
    questions: dict[str, dict[str, str | None]],
    question_ids: Sequence[str],
    prior: SupervisedParentPrior,
) -> None:
    for question_id in question_ids:
        start, end = dataset.groups[question_id]
        if start == end:
            continue
        features = prior.features_for_group(
            str(questions[question_id]["question"]), dataset.metas[start:end]
        )
        dataset.features[start:end, -2:] = np.asarray(features, dtype=np.float32)


def fit_selector(dataset: CandidateDataset, args: argparse.Namespace) -> Any:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=False,
        random_state=args.seed,
    )
    model.fit(
        dataset.features,
        dataset.labels,
        sample_weight=dataset.sample_weights,
    )
    return model


def select_answers(
    dataset: CandidateDataset,
    question_ids: Sequence[str],
    model: Any,
    threshold: float,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    if dataset.candidate_spans is None:
        raise ValueError("candidate spans were not retained for answer selection")
    predictions: list[dict[str, str]] = []
    selected = 0
    predicted_gains: list[float] = []
    for question_id in question_ids:
        start, end = dataset.groups[question_id]
        answer = dataset.base_answers[question_id]
        if start != end:
            gains = np.asarray(model.predict(dataset.features[start:end]))
            relative_index = int(np.argmax(gains))
            gain = float(gains[relative_index])
            predicted_gains.append(gain)
            if gain > threshold:
                third_span = dataset.candidate_spans[start + relative_index]
                answer = f"{answer}\n{third_span}".strip()
                selected += 1
        predictions.append({"id": question_id, "answer": answer})
    return predictions, {
        "selected_count": selected,
        "selected_rate": selected / len(question_ids),
        "mean_best_predicted_gain": (
            sum(predicted_gains) / len(predicted_gains) if predicted_gains else 0.0
        ),
    }


class _CachedStemmer:
    def __init__(self) -> None:
        from nltk.stem import PorterStemmer

        self.stemmer = PorterStemmer()

    @lru_cache(maxsize=None)
    def stem(self, word: str) -> str:
        return self.stemmer.stem(word)


class _CachedWordNet:
    @lru_cache(maxsize=None)
    def synsets(self, word: str) -> Any:
        from nltk.corpus import wordnet

        return wordnet.synsets(word)


def _official_meteor(reference: str, prediction: str) -> float:
    from nltk.translate.meteor_score import meteor_score

    return float(
        meteor_score(
            [reference.split()],
            prediction.split(),
            stemmer=_OFFICIAL_STEMMER,
            wordnet=_OFFICIAL_WORDNET,
        )
    )


_OFFICIAL_STEMMER = _CachedStemmer()
_OFFICIAL_WORDNET = _CachedWordNet()


def _rouge_tokens(text: str) -> list[str]:
    lowered = text.lower()
    normalized = _ROUGE_NON_ALNUM_RE.sub(" ", lowered)
    return [token for token in _ROUGE_SPACE_RE.split(normalized) if token]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    try:
        from rapidfuzz.distance import LCSseq

        return int(LCSseq.similarity(left, right))
    except ImportError:
        if len(left) < len(right):
            short, long = left, right
        else:
            short, long = right, left
        previous = [0] * (len(short) + 1)
        for token in long:
            current = [0]
            for index, short_token in enumerate(short, 1):
                current.append(
                    previous[index - 1] + 1
                    if token == short_token
                    else max(previous[index], current[-1])
                )
            previous = current
        return previous[-1]


def _official_rouge_l(reference: str, prediction: str) -> float:
    reference_tokens = _rouge_tokens(reference)
    prediction_tokens = _rouge_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    return 2.0 * lcs / (len(reference_tokens) + len(prediction_tokens))


def official_scores(
    predictions: Sequence[dict[str, str]],
    questions: dict[str, dict[str, str | None]],
) -> dict[str, float]:
    meteor_values: list[float] = []
    rouge_values: list[float] = []
    for prediction in predictions:
        reference = questions[prediction["id"]]["answer"]
        assert isinstance(reference, str)
        meteor_values.append(_official_meteor(reference, prediction["answer"]))
        rouge_values.append(_official_rouge_l(reference, prediction["answer"]))
    return {
        "meteor": sum(meteor_values) / len(meteor_values),
        "rouge_l": sum(rouge_values) / len(rouge_values),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _paths_collide(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=True) == right.resolve(strict=False):
            return True
    except OSError:
        pass
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    return False


def _protect_outputs(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    for index, output in enumerate(outputs):
        for other in outputs[index + 1 :]:
            if _paths_collide(output, other):
                raise ValueError("selector output paths must be distinct")
        for source in inputs:
            if _paths_collide(source, output):
                raise ValueError(f"selector output must not overwrite {source}")


def _model_summary(dataset: CandidateDataset) -> dict[str, float | int]:
    positive = int(np.count_nonzero(dataset.labels > 0.0))
    return {
        "candidate_rows": int(len(dataset.labels)),
        "positive_rows": positive,
        "positive_rate": positive / len(dataset.labels),
        "mean_marginal_proxy": float(np.mean(dataset.labels)),
        "max_marginal_proxy": float(np.max(dataset.labels)),
    }


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [args.output_report]
    if args.output_predictions is not None:
        outputs.append(args.output_predictions)
    _protect_outputs(
        [args.questions, args.dense, args.reranked, args.train_ids, args.eval_ids],
        outputs,
    )
    questions = _load_question_mapping(args.questions, require_answers=True)
    requested_train_ids = _load_id_array(args.train_ids, label="train IDs")
    eval_ids = _load_id_array(args.eval_ids, label="eval IDs")
    train_ids, removed_train_ids = validate_strict_split(
        questions, requested_train_ids, eval_ids
    )
    rankings = load_rankings(args.dense, args.reranked, list(questions))
    context = FeatureContext.fit(rankings, train_ids)
    store = ParentTextStore(
        args.parents_dir, max_cached_documents=args.parent_cache_documents
    )
    train_dataset = build_candidate_dataset(
        questions,
        rankings,
        train_ids,
        store,
        context,
        args,
        labeled=True,
        retain_answer_spans=False,
    )
    prior = SupervisedParentPrior()
    if args.supervised_prior:
        prior = add_oof_prior_features(
            train_dataset, questions, train_ids, args.prior_folds
        )
    model = fit_selector(train_dataset, args)

    eval_dataset = build_candidate_dataset(
        questions,
        rankings,
        eval_ids,
        store,
        context,
        args,
        labeled=True,
        retain_answer_spans=True,
    )
    if args.supervised_prior:
        apply_prior_features(eval_dataset, questions, eval_ids, prior)
    selected, selection = select_answers(
        eval_dataset, eval_ids, model, args.selection_threshold
    )
    baseline = [
        {"id": item, "answer": eval_dataset.base_answers[item]} for item in eval_ids
    ]
    report: dict[str, Any] = {
        "mode": "strict_train_eval",
        "requested_train_question_count": len(requested_train_ids),
        "train_question_count": len(train_ids),
        "eval_question_count": len(eval_ids),
        "removed_train_normalized_eval_duplicate_count": len(removed_train_ids),
        "removed_train_normalized_eval_duplicate_ids": removed_train_ids,
        "feature_names": list(FEATURE_NAMES),
        "supervised_prior": bool(args.supervised_prior),
        "selection_threshold": args.selection_threshold,
        "train_candidates": _model_summary(train_dataset),
        "eval_candidates": _model_summary(eval_dataset),
        "selection": selection,
        "baseline_official": official_scores(baseline, questions),
        "selected_official": official_scores(selected, questions),
        "assumptions": [
            "old RRF top two distinct parents are always retained",
            "candidate target is marginal exact-only METEOR proxy",
            "answers contain contiguous BTC parent/child spans only",
            "eval labels are used only after selection for reporting",
            "normalized train duplicates of eval are excluded before fitting",
        ],
    }
    _write_json(args.output_report, report)
    if args.output_predictions is not None:
        _write_json(args.output_predictions, selected)
    return report


def run_fit_public(args: argparse.Namespace) -> dict[str, Any]:
    _protect_outputs(
        [
            args.train_questions,
            args.train_dense,
            args.train_reranked,
            args.public_questions,
            args.public_dense,
            args.public_reranked,
        ],
        [args.output, args.output_report],
    )
    train_questions = _load_question_mapping(
        args.train_questions, require_answers=True
    )
    train_ids = list(train_questions)
    train_rankings = load_rankings(
        args.train_dense, args.train_reranked, train_ids
    )
    context = FeatureContext.fit(train_rankings, train_ids)
    store = ParentTextStore(
        args.parents_dir, max_cached_documents=args.parent_cache_documents
    )
    train_dataset = build_candidate_dataset(
        train_questions,
        train_rankings,
        train_ids,
        store,
        context,
        args,
        labeled=True,
        retain_answer_spans=False,
    )
    prior = SupervisedParentPrior()
    if args.supervised_prior:
        prior = add_oof_prior_features(
            train_dataset, train_questions, train_ids, args.prior_folds
        )
    model = fit_selector(train_dataset, args)
    del train_rankings
    gc.collect()

    public_questions = _load_question_mapping(
        args.public_questions, require_answers=False
    )
    public_ids = list(public_questions)
    public_rankings = load_rankings(
        args.public_dense, args.public_reranked, public_ids
    )
    public_dataset = build_candidate_dataset(
        public_questions,
        public_rankings,
        public_ids,
        store,
        context,
        args,
        labeled=False,
        retain_answer_spans=True,
    )
    if args.supervised_prior:
        # The prior contains only train labels; public questions never refit it.
        apply_prior_features(public_dataset, public_questions, public_ids, prior)
    predictions, selection = select_answers(
        public_dataset, public_ids, model, args.selection_threshold
    )
    _write_json(args.output, predictions)
    report: dict[str, Any] = {
        "mode": "fit_all_train_predict_public",
        "train_question_count": len(train_ids),
        "public_question_count": len(public_ids),
        "feature_names": list(FEATURE_NAMES),
        "supervised_prior": bool(args.supervised_prior),
        "selection_threshold": args.selection_threshold,
        "train_candidates": _model_summary(train_dataset),
        "selection": selection,
        "output": str(args.output),
        "submission_written": False,
    }
    _write_json(args.output_report, report)
    return report


def _add_selector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--parents-dir", type=Path, default=Path("data/processed_v3/parents")
    )
    parser.add_argument("--base-dense-weight", type=float, default=0.2)
    parser.add_argument("--base-bge-weight", type=float, default=0.8)
    parser.add_argument("--base-rrf-k", type=int, default=60)
    parser.add_argument("--base-parent-tokens", type=int, default=512)
    parser.add_argument("--third-parent-tokens", type=int, default=320)
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=0,
        help="Remaining distinct parents to label; 0 means all.",
    )
    parser.add_argument("--parent-cache-documents", type=int, default=2048)
    parser.add_argument("--selection-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--supervised-prior",
        action="store_true",
        help="Enable leakage-safe OOF same-doc/article TF-IDF prior features.",
    )
    parser.add_argument("--prior-folds", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--dense", type=Path, required=True)
    evaluate.add_argument("--reranked", type=Path, required=True)
    evaluate.add_argument("--train-ids", type=Path, required=True)
    evaluate.add_argument("--eval-ids", type=Path, required=True)
    evaluate.add_argument("--output-report", type=Path, required=True)
    evaluate.add_argument("--output-predictions", type=Path)
    _add_selector_options(evaluate)

    public = subparsers.add_parser("fit-public")
    public.add_argument("--train-questions", type=Path, required=True)
    public.add_argument("--train-dense", type=Path, required=True)
    public.add_argument("--train-reranked", type=Path, required=True)
    public.add_argument("--public-questions", type=Path, required=True)
    public.add_argument("--public-dense", type=Path, required=True)
    public.add_argument("--public-reranked", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    public.add_argument("--output-report", type=Path, required=True)
    _add_selector_options(public)
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    weights = (args.base_dense_weight, args.base_bge_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("base RRF weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("base RRF weights must sum to one")
    if args.base_rrf_k < 0:
        raise ValueError("base RRF k must be non-negative")
    positive_options = {
        "base-parent-tokens": args.base_parent_tokens,
        "third-parent-tokens": args.third_parent_tokens,
        "parent-cache-documents": args.parent_cache_documents,
        "max-iter": args.max_iter,
        "max-leaf-nodes": args.max_leaf_nodes,
        "min-samples-leaf": args.min_samples_leaf,
    }
    for name, value in positive_options.items():
        if value < 1:
            raise ValueError(f"--{name} must be positive")
    if args.candidate_depth < 0:
        raise ValueError("candidate depth must be non-negative")
    finite_options = (
        args.selection_threshold,
        args.learning_rate,
        args.l2_regularization,
    )
    if any(not math.isfinite(value) for value in finite_options):
        raise ValueError("numeric selector options must be finite")
    if args.learning_rate <= 0.0 or args.l2_regularization < 0.0:
        raise ValueError("learning rate must be positive and L2 non-negative")
    if args.supervised_prior and args.prior_folds < 2:
        raise ValueError("supervised-prior folds must be at least two")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_options(args)
        report = run_evaluate(args) if args.mode == "evaluate" else run_fit_public(args)
    except (OSError, TypeError, ValueError, LookupError) as exc:
        print(f"third-parent selector error: {exc}", file=sys.stderr)
        return 2
    print(
        f"mode={report['mode']} supervised_prior={report['supervised_prior']}"
    )
    print(args.output_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
