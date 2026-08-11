"""CPU-only legal lexical components used by Task 1 experiments.

The functions in this module deliberately consume only question text and
public corpus fields.  They never inspect relevance labels while ranking a
document (labels are only an input to the supervised KNN component).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    CountVectorizer,
    TfidfVectorizer,
)

_WORD_RE = re.compile(r"(?u)\b\w+\b")
_ARTICLE_RE = re.compile(r"\bđiều\s*(\d+[a-zđ]?)\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"\bkhoản\s*(\d+[a-zđ]?)\b", re.IGNORECASE)
_POINT_RE = re.compile(r"\bđiểm\s*([a-zđ])\b", re.IGNORECASE)
_INSTRUMENT_RE = re.compile(
    r"\b(nghị\s*định|thông\s*tư|quyết\s*định)\s*(?:số\s*)?"
    r"(\d+)(?:\s*/\s*(\d{4}))?\s*/\s*([a-zđ-]+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalContext:
    """One document, retaining the optional organizer title/name."""

    doc_id: str
    passage: str
    title: str = ""


@dataclass(frozen=True)
class CitationSignals:
    """Normalized legal citation tokens extracted from one text."""

    articles: frozenset[str]
    clauses: frozenset[str]
    points: frozenset[str]
    instruments: frozenset[str]

    @property
    def explicit(self) -> bool:
        """Return whether at least one legal citation was found."""

        return bool(self.articles or self.clauses or self.points or self.instruments)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_legal_citations(text: str) -> CitationSignals:
    """Extract normalized Vietnamese legal references from arbitrary text."""

    normalized = _normalized(text)
    instruments = {
        f"{kind.replace(' ', '')}:{number}"
        f"{'/' + year if year else ''}/{suffix.replace(' ', '')}"
        for kind, number, year, suffix in _INSTRUMENT_RE.findall(normalized)
    }
    return CitationSignals(
        articles=frozenset(_ARTICLE_RE.findall(normalized)),
        clauses=frozenset(_CLAUSE_RE.findall(normalized)),
        points=frozenset(_POINT_RE.findall(normalized)),
        instruments=frozenset(instruments),
    )


def citation_score(question: str, context: LegalContext) -> float:
    """Score solely from citation overlap in question and corpus metadata/text."""

    wanted = parse_legal_citations(question)
    if not wanted.explicit:
        return 0.0
    title = parse_legal_citations(context.title)
    body = parse_legal_citations(context.passage)
    score = 0.0
    # Full instrument references are much more discriminative than bare articles.
    for signals, multiplier in ((title, 2.0), (body, 1.0)):
        score += multiplier * 8.0 * len(wanted.instruments & signals.instruments)
        score += multiplier * 1.5 * len(wanted.articles & signals.articles)
        score += multiplier * 1.0 * len(wanted.clauses & signals.clauses)
        score += multiplier * 0.5 * len(wanted.points & signals.points)
    return score


def build_citation_rankings(
    questions: Mapping[str, str], contexts: Sequence[LegalContext]
) -> dict[str, list[str]]:
    """Rank contexts using citation overlap without relevance labels."""

    rankings: dict[str, list[str]] = {}
    for question_id, question in questions.items():
        scored = [
            (context.doc_id, citation_score(question, context)) for context in contexts
        ]
        rankings[question_id] = [
            doc_id
            for doc_id, score in sorted(scored, key=lambda pair: (-pair[1], pair[0]))
            if score > 0
        ]
    return rankings


def build_knn_rankings(
    questions: Mapping[str, str],
    labeled: Sequence[tuple[str, str, Sequence[str]]],
    *,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    neighbors: int = 20,
) -> dict[str, list[str]]:
    """Rank labelled documents via word or character TF-IDF KNN."""

    if not labeled:
        raise ValueError("labeled KNN pool must not be empty")
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform([item[1] for item in labeled])
    except ValueError as exc:
        raise ValueError(f"cannot build {analyzer} TF-IDF: {exc}") from exc
    ids = list(questions)
    similarities = (
        vectorizer.transform([questions[item] for item in ids]) @ matrix.T
    ).tocsr()
    results: dict[str, list[str]] = {}
    for row_index, question_id in enumerate(ids):
        row = similarities.getrow(row_index)
        ranked_neighbors = sorted(
            (
                (int(index), float(score))
                for index, score in zip(row.indices, row.data)
                if score > 0
            ),
            key=lambda item: (-item[1], labeled[item[0]][0]),
        )[:neighbors]
        scores: dict[str, tuple[float, int]] = {}
        for neighbor_rank, (index, score) in enumerate(ranked_neighbors, 1):
            for doc_id in labeled[index][2]:
                best = scores.get(doc_id, (-math.inf, 10**9))
                if score > best[0] or (score == best[0] and neighbor_rank < best[1]):
                    scores[doc_id] = (score, neighbor_rank)
        results[question_id] = sorted(
            scores, key=lambda doc_id: (-scores[doc_id][0], scores[doc_id][1], doc_id)
        )
    return results


def _bm25_weights(
    texts: Sequence[str], vocabulary: dict[str, int], lengths: np.ndarray
) -> Any:
    vectorizer = CountVectorizer(
        vocabulary=vocabulary,
        ngram_range=(1, 3),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    counts = vectorizer.transform(texts).tocsr()
    n_docs = len(texts)
    df = np.bincount(counts.indices, minlength=len(vocabulary)).astype(np.float32)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg = max(float(lengths.mean()), 1.0)
    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    norm = 0.7 * (1.0 - 0.3 + 0.3 * lengths / avg)
    counts.data = (
        (counts.data * 1.7) / (counts.data + norm[rows]) * idf[counts.indices]
    ).astype(np.float32)
    return counts


def build_bm25f_rankings(
    questions: Mapping[str, str],
    contexts: Sequence[LegalContext],
    *,
    title_weight: float = 0.0,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """BM25 body ranking plus an independently weighted title field."""

    if title_weight < 0:
        raise ValueError("title_weight must be non-negative")
    query_ids, query_texts = list(questions), list(questions.values())
    terms: set[str] = set()
    for question in query_texts:
        tokens = _WORD_RE.findall(_normalized(question))
        terms.update(
            " ".join(tokens[start : start + width])
            for width in (1, 2, 3)
            for start in range(len(tokens) - width + 1)
        )
    if not terms:
        raise ValueError("cannot build BM25 vocabulary from empty questions")
    vocabulary = {term: index for index, term in enumerate(sorted(terms))}
    vectorizer = CountVectorizer(
        vocabulary=vocabulary,
        ngram_range=(1, 3),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    query_counts = vectorizer.transform(query_texts).tocsr()
    query_counts.data.fill(1.0)
    passages, titles = (
        [item.passage for item in contexts],
        [item.title for item in contexts],
    )
    body = _bm25_weights(
        passages,
        vocabulary,
        np.asarray(
            [max(1, len(_WORD_RE.findall(value))) for value in passages],
            dtype=np.float32,
        ),
    )
    title = _bm25_weights(
        titles,
        vocabulary,
        np.asarray(
            [max(1, len(_WORD_RE.findall(value))) for value in titles], dtype=np.float32
        ),
    )
    score_matrix = (query_counts @ (body + title_weight * title).T).tocsr()
    output: dict[str, list[str]] = {}
    for index, question_id in enumerate(query_ids):
        row = score_matrix.getrow(index)
        scored = sorted(
            (
                (contexts[int(i)].doc_id, float(score))
                for i, score in zip(row.indices, row.data)
                if score > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        output[question_id] = [doc_id for doc_id, _ in scored[:top_k]]
    return output


def weighted_rrf(rankings: Iterable[Sequence[str]], *, rrf_k: int = 60) -> list[str]:
    """Fuse ordered rankings with equal-weight reciprocal rank fusion."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
