"""Task 2 scorer-compatible metrics with an explicit dependency boundary.

The fast METEOR proxy is suitable for candidate search and has no corpus
dependency. ``official_meteor`` deliberately fails when NLTK's WordNet data
is unavailable; promoted evaluation must never silently switch metrics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Sequence

_ROUGE_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ROUGE_SPACE_RE = re.compile(r"\s+")


def fast_exact_meteor(reference: str, prediction: str) -> float:
    """Return an exact-match proxy using organizer whitespace tokenization."""

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


class _CachedStemmer:
    def __init__(self) -> None:
        from nltk.stem import PorterStemmer

        self._stemmer = PorterStemmer()

    @lru_cache(maxsize=None)
    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)


class _CachedWordNet:
    @lru_cache(maxsize=None)
    def synsets(self, word: str) -> Any:
        from nltk.corpus import wordnet

        return wordnet.synsets(word)


@lru_cache(maxsize=1)
def _meteor_resources() -> tuple[Any, Any]:
    """Load and validate the exact corpora required by the organizer scorer."""

    import nltk

    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.data.find(f"corpora/{resource}.zip")
    return _CachedStemmer(), _CachedWordNet()


def official_meteor(reference: str, prediction: str) -> float:
    """Return organizer-compatible NLTK METEOR or fail if corpora are absent."""

    from nltk.translate.meteor_score import meteor_score

    stemmer, wordnet = _meteor_resources()
    return float(
        meteor_score(
            [reference.split()],
            prediction.split(),
            stemmer=stemmer,
            wordnet=wordnet,
        )
    )


def _rouge_tokens(text: str) -> list[str]:
    normalized = _ROUGE_NON_ALNUM_RE.sub(" ", text.lower())
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


def official_rouge_l(reference: str, prediction: str) -> float:
    """Return organizer-compatible ROUGE-L F1 without stemming."""

    reference_tokens = _rouge_tokens(reference)
    prediction_tokens = _rouge_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    return 2.0 * lcs / (len(reference_tokens) + len(prediction_tokens))


__all__ = ["fast_exact_meteor", "official_meteor", "official_rouge_l"]
