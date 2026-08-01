"""Deterministic, corpus-free diagnostic metrics for DSC2026 LegalQA.

These functions intentionally do not claim parity with the hidden competition
scorer.  They define a stable local profile that needs no network, stemming
model, WordNet corpus, or Vietnamese segmenter:

* normalize Unicode to NFC and case-fold;
* remove Unicode format characters (including soft hyphen and embedded BOM);
* map Unicode separator characters to ASCII space;
* extract surface tokens with the Unicode-aware ``\\w+`` regular expression.

The METEOR diagnostic uses exact token matches only, the commonly used
``alpha=0.9``, ``beta=3.0``, ``gamma=0.5`` parameters, and a documented greedy
alignment.  ROUGE-L is token-level LCS F1.  Profile names and parameters are
included in the evaluation report so results cannot be mistaken for official
Codabench scores.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

TOKENIZATION_PROFILE = "legalqa-nfc-casefold-cf-strip-word-regex-v1"
METEOR_DIAGNOSTIC_PROFILE = "legalqa-meteor-exact-diagnostic-v1"
ROUGE_L_DIAGNOSTIC_PROFILE = "legalqa-rouge-l-f1-diagnostic-v1"

METEOR_ALPHA = 0.9
METEOR_BETA = 3.0
METEOR_GAMMA = 0.5

_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class MeteorDiagnostic:
    """Complete contribution of one prediction to the local METEOR profile."""

    score: float
    precision: float
    recall: float
    harmonic_mean: float
    fragmentation_penalty: float
    matches: int
    chunks: int
    prediction_token_count: int
    reference_token_count: int


@dataclass(frozen=True, slots=True)
class RougeLDiagnostic:
    """Token-level ROUGE-L precision, recall, and F1 for one prediction."""

    f1: float
    precision: float
    recall: float
    lcs_length: int
    prediction_token_count: int
    reference_token_count: int


def normalize_legal_qa_text(text: str) -> str:
    """Return the exact normalized text used by both diagnostic metrics.

    Punctuation is retained at this stage and becomes a token boundary in
    :func:`tokenize_legal_qa_text`.  Format characters are removed rather than
    replaced so an accidental soft hyphen does not split one Vietnamese word.
    """

    if not isinstance(text, str):
        raise TypeError("LegalQA metric inputs must be strings")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "LegalQA metric inputs must contain valid Unicode scalar values"
        ) from exc
    normalized = unicodedata.normalize("NFC", text).casefold()
    characters: List[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category == "Cf":
            continue
        characters.append(" " if category.startswith("Z") else character)
    return "".join(characters)


def tokenize_legal_qa_text(text: str) -> List[str]:
    """Tokenize normalized Vietnamese text without external linguistic data."""

    return _WORD_PATTERN.findall(normalize_legal_qa_text(text))


def _exact_alignment(
    prediction_tokens: Sequence[str],
    reference_tokens: Sequence[str],
) -> List[Tuple[int, int]]:
    """Align all possible exact matches with deterministic fragmentation bias.

    For each prediction token, the alignment prefers the reference position
    immediately following the previous match, then the next unused forward
    occurrence, then the earliest remaining occurrence.  This preserves the
    maximum multiset match count while making duplicate-token behavior stable.
    """

    available: Dict[str, List[int]] = {}
    for reference_index, token in enumerate(reference_tokens):
        available.setdefault(token, []).append(reference_index)

    pairs: List[Tuple[int, int]] = []
    previous_reference_index = -1
    for prediction_index, token in enumerate(prediction_tokens):
        positions = available.get(token)
        if not positions:
            continue

        desired = previous_reference_index + 1
        desired_offset = bisect_left(positions, desired)
        if desired_offset < len(positions) and positions[desired_offset] == desired:
            chosen_offset = desired_offset
        else:
            forward_offset = bisect_right(positions, previous_reference_index)
            chosen_offset = forward_offset if forward_offset < len(positions) else 0

        reference_index = positions.pop(chosen_offset)
        pairs.append((prediction_index, reference_index))
        previous_reference_index = reference_index
    return pairs


def _count_chunks(alignment: Sequence[Tuple[int, int]]) -> int:
    """Count contiguous matched runs in prediction and reference order."""

    if not alignment:
        return 0
    chunks = 1
    previous_prediction, previous_reference = alignment[0]
    for prediction_index, reference_index in alignment[1:]:
        if (
            prediction_index != previous_prediction + 1
            or reference_index != previous_reference + 1
        ):
            chunks += 1
        previous_prediction = prediction_index
        previous_reference = reference_index
    return chunks


def meteor_diagnostic(
    prediction: str,
    reference: str,
) -> MeteorDiagnostic:
    """Calculate one corpus-free exact-match METEOR diagnostic.

    Empty or disjoint inputs score zero.  As in the standard fragmentation
    formulation, even an identical non-empty sentence can score slightly below
    one because its single matched chunk receives a small penalty.
    """

    prediction_tokens = tokenize_legal_qa_text(prediction)
    reference_tokens = tokenize_legal_qa_text(reference)
    prediction_count = len(prediction_tokens)
    reference_count = len(reference_tokens)
    if not prediction_tokens or not reference_tokens:
        return MeteorDiagnostic(
            score=0.0,
            precision=0.0,
            recall=0.0,
            harmonic_mean=0.0,
            fragmentation_penalty=0.0,
            matches=0,
            chunks=0,
            prediction_token_count=prediction_count,
            reference_token_count=reference_count,
        )

    if prediction_tokens == reference_tokens:
        alignment = [(index, index) for index in range(prediction_count)]
    else:
        alignment = _exact_alignment(prediction_tokens, reference_tokens)
    matches = len(alignment)
    if matches == 0:
        return MeteorDiagnostic(
            score=0.0,
            precision=0.0,
            recall=0.0,
            harmonic_mean=0.0,
            fragmentation_penalty=0.0,
            matches=0,
            chunks=0,
            prediction_token_count=prediction_count,
            reference_token_count=reference_count,
        )

    precision = matches / prediction_count
    recall = matches / reference_count
    denominator = METEOR_ALPHA * precision + (1 - METEOR_ALPHA) * recall
    harmonic_mean = precision * recall / denominator
    chunks = _count_chunks(alignment)
    fragmentation_penalty = METEOR_GAMMA * math.pow(
        chunks / matches,
        METEOR_BETA,
    )
    score = harmonic_mean * (1 - fragmentation_penalty)
    return MeteorDiagnostic(
        score=max(0.0, min(1.0, score)),
        precision=precision,
        recall=recall,
        harmonic_mean=harmonic_mean,
        fragmentation_penalty=fragmentation_penalty,
        matches=matches,
        chunks=chunks,
        prediction_token_count=prediction_count,
        reference_token_count=reference_count,
    )


def meteor_diagnostic_score(prediction: str, reference: str) -> float:
    """Return only the score from :func:`meteor_diagnostic`."""

    return meteor_diagnostic(prediction, reference).score


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute LCS length with O(min(n, m)) memory."""

    if left == right:
        return len(left)
    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for long_token in long:
        current = [0]
        for index, short_token in enumerate(short, start=1):
            if short_token == long_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_diagnostic(
    prediction: str,
    reference: str,
) -> RougeLDiagnostic:
    """Calculate token-level ROUGE-L F1 under the declared local profile."""

    prediction_tokens = tokenize_legal_qa_text(prediction)
    reference_tokens = tokenize_legal_qa_text(reference)
    prediction_count = len(prediction_tokens)
    reference_count = len(reference_tokens)
    if prediction_count == 0 and reference_count == 0:
        return RougeLDiagnostic(
            f1=1.0,
            precision=1.0,
            recall=1.0,
            lcs_length=0,
            prediction_token_count=0,
            reference_token_count=0,
        )
    if prediction_count == 0 or reference_count == 0:
        return RougeLDiagnostic(
            f1=0.0,
            precision=0.0,
            recall=0.0,
            lcs_length=0,
            prediction_token_count=prediction_count,
            reference_token_count=reference_count,
        )

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / prediction_count
    recall = lcs / reference_count
    f1 = 2 * precision * recall / (precision + recall) if lcs else 0.0
    return RougeLDiagnostic(
        f1=f1,
        precision=precision,
        recall=recall,
        lcs_length=lcs,
        prediction_token_count=prediction_count,
        reference_token_count=reference_count,
    )


def rouge_l_f1_score(prediction: str, reference: str) -> float:
    """Return only the F1 score from :func:`rouge_l_diagnostic`."""

    return rouge_l_diagnostic(prediction, reference).f1


__all__ = [
    "METEOR_ALPHA",
    "METEOR_BETA",
    "METEOR_DIAGNOSTIC_PROFILE",
    "METEOR_GAMMA",
    "ROUGE_L_DIAGNOSTIC_PROFILE",
    "TOKENIZATION_PROFILE",
    "MeteorDiagnostic",
    "RougeLDiagnostic",
    "meteor_diagnostic",
    "meteor_diagnostic_score",
    "normalize_legal_qa_text",
    "rouge_l_diagnostic",
    "rouge_l_f1_score",
    "tokenize_legal_qa_text",
]
