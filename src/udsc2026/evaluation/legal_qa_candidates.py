"""Label-free answer candidates and features for Task 2 LegalQA.

The public pipeline may call the functions in this module because they only
consume the question, retrieved evidence, and model predictions.  Reference
answers are deliberately absent from every function signature.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")
_LEGAL_MARKER_RE = re.compile(
    r"\b(?:điều|khoản|điểm|nghị định|thông tư|luật|quyết định|phạt|"
    r"thời hạn|trách nhiệm|điều kiện|hồ sơ|thủ tục)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerCandidate:
    """One deterministic candidate for a LegalQA question."""

    profile: str
    answer: str


def word_tokens(text: str) -> list[str]:
    """Return stable Unicode word tokens for label-free feature extraction."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    return _TOKEN_RE.findall(normalized)


def _sentence_units(text: str) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    for raw in _SENTENCE_BOUNDARY_RE.split(text):
        unit = " ".join(raw.split()).strip()
        tokens = word_tokens(unit)
        if len(tokens) < 3:
            continue
        key = " ".join(tokens)
        if key in seen:
            continue
        seen.add(key)
        units.append(unit)
    return units


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _bounded_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit]).strip()


def trim_at_sentence_boundary(text: str, word_limit: int) -> str:
    """Trim an answer without leaving a visibly broken final sentence."""

    if word_limit < 1:
        raise ValueError("word_limit must be positive")
    words = text.split()
    if len(words) <= word_limit:
        return " ".join(words).strip()
    selected: list[str] = []
    count = 0
    for unit in _sentence_units(text):
        unit_words = unit.split()
        if selected and count + len(unit_words) > word_limit:
            break
        if not selected and len(unit_words) > word_limit:
            return _bounded_words(unit, word_limit)
        selected.append(unit)
        count += len(unit_words)
    return "\n".join(selected).strip() or _bounded_words(text, word_limit)


def select_evidence_sentences(
    question: str,
    evidence: str,
    generated: str,
    *,
    word_limit: int,
) -> str:
    """Select relevant, non-duplicative evidence sentences within a word cap."""

    if word_limit < 1:
        raise ValueError("word_limit must be positive")
    query_tokens = word_tokens(question)
    query = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    generated_tokens = word_tokens(generated)
    generated_set = set(generated_tokens)
    generated_bigrams = set(zip(generated_tokens, generated_tokens[1:]))
    query_numbers = {token for token in query if any(char.isdigit() for char in token)}

    scored: list[tuple[float, int, str, list[str]]] = []
    for index, unit in enumerate(_sentence_units(evidence)):
        tokens = word_tokens(unit)
        token_set = set(tokens)
        bigrams = set(zip(tokens, tokens[1:]))
        query_coverage = len(query & token_set) / max(1, len(query))
        query_bigram = len(query_bigrams & bigrams) / max(1, len(query_bigrams))
        number_coverage = len(query_numbers & token_set) / max(1, len(query_numbers))
        generated_overlap = _jaccard(tokens, generated_tokens)
        generated_bigram_overlap = len(generated_bigrams & bigrams) / max(
            1, len(bigrams)
        )
        novel_relevant = len((query & token_set) - generated_set) / max(1, len(query))
        legal_markers = min(3, len(_LEGAL_MARKER_RE.findall(unit))) / 3.0
        length_penalty = max(0.0, (len(tokens) - 90) / 180.0)
        score = (
            3.0 * query_coverage
            + 2.0 * query_bigram
            + 1.2 * number_coverage
            + 0.8 * legal_markers
            + 0.7 * novel_relevant
            - 0.9 * generated_overlap
            - 0.4 * generated_bigram_overlap
            - 0.3 * length_penalty
            - 0.0005 * index
        )
        scored.append((score, index, unit, tokens))

    selected: list[tuple[int, str, list[str]]] = []
    used = 0
    for _, index, unit, tokens in sorted(scored, key=lambda row: (-row[0], row[1])):
        if any(
            _jaccard(tokens, prior_tokens) >= 0.82
            for _, _, prior_tokens in selected
        ):
            continue
        remaining = word_limit - used
        if remaining <= 0:
            break
        if len(tokens) > remaining:
            if not selected and remaining >= 12:
                selected.append(
                    (index, _bounded_words(unit, remaining), tokens[:remaining])
                )
                used += remaining
            continue
        selected.append((index, unit, tokens))
        used += len(tokens)
    selected.sort(key=lambda row: row[0])
    return "\n".join(unit for _, unit, _ in selected).strip()


def _deduplicate_join(first: str, second: str) -> str:
    first = first.strip()
    second = second.strip()
    if not first:
        return second
    if not second:
        return first
    first_tokens = word_tokens(first)
    second_tokens = word_tokens(second)
    if _jaccard(first_tokens, second_tokens) >= 0.9:
        return first if len(first_tokens) <= len(second_tokens) else second
    return f"{first}\n{second}"


def _plain_join(first: str, second: str) -> str:
    """Join two non-empty sources without changing the historical baseline."""

    return f"{first.strip()}\n{second.strip()}".strip()


def build_answer_candidates(
    question: str,
    qwen_answer: str,
    extractive_answer: str,
) -> list[AnswerCandidate]:
    """Build the fixed P15 candidate family without reading gold answers."""

    if not question.strip() or not qwen_answer.strip() or not extractive_answer.strip():
        raise ValueError("question and source answers must be non-blank")
    candidates: list[AnswerCandidate] = [AnswerCandidate("qwen", qwen_answer.strip())]
    for limit in (256, 320, 384, 448, 512):
        candidates.append(
            AnswerCandidate(
                f"qwen_trim_{limit}",
                trim_at_sentence_boundary(qwen_answer, limit),
            )
        )
    for limit in (48, 80, 112, 160, 224):
        evidence = select_evidence_sentences(
            question,
            extractive_answer,
            qwen_answer,
            word_limit=limit,
        )
        candidates.append(
            AnswerCandidate(
                f"smart_prefix_{limit}",
                _deduplicate_join(evidence, qwen_answer),
            )
        )
        candidates.append(
            AnswerCandidate(
                f"smart_suffix_{limit}",
                _deduplicate_join(qwen_answer, evidence),
            )
        )
    # Preserve the strongest P14 family exactly. The smart variants above
    # improve concision, while these raw prefixes provide a stable METEOR
    # anchor and make regressions against the submitted system visible.
    for limit in (96, 160, 224, 288, 352, 416):
        candidates.append(
            AnswerCandidate(
                f"raw_prefix_{limit}",
                _plain_join(_bounded_words(extractive_answer, limit), qwen_answer),
            )
        )
    for limit in (256, 384, 512):
        candidates.append(
            AnswerCandidate(
                f"evidence_{limit}",
                select_evidence_sentences(
                    question,
                    extractive_answer,
                    "",
                    word_limit=limit,
                ),
            )
        )
    unique: list[AnswerCandidate] = []
    seen_profiles: set[str] = set()
    for candidate in candidates:
        answer = candidate.answer.strip()
        if answer and candidate.profile not in seen_profiles:
            seen_profiles.add(candidate.profile)
            unique.append(AnswerCandidate(candidate.profile, answer))
    return unique


def answer_features(
    question: str,
    answer: str,
    qwen_answer: str,
    extractive_answer: str,
    *,
    profile: str,
    profile_names: Sequence[str],
) -> list[float]:
    """Return deterministic reference-free features for an answer selector."""

    question_tokens = word_tokens(question)
    answer_tokens = word_tokens(answer)
    qwen_tokens = word_tokens(qwen_answer)
    extractive_tokens = word_tokens(extractive_answer)
    answer_counts = Counter(answer_tokens)
    repeated = sum(max(0, count - 1) for count in answer_counts.values())
    question_numbers = {
        token for token in question_tokens if any(char.isdigit() for char in token)
    }
    answer_numbers = {
        token for token in answer_tokens if any(char.isdigit() for char in token)
    }
    profile_vector = [float(profile == name) for name in profile_names]
    return profile_vector + [
        math.log1p(len(answer_tokens)),
        len(answer_tokens) / max(1, len(qwen_tokens)),
        len(answer_tokens) / max(1, len(extractive_tokens)),
        len(set(question_tokens) & set(answer_tokens))
        / max(1, len(set(question_tokens))),
        _jaccard(answer_tokens, qwen_tokens),
        _jaccard(answer_tokens, extractive_tokens),
        repeated / max(1, len(answer_tokens)),
        len(question_numbers & answer_numbers) / max(1, len(question_numbers)),
        min(10, len(_LEGAL_MARKER_RE.findall(answer))) / 10.0,
        float("\n" in answer),
    ]


def validate_candidate_bank(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[str]]:
    """Validate one candidate row per question/profile and return stable axes."""

    if not rows:
        raise ValueError("candidate bank is empty")
    question_ids: list[str] = []
    profiles: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        question_id = str(row.get("id", "")).strip()
        profile = str(row.get("profile", "")).strip()
        answer = row.get("answer")
        if (
            not question_id
            or not profile
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError("candidate rows require non-blank id, profile, and answer")
        key = (question_id, profile)
        if key in seen:
            raise ValueError(f"duplicate candidate row {key!r}")
        seen.add(key)
        if question_id not in question_ids:
            question_ids.append(question_id)
        if profile not in profiles:
            profiles.append(profile)
    expected = set(profiles)
    for question_id in question_ids:
        observed = {profile for row_id, profile in seen if row_id == question_id}
        if observed != expected:
            raise ValueError(f"candidate profile coverage mismatch for {question_id!r}")
    return question_ids, profiles


__all__ = [
    "AnswerCandidate",
    "answer_features",
    "build_answer_candidates",
    "select_evidence_sentences",
    "trim_at_sentence_boundary",
    "validate_candidate_bank",
    "word_tokens",
]
