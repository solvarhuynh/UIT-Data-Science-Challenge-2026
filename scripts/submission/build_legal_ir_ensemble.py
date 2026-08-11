"""Build a CPU-only four-source ensemble for DSC2026 LegalIR.

The output is the internal ``[{"id": ..., "documents": [...]}]`` hand-off
accepted by ``write_legal_ir_submission.py``.  Dense and BGE rankings are
loaded from existing prediction JSONL files; no neural model is loaded here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import normalize_legal_ir_matching_question  # noqa: E402, I001
from udsc2026.evaluation.legal_ir_lexical import (  # noqa: E402, I001
    LegalContext,
    build_bm25f_rankings,
    build_knn_rankings,
)


OUTPUT_DOCUMENTS = 5
KNN_NEIGHBORS = 20
BM25_TOP_K = 100
BM25_K1 = 0.7
BM25_B = 0.3
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WORD_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class LabeledQuestion:
    """One validated question and its relevant document IDs."""

    question_id: str
    question: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class BuildStats:
    """Small set of counters printed by the CLI."""

    exact_matches: int
    overlay_documents_added: int


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


def _validate_opaque_id(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty without surrounding whitespace")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _document_id(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{label} must be a string or integer")
    return _validate_opaque_id(str(value), label=label)


def _question_text(record: object, *, label: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{label} must be an object")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{label} must have a non-empty string 'question'")
    return question.strip()


def _load_questions(path: Path) -> dict[str, str]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions root must be a non-empty object mapping IDs")
    result: dict[str, str] = {}
    for raw_id, record in payload.items():
        question_id = _validate_opaque_id(raw_id, label="question ID")
        result[question_id] = _question_text(record, label=f"question {question_id!r}")
    return result


def _load_exclusions(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    excluded: set[str] = set()
    excluded_questions: set[str] = set()
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload, dict):
            values: list[tuple[object, object | None]] = [
                (raw_id, record) for raw_id, record in payload.items()
            ]
        elif isinstance(payload, list):
            values = []
            for index, item in enumerate(payload):
                if isinstance(item, str):
                    values.append((item, None))
                elif isinstance(item, dict):
                    raw_id = item.get("id", item.get("question_id"))
                    if raw_id is None:
                        raise ValueError(
                            f"{path}[{index}] needs an 'id' or 'question_id'"
                        )
                    values.append((raw_id, item))
                else:
                    raise TypeError(
                        f"{path}[{index}] must be a string or question object"
                    )
        else:
            raise TypeError(f"exclusion source {path} must be an object or an array")
        for value, record in values:
            excluded.add(_validate_opaque_id(value, label="excluded question ID"))
            if isinstance(record, dict) and isinstance(record.get("question"), str):
                normalized = normalize_question(record["question"])
                if normalized:
                    excluded_questions.add(normalized)
    return excluded, excluded_questions


def _load_labeled_questions(
    paths: Sequence[Path],
    excluded_ids: set[str],
    excluded_questions: set[str],
) -> list[LabeledQuestion]:
    if not paths:
        raise ValueError("at least one --labeled mapping is required")
    by_id: dict[str, LabeledQuestion] = {}
    order: list[str] = []
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"labeled mapping must be a non-empty object: {path}")
        for raw_id, record in payload.items():
            question_id = _validate_opaque_id(raw_id, label="labeled question ID")
            if question_id in excluded_ids:
                continue
            question = _question_text(record, label=f"labeled question {question_id!r}")
            if normalize_question(question) in excluded_questions:
                continue
            assert isinstance(record, dict)
            raw_documents = record.get("answer")
            if not isinstance(raw_documents, list) or not raw_documents:
                raise ValueError(
                    f"labeled question {question_id!r} needs a non-empty answer array"
                )
            documents = tuple(
                _document_id(value, label=f"answer for {question_id!r}")
                for value in raw_documents
            )
            if len(documents) != len(set(documents)):
                raise ValueError(f"answer for {question_id!r} contains duplicates")
            if len(documents) > OUTPUT_DOCUMENTS:
                raise ValueError(
                    f"answer for {question_id!r} exceeds {OUTPUT_DOCUMENTS} documents"
                )
            candidate = LabeledQuestion(question_id, question, documents)
            previous = by_id.get(question_id)
            if previous is None:
                by_id[question_id] = candidate
                order.append(question_id)
            elif previous != candidate:
                raise ValueError(
                    f"conflicting duplicate labeled question ID {question_id!r}"
                )
    if not order:
        raise ValueError("question-source exclusions removed every labeled question")
    deduplicated: list[LabeledQuestion] = []
    seen_pairs: set[tuple[str, tuple[str, ...]]] = set()
    for question_id in order:
        record = by_id[question_id]
        pair = (normalize_question(record.question), record.documents)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduplicated.append(record)
    return deduplicated


def _load_contexts(contexts_dir: Path) -> tuple[list[str], list[str]]:
    """Backward-compatible passage-only context loader."""

    contexts = _load_context_records(contexts_dir)
    return [context.doc_id for context in contexts], [
        context.passage for context in contexts
    ]


def _load_context_records(contexts_dir: Path) -> list[LegalContext]:
    """Load IDs, optional name/title, and passages without altering raw data."""
    if not contexts_dir.is_dir():
        raise NotADirectoryError(f"contexts directory does not exist: {contexts_dir}")
    paths = sorted(contexts_dir.glob("*.json"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"contexts directory contains no JSON files: {contexts_dir}")
    documents: dict[str, LegalContext] = {}
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"context root must be an object: {path}")
        document_id = _document_id(payload.get("id"), label=f"context ID in {path}")
        passage = payload.get("passage")
        if not isinstance(passage, str):
            raise TypeError(f"context {document_id!r} passage must be a string")
        raw_title = payload.get("title", payload.get("name", ""))
        if raw_title is None:
            raw_title = ""
        if not isinstance(raw_title, str):
            raise TypeError(f"context {document_id!r} title/name must be a string")
        if document_id in documents:
            raise ValueError(f"duplicate context document ID {document_id!r}")
        documents[document_id] = LegalContext(document_id, passage, raw_title)
    ordered_ids = sorted(documents, key=lambda value: (len(value), value))
    return [documents[document_id] for document_id in ordered_ids]


def _load_prediction_rankings(
    path: Path,
    expected_question_ids: Sequence[str],
    allowed_document_ids: set[str],
    *,
    rank_limit: int,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    rows: dict[str, list[str]] = {}
    chunk_pools: dict[str, set[str]] = {}
    try:
        stream = path.open(encoding="utf-8-sig")
    except OSError:
        raise
    with stream:
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
            question_id = _validate_opaque_id(
                row.get("question_id"), label=f"question_id at {path}:{line_number}"
            )
            if question_id in rows:
                raise ValueError(
                    f"duplicate prediction question ID {question_id!r} in {path}"
                )
            hits = row.get("hits")
            if not isinstance(hits, list) or not hits:
                raise ValueError(f"question {question_id!r} has no prediction hits")
            seen_documents: set[str] = set()
            seen_chunks: set[str] = set()
            ranking: list[str] = []
            for hit_index, hit in enumerate(hits):
                if not isinstance(hit, dict):
                    raise TypeError(
                        f"hit {hit_index} for {question_id!r} must be an object"
                    )
                document_id = _document_id(
                    hit.get("doc_id"), label=f"doc_id for {question_id!r}"
                )
                if document_id not in allowed_document_ids:
                    raise ValueError(
                        f"prediction for {question_id!r} references unknown "
                        f"document {document_id!r}"
                    )
                chunk_id = _validate_opaque_id(
                    hit.get("chunk_id"), label=f"chunk_id for {question_id!r}"
                )
                if chunk_id in seen_chunks:
                    raise ValueError(
                        f"prediction for {question_id!r} repeats chunk {chunk_id!r}"
                    )
                seen_chunks.add(chunk_id)
                if document_id not in seen_documents:
                    seen_documents.add(document_id)
                    if len(ranking) < rank_limit:
                        ranking.append(document_id)
            rows[question_id] = ranking
            chunk_pools[question_id] = seen_chunks
    if not rows:
        raise ValueError(f"prediction JSONL is empty: {path}")
    expected = set(expected_question_ids)
    observed = set(rows)
    if expected != observed:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            f"prediction coverage mismatch in {path}: missing={missing}, extra={extra}"
        )
    return rows, chunk_pools


def normalize_question(text: str) -> str:
    """Normalize with the shared strict-CV and supervised matching contract."""

    return normalize_legal_ir_matching_question(text)


def _build_knn_rankings(
    questions: dict[str, str], labeled: Sequence[LabeledQuestion]
) -> dict[str, list[str]]:
    return build_knn_rankings(
        questions,
        [(record.question_id, record.question, record.documents) for record in labeled],
        analyzer="word",
        ngram_range=(1, 2),
        neighbors=KNN_NEIGHBORS,
    )


def _build_char_knn_rankings(
    questions: dict[str, str], labeled: Sequence[LabeledQuestion]
) -> dict[str, list[str]]:
    """Legal spelling/format tolerant KNN, separate from word KNN."""

    return build_knn_rankings(
        questions,
        [(record.question_id, record.question, record.documents) for record in labeled],
        analyzer="char_wb",
        ngram_range=(3, 5),
        neighbors=KNN_NEIGHBORS,
    )


def _build_bm25_rankings(
    questions: dict[str, str],
    document_ids: Sequence[str],
    passages: Sequence[str],
) -> dict[str, list[str]]:
    """Backward-compatible passage-only (title weight 0) BM25 ranking."""

    return build_bm25f_rankings(
        questions,
        [
            LegalContext(doc_id, passage)
            for doc_id, passage in zip(document_ids, passages)
        ],
        title_weight=0.0,
        top_k=BM25_TOP_K,
    )


def _build_bm25f_rankings(
    questions: dict[str, str], contexts: Sequence[LegalContext], title_weight: float
) -> dict[str, list[str]]:
    return build_bm25f_rankings(
        questions, contexts, title_weight=title_weight, top_k=BM25_TOP_K
    )


def _legacy_build_bm25_rankings(
    questions: dict[str, str], document_ids: Sequence[str], passages: Sequence[str]
) -> dict[str, list[str]]:

    query_ids = list(questions)
    query_texts = [questions[item] for item in query_ids]
    vocabulary_terms: dict[str, int] = {}
    term_lengths: dict[str, int] = {}
    for question in query_texts:
        tokens = _WORD_RE.findall(unicodedata.normalize("NFKC", question).casefold())
        for ngram_length in (1, 2, 3):
            for start in range(len(tokens) - ngram_length + 1):
                term = " ".join(tokens[start : start + ngram_length])
                term_lengths[term] = ngram_length
    for index, term in enumerate(sorted(term_lengths)):
        vocabulary_terms[term] = index
    vectorizer = CountVectorizer(
        vocabulary=vocabulary_terms,
        ngram_range=(1, 3),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    if not vocabulary_terms:
        raise ValueError("cannot build BM25 query vocabulary from empty questions")
    query_counts = vectorizer.transform(query_texts).tocsr()
    document_counts = vectorizer.transform(passages).tocsr()

    document_count = len(document_ids)
    document_lengths = np.asarray(
        [max(1, len(_WORD_RE.findall(passage))) for passage in passages],
        dtype=np.float32,
    )
    average_length = float(document_lengths.mean())
    document_frequency = np.bincount(
        document_counts.indices,
        minlength=document_counts.shape[1],
    ).astype(np.float32)
    inverse_document_frequency = np.log1p(
        (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)
    phrase_weights = np.asarray(
        [term_lengths[term] for term in sorted(term_lengths)],
        dtype=np.float32,
    )

    weighted_documents = document_counts.copy()
    row_indices = np.repeat(
        np.arange(document_count, dtype=np.int64),
        np.diff(weighted_documents.indptr),
    )
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_lengths / average_length)
    term_frequency = weighted_documents.data
    term_indices = weighted_documents.indices
    weighted_documents.data = (
        (term_frequency * (BM25_K1 + 1.0))
        / (term_frequency + length_norm[row_indices])
        * inverse_document_frequency[term_indices]
        * phrase_weights[term_indices]
    ).astype(np.float32)

    query_presence = query_counts.copy()
    query_presence.data.fill(1.0)
    score_matrix = (query_presence @ weighted_documents.T).tocsr()
    rankings: dict[str, list[str]] = {}
    for row_index, question_id in enumerate(query_ids):
        row = score_matrix.getrow(row_index)
        candidates = [
            (document_ids[int(index)], float(score))
            for index, score in zip(row.indices, row.data)
            if score > 0.0
        ]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        rankings[question_id] = [
            document_id for document_id, _ in candidates[:BM25_TOP_K]
        ]
    return rankings


def _exact_label_map(
    labeled: Sequence[LabeledQuestion],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for record in labeled:
        normalized = normalize_question(record.question)
        if not normalized:
            continue
        documents = mapping.setdefault(normalized, [])
        for document_id in record.documents:
            if document_id not in documents:
                documents.append(document_id)
        if len(documents) > OUTPUT_DOCUMENTS:
            raise ValueError(
                "normalized duplicate labeled questions disagree on more than "
                f"{OUTPUT_DOCUMENTS} unique documents: {record.question_id!r}"
            )
    return mapping


def _weighted_rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float],
    rrf_k: int,
) -> list[str]:
    scores: dict[str, float] = {}
    source_ranks: list[dict[str, int]] = []
    for ranking, weight in zip(rankings, weights):
        rank_map = {document_id: rank for rank, document_id in enumerate(ranking, 1)}
        source_ranks.append(rank_map)
        if weight == 0.0:
            continue
        for document_id, rank in rank_map.items():
            scores[document_id] = scores.get(document_id, 0.0) + weight / (rrf_k + rank)
    return sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            *(ranks.get(document_id, 10**9) for ranks in source_ranks),
            document_id,
        ),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument(
        "--labeled",
        type=Path,
        action="append",
        required=True,
        help="Repeatable organizer mapping id -> {question, answer:[doc IDs]}.",
    )
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--components-output",
        type=Path,
        help=(
            "Optional cached JSON with per-ID dense/BGE/KNN/BM25 rankings; "
            "use it to retune RRF without rescanning the context corpus."
        ),
    )
    parser.add_argument(
        "--exclude-question-sources",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeatable mapping/ID-list whose IDs are removed from --labeled. "
            "Pass the validation query source to prevent self-label leakage."
        ),
    )
    parser.add_argument(
        "--exact-label-overlay",
        action="store_true",
        help=(
            "For normalized exact question matches, force all known gold docs "
            "into top 5."
        ),
    )
    parser.add_argument("--dense-weight", type=float, default=0.20)
    parser.add_argument("--bge-weight", type=float, default=0.50)
    parser.add_argument("--knn-weight", type=float, default=0.20)
    parser.add_argument("--bm25-weight", type=float, default=0.10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--component-top-k",
        type=int,
        default=200,
        help="Document-deduplicated dense/BGE depth used and cached (default: 200).",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], BuildStats]:
    weights = (
        args.dense_weight,
        args.bge_weight,
        args.knn_weight,
        args.bm25_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("component weights must be finite and non-negative")
    if not any(value > 0.0 for value in weights):
        raise ValueError("at least one component weight must be positive")
    if args.rrf_k < 0:
        raise ValueError("--rrf-k must be non-negative")
    if args.component_top_k < OUTPUT_DOCUMENTS:
        raise ValueError(f"--component-top-k must be at least {OUTPUT_DOCUMENTS}")

    protected = [
        args.questions,
        args.dense,
        args.reranked,
        *args.labeled,
        *args.exclude_question_sources,
    ]
    destinations = [args.output]
    if args.components_output is not None:
        destinations.append(args.components_output)
        if _paths_collide(args.output, args.components_output):
            raise ValueError("--output and --components-output must be different files")
    for destination in destinations:
        for source in protected:
            if _paths_collide(source, destination):
                raise ValueError(f"output must not overwrite input {source}")

    questions = _load_questions(args.questions)
    excluded_ids, excluded_questions = _load_exclusions(args.exclude_question_sources)
    labeled = _load_labeled_questions(
        args.labeled,
        excluded_ids,
        excluded_questions,
    )
    document_ids, passages = _load_contexts(args.contexts_dir)
    allowed_documents = set(document_ids)
    for record in labeled:
        unknown = set(record.documents) - allowed_documents
        if unknown:
            raise ValueError(
                f"labeled question {record.question_id!r} references unknown "
                f"documents {sorted(unknown)}"
            )

    question_ids = list(questions)
    dense, dense_chunks = _load_prediction_rankings(
        args.dense,
        question_ids,
        allowed_documents,
        rank_limit=args.component_top_k,
    )
    bge, bge_chunks = _load_prediction_rankings(
        args.reranked,
        question_ids,
        allowed_documents,
        rank_limit=args.component_top_k,
    )
    for question_id in question_ids:
        if dense_chunks[question_id] != bge_chunks[question_id]:
            raise ValueError(
                f"dense/BGE candidate chunk pool mismatch at {question_id!r}"
            )

    knn = _build_knn_rankings(questions, labeled)
    bm25 = _build_bm25_rankings(questions, document_ids, passages)
    exact_labels = _exact_label_map(labeled) if args.exact_label_overlay else {}

    output: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    exact_matches = 0
    overlay_documents_added = 0
    for question_id, question in questions.items():
        source_rankings = (
            dense[question_id],
            bge[question_id],
            knn[question_id],
            bm25[question_id],
        )
        fused = _weighted_rrf(source_rankings, weights, args.rrf_k)
        if len(fused) < OUTPUT_DOCUMENTS:
            raise ValueError(
                f"fusion for {question_id!r} produced only {len(fused)} documents"
            )
        selected = fused[:OUTPUT_DOCUMENTS]
        known = exact_labels.get(normalize_question(question), [])
        if known:
            exact_matches += 1
            overlay_documents_added += len(set(known) - set(selected))
            selected = (known + [item for item in selected if item not in known])[
                :OUTPUT_DOCUMENTS
            ]
        if len(selected) != OUTPUT_DOCUMENTS or len(selected) != len(set(selected)):
            raise AssertionError(f"invalid final ranking for {question_id!r}")
        if not set(selected) <= allowed_documents:
            raise AssertionError(f"unknown final document for {question_id!r}")
        output.append({"id": question_id, "documents": selected})
        components.append(
            {
                "id": question_id,
                "dense": dense[question_id],
                "bge": bge[question_id],
                "knn": knn[question_id][: args.component_top_k],
                "bm25": bm25[question_id],
            }
        )

    if len(output) != len(questions):
        raise AssertionError("internal output lost questions")
    _write_json(args.output, output)
    if args.components_output is not None:
        _write_json(args.components_output, components)
    return output, BuildStats(exact_matches, overlay_documents_added)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output, stats = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR ensemble error: {exc}", file=sys.stderr)
        return 2
    print(
        f"questions={len(output)} documents_per_question={OUTPUT_DOCUMENTS} "
        f"exact_matches={stats.exact_matches} "
        f"overlay_added={stats.overlay_documents_added}"
    )
    print(args.output)
    if args.components_output is not None:
        print(f"components={args.components_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
