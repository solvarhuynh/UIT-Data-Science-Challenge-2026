"""Leakage-safe, cached-ranking hard/semi-hard negative mining for Task 1.

This script neither loads a retrieval model nor trains a model.  Each
``fold_N.jsonl`` contains only queries outside validation fold N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from udsc2026.evaluation.task1_canonical_evidence import (  # type: ignore[import-untyped]
    CanonicalEvidenceResolver,
    corpus_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS = Path("artifacts/task1/evaluation/strict_cv_v2/folds.json")
DEFAULT_RANKINGS = Path("artifacts/task1/train500_dense200_predictions.jsonl")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    text: str
    law_name: str
    rank: int
    score: float | None
    source: str
    ambiguous: bool


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=Path("data/raw/btc/LegalIR/train.json")
    )
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--rankings",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "Cached source rankings; repeat for parent, bm25f, citation, reranker, etc."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task1/training/negatives")
    )
    parser.add_argument("--hard-rank-max", type=_positive, default=20)
    parser.add_argument("--semi-hard-min-rank", type=_positive, default=10)
    parser.add_argument("--semi-hard-max-rank", type=_positive, default=100)
    parser.add_argument("--max-hard-per-query", type=_positive, default=10)
    parser.add_argument("--max-semi-hard-per-query", type=_positive, default=10)
    parser.add_argument(
        "--easy-rank-min",
        type=_positive,
        default=101,
        help="Lowest candidate rank eligible for optional easy/random negatives.",
    )
    parser.add_argument(
        "--max-easy-per-query",
        type=int,
        default=0,
        help="Optional easy_random negatives per query; zero preserves P12 default.",
    )
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.90)
    parser.add_argument("--max-queries", type=_positive)
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed_v3")
    )
    parser.add_argument("--evidence-limit", type=int, choices=(1, 2), default=2)
    return parser


def parse_ranking_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    if not values:
        return [("hcmute", DEFAULT_RANKINGS)]
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        source, separator, raw_path = value.partition("=")
        source = source.strip().casefold()
        if not separator or not source or not raw_path.strip():
            raise ValueError("--rankings must use SOURCE=PATH")
        if source in seen:
            raise ValueError(f"duplicate ranking source: {source}")
        seen.add(source)
        parsed.append((source, Path(raw_path.strip())))
    return parsed


def load_fold_map(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "legal-ir-strict-cv-v2":
        raise ValueError(f"{path} is not a strict_cv_v2 folds artifact")
    result: dict[str, int] = {}
    for row in payload.get("folds", []):
        fold = row.get("fold")
        for query_id in row.get("validation_ids", []):
            if str(query_id) in result:
                raise ValueError(f"duplicate fold query ID: {query_id}")
            result[str(query_id)] = int(fold)
    if not result:
        raise ValueError("strict folds contain no query IDs")
    return result


def load_rankings(
    source: str, path: Path, *, rank_limit: int | None = None
) -> dict[str, list[Candidate]]:
    if not path.is_file():
        raise FileNotFoundError(f"cached ranking not found: {path}")
    records: Iterable[Any]
    if path.suffix.casefold() == ".jsonl":

        def jsonl_records() -> Iterable[Any]:
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

        records = jsonl_records()
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = (
            payload if isinstance(payload, list) else payload.get("predictions", [])
        )
    output: dict[str, list[Candidate]] = {}
    record_count = 0
    for record_count, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        query_id = str(record.get("question_id", record.get("id", "")))
        if not query_id:
            continue
        entries = record.get("hits", record.get("documents", []))
        if not isinstance(entries, list):
            continue
        per_doc: dict[str, Candidate] = {}
        for index, raw in enumerate(entries, 1):
            item = raw if isinstance(raw, dict) else {"doc_id": raw}
            raw_rank = item.get("rank", index)
            rank = (
                int(raw_rank)
                if isinstance(raw_rank, int) and not isinstance(raw_rank, bool)
                else index
            )
            if rank_limit is not None and rank > rank_limit:
                continue
            evidence_rows = item.get("evidence", [])
            evidence_text = (
                "\n\n".join(
                    str(part.get("text", "")).strip()
                    for part in evidence_rows
                    if isinstance(part, dict) and str(part.get("text", "")).strip()
                )
                if isinstance(evidence_rows, list)
                else ""
            )
            doc_id = str(
                item.get("doc_id", item.get("document_id", item.get("id", "")))
            ).strip()
            if not doc_id or doc_id in per_doc:
                continue
            raw_metadata = item.get("metadata")
            metadata: dict[str, Any] = (
                dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            )
            raw_warnings = metadata.get("structure_warnings", [])
            raw_reasons = metadata.get("review_reasons", [])
            warnings = raw_warnings if isinstance(raw_warnings, list) else []
            reasons = raw_reasons if isinstance(raw_reasons, list) else []
            ambiguous = any(
                "ambiguous" in str(value).casefold() for value in [*warnings, *reasons]
            )
            score = next(
                (
                    item[key]
                    for key in ("rerank_score", "final_score", "dense_score", "score")
                    if isinstance(item.get(key), (int, float))
                ),
                None,
            )
            per_doc[doc_id] = Candidate(
                doc_id=doc_id,
                text=str(item.get("text", "")).strip() or evidence_text,
                law_name=str(
                    item.get("law_name", metadata.get("law_name", ""))
                ).strip(),
                rank=rank,
                score=float(score) if score is not None else None,
                source=source,
                ambiguous=ambiguous,
            )
        output[query_id] = sorted(
            per_doc.values(), key=lambda item: (item.rank, item.doc_id)
        )
        if record_count % 500 == 0:
            print(
                f"loaded ranking {source}: {record_count} queries",
                flush=True,
            )
    if record_count:
        print(
            f"loaded ranking {source}: {record_count} queries complete",
            flush=True,
        )
    return output


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value) if len(token) > 1}


def _near_duplicate(
    candidate: Candidate, positives: Sequence[Candidate], threshold: float
) -> bool:
    if not candidate.text:
        return False
    candidate_tokens = _tokens(candidate.text)
    if not candidate_tokens:
        return False
    for positive in positives:
        positive_tokens = _tokens(positive.text)
        if positive_tokens:
            similarity = len(candidate_tokens & positive_tokens) / len(
                candidate_tokens | positive_tokens
            )
            if similarity >= threshold:
                return True
    return False


def classify_negative(
    candidate: Candidate,
    *,
    query: str,
    positive_laws: set[str],
    hard_rank_max: int,
    semi_min: int,
    semi_max: int,
    easy_rank_min: int,
) -> tuple[str, str] | None:
    if candidate.law_name and candidate.law_name.casefold() in positive_laws:
        return "same_law", "hard" if candidate.rank <= hard_rank_max else "semi_hard"
    source_is_dense = any(
        token in candidate.source for token in ("dense", "hcmute", "parent", "semantic")
    )
    if source_is_dense and candidate.rank <= hard_rank_max:
        return "semantic_confuser", "hard"
    if candidate.rank <= hard_rank_max:
        return "hard_false_positive", "hard"
    if semi_min <= candidate.rank <= semi_max:
        if len(_tokens(query) & _tokens(candidate.text)) >= 2:
            return "lexical_confuser", "semi_hard"
        return "semi_hard", "semi_hard"
    if candidate.rank >= easy_rank_min:
        return "easy_random", "easy"
    return None


def mine_query(
    query_id: str,
    query: str,
    gold_docs: Sequence[str],
    by_source: dict[str, list[Candidate]],
    *,
    hard_rank_max: int,
    semi_min: int,
    semi_max: int,
    max_hard: int,
    max_semi: int,
    easy_rank_min: int,
    max_easy: int,
    near_duplicate_jaccard: float,
    resolver: CanonicalEvidenceResolver | None = None,
) -> list[dict[str, Any]]:
    gold = set(gold_docs)
    all_candidates = [
        candidate for candidates in by_source.values() for candidate in candidates
    ]
    positive_by_doc: dict[str, Candidate] = {}
    for candidate in all_candidates:
        if candidate.doc_id in gold:
            positive_by_doc.setdefault(candidate.doc_id, candidate)
    if resolver is not None:
        for document_id in gold_docs:
            if document_id in positive_by_doc:
                continue
            canonical = resolver.resolve(document_id)
            if canonical is not None:
                positive_by_doc[document_id] = Candidate(
                    doc_id=document_id,
                    text=canonical.text,
                    law_name=canonical.title or "",
                    rank=10**9,
                    score=None,
                    source=canonical.source,
                    ambiguous=False,
                )
    positive_laws = {
        candidate.law_name.casefold()
        for candidate in positive_by_doc.values()
        if candidate.law_name
    }
    selected: dict[str, tuple[Candidate, str, str]] = {}
    for candidate in sorted(
        all_candidates, key=lambda item: (item.rank, item.source, item.doc_id)
    ):
        if (
            candidate.doc_id in gold
            or candidate.ambiguous
            or _near_duplicate(
                candidate, list(positive_by_doc.values()), near_duplicate_jaccard
            )
        ):
            continue
        classified = classify_negative(
            candidate,
            query=query,
            positive_laws=positive_laws,
            hard_rank_max=hard_rank_max,
            semi_min=semi_min,
            semi_max=semi_max,
            easy_rank_min=easy_rank_min,
        )
        if classified is None:
            continue
        negative_type, strength = classified
        previous = selected.get(candidate.doc_id)
        if previous is None or candidate.rank < previous[0].rank:
            selected[candidate.doc_id] = (candidate, negative_type, strength)
    hard = [item for item in selected.values() if item[2] == "hard"][:max_hard]
    semi = [item for item in selected.values() if item[2] == "semi_hard"][:max_semi]
    easy = [item for item in selected.values() if item[2] == "easy"][:max_easy]
    rows: list[dict[str, Any]] = []
    for positive_doc in gold_docs:
        canonical = resolver.resolve(positive_doc) if resolver else None
        for negative, negative_type, strength in [*hard, *semi, *easy]:
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "positive_doc": positive_doc,
                    "positive_document_id": positive_doc,
                    "positive_evidence": canonical.text if canonical else None,
                    "positive_evidence_ids": list(canonical.evidence_ids)
                    if canonical
                    else [],
                    "positive_evidence_source": canonical.source if canonical else None,
                    "positive_resolution_status": "resolved"
                    if canonical
                    else "unresolved",
                    "negative_doc": negative.doc_id,
                    "negative_evidence": negative.text or None,
                    "negative_type": negative_type,
                    "strength": strength,
                    "source_rank": negative.rank,
                    "source_score": negative.score,
                    "provenance": {
                        "source": negative.source,
                        "law_name": negative.law_name or None,
                        "safety": "not_gold_not_ambiguous_not_near_duplicate",
                    },
                }
            )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.semi_hard_min_rank > args.semi_hard_max_rank:
        raise SystemExit("--semi-hard-min-rank must be <= --semi-hard-max-rank")
    if args.max_easy_per_query < 0:
        raise SystemExit("--max-easy-per-query must be zero or greater")
    if not 0.0 <= args.near_duplicate_jaccard <= 1.0:
        raise SystemExit("--near-duplicate-jaccard must be between 0 and 1")
    train = json.loads(args.train.read_text(encoding="utf-8"))
    resolver = CanonicalEvidenceResolver(
        args.processed_root, evidence_limit=args.evidence_limit
    )
    if not isinstance(train, dict):
        raise SystemExit("--train must contain the LegalIR object mapping")
    fold_map = load_fold_map(args.folds)
    specs = parse_ranking_specs(args.rankings)
    ranking_limit = max(args.hard_rank_max, args.semi_hard_max_rank)
    if args.max_easy_per_query:
        ranking_limit = max(
            ranking_limit,
            args.easy_rank_min + args.max_easy_per_query - 1,
        )
    rankings = {
        source: load_rankings(source, path, rank_limit=ranking_limit)
        for source, path in specs
    }
    query_ids = sorted(
        set(fold_map)
        & set(train)
        & set().union(*(set(rows) for rows in rankings.values()))
    )
    if args.max_queries:
        query_ids = query_ids[: args.max_queries]
    if not query_ids:
        raise SystemExit("no queries overlap train, folds, and cached rankings")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "folds": {},
        "negative_types": Counter(),
        "rank_bands": Counter(),
        "same_law_count": 0,
        "query_count": len(query_ids),
        "record_count": 0,
    }
    for training_fold in sorted(set(fold_map.values())):
        print(f"mining fold {training_fold}/4", flush=True)
        records: list[dict[str, Any]] = []
        for query_id in query_ids:
            if fold_map[query_id] == training_fold:
                continue
            item = train[query_id]
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("answer"), list)
                or not item["answer"]
            ):
                continue
            query = str(item.get("question", "")).strip()
            if not query:
                continue
            rows = mine_query(
                query_id,
                query,
                [str(value) for value in item["answer"]],
                {
                    source: values.get(query_id, [])
                    for source, values in rankings.items()
                },
                hard_rank_max=args.hard_rank_max,
                semi_min=args.semi_hard_min_rank,
                semi_max=args.semi_hard_max_rank,
                max_hard=args.max_hard_per_query,
                max_semi=args.max_semi_hard_per_query,
                easy_rank_min=args.easy_rank_min,
                max_easy=args.max_easy_per_query,
                near_duplicate_jaccard=args.near_duplicate_jaccard,
                resolver=resolver,
            )
            for row in rows:
                row["fold"] = training_fold
                row["query_validation_fold"] = fold_map[query_id]
                row["split"] = "train"
                stats["negative_types"][row["negative_type"]] += 1
                stats["rank_bands"][
                    "hard_1_20"
                    if row["strength"] == "hard"
                    else (
                        "easy_101_plus" if row["strength"] == "easy" else "semi_10_100"
                    )
                ] += 1
                stats["same_law_count"] += row["negative_type"] == "same_law"
            records.extend(rows)
        path = output_dir / f"fold_{training_fold}.jsonl"
        _write_jsonl(path, records)
        stats["folds"][str(training_fold)] = {
            "path": str(path),
            "record_count": len(records),
            "excluded_validation_fold": training_fold,
        }
        stats["record_count"] += len(records)
        print(
            f"mined fold {training_fold}: {len(records)} records",
            flush=True,
        )
    stats["negative_types"] = dict(stats["negative_types"])
    stats["rank_bands"] = dict(stats["rank_bands"])
    stats["same_law_ratio"] = (
        stats["same_law_count"] / stats["record_count"]
        if stats["record_count"]
        else 0.0
    )
    manifest = {
        "schema_version": "task1-negative-mining-v1",
        "train": str(args.train),
        "train_sha256": _sha256(args.train),
        "strict_folds": str(args.folds),
        "strict_folds_sha256": _sha256(args.folds),
        "canonical_corpus": {
            "root": str(args.processed_root),
            "sha256": corpus_fingerprint(args.processed_root),
            "evidence_limit": args.evidence_limit,
            "positive_evidence_source": "canonical_corpus",
        },
        "ranking_sources": [
            {"source": source, "path": str(path), "sha256": _sha256(path)}
            for source, path in specs
        ],
        "safety": {
            "exclude_gold": True,
            "exclude_ambiguous": True,
            "near_duplicate_jaccard": args.near_duplicate_jaccard,
        },
        "sampling": {
            "hard_rank_max": args.hard_rank_max,
            "semi_hard_min_rank": args.semi_hard_min_rank,
            "semi_hard_max_rank": args.semi_hard_max_rank,
            "max_hard_per_query": args.max_hard_per_query,
            "max_semi_hard_per_query": args.max_semi_hard_per_query,
            "easy_rank_min": args.easy_rank_min,
            "max_easy_per_query": args.max_easy_per_query,
        },
        "leakage_policy": (
            "fold_F training output excludes all validation IDs assigned to F"
        ),
        "statistics": stats,
    }
    _write_json(output_dir / "mining_manifest.json", manifest)
    (output_dir / "statistics.md").write_text(
        "# Task1 negative mining statistics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        "| Queries | %d |\n| Records | %d |\n"
        "| Same-law ratio | %.4f |\n\n"
        "## Negative types\n\n%s\n\n## Rank bands\n\n%s\n"
        % (
            stats["query_count"],
            stats["record_count"],
            stats["same_law_ratio"],
            "\n".join(
                f"- {key}: {value}"
                for key, value in sorted(stats["negative_types"].items())
            )
            or "- none",
            "\n".join(
                f"- {key}: {value}"
                for key, value in sorted(stats["rank_bands"].items())
            )
            or "- none",
        ),
        encoding="utf-8",
    )
    print(
        "Mined "
        f"{stats['record_count']} records for {stats['query_count']} queries "
        f"-> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
