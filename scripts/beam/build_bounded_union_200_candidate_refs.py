from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable, Iterator


PUBLIC_DIR = Path("artifacts/task1/recovery_096/final_public_v3")
PUBLIC_BM25 = PUBLIC_DIR / "public_bm25_v3_exact.jsonl"
PUBLIC_KNN_WORD = PUBLIC_DIR / "public_knn_word_v3_exact.jsonl"
PUBLIC_KNN_CHAR = PUBLIC_DIR / "public_knn_char_v3_exact.jsonl"
PUBLIC_ADAPTIVE = PUBLIC_DIR / "public_adaptive_k500.jsonl"
# Fresh label-free public dense retrieval output.  The old
# artifacts/task1/raw_k500.jsonl is a 7,000-query/train artifact and must not
# be used for the public union.
PUBLIC_DENSE_RAW = PUBLIC_DIR / "public_raw_k500.jsonl"
PUBLIC_DENSE = PUBLIC_DENSE_RAW
PUBLIC_OUTPUT = PUBLIC_DIR / "public_candidate_union.jsonl"
EXPECTED_PUBLIC_QUERIES = 1000
RRF_K = 60
CAP = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)

                # BTC raw_k500 uses question_id
                if "query_id" not in row and "question_id" in row:
                    row["query_id"] = str(row["question_id"])

                # Normalize raw_k500 hits to the ranking representation used
                # internally by the union builder. Keep the original hits so
                # existing callers that inspect them remain compatible.
                if "ranking" not in row and "hits" in row:
                    hits = row["hits"]
                    if not isinstance(hits, list):
                        raise ValueError(f"{path}: row {row_number} has no hits array")
                    ranked_hits = []
                    seen_docs: set[str] = set()
                    for hit_number, hit in enumerate(hits, 1):
                        if not isinstance(hit, dict) or "doc_id" not in hit or "rank" not in hit:
                            raise ValueError(f"{path}: malformed hit {hit_number} in row {row_number}")
                        doc_id = str(hit["doc_id"])
                        rank = hit["rank"]
                        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                            raise ValueError(f"{path}: invalid 1-based rank in row {row_number}")
                        if doc_id in seen_docs:
                            # Dense retrieval is chunk-level, so multiple hits
                            # may refer to the same document.  Keep the first
                            # occurrence in rank order; the union itself is
                            # document-level and deduplicates by doc_id.
                            continue
                        seen_docs.add(doc_id)
                        ranked_hits.append((rank, doc_id))
                    row["ranking"] = [doc_id for _, doc_id in sorted(ranked_hits)]

                yield row


def _query_map(rows: Iterable[dict], path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row_number, row in enumerate(rows, 1):
        if "query_id" not in row:
            raise ValueError(f"{path}: row {row_number} has no query_id")
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"{path}: duplicate query_id={query_id}")
        result[query_id] = row
        if len(result) > EXPECTED_PUBLIC_QUERIES:
            raise ValueError(f"{path}: public query count exceeds {EXPECTED_PUBLIC_QUERIES}")
    if len(result) != EXPECTED_PUBLIC_QUERIES:
        raise ValueError(f"{path}: public query count must be {EXPECTED_PUBLIC_QUERIES}, got {len(result)}")
    return result


def _ranking_from_array(row: dict, path: Path) -> list[tuple[str, int]]:
    ranking = row.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"{path}: query {row.get('query_id')} has no ranking array")
    seen: set[str] = set()
    result = []
    for rank, doc_id in enumerate(ranking, 1):
        doc_id = str(doc_id)
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate doc_id={doc_id} in query {row['query_id']}")
        seen.add(doc_id)
        result.append((doc_id, rank))
    return result


def _ranking_from_hits(row: dict, path: Path) -> list[tuple[str, int]]:
    hits = row.get("hits")
    if not isinstance(hits, list):
        raise ValueError(f"{path}: query {row.get('query_id')} has no hits array")
    seen: set[str] = set()
    result = []
    for hit_number, hit in enumerate(hits, 1):
        if not isinstance(hit, dict) or "doc_id" not in hit or "rank" not in hit:
            raise ValueError(f"{path}: malformed hit {hit_number} in query {row['query_id']}")
        doc_id = str(hit["doc_id"])
        rank = hit["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"{path}: invalid 1-based rank in query {row['query_id']}")
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate doc_id={doc_id} in query {row['query_id']}")
        seen.add(doc_id)
        result.append((doc_id, rank))
    return result


def _validate_adaptive_rows(path: Path) -> dict[str, dict]:
    rows = _query_map(_read_jsonl(path), path)
    for query_id, row in rows.items():
        if set(row) & {"answer", "gold", "label", "fold"}:
            raise ValueError(f"{path}: public adaptive row contains train-only field: {query_id}")
        required = {"expand_to_k500", "k200_documents", "k500_added_documents", "ranking"}
        if not required.issubset(row):
            raise ValueError(f"{path}: query {query_id} missing {sorted(required - set(row))}")
        k200 = [str(value) for value in row["k200_documents"]]
        added = [str(value) for value in row["k500_added_documents"]]
        ranking = [str(value) for value in row["ranking"]]
        if len(set(k200)) != len(k200) or len(set(added)) != len(added) or len(set(ranking)) != len(ranking):
            raise ValueError(f"{path}: duplicate adaptive document for query {query_id}")
        if ranking != k200 + added:
            raise ValueError(f"{path}: ranking is not k200_documents + k500_added_documents for {query_id}")
        if bool(row["expand_to_k500"]) != bool(added):
            raise ValueError(f"{path}: expansion flag/additions mismatch for {query_id}")
    return rows


def _doc_sort_key(doc_id: str) -> tuple[int, int | str]:
    return (0, int(doc_id)) if doc_id.isdigit() else (1, doc_id)


def _ensure_public_dense_k500(
    dense_raw_path: Path = PUBLIC_DENSE_RAW,
    output_path: Path = PUBLIC_DENSE,
    bm25_path: Path = PUBLIC_BM25,
) -> Path:
    """Validate the fresh public dense K500 artifact.

    It already contains exactly the 1,000 public queries, so the old full
    train/evaluation dense artifact is never consulted.
    """
    if output_path.is_file():
        if output_path == PUBLIC_DENSE:
            rows = _query_map(_read_jsonl(output_path), output_path)
            for query_id, row in rows.items():
                hits = row.get("hits")
                if not isinstance(hits, list) or len(hits) != 500:
                    raise ValueError(
                        f"{output_path}: query {query_id} must contain exactly 500 hits"
                    )
        return output_path
    if not dense_raw_path.is_file():
        raise FileNotFoundError(f"dense raw source missing: {dense_raw_path}")
    # The authoritative public query set is the bm25 public source.
    public_ids = set(_query_map(_read_jsonl(bm25_path), bm25_path))
    if len(public_ids) != EXPECTED_PUBLIC_QUERIES:
        raise ValueError(
            f"public bm25 query count must be {EXPECTED_PUBLIC_QUERIES}, got {len(public_ids)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with dense_raw_path.open(encoding="utf-8") as source:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row.get("query_id") or row.get("question_id"))
                if query_id not in public_ids:
                    continue
                handle.write(line.rstrip("\n") + "\n")
                emitted += 1
    if emitted != EXPECTED_PUBLIC_QUERIES:
        if temporary.is_file():
            temporary.unlink()
        raise ValueError(
            f"dense public subset must contain {EXPECTED_PUBLIC_QUERIES} queries, got {emitted}"
        )
    temporary.replace(output_path)
    return output_path


def _validate_public_query_sets(source_maps: dict[str, dict[str, dict]]) -> list[str]:
    query_sets = {name: set(mapping) for name, mapping in source_maps.items()}
    public_ids = query_sets["bm25"]
    if len(public_ids) != EXPECTED_PUBLIC_QUERIES:
        raise ValueError(f"public query count must be {EXPECTED_PUBLIC_QUERIES}, got {len(public_ids)}")
    for name, query_ids in query_sets.items():
        missing = sorted(public_ids - query_ids, key=_doc_sort_key)
        extra = sorted(query_ids - public_ids, key=_doc_sort_key)
        if missing or extra:
            raise ValueError(f"{name}: query set mismatch; missing={missing[:10]}, extra={extra[:10]}")
    return sorted(public_ids, key=_doc_sort_key)


def build_public_union(
    bm25_path: Path = PUBLIC_BM25,
    knn_word_path: Path = PUBLIC_KNN_WORD,
    knn_char_path: Path = PUBLIC_KNN_CHAR,
    adaptive_path: Path = PUBLIC_ADAPTIVE,
    output_path: Path = PUBLIC_OUTPUT,
    rebuild_incompatible: bool = False,
) -> dict[str, object]:
    source_paths = {
        "adaptive_k500": adaptive_path,
        "bm25": bm25_path,
        "knn_word": knn_word_path,
        "knn_char": knn_char_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("public union input missing:\n" + "\n".join(missing))

    source_maps = {name: (_validate_adaptive_rows(path) if name == "adaptive_k500" else _query_map(_read_jsonl(path), path)) for name, path in source_paths.items()}
    query_ids = _validate_public_query_sets(source_maps)
    rows = []
    for query_id in query_ids:
        source_rows = [
            ("adaptive_k500", _ranking_from_array(source_maps["adaptive_k500"][query_id], adaptive_path)),
            ("bm25", _ranking_from_array(source_maps["bm25"][query_id], bm25_path)),
            ("knn_word", _ranking_from_array(source_maps["knn_word"][query_id], knn_word_path)),
            ("knn_char", _ranking_from_array(source_maps["knn_char"][query_id], knn_char_path)),
        ]
        scores: defaultdict[str, float] = defaultdict(float)
        source_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for source_name, ranking in source_rows:
            for doc_id, rank in ranking:
                scores[doc_id] += 1.0 / (RRF_K + rank)
                source_ranks[doc_id][source_name] = rank
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], _doc_sort_key(doc_id)))[:CAP]
        if not ordered:
            raise ValueError(f"query {query_id} produced no candidates")
        candidates = [
            {
                "doc_id": doc_id,
                "union_rank": rank,
                "rrf_score": scores[doc_id],
                "source_support": len(source_ranks[doc_id]),
                "source_ranks": source_ranks[doc_id],
            }
            for rank, doc_id in enumerate(ordered, 1)
        ]
        rows.append({"query_id": query_id, "candidates": candidates})

    if output_path.is_file():
        existing = next(_read_jsonl(output_path), None)
        existing_keys = set()
        if existing:
            for candidate in existing.get("candidates", []):
                existing_keys.update((candidate.get("source_ranks") or {}).keys())
        if "adaptive_k500" not in existing_keys and not rebuild_incompatible:
            raise RuntimeError(
                f"refusing to overwrite incompatible public union {output_path}; "
                "rerun with --rebuild-compatible after reviewing the backup"
            )
        if "adaptive_k500" not in existing_keys:
            backup = output_path.with_name("public_candidate_union_dense_k500_incompatible.jsonl")
            if not backup.exists():
                shutil.copy2(output_path, backup)
            manifest = output_path.with_name("public_candidate_union_dense_k500_incompatible_manifest.json")
            manifest.write_text(json.dumps({"source": str(output_path), "backup": str(backup), "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "reason": "old union used dense_k500 instead of historical adaptive_k500", "not_for_v3a": True}, indent=2) + "\n", encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output_path)
    source_sha256 = {name: _sha256(path) for name, path in source_paths.items()}
    manifest_path = output_path.with_name("public_candidate_union_manifest.json")
    manifest_path.write_text(json.dumps({
        "schema_version": "public-candidate-union-v2",
        "status": "PUBLIC_CANDIDATE_UNION_PASS",
        "query_count": len(rows),
        "rrf_k": RRF_K,
        "cap": CAP,
        "source_keys": ["adaptive_k500", "bm25", "knn_word", "knn_char"],
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_sha256,
        "output_sha256": _sha256(output_path),
        "no_answer_gold_label_fold_access": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "PUBLIC_CANDIDATE_UNION_PASS",
        "query_count": len(rows),
        "max_candidates": max(len(row["candidates"]) for row in rows),
        "rrf_k": RRF_K,
        "cap": CAP,
        "source_keys": ["adaptive_k500", "bm25", "knn_word", "knn_char"],
        "sha256": _sha256(output_path),
        "source_sha256": source_sha256,
        "manifest": str(manifest_path),
        "no_answer_gold_label_fold_access": True,
    }


def _write_toy_sources(directory: Path) -> tuple[Path, Path, Path, Path]:
    paths = [directory / name for name in ("bm25.jsonl", "word.jsonl", "char.jsonl", "adaptive.jsonl")]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    try:
        for query_number in range(EXPECTED_PUBLIC_QUERIES):
            query_id = str(query_number)
            json.dump({"query_id": query_id, "ranking": ["10", "2"]}, handles[0]); handles[0].write("\n")
            json.dump({"query_id": query_id, "ranking": ["2", "3"]}, handles[1]); handles[1].write("\n")
            json.dump({"query_id": query_id, "ranking": ["3", "10"]}, handles[2]); handles[2].write("\n")
            json.dump({"query_id": query_id, "expand_to_k500": False, "k200_documents": ["10"], "k500_added_documents": [], "ranking": ["10"]}, handles[3]); handles[3].write("\n")
    finally:
        for handle in handles:
            handle.close()
    return tuple(paths)  # type: ignore[return-value]


def self_test() -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="public_union_") as temporary:
        directory = Path(temporary)
        raw_k500 = directory / "raw_k500.jsonl"
        raw_k500.write_text(
            json.dumps(
                {
                    "question_id": "raw-1",
                    "hits": [
                        {"doc_id": "late", "rank": 2, "score": 0.2},
                        {"doc_id": "first", "rank": 1, "score": 0.3},
                        {"doc_id": "first", "rank": 3, "score": 0.1},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        adapted = next(_read_jsonl(raw_k500))
        raw_k500_adapter = adapted["query_id"] == "raw-1" and adapted["ranking"] == ["first", "late"]

        bm25, word, char, dense = _write_toy_sources(directory)
        output = directory / "union.jsonl"
        report = build_public_union(bm25, word, char, dense, output)
        rows = list(_read_jsonl(output))
        unique_queries = {row["query_id"] for row in rows}
        no_duplicate_doc_query = all(len({candidate["doc_id"] for candidate in row["candidates"]}) == len(row["candidates"]) for row in rows)
        no_forbidden_field = all(not ({"gold", "answer", "label", "fold"} & set(row)) for row in rows)
        return {
            "raw_k500_adapter": raw_k500_adapter,
            "count_query_1000": report["query_count"] == EXPECTED_PUBLIC_QUERIES and len(unique_queries) == EXPECTED_PUBLIC_QUERIES,
            "candidate_nonempty": all(row["candidates"] for row in rows),
            "no_duplicate_doc_query": no_duplicate_doc_query,
            "no_gold_field": no_forbidden_field,
        }


def preflight() -> dict[str, object]:
    _ensure_public_dense_k500()
    _validate_adaptive_rows(PUBLIC_ADAPTIVE)
    paths = [PUBLIC_BM25, PUBLIC_KNN_WORD, PUBLIC_KNN_CHAR, PUBLIC_ADAPTIVE, PUBLIC_DENSE_RAW]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("public union input missing:\n" + "\n".join(missing))
    return {"status": "READY_PUBLIC_CANDIDATE_UNION", "required_inputs": [str(path) for path in paths], "query_count_expected": EXPECTED_PUBLIC_QUERIES, "rrf_k": RRF_K, "cap": CAP, "source_keys": ["adaptive_k500", "bm25", "knn_word", "knn_char"], "answer_gold_label_fold_access": False, "beam_launched": False}


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks}, indent=2))
    elif "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    elif "--public" in sys.argv:
        print(json.dumps(build_public_union(rebuild_incompatible="--rebuild-compatible" in sys.argv), indent=2))
    else:
        raise SystemExit("public mode is required; use --public, --self-test, or --preflight")
