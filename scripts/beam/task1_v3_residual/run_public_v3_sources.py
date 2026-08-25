"""Generate exact public V3 BM25, KNN-word, and KNN-char source rankings.

This is CPU-only source materialization.  It uses the locked lexical helpers;
it does not read public answers, dense retrieval, or any public labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.submission import build_legal_ir_ensemble as _public_helpers
from udsc2026.evaluation.legal_ir_lexical import (
    LegalContext,
    build_bm25f_rankings,
    build_knn_rankings,
)

DEFAULT_OUT = ROOT / "artifacts/task1/recovery_096/final_public_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or len(payload) != 1000:
        raise ValueError("public questions must contain exactly 1000 records")
    result: dict[str, str] = {}
    for raw_id, row in payload.items():
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ValueError(f"invalid public question: {raw_id}")
        # Deliberately do not access row["answer"].
        result[str(raw_id)] = str(row["question"])
    if len(result) != 1000:
        raise ValueError("duplicate public question IDs")
    return result


def load_contexts(path: Path) -> list[LegalContext]:
    contexts: list[LegalContext] = []
    for item in sorted(path.glob("*.json")):
        row = json.loads(item.read_text(encoding="utf-8-sig"))
        contexts.append(LegalContext(str(row["id"]), str(row["passage"]), str(row.get("title", row.get("name", "")) or "")))
    if not contexts:
        raise ValueError(f"no contexts found: {path}")
    return contexts


def load_labeled_pool(paths: list[Path], public_ids: set[str]) -> list[tuple[str, str, list[str]]]:
    labeled = _public_helpers._load_labeled_questions(paths, public_ids, set())
    pool = [(str(item.question_id), str(item.question), [str(doc) for doc in item.documents]) for item in labeled]
    if not pool:
        raise ValueError("empty labeled KNN pool")
    return pool


def write_rankings(path: Path, rankings: dict[str, list[str]], expected_ids: list[str]) -> None:
    if list(rankings) != expected_ids or len(rankings) != 1000:
        raise ValueError(f"ranking coverage mismatch: {path}")
    for qid in expected_ids:
        docs = [str(doc) for doc in rankings[qid]]
        if len(docs) != len(set(docs)):
            raise ValueError(f"duplicate document ranking: {qid}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text("".join(json.dumps({"query_id": qid, "ranking": rankings[qid]}, ensure_ascii=False) + "\n" for qid in expected_ids), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data/raw/btc/LegalIR/public-official.json")
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--warmup", type=Path, default=ROOT / "data/task1/warmup.json")
    parser.add_argument("--contexts-dir", type=Path, default=ROOT / "data/raw/btc/LegalIR/selected-contexts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    questions = load_questions(args.questions)
    ids = list(questions)
    contexts = load_contexts(args.contexts_dir)
    pool = load_labeled_pool([args.train, args.warmup], set(ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "public_bm25_v3_exact.jsonl": build_bm25f_rankings(questions, contexts, title_weight=0.0, top_k=200),
        "public_knn_word_v3_exact.jsonl": build_knn_rankings(questions, pool, analyzer="word", ngram_range=(1, 2), neighbors=20),
        "public_knn_char_v3_exact.jsonl": build_knn_rankings(questions, pool, analyzer="char_wb", ngram_range=(3, 5), neighbors=20),
    }
    hashes: dict[str, str] = {}
    for name, rankings in outputs.items():
        path = args.output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite production source: {path}")
        write_rankings(path, rankings, ids)
        hashes[name] = sha256(path)
    report: dict[str, Any] = {
        "status": "PUBLIC_V3_LEXICAL_SOURCES_COMPLETE",
        "query_count": 1000,
        "public_ids": ids,
        "labeled_knn_pool_count": len(pool),
        "bm25": {"producer": "src.udsc2026.evaluation.legal_ir_lexical.build_bm25f_rankings", "title_weight": 0.0, "top_k": 200, "ngram_range": [1, 3], "k1": 0.7, "b": 0.3},
        "knn_word": {"producer": "src.udsc2026.evaluation.legal_ir_lexical.build_knn_rankings", "analyzer": "word", "ngram_range": [1, 2], "neighbors": 20},
        "knn_char": {"producer": "src.udsc2026.evaluation.legal_ir_lexical.build_knn_rankings", "analyzer": "char_wb", "ngram_range": [3, 5], "neighbors": 20},
        "no_public_answers_read": True,
        "no_gpu": True,
        "no_retrieval_index_called": True,
        "sha256": hashes,
    }
    (args.output_dir / "public_v3_source_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "query_count": 1000, "output_dir": str(args.output_dir), "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
