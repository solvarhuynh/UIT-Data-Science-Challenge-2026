#!/usr/bin/env python3
"""CPU-only, local B4 payload/parser/S2 preflight.

This intentionally performs no model loading and no Beam/GPU call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_vector_payloads_vs_raw_chunks_b4 import iter_payloads  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
DEFAULT_WORKLIST = ROOT / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
DEFAULT_TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
EXPECTED_WORKLIST_SHA = "34d3ba1f0f9e202222c19a921f10725faace86af8869809d3c947f66b035ba32"
EXPECTED_PAYLOADS = 1_270_356
EXPECTED_DOCS = 6_543
EXPECTED_EVIDENCE = 161_763
PROTECTED = re.compile(r"(?:\b(?:(?:Điều|Khoản)\s+\d+[a-zđ]?|Điểm\s+[a-zđ])\b|\b\d{1,4}/\d{4}/[a-zđ]{1,12}\d{0,4}\b)", re.IGNORECASE)


def tokenize_vi(text: str) -> list[str]:
    from pyvi import ViTokenizer
    if not text.strip():
        return []
    out: list[str] = []
    cursor = 0
    for match in PROTECTED.finditer(text):
        out.extend(t.lower() for t in ViTokenizer.tokenize(text[cursor:match.start()]).split() if t)
        out.append(match.group(0))
        cursor = match.end()
    out.extend(t.lower() for t in ViTokenizer.tokenize(text[cursor:]).split() if t)
    return out


def bm25(corpus: list[list[str]], query: list[str]) -> list[float]:
    lengths = [len(x) for x in corpus]
    avg = sum(lengths) / len(lengths) if lengths else 0.0
    if avg <= 0:
        return [0.0] * len(corpus)
    dfs = Counter(term for doc in corpus for term in set(doc))
    idf = {term: math.log(1 + (len(corpus) - n + .5) / (n + .5)) for term, n in dfs.items()}
    result = []
    for doc, length in zip(corpus, lengths):
        tf = Counter(doc); norm = 1.5 * (.25 + .75 * length / avg); score = 0.0
        for term in query:
            n = tf.get(term, 0)
            if n:
                score += idf.get(term, 0.0) * n * 2.5 / (n + norm)
        result.append(score)
    return result


def select_s2(question: str, chunks: list[dict]) -> list[dict]:
    corpus = [tokenize_vi(str(x.get("text", ""))) for x in chunks]
    scores = bm25(corpus, tokenize_vi(question))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], str(chunks[i].get("chunk_id", ""))))
    return [chunks[i] for i in order[:3] if str(chunks[i].get("text", "")).strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    ap.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    args = ap.parse_args()
    try:
        import pyvi  # noqa: F401
    except ModuleNotFoundError as exc:
        print(json.dumps({
            "status": "MISSING_DEPENDENCY",
            "package": "pyvi",
            "error": str(exc),
            "install": "python -m pip install 'pyvi>=0.1,<1.0'",
        }, ensure_ascii=False, indent=2))
        return 2
    work = json.loads(args.worklist.read_text(encoding="utf-8"))
    rows = work["rows"]
    work_sha = hashlib.sha256("".join(f"{r['query_id']}\t{r['doc_id']}\n" for r in sorted(rows, key=lambda r: (str(r['query_id']), str(r['doc_id'])))).encode()).hexdigest()
    wanted = {str(r["doc_id"]) for r in rows}
    docs: dict[str, list[dict]] = defaultdict(list)
    total = 0
    print(f"[START] streaming payloads: {args.payloads}", flush=True)
    for payload in iter_payloads(args.payloads):
        total += 1
        doc = str(payload.get("doc_id", ""))
        if doc in wanted:
            docs[doc].append(payload)
        if total % 100_000 == 0:
            print(f"[PAYLOAD] {total}/{EXPECTED_PAYLOADS} docs={len(docs)}", flush=True)
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    evidence = 0
    for n, qid in enumerate(sorted(by_query), 1):
        for row in by_query[qid]:
            selected = select_s2(str(train[qid]["question"]), docs[str(row["doc_id"])])
            evidence += len(selected)
        if n % 500 == 0:
            print(f"[S2] {n}/{len(by_query)} evidence={evidence}", flush=True)
    result = {
        "status": "PASS" if (total == EXPECTED_PAYLOADS and len(docs) == EXPECTED_DOCS and len(rows) == 53924 and work_sha == EXPECTED_WORKLIST_SHA and evidence == EXPECTED_EVIDENCE) else "FAIL",
        "payload_total": total,
        "b4_doc_count": len(docs),
        "b4_doc_occurrences": len(rows),
        "query_count": len(by_query),
        "s2_selected_chunks": evidence,
        "expected": {"payload_total": EXPECTED_PAYLOADS, "b4_doc_count": EXPECTED_DOCS, "s2_selected_chunks": EXPECTED_EVIDENCE},
        "worklist_sha256": work_sha,
        "worklist_sha_matches": work_sha == EXPECTED_WORKLIST_SHA,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
