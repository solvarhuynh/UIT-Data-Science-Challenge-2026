"""No-GPU parity check for raw JSONL versus FAISS payload storage in B4.

The check is deterministic and restricted to the locked B4 candidate documents.
It verifies chunk keys/text and the exact S2 BM25 ordering used by the runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/processed_v3/chunks"
PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
WORKLIST = ROOT / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
OUT = ROOT / "artifacts/task1/recovery_096/vector_payloads_vs_raw_chunks_b4/parity_report.json"
PROTECTED = __import__("re").compile(r"(?:\b(?:(?:Điều|Khoản)\s+\d+[a-zđ]?|Điểm\s+[a-zđ])\b|\b\d{1,4}/\d{4}/[a-zđ]{1,12}\d{0,4}\b)", __import__("re").IGNORECASE)


def iter_payloads(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buf = ""; eof = False
        def refill() -> None:
            nonlocal buf, eof
            part = handle.read(1 << 20)
            if part: buf += part
            else: eof = True
        while '"payloads"' not in buf and not eof: refill()
        marker = buf.find('"payloads"')
        if marker < 0: raise ValueError("missing payloads field")
        buf = buf[marker + len('"payloads"'):]
        while "{" not in buf and not eof: refill()
        opening = buf.find("{")
        if opening < 0: raise ValueError("payloads is not an object")
        buf = buf[opening + 1:]
        while True:
            buf = buf.lstrip()
            if not buf and not eof: refill(); continue
            if not buf: raise ValueError("unexpected EOF")
            if buf[0] == "}": return
            while True:
                try: _, pos = decoder.raw_decode(buf); break
                except json.JSONDecodeError:
                    if eof: raise
                    refill()
            buf = buf[pos:].lstrip()
            if not buf.startswith(":"): raise ValueError("invalid payload separator")
            buf = buf[1:].lstrip()
            while True:
                try: value, pos = decoder.raw_decode(buf); break
                except json.JSONDecodeError:
                    if eof: raise
                    refill()
            if not isinstance(value, dict): raise ValueError("payload row is not object")
            yield value
            buf = buf[pos:].lstrip()
            if buf.startswith(","): buf = buf[1:]


def tokenize_vi(text: str) -> list[str]:
    from pyvi import ViTokenizer
    out: list[str] = []; cursor = 0
    for match in PROTECTED.finditer(text):
        out.extend(t.lower() for t in ViTokenizer.tokenize(text[cursor:match.start()]).split() if t)
        out.append(match.group(0)); cursor = match.end()
    out.extend(t.lower() for t in ViTokenizer.tokenize(text[cursor:]).split() if t)
    return out


def bm25(corpus: list[list[str]], query: list[str]) -> list[float]:
    lengths = [len(x) for x in corpus]; avg = sum(lengths) / len(lengths) if lengths else 0.0
    if not avg: return [0.0] * len(corpus)
    dfs = Counter(term for doc in corpus for term in set(doc)); k1, b = 1.5, .75
    result = []
    for doc, length in zip(corpus, lengths):
        tf = Counter(doc); norm = k1 * (1 - b + b * length / avg); score = 0.0
        for term in query:
            count = tf.get(term, 0)
            if count: score += math.log(1 + (len(corpus) - dfs[term] + .5) / (dfs[term] + .5)) * count * (k1 + 1) / (count + norm)
        result.append(score)
    return result


def select(question: str, chunks: list[dict[str, Any]]) -> list[str]:
    scores = bm25([tokenize_vi(str(x.get("text", ""))) for x in chunks], tokenize_vi(question))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], str(chunks[i].get("chunk_id", ""))))
    return [str(chunks[i]["chunk_id"]) for i in order[:3] if str(chunks[i].get("text", "")).strip()]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUT); ap.add_argument("--progress-every", type=int, default=100000)
    args = ap.parse_args(); work = json.loads(WORKLIST.read_text(encoding="utf-8")); docs = {str(r["doc_id"]) for r in work["rows"]}
    train = json.loads((ROOT / "data/raw/btc/LegalIR/train.json").read_text(encoding="utf-8-sig"))
    qdocs: dict[str, set[str]] = defaultdict(set)
    for row in work["rows"]: qdocs[str(row["query_id"])].add(str(row["doc_id"]))
    raw: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        source = RAW_DIR / f"{doc}.jsonl"; raw[doc] = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    payload: dict[str, list[dict[str, Any]]] = defaultdict(list); seen = 0
    for item in iter_payloads(PAYLOADS):
        if str(item.get("doc_id", "")) in docs: payload[str(item["doc_id"])].append(item)
        seen += 1
        if seen % args.progress_every == 0: print(f"[PAYLOAD] scanned={seen}", flush=True)
    doc_ok = chunk_ok = text_ok = top1_ok = top3_ok = bge_ok = 0; total_docs = len(docs); total_qdocs = sum(len(v) for v in qdocs.values())
    for doc in docs:
        a, b = raw[doc], payload[doc]; doc_ok += int({str(x["chunk_id"]) for x in a} == {str(x["chunk_id"]) for x in b}) and int(len(a) == len(b))
        amap = {str(x["chunk_id"]): x for x in a}; bmap = {str(x["chunk_id"]): x for x in b}
        chunk_ok += int(set(amap) == set(bmap)); text_ok += int(set(amap) == set(bmap) and all(str(amap[k].get("text", "")) == str(bmap[k].get("text", "")) for k in amap))
    for qid, qset in qdocs.items():
        question = str(train[qid]["question"])
        for doc in qset:
            a = select(question, raw[doc]); b = select(question, payload[doc]); top1_ok += int(a[:1] == b[:1]); top3_ok += int(set(a) == set(b))
            amap = {str(x["chunk_id"]): str(x.get("text", "")) for x in raw[doc]}; bmap = {str(x["chunk_id"]): str(x.get("text", "")) for x in payload[doc]}
            bge_ok += int(a == b and all(amap.get(cid) == bmap.get(cid) for cid in a))
    report = {"status": "STORAGE_BACKEND_PARITY_PASS" if all(x == y for x, y in ((doc_ok,total_docs),(chunk_ok,total_docs),(text_ok,total_docs),(top1_ok,total_qdocs),(top3_ok,total_qdocs),(bge_ok,total_qdocs))) else "STORAGE_BACKEND_PARITY_FAIL", "doc_parity_rate": doc_ok/total_docs, "chunk_key_parity_rate": chunk_ok/total_docs, "text_parity_rate": text_ok/total_docs, "s2_top1_parity_rate": top1_ok/total_qdocs, "s2_top3_set_parity_rate": top3_ok/total_qdocs, "bge_input_text_parity_rate": bge_ok/total_qdocs, "b4_unique_docs": total_docs, "b4_doc_occurrences": len(work["rows"]), "worklist_sha256": "34d3ba1f0f9e202222c19a921f10725faace86af8869809d3c947f66b035ba32"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False)); return 0 if report["status"] == "STORAGE_BACKEND_PARITY_PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
