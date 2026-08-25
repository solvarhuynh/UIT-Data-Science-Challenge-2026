"""Checkpointed no-GPU preflight for the bounded-union BGE V2B scorer.

It deliberately creates only an inference-only worklist.  It never loads a
model or scores a chunk.  Missing lexical candidates have no existing
query-to-chunk evidence, so a deterministic lexical-overlap cap is used to
choose at most ``--max-chunks-per-doc`` chunks per (query, doc).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from analyze_post_k500_bounded_union_v2 import RECOVERY, doc_features, index, load_jsonl, rrf

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data/processed_v3/chunks"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, ensure_ascii=False)
        os.replace(name, path)
    finally: Path(name).unlink(missing_ok=True)


def words(text: str) -> set[str]:
    return {x for x in text.lower().split() if len(x) > 1}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=RECOVERY / "selective_bge_bounded_union_v2b_preflight")
    p.add_argument("--max-chunks-per-doc", type=int, default=3)
    args = p.parse_args()
    if args.max_chunks_per_doc < 1: raise ValueError("max-chunks-per-doc must be positive")
    base = index(load_jsonl(RECOVERY / "baseline_093_oof/predictions.jsonl"))
    decisions = index(load_jsonl(RECOVERY / "adaptive_k500_v1/adaptive_k500_decisions.jsonl"))
    k200 = index(load_jsonl(RECOVERY / "baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl") + load_jsonl(RECOVERY / "baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"))
    k500 = index(load_jsonl(RECOVERY / "selective_bge_k500_v1_from_beam/new_chunk_scores.jsonl"))
    train = json.loads((ROOT / "data/raw/btc/LegalIR/train.json").read_text(encoding="utf-8-sig"))
    ckpt = RECOVERY / "candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
    lexical = {name: {} for name in ("bm25", "knn_word", "knn_char")}
    for name in lexical:
        for fold in range(5):
            x = json.loads((ckpt / f"{name}_fold{fold}.json").read_text(encoding="utf-8"))
            lexical[name].update({str(q): [str(d) for d in docs] for q, docs in x["rankings"].items()})
    query_docs: list[tuple[str, str]] = []
    for qid in base:
        d = decisions[qid]; adaptive = [str(x) for x in d["k200_documents"]] + [str(x) for x in d["k500_added_documents"]]
        known = set(doc_features(k200[qid]["hits"])) | (set(doc_features(k500[qid]["new_hits"])) if qid in k500 else set())
        for doc in rrf([adaptive, lexical["bm25"][qid], lexical["knn_word"][qid], lexical["knn_char"][qid]])[:200]:
            if doc not in known: query_docs.append((qid, doc))
    # Reuse document chunk reads across every query that selected the same doc.
    needed: dict[str, list[str]] = {}
    for qid, doc in query_docs: needed.setdefault(doc, []).append(qid)
    state = args.output_dir / "checkpoints"; state.mkdir(parents=True, exist_ok=True)
    worklist = args.output_dir / "worklist.jsonl"
    completed = {p.stem for p in state.glob("*.json")}
    print(
        f"[START] missing_doc_occurrences={len(query_docs)} "
        f"unique_missing_docs={len(needed)} completed_checkpoints={len(completed)}",
        flush=True,
    )
    for n, (doc, qids) in enumerate(needed.items(), 1):
        path = state / f"{doc}.json"
        if doc in completed: continue
        source = CHUNKS / f"{doc}.jsonl"
        if not source.is_file():
            atomic_json(path, {"doc_id": doc, "rows": [], "missing_source": True}); continue
        chunks = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
        rows = []
        for qid in qids:
            qwords = words(str(train[qid]["question"]))
            ranked = sorted(chunks, key=lambda x: (-len(qwords & words(str(x.get("text", "")))), str(x.get("chunk_id", ""))))[:args.max_chunks_per_doc]
            rows += [{"query_id": qid, "doc_id": doc, "chunk_id": str(x["chunk_id"]), "text": str(x["text"]), "evidence_selector": "token_overlap_topk_v1"} for x in ranked if str(x.get("text", "")).strip()]
        atomic_json(path, {"doc_id": doc, "rows": rows, "missing_source": False})
        if n % 100 == 0: print(f"[CHECKPOINT] documents={n}/{len(needed)}", flush=True)
    with worklist.open("w", encoding="utf-8") as out:
        for doc in needed:
            for row in json.loads((state / f"{doc}.json").read_text(encoding="utf-8"))["rows"]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    chunks = sum(1 for _ in worklist.open(encoding="utf-8"))
    report = {"schema_version": "bounded-union-bge-v2b-preflight-v1", "status": "PREFLIGHT_COMPLETE_DO_NOT_RUN_GPU_AUTOMATICALLY", "candidate_pool": "locked bounded union @200", "missing_doc_occurrences": len(query_docs), "unique_missing_docs": len(needed), "max_chunks_per_doc": args.max_chunks_per_doc, "chunks_to_score": chunks, "mean_chunks_per_query": chunks / len(base), "p95_chunks_per_query": "see worklist; intentionally not inferred before exact worklist", "reuse": {"k200_bge_doc_occurrences": 429291, "k500_v1_bge_doc_occurrences": 62044, "new_scoring_doc_occurrences": len(query_docs)}, "relative_cost_vs_k500_v1_487914_chunks": chunks / 487914, "constraints": {"no_bge_scoring": True, "no_rescore_k200": True, "no_rescore_k500_v1": True, "deduplicated_document_reads": True, "checkpoint_resume": True}}
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
