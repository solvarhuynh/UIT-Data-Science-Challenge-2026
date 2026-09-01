"""CPU-only B2a-0 context audit using the frozen Qwen3-Reranker format.

This script intentionally reads only query text, candidate document references,
and corpus passages. It never reads answer/relevance fields.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "reports/task1/workflow_b/b2a/audit"
MODEL = ROOT / "models/reranker"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a Vietnamese legal question, determine whether the Document contains legal provisions relevant to answering the Query."
SYSTEM = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def quantile(values: list[int], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    x = (len(s) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def summary(values: list[int]) -> dict:
    return {
        "pair_count": len(values),
        "minimum": min(values) if values else None,
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "p99_5": quantile(values, 0.995),
        "p99_9": quantile(values, 0.999),
        "maximum": max(values) if values else None,
        "count_gt_4096": sum(x > 4096 for x in values),
        "count_gt_8192": sum(x > 8192 for x in values),
        "count_gt_16384": sum(x > 16384 for x in values),
        "count_gt_32768": sum(x > 32768 for x in values),
    }


def main() -> None:
    from transformers import AutoTokenizer

    prov_path = AUDIT / "model_provenance_audit.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("status") != "PASS" or prov.get("model_id") != MODEL_ID or prov.get("model_revision_sha") != REVISION:
        raise SystemExit("BLOCKED_B2A_PROVENANCE_AUDIT_INVALID")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, revision=REVISION, padding_side="left")
    transformers_version = __import__("transformers").__version__
    if tuple(int(x) for x in transformers_version.split(".")[:2]) < (4, 51):
        raise SystemExit("BLOCKED_B2A_TRANSFORMERS_TOO_OLD")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    tokenizer_hash = sha256(MODEL / "tokenizer.json")

    train = json.loads((ROOT / "data/raw/btc/LegalIR/train.json").read_text(encoding="utf-8"))
    questions = {str(k): str(v["question"]) for k, v in train.items() if isinstance(v, dict) and "question" in v}
    refs_path = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
    ctx_root = ROOT / "data/raw/btc/LegalIR/selected-contexts"
    rows = []
    for line in refs_path.open(encoding="utf-8"):
        r = json.loads(line)
        if int(r.get("fold", -1)) in (1, 2, 3, 4):
            rows.append(r)
    if len(rows) != 5600:
        raise SystemExit("BLOCKED_B2A_K77_DOCUMENT_CONTRACT_UNRESOLVED")
    rows.sort(key=lambda x: int(x["query_id"]))
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    missing_docs = 0
    doc_cache: dict[str, str] = {}
    lengths: list[int] = []
    by_fold: dict[str, list[int]] = {f"F{i}": [] for i in (1, 2, 3, 4)}
    per_query_counts = []

    # Tokenize each query prefix and each unique document once.  The frozen
    # separators make this decomposition exact; verify it on representative
    # complete pairs before using it for the full population.
    query_part_len = {
        qid: len(tokenizer.encode(f"<Instruct>: {INSTRUCTION}\n\n<Query>: {q}\n\n", add_special_tokens=False))
        for qid, q in questions.items()
    }
    doc_part_len: dict[str, int] = {}
    for di, did in enumerate(sorted({str(c["doc_id"]) for r in rows for c in r["candidates"][:77]}), 1):
        if did not in doc_cache:
            p = ctx_root / f"context_{did}.json"
            if not p.exists():
                missing_docs += 1
                doc_cache[did] = ""
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                doc_cache[did] = str(obj.get("passage", obj.get("text", "")))
        doc_part_len[did] = len(tokenizer.encode(f"<Document>: {doc_cache[did]}", add_special_tokens=False))
        if di % 500 == 0:
            print(f"tokenized_documents={di}", flush=True)
    # Verify the separator decomposition against complete official inputs.
    for r in rows[:5]:
        qid = str(r["query_id"]); q = questions[qid]
        for c in sorted(r["candidates"], key=lambda x: int(x.get("union_rank", 10**9)))[:3]:
            did = str(c["doc_id"]); d = doc_cache[did]
            full = tokenizer.encode(f"<Instruct>: {INSTRUCTION}\n\n<Query>: {q}\n\n<Document>: {d}", add_special_tokens=False)
            if len(full) != query_part_len[qid] + doc_part_len[did]:
                raise SystemExit("BLOCKED_B2A_TOKEN_DECOMPOSITION_MISMATCH")

    for ri, r in enumerate(rows, 1):
        qid = str(r["query_id"])
        if qid not in questions:
            raise SystemExit("BLOCKED_B2A_QUERY_TEXT_UNRESOLVED")
        candidates = sorted(r["candidates"], key=lambda x: int(x.get("union_rank", 10**9)))[:77]
        docs = []
        for c in candidates:
            did = str(c["doc_id"])
            key = (qid, did)
            if key in seen_pairs:
                duplicate_pairs += 1
            seen_pairs.add(key)
            docs.append(doc_cache[did])
        if len(candidates) != 77:
            raise SystemExit("BLOCKED_B2A_K77_DOCUMENT_CONTRACT_UNRESOLVED")
        vals = [len(prefix_tokens) + query_part_len[qid] + doc_part_len[str(c["doc_id"])] + len(suffix_tokens) for c in candidates]
        lengths.extend(vals)
        by_fold[f"F{int(r['fold'])}"].extend(vals)
        per_query_counts.append(len(vals))
        if ri % 10 == 0:
            print(f"audited_queries={ri}/{len(rows)} pairs={len(lengths)}", flush=True)

    if sum(per_query_counts) != 431200 or len(seen_pairs) != 431200 or duplicate_pairs != 0 or missing_docs != 0:
        raise SystemExit("BLOCKED_B2A_K77_DOCUMENT_CONTRACT_UNRESOLVED")
    overall = summary(lengths)
    p99 = overall["p99"]
    if p99 <= 8192:
        selected = 8192
    elif p99 <= 16384:
        selected = 16384
    elif p99 <= 32768:
        selected = 32768
    else:
        selected = None
    truncated = sum(x > selected for x in lengths) if selected is not None else None
    trunc_rate = (truncated / len(lengths)) if truncated is not None else None
    status = "PASS" if selected is not None and trunc_rate <= 0.01 else ("BLOCKED_B2A_TRUNCATION_CONFOUND" if selected is not None else "BLOCKED_B2A_LONG_DOCUMENT_HANDLING_REQUIRED")
    out = {
        "status": status,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "tokenizer_sha256": tokenizer_hash,
        "python": platform.python_version(),
        "transformers": transformers_version,
        "instruction": INSTRUCTION,
        "system_semantics": SYSTEM,
        "prefix_token_count": len(prefix_tokens),
        "suffix_token_count": len(suffix_tokens),
        "query_count": len(rows),
        "candidate_count_per_query": 77,
        "pair_count": len(lengths),
        "duplicate_query_document_pairs": duplicate_pairs,
        "missing_document_texts": missing_docs,
        **overall,
        "max": overall["maximum"],
        "selected_max_length": selected if status == "PASS" else None,
        "truncated_pair_count": truncated,
        "truncation_rate": trunc_rate,
        "folds": {k: summary(v) for k, v in by_fold.items()},
        "labels_used": False,
        "Fold0_used": False,
        "public_labels_used": False,
        "CUDA_required": False,
        "source_candidates": str(refs_path.relative_to(ROOT)).replace("\\", "/"),
        "source_corpus": str(ctx_root.relative_to(ROOT)).replace("\\", "/"),
        "format": "Qwen3 official Transformers prefix + formatted pair + suffix; all tokens counted",
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "context_length_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# B2a-0 CPU context-length audit", "", f"Status: **{status}**", "", f"Pairs: **{len(lengths)}** (5600 queries × 77 documents)", f"Tokenizer: `{transformers_version}`, revision `{REVISION}`", f"Prefix/suffix tokens: {len(prefix_tokens)} / {len(suffix_tokens)}", "", "All lengths are exact tokenizer counts for the official Qwen prefix + formatted `<Instruct>/<Query>/<Document>` content + suffix. No labels, Fold0, or CUDA were used.", "", "## Distribution", "", "| statistic | tokens |", "|---|---:|"]
    for k in ("minimum", "p50", "p90", "p95", "p99", "p99_5", "p99_9", "maximum"):
        md.append(f"| {k} | {out[k]:.3f} |" if isinstance(out[k], float) else f"| {k} | {out[k]} |")
    trunc_display = f"{truncated} ({trunc_rate:.6%})" if trunc_rate is not None else "not selected (p99 > 32768)"
    md += ["", f"Selected max_length: **{out['selected_max_length']}**", f"Truncated pairs: **{trunc_display}**", "", "## Per fold", "", "| fold | pairs | p99 | max |", "|---|---:|---:|---:|"]
    for k in ("F1", "F2", "F3", "F4"):
        md.append(f"| {k} | {out['folds'][k]['pair_count']} | {out['folds'][k]['p99']:.3f} | {out['folds'][k]['maximum']} |")
    (AUDIT / "context_length_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "pairs": len(lengths), "p99": p99, "max": overall["maximum"], "selected_max_length": out["selected_max_length"], "truncation_rate": trunc_rate, "transformers": transformers_version}, ensure_ascii=False))


if __name__ == "__main__":
    main()
