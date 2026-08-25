"""No-GPU evidence-selector recovery and leakage-safe V2B Slim redesign.

The candidate policies are fixed below before gold is read.  Gold is used only
for diagnostics and the explicitly cross-fitted D2 selection.  ``--audit`` is
checkpointed and may be slow because it reads original chunk JSONL documents;
it never loads a cross-encoder, scores BGE, or invokes Beam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from analyze_post_k500_bounded_union_v2 import RECOVERY, index, load_jsonl
from analyze_v2b_slim_preflight import K500_COST, build_candidates, checkpoint_chunk_counts, load_rankings, words

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data/processed_v3/chunks"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stats(values: list[int | float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "median": median(values) if values else None,
            "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if values else None}


def provenance_bucket(row: dict[str, Any]) -> str:
    sources = set(row["sources"])
    has_bm25 = "bm25" in sources
    has_knn = bool({"knn_word", "knn_char"} & sources)
    if has_bm25 and has_knn:
        return "bm25_knn_multi_source"
    if has_bm25:
        return "bm25_only_or_heavy"
    if has_knn:
        return "knn_only"
    if "adaptive" in sources:
        return "adaptive_dense_derived"
    return "other"


def rank_bucket(rank: int) -> str:
    if rank <= 25: return "rank_001_025"
    if rank <= 50: return "rank_026_050"
    if rank <= 100: return "rank_051_100"
    return "rank_101_200"


def fixed_policies(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    per_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: per_q[row["query_id"]].append(row)
    top20 = [x for q in sorted(per_q) for x in sorted(per_q[q], key=lambda y: (y["union_rank"], y["doc_id"]))[:20]]
    rank50 = [x for x in rows if x["union_rank"] <= 50]
    multi = [x for x in rows if x["source_support"] >= 2]
    def union(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list({(x["query_id"], x["doc_id"]): x for x in a + b}.values())
    return {"rank_le_50": rank50, "top20_missing_per_query": top20,
            "rank_le_50_or_multi_source_ge_2": union(rank50, multi),
            "top20_missing_per_query_or_multi_source_ge_2": union(top20, multi)}


def plan_report(selected: list[dict[str, Any]], counts: dict[tuple[str, str], int], baseline: dict[str, dict[str, Any]], existing: dict[str, set[str]]) -> dict[str, Any]:
    selected_by_q: dict[str, set[str]] = defaultdict(set)
    for r in selected: selected_by_q[r["query_id"]].add(r["doc_id"])
    chunks = {cap: sum(min(cap, counts[(r["query_id"], r["doc_id"])]) for r in selected) for cap in (1, 2, 3)}
    total = covered = 0
    rescue_total = rescue_covered = 0
    for qid, base in baseline.items():
        gold = set(map(str, base["gold_documents"])); total += len(gold)
        covered += len(gold & (existing[qid] | selected_by_q[qid]))
        rescue = gold - set(map(str, base["top5"])) - existing[qid]
        rescue_total += len(rescue); rescue_covered += len(rescue & selected_by_q[qid])
    return {"candidate_doc_occurrences": len(selected), "unique_docs": len({r["doc_id"] for r in selected}),
            "chunks": {str(cap): chunks[cap] for cap in (1,2,3)}, "relative_cost_vs_k500_v1": {str(cap): chunks[cap] / K500_COST for cap in (1,2,3)},
            "candidate_oracle_ceiling": covered / total, "rescued_missing_bge_gold_capture": {"covered": rescue_covered, "total": rescue_total, "rate": rescue_covered / rescue_total if rescue_total else None}}


def target_headroom(oracle: float, baseline: float = .9244452380952382) -> dict[str, Any]:
    result = {"candidate_oracle_ceiling": oracle, "headroom_over_strict_baseline": oracle - baseline, "targets": {}}
    for target in (.94, .95, .96, .98):
        gain_needed = target - baseline
        available = oracle - baseline
        result["targets"][f"{target:.2f}"] = {"possible_by_candidate_ceiling": oracle >= target,
            "fraction_of_plan_headroom_that_ranking_must_capture": gain_needed / available if available > 0 else None}
    return result


def crossfit(policies: dict[str, list[dict[str, Any]]], counts: dict[tuple[str, str], int], baseline: dict[str, dict[str, Any]], existing: dict[str, set[str]]) -> dict[str, Any]:
    # D2: among predeclared policies select maximum training-fold oracle subject to <=1x cost.
    chosen: dict[str, str] = {}; held: dict[str, Any] = {}
    for fold in range(5):
        train_ids = {q for q, b in baseline.items() if int(b["fold"]) != fold}
        candidates = []
        for name, rows in policies.items():
            filtered = [r for r in rows if r["query_id"] in train_ids]
            report = plan_report(filtered, counts, {q: baseline[q] for q in train_ids}, {q: existing[q] for q in train_ids})
            if report["relative_cost_vs_k500_v1"]["3"] <= 1.0:
                candidates.append((report["candidate_oracle_ceiling"], name))
        if not candidates: raise RuntimeError("no predeclared policy under 1x training cost")
        name = max(candidates, key=lambda x: (x[0], x[1]))[1]; chosen[str(fold)] = name
        ids = {q for q, b in baseline.items() if int(b["fold"]) == fold}
        held[str(fold)] = plan_report([r for r in policies[name] if r["query_id"] in ids], counts, {q: baseline[q] for q in ids}, {q: existing[q] for q in ids})
    return {"selection_rule": "max training-fold candidate oracle among fixed D1 policies with chunks@3 <=1.0x K500; held fold never used for selection", "chosen_policy_by_outer_fold": chosen, "held_out": held}


def tokenize_for_bm25(text: str) -> list[str]:
    try:
        from udsc2026.retrieval.sparse.tokenizer import tokenize_vi
        return tokenize_vi(text)
    except Exception:
        return sorted(words(text))


def bm25_scores(corpus: list[list[str]], query_tokens: Iterable[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not corpus:
        return []
    query = list(query_tokens)
    doc_lens = [len(doc) for doc in corpus]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    if avgdl <= 0:
        return [0.0 for _ in corpus]
    dfs = Counter(term for doc in corpus for term in set(doc))
    idf = {term: math.log(1.0 + (len(corpus) - df + 0.5) / (df + 0.5)) for term, df in dfs.items()}
    output: list[float] = []
    for doc, doc_len in zip(corpus, doc_lens):
        tf = Counter(doc)
        score = 0.0
        denom_norm = k1 * (1.0 - b + b * doc_len / avgdl)
        for term in query:
            freq = tf.get(term, 0)
            if freq:
                score += idf.get(term, 0.0) * freq * (k1 + 1.0) / (freq + denom_norm)
        output.append(score)
    return output


def bm25_within_document(query: str, chunks: list[dict[str, Any]], topk: int = 3) -> list[str]:
    corpus = [tokenize_for_bm25(str(row.get("text", ""))) for row in chunks]
    query_tokens = tokenize_for_bm25(query)
    if not chunks or not query_tokens: return []
    scores = bm25_scores(corpus, query_tokens)
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), str(chunks[i].get("chunk_id", ""))))
    return [str(chunks[i]["chunk_id"]) for i in order[:topk] if str(chunks[i].get("text", "")).strip()]


def token_overlap(query: str, chunks: list[dict[str, Any]], topk: int = 3) -> list[str]:
    q = words(query)
    ranked = sorted(chunks, key=lambda x: (-len(q & words(str(x.get("text", "")))), str(x.get("chunk_id", ""))))
    return [str(x["chunk_id"]) for x in ranked[:topk] if str(x.get("text", "")).strip()]


def selector_audit(k200: dict[str, dict[str, Any]], k500: dict[str, dict[str, Any]], lexical: dict[str, dict[str, list[str]]], train: dict[str, Any], sample_per_cache: int, checkpoint: Path) -> dict[str, Any]:
    """Audit S0/S2 using scored pairs; stratification is inference provenance only."""
    output: dict[str, Any] = {}
    for cache_name, records, hit_key in (("k200", k200, "hits"), ("k500_v1", k500, "new_hits")):
        grouped: dict[str, list[tuple[str, str, list[dict[str, Any]]]]] = defaultdict(list)
        for qid, record in records.items():
            by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for hit in record[hit_key]: by_doc[str(hit["doc_id"])].append(hit)
            for doc, hits in by_doc.items():
                bucket = "bm25_heavy" if doc in set(lexical["bm25"][qid][:20]) else "dense_heavy"
                grouped[bucket].append((qid, doc, hits))
        all_results: dict[str, Any] = {}
        for bucket, pairs in grouped.items():
            pairs.sort(key=lambda item: hashlib.sha256(f"{cache_name}:{bucket}:{item[0]}:{item[1]}".encode()).hexdigest())
            pairs = pairs[:sample_per_cache]
            counters = {name: Counter() for name in ("S0_token_overlap_topk_v1", "S2_bm25_within_document")}
            for n, (qid, doc, hits) in enumerate(pairs, 1):
                cp = checkpoint / cache_name / bucket / f"{qid}_{doc}.json"
                if cp.is_file():
                    row = json.loads(cp.read_text(encoding="utf-8"))
                    for name, value in row.items(): counters[name].update(value)
                    continue
                source = CHUNKS / f"{doc}.jsonl"
                chunks = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()] if source.is_file() else []
                actual = {str(x["chunk_id"]) for x in hits}
                best = max(hits, key=lambda x: (float(x["bge_score"]), -int(x.get("dense_rank", 10**9)), str(x["chunk_id"])))
                payload: dict[str, dict[str, int | float]] = {}
                for name, selector in (("S0_token_overlap_topk_v1", token_overlap), ("S2_bm25_within_document", bm25_within_document)):
                    selected = selector(str(train[qid]["question"]), chunks)
                    inter = set(selected) & actual
                    payload[name] = {"valid": int(bool(selected)), "exact_best_at1": int(bool(selected) and selected[0] == str(best["chunk_id"])), "any_scored_at1": int(bool(selected) and selected[0] in actual), "any_scored_at3": int(bool(inter)), "best_at3": int(str(best["chunk_id"]) in set(selected)), "overlap_count": len(inter), "jaccard_x1e6": int(1_000_000 * len(inter) / len(set(selected) | actual)) if selected else 0}
                atomic_json(cp, payload)
                for name, value in payload.items(): counters[name].update(value)
                if n % 250 == 0: print(f"[AUDIT] {cache_name}/{bucket} {n}/{len(pairs)}", flush=True)
            all_results[bucket] = {}
            for name, c in counters.items():
                v = c["valid"]
                all_results[bucket][name] = {"sample_count": len(pairs), "valid": v,
                    "exact_best_bge_chunk_at1": c["exact_best_at1"] / v if v else None, "any_previously_scored_at1": c["any_scored_at1"] / v if v else None,
                    "any_previously_scored_at3": c["any_scored_at3"] / v if v else None, "best_bge_chunk_in_selected_at3": c["best_at3"] / v if v else None,
                    "mean_overlap_count": c["overlap_count"] / v if v else None, "mean_jaccard": c["jaccard_x1e6"] / (1_000_000 * v) if v else None}
        output[cache_name] = all_results
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RECOVERY / "evidence_selector_recovery_v1")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--sample-per-cache-stratum", type=int, default=1000)
    args = parser.parse_args()
    print("[START] loading rankings/caches", flush=True)
    baseline, decisions, k200, k500, lexical = load_rankings()
    print("[START] building candidate policies", flush=True)
    rows, existing = build_candidates(baseline, decisions, k200, k500, lexical)
    print("[START] loading/checkpointing chunk counts", flush=True)
    counts = checkpoint_chunk_counts()
    print("[START] computing fixed plans and crossfit report", flush=True)
    policies = fixed_policies(rows)
    target_distribution: dict[str, Any] = {}
    for name, selected in policies.items():
        target_distribution[name] = {"provenance": dict(Counter(provenance_bucket(r) for r in selected)), "union_rank": dict(Counter(rank_bucket(int(r["union_rank"])) for r in selected))}
    plans = {name: plan_report(selected, counts, baseline, existing) for name, selected in policies.items()}
    report: dict[str, Any] = {"schema_version": "evidence-selector-recovery-v1", "constraints": {"no_gpu": True, "no_neural_training": True, "no_submission": True, "policy_fixed_before_gold": True},
        "exact_evidence_provenance": {"raw_k200_k500": "Global FAISS dense chunk retrieval using huyydangg/DEk21_hcmute_embedding_v2; raw hit order is dense rank.", "k200_bge": "Original BGE scores every usable raw dense hit rank 1..200; no document-level or lexical chunk selector intervenes.", "k500_v1_bge": "For selected queries only, original BGE scores raw dense hits rank 201..500 after excluding chunk IDs already present in K200.", "bge_pair_format": "AutoTokenizer([question] * batch, [raw_chunk_text] * batch, padding=True, truncation=True, max_length=512, return_tensors='pt'); logits[1]-logits[0] for 2 labels.", "reranker_revision": "archive SHA256 recorded in selective_bge_k500_v1_from_beam/manifest.json; transformers==5.0.0; fp16 autocast; batch=16.", "outside_dense_top500": "No per-(query,chunk) dense score/embedding cache for lexical candidate documents was found. Existing FAISS index/payload supports global dense retrieval, but cannot reproduce the original candidate-conditioned selector outside its retrieved top500 without a new query encoding/retrieval operation. Therefore S1/S3 are not available as cache-only selectors."},
        "target_worklist_source_distribution": target_distribution, "fixed_cost_plans": plans, "cross_fitted_d2": crossfit(policies, counts, baseline, existing),
        "target_headroom": {name: target_headroom(value["candidate_oracle_ceiling"]) for name, value in plans.items()},
        "selector_family": {"S0": "token_overlap_topk_v1", "S1": "exact original dense selector: unavailable outside dense top500 cache", "S2": "BM25-within-document using repository Vietnamese tokenizer or local BM25 fallback", "S3": "dense-within-document: unavailable cache-only", "S4": "source-aware: blocked until KNN provenance-specific chunk evidence is available"}}
    report["selector_status"] = "AUDIT_PENDING_NO_GPU"
    atomic_json(args.output_dir / "report.partial.json", report)
    if args.audit:
        train = json.loads((ROOT / "data/raw/btc/LegalIR/train.json").read_text(encoding="utf-8-sig"))
        print("[START] auditing evidence selectors", flush=True)
        report["selector_audit"] = selector_audit(k200, k500, lexical, train, args.sample_per_cache_stratum, args.output_dir / "checkpoints")
        report["selector_status"] = "SELECTOR_STILL_UNRESOLVED"
        report["selector_decision"] = "S2 can be measured for BM25-heavy candidates, but exact dense reproduction is unavailable outside top500 and no source-grounded KNN chunk selector exists; GPU remains blocked."
    else:
        report["selector_status"] = "AUDIT_PENDING_NO_GPU"
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": report["selector_status"], "output": str(args.output_dir), "plans": {k: round(v["relative_cost_vs_k500_v1"]["3"], 3) for k,v in plans.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
