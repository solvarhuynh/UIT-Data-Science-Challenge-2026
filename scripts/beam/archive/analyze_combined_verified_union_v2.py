"""Inference-only bounded candidate union from completed lexical checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Sequence


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    return float(ordered[math.ceil(q * len(ordered)) - 1])


def oracle(gold: set[str], docs: Sequence[str]) -> float:
    return len(gold & set(docs)) / len(gold)


def rrf(sources: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    first: dict[str, tuple[int, int]] = {}
    for source_index, ranking in enumerate(sources):
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            first.setdefault(doc, (source_index, rank))
    return sorted(scores, key=lambda doc: (-scores[doc], first[doc], doc))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--adaptive", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    baseline = {str(x["query_id"]): x for x in load_jsonl(args.baseline)}
    adaptive = {str(x["query_id"]): x for x in load_jsonl(args.adaptive)}
    source_names = ("bm25", "knn_word", "knn_char")
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in source_names}
    for name in source_names:
        for fold in range(5):
            row = json.loads((args.checkpoint_dir / f"{name}_fold{fold}.json").read_text(encoding="utf-8"))
            rankings[name].update({str(q): [str(d) for d in docs] for q, docs in row["rankings"].items()})
    if any(set(value) != set(baseline) for value in rankings.values()) or set(adaptive) != set(baseline):
        raise ValueError("checkpoint/baseline/adaptive coverage mismatch")
    budgets = (200, 300, 500)
    details: dict[str, Any] = {}
    candidate_counts: list[int] = []
    additions = {name: set() for name in source_names}
    pair_overlap: dict[str, int] = {}
    per_fold = {str(fold): {"unbounded": [], **{str(x): [] for x in budgets}} for fold in range(5)}
    scores = {"unbounded": [], **{str(x): [] for x in budgets}}
    miss_rescue = {"unbounded": 0, **{str(x): 0 for x in budgets}}
    multi_rescue = {"unbounded": 0, **{str(x): 0 for x in budgets}}
    for qid, base in baseline.items():
        raw = adaptive[qid]
        k200 = list(raw["k200_documents"])
        adaptive_docs = k200 + list(raw["k500_added_documents"])
        source_lists = [adaptive_docs, rankings["bm25"][qid], rankings["knn_word"][qid], rankings["knn_char"][qid]]
        merged = rrf(source_lists)
        candidate_counts.append(len(merged))
        added_by_source = {name: set(rankings[name][qid]) - set(adaptive_docs) for name in source_names}
        for name, docs in added_by_source.items(): additions[name].update(docs)
        gold = set(base["gold_documents"])
        for label, docs in [("unbounded", merged), *[(str(budget), merged[:budget]) for budget in budgets]]:
            value = oracle(gold, docs)
            scores[label].append(value); per_fold[str(base["fold"])][label].append(value)
            if raw["miss_k200"] and value > oracle(gold, k200): miss_rescue[label] += 1
            if len(gold) > 1 and value > oracle(gold, k200): multi_rescue[label] += 1
    for a, b in (("bm25", "knn_word"), ("bm25", "knn_char"), ("knn_word", "knn_char")):
        pair_overlap[f"{a}__{b}"] = len(additions[a] & additions[b])
    result = {"schema_version": "combined-verified-union-v2", "merge_rule": "RRF rank-only, sources [adaptive K500, BM25 body, KNN word, KNN char], equal weights, rrf_k=60; cap is applied after merge; no labels in merge/cap.", "query_count": len(baseline), "candidate_count": {"mean": sum(candidate_counts) / len(candidate_counts), "median": median(candidate_counts), "p95": percentile(candidate_counts, .95), "max": max(candidate_counts)}, "source_unique_docs_added_after_adaptive_dedup": {name: len(docs) for name, docs in additions.items()}, "addition_overlap": pair_overlap, "oracle": {}, "per_fold_candidate_oracle": {fold: {label: sum(vals) / len(vals) for label, vals in value.items()} for fold, value in per_fold.items()}}
    reference = {"k200": .975895238095238, "previous_adaptive_producer": .9820738095238095, "bm25_adaptive_union": .9893642857142857}
    for label, values in scores.items():
        value = sum(values) / len(values)
        result["oracle"][label] = {"value": value, "gain_vs": {name: value - ref for name, ref in reference.items()}, "k200_misses_rescued": miss_rescue[label], "multi_gold_misses_rescued": multi_rescue[label]}
    passing = [int(budget) for budget in budgets if result["oracle"][str(budget)]["value"] >= .99]
    result["candidate_oracle_gate"] = "CANDIDATE_ORACLE_GATE_PASSED" if passing else ("UNBOUNDED_ORACLE_ONLY_NOT_PROMOTABLE" if result["oracle"]["unbounded"]["value"] >= .99 else "CANDIDATE_ORACLE_GATE_NOT_YET_PASSED")
    result["smallest_passing_budget"] = min(passing) if passing else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
