"""No-GPU forensic and score-coverage audit after Selective BGE K500 V1.

Uses only completed local caches.  Gold is used exclusively for post-hoc
diagnostics; neither the RRF union nor score-coverage labels use it.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "artifacts/task1/recovery_096"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["query_id"]): row for row in rows}
    if len(result) != len(list(rows)) if not isinstance(rows, list) else False:
        raise ValueError("duplicate query_id")
    return result


def doc_features(hits: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["doc_id"])].append(hit)
    result = {}
    for doc, values in grouped.items():
        scores = sorted((float(x["bge_score"]) for x in values), reverse=True)
        result[doc] = {
            "best_bge": scores[0],
            "mean_top3_bge": sum(scores[:3]) / min(3, len(scores)),
            "support": float(len(scores)),
            "dense_rank": float(min(int(x["dense_rank"]) for x in values)),
        }
    return result


def rrf(sources: list[list[str]], rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first: dict[str, tuple[int, int]] = {}
    for source_idx, docs in enumerate(sources):
        for rank, doc in enumerate(docs, 1):
            scores[doc] += 1.0 / (rrf_k + rank)
            first.setdefault(doc, (source_idx, rank))
    return sorted(scores, key=lambda doc: (-scores[doc], first[doc], doc))


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "median": median(values) if values else None, "p95": pct(values, .95),
            "min": min(values) if values else None, "max": max(values) if values else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RECOVERY / "post_k500_bounded_union_v2_audit")
    args = parser.parse_args()
    baseline = index(load_jsonl(RECOVERY / "baseline_093_oof/predictions.jsonl"))
    decisions = index(load_jsonl(RECOVERY / "adaptive_k500_v1/adaptive_k500_decisions.jsonl"))
    k200 = index(load_jsonl(RECOVERY / "baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl") + load_jsonl(RECOVERY / "baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"))
    k500 = index(load_jsonl(RECOVERY / "selective_bge_k500_v1_from_beam/new_chunk_scores.jsonl"))
    old_eval = json.loads((RECOVERY / "selective_bge_k500_v1_evaluation_v1/report.json").read_text(encoding="utf-8"))
    changed = load_jsonl(RECOVERY / "selective_bge_k500_v1_evaluation_v1/changed_docs.jsonl")
    checkpoint = RECOVERY / "candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
    lexical: dict[str, dict[str, list[str]]] = {name: {} for name in ("bm25", "knn_word", "knn_char")}
    for name in lexical:
        for fold in range(5):
            payload = json.loads((checkpoint / f"{name}_fold{fold}.json").read_text(encoding="utf-8"))
            lexical[name].update({str(q): [str(d) for d in docs] for q, docs in payload["rankings"].items()})
    expected = set(baseline)
    if any(set(source) != expected for source in [decisions, k200, *lexical.values()]):
        raise ValueError("coverage cache mismatch")

    coverage = Counter()
    missing_per_query: list[float] = []
    selected_missing: list[float] = []
    unselected_missing: list[float] = []
    rank_by_bucket: dict[str, list[float]] = defaultdict(list)
    gold_by_bucket = Counter()
    gold_union_rank: list[float] = []
    lexical_added = total_union = 0
    source_sets = {name: 0 for name in ("adaptive", "bm25", "knn_word", "knn_char", "multi_source")}
    selected_forensics = {"selected_queries": 0, "new_pool_adds_gold_outside_k200": 0,
                         "gold_docs_with_new_bge": 0, "gold_docs_competitive_top5_by_new_bge": 0,
                         "new_pool_gold_oracle_recall_values": []}
    gold_new_doc_ranks: list[float] = []
    gold_new_scores: list[float] = []
    for qid, base in baseline.items():
        decision = decisions[qid]
        adaptive = [str(x) for x in decision["k200_documents"]] + [str(x) for x in decision["k500_added_documents"]]
        sources = {"adaptive": adaptive, "bm25": lexical["bm25"][qid], "knn_word": lexical["knn_word"][qid], "knn_char": lexical["knn_char"][qid]}
        union = rrf(list(sources.values()))[:200]
        total_union += len(union)
        old = doc_features(k200[qid]["hits"])
        new = doc_features(k500[qid]["new_hits"]) if qid in k500 else {}
        missing = 0
        for rank, doc in enumerate(union, 1):
            membership = [name for name, docs in sources.items() if doc in docs]
            key = "multi_source" if len(membership) > 1 else membership[0]
            source_sets[key] += 1
            category = "k200_bge" if doc in old else ("k500_v1_bge" if doc in new else "missing_bge")
            coverage[category] += 1
            coverage[f"source_{key}"] += 1
            rank_by_bucket[category].append(rank)
            if category == "missing_bge":
                missing += 1
            if doc not in old and any(name != "adaptive" for name in membership):
                lexical_added += 1
        missing_per_query.append(missing)
        (selected_missing if decision["expand_to_k500"] else unselected_missing).append(missing)
        gold = set(map(str, base["gold_documents"]))
        # Rescue diagnostic: gold in union but absent from the actual baseline top5.
        for doc in gold & set(union) - set(map(str, base["top5"])):
            category = "k200_bge" if doc in old else ("k500_v1_bge" if doc in new else "missing_bge")
            gold_by_bucket[category] += 1
            gold_union_rank.append(union.index(doc) + 1)
        if qid in k500:
            selected_forensics["selected_queries"] += 1
            new_docs = new
            extra_gold = gold & set(new_docs) - set(map(str, decision["k200_documents"]))
            if extra_gold:
                selected_forensics["new_pool_adds_gold_outside_k200"] += 1
            selected_forensics["gold_docs_with_new_bge"] += len(extra_gold)
            selected_forensics["new_pool_gold_oracle_recall_values"].append(len(gold & set(new_docs)) / len(gold))
            ranked_new = sorted(new_docs, key=lambda d: (-new_docs[d]["best_bge"], -new_docs[d]["support"], new_docs[d]["dense_rank"], d))
            incumbent = old.get(str(base["top5"][-1]), {"best_bge": float("inf")})["best_bge"]
            for doc in extra_gold:
                rank = ranked_new.index(doc) + 1
                gold_new_doc_ranks.append(rank); gold_new_scores.append(new_docs[doc]["best_bge"])
                if new_docs[doc]["best_bge"] >= incumbent:
                    selected_forensics["gold_docs_competitive_top5_by_new_bge"] += 1

    action_margins: dict[str, list[float]] = defaultdict(list)
    action_scores: dict[str, list[float]] = defaultdict(list)
    action_support: dict[str, list[float]] = defaultdict(list)
    for row in changed:
        before = len(set(row["gold_documents"]) & set(row["baseline_top5"]))
        after = len(set(row["gold_documents"]) & set(row["policy_top5"]))
        bucket = "better" if after > before else ("worse" if after < before else "neutral")
        action_margins[bucket].append(float(row["bge_margin"]))
        action_scores[bucket].append(float(row["inserted"]["best_bge"]))
        action_support[bucket].append(float(row["inserted"]["support_count"]))
    selected_forensics["new_pool_gold_oracle_recall"] = sum(selected_forensics.pop("new_pool_gold_oracle_recall_values")) / len(k500)
    report = {
        "schema_version": "post-k500-bounded-union-audit-v2a",
        "constraints": {"no_new_bge_scoring": True, "neural_training": False, "public_submission_created": False},
        "k500_forensics": {"k500_v1_status": old_eval["status"], "selected_policy_by_outer_fold": {f: v["selected_policy"] for f, v in old_eval["per_fold"].items()},
            "per_fold": old_eval["per_fold"], "outcomes": old_eval["outcomes_vs_baseline"], "changed_documents": changed,
            "realized_insert_distribution": {bucket: {"bge_margin": distribution(action_margins[bucket]), "inserted_best_bge": distribution(action_scores[bucket]), "inserted_support": distribution(action_support[bucket])} for bucket in ("better", "worse", "neutral")},
            "counterfactual_selected_pool": {**selected_forensics, "gold_new_doc_bge_rank": distribution(gold_new_doc_ranks), "gold_new_doc_best_bge": distribution(gold_new_scores)}},
        "bounded_union_200_score_coverage": {"merge_rule": "RRF rank-only/equal weights/rrf_k=60, adaptive+BM25+KNN word+KNN char, cap=200",
            "total_candidates": total_union, "mean_candidates_per_query": total_union / len(baseline), "docs_by_score_coverage": dict(coverage),
            "missing_score_docs_per_query": distribution(missing_per_query), "missing_score_selected_queries": distribution(selected_missing), "missing_score_nonselected_queries": distribution(unselected_missing),
            "lexical_added_docs_outside_k200": lexical_added, "source_membership": source_sets,
            "union_rank_by_score_coverage": {name: distribution(values) for name, values in rank_by_bucket.items()},
            "gold_rescue_not_in_baseline_top5": {"docs_by_score_coverage": dict(gold_by_bucket), "union_rank": distribution(gold_union_rank)}},
        "decision": "V2A_REUSE_ONLY_READY" if coverage["missing_bge"] == 0 else "V2A_REUSE_ONLY_REQUIRED_BEFORE_V2B_PREFLIGHT",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text("# Post-K500 forensic & bounded-union score coverage\n\n" + json.dumps({"k500_status": old_eval["status"], "coverage": report["bounded_union_200_score_coverage"]["docs_by_score_coverage"], "decision": report["decision"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output_dir), "decision": report["decision"], "coverage": report["bounded_union_200_score_coverage"]["docs_by_score_coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
