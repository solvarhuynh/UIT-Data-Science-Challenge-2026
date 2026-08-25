"""Ablation candidate-oracle từ cache producer đã có; không chạy neural."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def oracle(gold: set[str], docs: Sequence[str]) -> float:
    return len(gold & set(docs)) / len(gold)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--v3", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--compact-a", type=Path, required=True)
    p.add_argument("--compact-b", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    baseline = {str(x["query_id"]): x for x in jsonl(args.baseline)}
    v3 = {str(x["query_id"]): x for x in jsonl(args.v3)}
    decisions = {str(x["query_id"]): x for x in jsonl(args.decisions)}
    bge: dict[str, list[str]] = {}
    for row in jsonl(args.compact_a) + jsonl(args.compact_b):
        score: dict[str, float] = defaultdict(lambda: float("-inf"))
        for hit in row["hits"]:
            doc = str(hit["doc_id"])
            score[doc] = max(score[doc], float(hit["bge_score"]))
        bge[str(row["query_id"])] = sorted(score, key=lambda doc: (-score[doc], doc))
    if set(baseline) != set(v3) or set(baseline) != set(decisions) or set(baseline) != set(bge):
        raise ValueError("coverage cache không khớp")
    source_docs: dict[str, dict[str, list[str]]] = {
        "raw_dense_k200": {q: decisions[q]["k200_documents"] for q in baseline},
        "bge_chunk200": bge,
        "producer_baseline_top5": {q: baseline[q]["top5"] for q in baseline},
        "v3_delta_top5": {q: v3[q]["top5"] for q in baseline},
        "adaptive_k500_selected_candidate": {q: decisions[q]["k200_documents"] + decisions[q]["k500_added_documents"] for q in baseline},
    }
    variants = {
        **source_docs,
        "raw_dense_plus_bge": {q: source_docs["raw_dense_k200"][q] + source_docs["bge_chunk200"][q] for q in baseline},
        "producer_plus_adaptive_candidate": {q: source_docs["producer_baseline_top5"][q] + source_docs["adaptive_k500_selected_candidate"][q] for q in baseline},
        "all_verified_cache_sources": {q: source_docs["raw_dense_k200"][q] + source_docs["bge_chunk200"][q] + source_docs["producer_baseline_top5"][q] + source_docs["v3_delta_top5"][q] + source_docs["adaptive_k500_selected_candidate"][q] for q in baseline},
    }
    scores = {name: mean([oracle(set(baseline[q]["gold_documents"]), docs) for q, docs in value.items()]) for name, value in variants.items()}
    report = {
        "schema_version": "candidate-union-cache-ablation-v1",
        "status": "PARTIAL_VERIFIED_CACHE_ONLY",
        "query_count": len(baseline),
        "candidate_oracle": scores,
        "top5_baseline_recall": scores["producer_baseline_top5"],
        "reranker_gap_k200": scores["raw_dense_k200"] - scores["producer_baseline_top5"],
        "reranker_gap_adaptive_candidate": scores["producer_plus_adaptive_candidate"] - scores["producer_baseline_top5"],
        "lexical_source_status": {
            "bm25_words": "NOT_COMPLETED_FULL_OOF_TIMEOUT_30M",
            "knn_words": "NOT_COMPLETED_FULL_OOF_TIMEOUT_30M",
            "knn_char": "NOT_COMPLETED_FULL_OOF_TIMEOUT_30M",
            "citation": "NOT_COMPLETED_FULL_OOF_TIMEOUT_30M",
            "reason": "ablate_legal_ir_lexical.py không checkpoint; không dùng kết quả partial hay proxy để kết luận incremental gain.",
        },
        "decision": "Ưu tiên nếu tiếp tục: cache/checkpoint lexical retrieval hoặc selective BGE K500 cho tập adaptive; không thay top5 hiện tại chỉ từ candidate oracle.",
        "constraints": {"neural_training": False, "public_submission_created": False, "blanket_k500_bge": False},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_union_ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "candidate_union_ablation.md").write_text("# Candidate union ablation (cache đã xác minh)\n\n" + "| Nguồn | Candidate-oracle Recall |\n|---|---:|\n" + "\n".join(f"| {name} | {score:.6f} |" for name, score in scores.items()) + "\n\nCác nguồn BM25/KNN/citation chưa có full ranking mới; run strict-OOF 30 phút đã timeout trước checkpoint nên không suy diễn gain.\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
