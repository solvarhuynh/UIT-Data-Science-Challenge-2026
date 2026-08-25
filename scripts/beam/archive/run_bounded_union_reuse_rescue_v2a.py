"""Strict nested-OOF bounded union rescue V2A, reusing BGE scores only."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_post_k500_bounded_union_v2 import RECOVERY, doc_features, index, load_jsonl, rrf

ANCHOR = 0.9244452380952382


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    recalls = [len(set(r["gold"]) & set(r[field])) / len(r["gold"]) for r in rows]
    precisions = [len(set(r["gold"]) & set(r[field])) / 5 for r in rows]
    return {"macro_recall": sum(recalls) / len(recalls), "macro_precision": sum(precisions) / len(precisions), "query_count": len(rows)}


def candidate(row: dict[str, Any], config: tuple[float, float, int, int] | None) -> tuple[list[str], dict[str, Any] | None]:
    base = row["baseline"]
    if config is None:
        return base, None
    min_bge, margin, min_support, rank_cap = config
    incumbent = row["scores"].get(base[-1])
    if incumbent is None:
        return base, None
    pool = []
    for rank, doc in enumerate(row["union"][:rank_cap], 1):
        feat = row["scores"].get(doc)
        if doc in base or feat is None or feat["best_bge"] < min_bge or feat["support"] < min_support:
            continue
        if feat["best_bge"] - incumbent["best_bge"] < margin:
            continue
        pool.append((doc, feat, rank))
    if not pool:
        return base, None
    doc, feat, rank = min(pool, key=lambda x: (-x[1]["best_bge"], -x[1]["support"], x[2], x[0]))
    return base[:4] + [doc], {"removed_doc_id": base[-1], "inserted_doc_id": doc, "union_rank": rank, "inserted": feat, "removed": incumbent, "bge_margin": feat["best_bge"] - incumbent["best_bge"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RECOVERY / "bounded_union_reuse_rescue_v2a")
    args = parser.parse_args()
    baseline = index(load_jsonl(RECOVERY / "baseline_093_oof/predictions.jsonl"))
    decisions = index(load_jsonl(RECOVERY / "adaptive_k500_v1/adaptive_k500_decisions.jsonl"))
    k200 = index(load_jsonl(RECOVERY / "baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl") + load_jsonl(RECOVERY / "baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"))
    k500 = index(load_jsonl(RECOVERY / "selective_bge_k500_v1_from_beam/new_chunk_scores.jsonl"))
    ckpt = RECOVERY / "candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
    lexical = {name: {} for name in ("bm25", "knn_word", "knn_char")}
    for name in lexical:
        for fold in range(5):
            payload = json.loads((ckpt / f"{name}_fold{fold}.json").read_text(encoding="utf-8"))
            lexical[name].update({str(q): [str(d) for d in docs] for q, docs in payload["rankings"].items()})
    rows = []
    for qid, base in baseline.items():
        d = decisions[qid]
        adaptive = [str(x) for x in d["k200_documents"]] + [str(x) for x in d["k500_added_documents"]]
        scores = doc_features(k200[qid]["hits"])
        if qid in k500:
            scores.update({doc: feat for doc, feat in doc_features(k500[qid]["new_hits"]).items() if doc not in scores})
        rows.append({"query_id": qid, "fold": int(base["fold"]), "gold": [str(x) for x in base["gold_documents"]], "baseline": [str(x) for x in base["top5"]], "scores": scores, "union": rrf([adaptive, lexical["bm25"][qid], lexical["knn_word"][qid], lexical["knn_char"][qid]])[:200], "miss_k200": bool(d["miss_k200"])})
    # Deliberately small, hand-specified policy family; no held-out fitting.
    configs = [("KEEP_BASELINE", None), ("bge4_margin0_s1_rank200", (4., 0., 1, 200)), ("bge5_margin0_s1_rank200", (5., 0., 1, 200)), ("bge6_margin0_s1_rank200", (6., 0., 1, 200)), ("bge5_margin05_s1_rank200", (5., .5, 1, 200)), ("bge6_margin05_s2_rank200", (6., .5, 2, 200)), ("bge6_margin1_s2_rank200", (6., 1., 2, 200)), ("bge5_margin05_s2_rank50", (5., .5, 2, 50)), ("bge6_margin05_s2_rank50", (6., .5, 2, 50))]
    chosen = {}
    for fold in range(5):
        train = [r for r in rows if r["fold"] != fold]
        scores = []
        for name, config in configs:
            simulated = []
            harms = changes = 0
            for row in train:
                top5, detail = candidate(row, config)
                before = len(set(row["gold"]) & set(row["baseline"])) / len(row["gold"])
                after = len(set(row["gold"]) & set(top5)) / len(row["gold"])
                harms += after < before; changes += detail is not None
                simulated.append({**row, "policy": top5})
            m = metric(simulated, "policy")
            scores.append(((m["macro_recall"], m["macro_precision"], -harms, -changes), name, config))
        _, name, config = max(scores, key=lambda x: x[0]); chosen[fold] = (name, config)
    changed = []
    for row in rows:
        name, config = chosen[row["fold"]]
        top5, detail = candidate(row, config)
        row["policy"] = top5; row["selected_policy"] = name
        if detail:
            changed.append({"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold"], "baseline_top5": row["baseline"], "policy_top5": top5, "selected_policy": name, **detail})
    base_m, policy_m = metric(rows, "baseline"), metric(rows, "policy")
    per_fold = {}
    for fold in range(5):
        part = [r for r in rows if r["fold"] == fold]
        b, p = metric(part, "baseline"), metric(part, "policy")
        per_fold[str(fold)] = {"baseline": b, "policy": p, "delta": p["macro_recall"] - b["macro_recall"], "selected_policy": chosen[fold][0]}
    better = sum(len(set(r["gold"]) & set(r["policy"])) > len(set(r["gold"]) & set(r["baseline"])) for r in rows)
    worse = sum(len(set(r["gold"]) & set(r["policy"])) < len(set(r["gold"]) & set(r["baseline"])) for r in rows)
    gate = {"pooled_recall_above_anchor": policy_m["macro_recall"] > ANCHOR, "no_negative_fold": all(v["delta"] >= 0 for v in per_fold.values()), "improvements_at_least_harms": better >= worse}
    gate["promote"] = all(gate.values())
    status = "PROMOTE_BOUNDED_UNION_REUSE_RESCUE_V2A" if gate["promote"] else "REJECT_BOUNDED_UNION_REUSE_RESCUE_V2A"
    report = {"schema_version": "bounded-union-reuse-rescue-v2a", "status": status, "baseline": base_m, "policy": policy_m, "recall_delta_vs_anchor": policy_m["macro_recall"] - ANCHOR, "per_fold": per_fold, "outcomes_vs_baseline": {"better": better, "same": len(rows)-better-worse, "worse": worse, "changed_queries": len(changed)}, "gate": gate, "constraints": {"strict_nested_outer_folds": 5, "no_new_bge_scoring": True, "gold_as_inference_feature": False, "candidate_pool": "locked_bounded_union_at_200"}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "changed_docs.jsonl").open("w", encoding="utf-8") as f:
        for row in changed: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": status, "baseline": base_m["macro_recall"], "policy": policy_m["macro_recall"], "delta": report["recall_delta_vs_anchor"], "changed_queries": len(changed), "gate": gate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
