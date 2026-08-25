from __future__ import annotations

from beam import function, Image, Volume

import hashlib
import json
import math
from pathlib import Path

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"

FULL_SCORES = (
    RUNTIME
    / "artifacts/task1/diagnostics/p13_recovery_v2_fusion_fold0/"
      "fold0_original_bge_top100_scores.jsonl"
)
CANDIDATES = (
    RUNTIME
    / "artifacts/task1/candidates/train7000_document_candidates.jsonl"
)
OUT_DIR = (
    RUNTIME
    / "artifacts/task1/diagnostics/p13_recovery_v2b_calibration_fold0"
)
REPORT = OUT_DIR / "calibration_report.json"

image = Image(python_version="python3.11")


def _z(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return [0.0 for _ in values]
    return [(x - mean) / sd for x in values]


def _rank_score(values, *, higher_better=True):
    order = sorted(
        range(len(values)),
        key=lambda i: values[i],
        reverse=higher_better,
    )
    rank = [0] * len(values)
    for r, i in enumerate(order, 1):
        rank[i] = r
    n = max(1, len(values))
    # 1.0 for top rank, approaching 0 for bottom rank.
    return [1.0 - (r - 1) / n for r in rank]


def _recall(gold, top5):
    truth = set(gold)
    return len(truth & set(top5[:5])) / len(truth)


def _metrics(rows, tops):
    recalls = []
    precisions = []
    multi = []
    full = 0
    for row, top in zip(rows, tops):
        gold = set(row["gold_documents"])
        pred = set(top[:5])
        hit = gold & pred
        rec = len(hit) / len(gold)
        pre = len(hit) / len(pred) if pred else 0.0
        recalls.append(rec)
        precisions.append(pre)
        if len(gold) > 1:
            multi.append(rec)
        if hit == gold:
            full += 1
    return {
        "official_macro_recall": sum(recalls) / len(recalls),
        "official_macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": sum(multi) / len(multi) if multi else None,
        "full_gold_query_count": full,
        "query_count": len(rows),
    }


def _top_by(scores, docs):
    return [
        docs[i]
        for i in sorted(
            range(len(docs)),
            key=lambda i: (-scores[i], i, docs[i]),
        )[:5]
    ]


@function(
    name="udsc-p13-recovery-v2b-calibration-fold0",
    cpu=4,
    memory="16Gi",
    image=image,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    print("=== P13 RECOVERY V2B: SCORE CALIBRATION, FOLD0 ===", flush=True)
    print("CPU only. Reuses cached original-BGE top100 scores. No training.", flush=True)

    for path in (FULL_SCORES, CANDIDATES):
        print(f"[CHECK] {path} exists={path.is_file()}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with FULL_SCORES.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != 1400:
        raise RuntimeError(f"Expected 1400 cached fold0 rows, got {len(rows)}")
    qids = {str(r["query_id"]) for r in rows}

    # Load only fold0 candidate records. The cache contains the original
    # collapse-v1 document order plus best dense score for each document.
    candidate_meta = {}
    with CANDIDATES.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec.get("question_id", rec.get("id", "")))
            if qid not in qids:
                continue
            docs = rec.get("documents", rec.get("hits", []))
            meta = {}
            for idx, item in enumerate(docs, 1):
                if not isinstance(item, dict):
                    continue
                doc = str(item.get("doc_id", item.get("document_id", ""))).strip()
                if not doc:
                    continue
                raw = item.get("dense_score", item.get("score"))
                dense = float(raw) if isinstance(raw, (int, float)) else None
                rank = int(item.get("rank", idx))
                meta[doc] = {"dense_score": dense, "prior_rank": rank}
            candidate_meta[qid] = meta

    if set(candidate_meta) != qids:
        missing = sorted(qids - set(candidate_meta))[:10]
        raise RuntimeError(f"Missing candidate metadata for fold0 qids: {missing}")

    # Build aligned feature arrays.
    prepared = []
    for row in rows:
        qid = str(row["query_id"])
        docs = list(map(str, row["candidate_documents"]))
        bge = [float(x) for x in row["bge_scores_aligned_to_candidates"]]
        if len(docs) != len(bge):
            raise RuntimeError(f"BGE score alignment mismatch at {qid}")

        meta = candidate_meta[qid]
        dense = []
        prior_rank = []
        for i, doc in enumerate(docs, 1):
            m = meta.get(doc, {})
            value = m.get("dense_score")
            dense.append(float(value) if isinstance(value, (int, float)) else float("nan"))
            prior_rank.append(int(m.get("prior_rank", i)))

        # Replace missing dense values conservatively with the query minimum.
        finite = [x for x in dense if math.isfinite(x)]
        fallback = min(finite) if finite else 0.0
        dense = [x if math.isfinite(x) else fallback for x in dense]

        prepared.append(
            {
                **row,
                "candidate_documents": docs,
                "_bge": bge,
                "_dense": dense,
                "_prior_rank": prior_rank,
                "_bge_z": _z(bge),
                "_dense_z": _z(dense),
                "_bge_rank_score": _rank_score(bge),
                "_dense_rank_score": _rank_score(dense),
                "_prior_rank_score": [
                    1.0 / math.log2(2.0 + max(1, r)) for r in prior_rank
                ],
            }
        )

    base_tops = [list(r["bge_top5"]) for r in prepared]
    base = _metrics(prepared, base_tops)

    configs = []

    # Score calibration: preserve BGE as the anchor and inject only modest prior.
    for prior_kind in ("dense_z", "dense_rank", "prior_rank"):
        for bge_kind in ("bge_z", "bge_rank"):
            for alpha_i in range(50, 101, 5):
                alpha = alpha_i / 100.0
                configs.append(
                    {
                        "kind": "linear",
                        "bge_feature": bge_kind,
                        "prior_feature": prior_kind,
                        "bge_weight": alpha,
                        "prior_weight": 1.0 - alpha,
                    }
                )

    def tops_for(config):
        out = []
        for row in prepared:
            if config["bge_feature"] == "bge_z":
                bf = row["_bge_z"]
            else:
                bf = row["_bge_rank_score"]

            pf_name = config["prior_feature"]
            if pf_name == "dense_z":
                pf = row["_dense_z"]
            elif pf_name == "dense_rank":
                pf = row["_dense_rank_score"]
            else:
                pf = row["_prior_rank_score"]

            a = config["bge_weight"]
            scores = [a * x + (1.0 - a) * y for x, y in zip(bf, pf)]
            out.append(_top_by(scores, row["candidate_documents"]))
        return out

    scored = []
    tops_cache = {}
    for config in configs:
        key = (
            config["bge_feature"],
            config["prior_feature"],
            config["bge_weight"],
        )
        tops = tops_for(config)
        tops_cache[key] = tops
        m = _metrics(prepared, tops)
        scored.append({**config, **m})

    scored.sort(
        key=lambda x: (
            x["official_macro_recall"],
            x["official_macro_precision"],
            x["multi_gold_recall"] if x["multi_gold_recall"] is not None else -1.0,
            x["bge_weight"],
        ),
        reverse=True,
    )
    best = scored[0]

    # Deterministic five-way inner CV.
    buckets = [[] for _ in range(5)]
    for i, row in enumerate(prepared):
        bucket = int(hashlib.sha256(str(row["query_id"]).encode()).hexdigest(), 16) % 5
        buckets[bucket].append(i)

    base_recalls = [_recall(r["gold_documents"], t) for r, t in zip(prepared, base_tops)]
    cv = []
    weighted_delta = 0.0
    weighted_n = 0

    for holdout in range(5):
        train_idx = [
            i
            for b in range(5)
            if b != holdout
            for i in buckets[b]
        ]
        test_idx = buckets[holdout]

        def train_score(config):
            key = (
                config["bge_feature"],
                config["prior_feature"],
                config["bge_weight"],
            )
            tops = tops_cache[key]
            return sum(
                _recall(prepared[i]["gold_documents"], tops[i])
                for i in train_idx
            ) / len(train_idx)

        chosen = max(
            configs,
            key=lambda c: (
                train_score(c),
                c["bge_weight"],  # conservative tie-break
            ),
        )
        key = (
            chosen["bge_feature"],
            chosen["prior_feature"],
            chosen["bge_weight"],
        )
        tops = tops_cache[key]
        fusion_test = sum(
            _recall(prepared[i]["gold_documents"], tops[i])
            for i in test_idx
        ) / len(test_idx)
        base_test = sum(base_recalls[i] for i in test_idx) / len(test_idx)
        delta = fusion_test - base_test
        weighted_delta += delta * len(test_idx)
        weighted_n += len(test_idx)
        cv.append(
            {
                "holdout": holdout,
                "query_count": len(test_idx),
                "selected": chosen,
                "base_recall": base_test,
                "fusion_recall": fusion_test,
                "delta": delta,
            }
        )

    best_key = (
        best["bge_feature"],
        best["prior_feature"],
        best["bge_weight"],
    )
    best_tops = tops_cache[best_key]
    better = worse = same = 0
    for row, bt, ft in zip(prepared, base_tops, best_tops):
        br = _recall(row["gold_documents"], bt)
        fr = _recall(row["gold_documents"], ft)
        if fr > br:
            better += 1
        elif fr < br:
            worse += 1
        else:
            same += 1

    report = {
        "schema_version": "p13-recovery-v2b-calibration-fold0-v1",
        "status": "COMPLETE",
        "note": "CPU-only calibration over cached original-BGE scores; no model training.",
        "base_original_bge": base,
        "same_fold_best": best,
        "same_fold_delta": (
            best["official_macro_recall"] - base["official_macro_recall"]
        ),
        "same_fold_query_outcomes": {
            "better": better,
            "worse": worse,
            "same": same,
        },
        "inner_cv": {
            "folds": cv,
            "weighted_recall_delta": weighted_delta / weighted_n,
        },
        "top_20_configs": scored[:20],
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={REPORT}", flush=True)
    return report


if __name__ == "__main__":
    print("Enqueuing P13 Recovery V2B calibration on Beam CPU...", flush=True)
    result = run.remote()
    print("REMOTE RESULT:", result, flush=True)
