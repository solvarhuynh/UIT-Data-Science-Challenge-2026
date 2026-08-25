from __future__ import annotations

from beam import function, Image, Volume

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
FOLDS = RUNTIME / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
CANDIDATES = RUNTIME / "artifacts/task1/candidates/train7000_document_candidates.jsonl"
RUNTIME_TRAINER = RUNTIME / "scripts/training/finetune_task1_bge_reranker.py"
MODEL_ZIP = VOLUME_ROOT / "udsc_p13_reranker.zip"

OUT = (
    RUNTIME
    / "artifacts/task1/diagnostics/p13_recovery_v2_fusion_fold0"
)
FULL_SCORES = OUT / "fold0_original_bge_top100_scores.jsonl"
REPORT = OUT / "fusion_report.json"

image = Image(
    python_version="python3.11",
    python_packages=[
        "torch",
        "transformers==5.0.0",
        "sentence-transformers==5.4.1",
        "accelerate>=1.1,<2",
        "huggingface-hub>=1.3,<2",
        "PyYAML>=6",
        "tqdm>=4.66,<5",
    ],
)


def _load_runtime_trainer():
    sys.path.insert(0, str(RUNTIME / "src"))
    spec = importlib.util.spec_from_file_location(
        "p13_runtime_trainer_for_fusion",
        RUNTIME_TRAINER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runtime trainer: {RUNTIME_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stage_original_model() -> Path:
    local_stage = Path("/tmp/p13_recovery_v2_model")
    local_zip = Path("/tmp/p13_recovery_v2_reranker.zip")
    local_model = local_stage / "models/reranker"

    shutil.rmtree(local_stage, ignore_errors=True)
    if local_zip.exists():
        local_zip.unlink()
    if not MODEL_ZIP.is_file():
        raise FileNotFoundError(MODEL_ZIP)

    shutil.copy2(MODEL_ZIP, local_zip)
    local_stage.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/python3.11", "-m", "zipfile", "-e", str(local_zip), str(local_stage)],
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if not (local_model / "config.json").is_file():
        raise FileNotFoundError(f"Original model staging failed: {local_model}")
    return local_model


def _metric(rows, key: str) -> dict:
    recalls = []
    precisions = []
    multi = []
    full = 0
    for row in rows:
        gold = set(row["gold_documents"])
        predicted = set(row[key][:5])
        matched = gold & predicted
        recalls.append(len(matched) / len(gold))
        precisions.append(len(matched) / len(predicted) if predicted else 0.0)
        if len(gold) > 1:
            multi.append(len(matched) / len(gold))
        if matched == gold:
            full += 1
    return {
        "official_macro_recall": sum(recalls) / len(recalls),
        "official_macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": sum(multi) / len(multi) if multi else None,
        "full_gold_query_count": full,
        "query_count": len(rows),
    }


def _rrf_top5(row: dict, *, k: int, bge_weight: float) -> list[str]:
    prior = list(row["candidate_documents"])
    bge = list(row["bge_ranking"])
    prior_rank = {doc: i + 1 for i, doc in enumerate(prior)}
    bge_rank = {doc: i + 1 for i, doc in enumerate(bge)}
    docs = set(prior) | set(bge)
    prior_weight = 1.0 - bge_weight

    scores = {}
    for doc in docs:
        score = 0.0
        if doc in prior_rank:
            score += prior_weight / (k + prior_rank[doc])
        if doc in bge_rank:
            score += bge_weight / (k + bge_rank[doc])
        scores[doc] = score

    return sorted(
        docs,
        key=lambda doc: (
            -scores[doc],
            prior_rank.get(doc, 10**9),
            bge_rank.get(doc, 10**9),
            doc,
        ),
    )[:5]


def _recall_for_indices(rows, tops, indices) -> float:
    total = 0.0
    for i in indices:
        gold = set(rows[i]["gold_documents"])
        total += len(gold & set(tops[i][:5])) / len(gold)
    return total / len(indices)


@function(
    name="udsc-p13-recovery-v2-fusion-fold0",
    cpu=4,
    memory="64Gi",
    gpu="RTX4090",
    image=image,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    print("=== P13 RECOVERY V2: ORIGINAL BGE + PRIOR FUSION, FOLD0 ===", flush=True)
    print("No training. Original reranker is read-only.", flush=True)

    for path in (TRAIN, FOLDS, CANDIDATES, RUNTIME_TRAINER, MODEL_ZIP):
        print(f"[CHECK] {path} exists={path.exists()}", flush=True)
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    trainer = _load_runtime_trainer()

    train = trainer.load_train(TRAIN)
    fold_map = trainer.load_fold_map(FOLDS)
    candidates = trainer.load_document_candidates(
        CANDIDATES,
        candidate_depth=100,
        evidence_limit=2,
    )
    validation_ids = sorted(
        qid for qid, fold in fold_map.items()
        if fold == 0 and qid in candidates and qid in train
    )
    if not validation_ids:
        raise RuntimeError("No fold0 validation IDs with candidates")

    rows = []
    if FULL_SCORES.is_file():
        try:
            with FULL_SCORES.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if [r.get("query_id") for r in rows] != validation_ids:
                print("[CACHE] stale score file -> recompute", flush=True)
                rows = []
            else:
                print(f"[CACHE] reuse {len(rows)} original-BGE scored queries", flush=True)
        except Exception:
            rows = []

    if not rows:
        model = _stage_original_model()
        backend = trainer.TorchBGERerankerBackend(
            str(model),
            device="cuda",
            batch_size=8,
            gradient_accumulation=1,
            max_length=512,
            learning_rate=1e-6,
            warmup_ratio=0.0,
            fp16=True,
            seed=2026,
        )
        try:
            from tqdm import tqdm
            for qid in tqdm(validation_ids, desc="Original BGE top100 scoring", unit="q"):
                query = train[qid]["question"]
                pool = candidates[qid]
                docs = [
                    trainer.format_document_evidence(item)
                    for item in pool
                ]
                scores = backend.score(query, docs)
                if len(scores) != len(pool):
                    raise RuntimeError("BGE score count mismatch")

                scored = [
                    trainer.replace(candidate, rerank_score=float(score))
                    for candidate, score in zip(pool, scores)
                ]
                ranked = trainer.final_document_ranking(scored)
                bge_ranking = [item.doc_id for item in ranked]
                top5 = trainer.final_submission_documents(ranked)

                rows.append(
                    {
                        "query_id": qid,
                        "gold_documents": train[qid]["gold"],
                        "candidate_documents": [item.doc_id for item in pool],
                        "bge_scores_aligned_to_candidates": [float(x) for x in scores],
                        "bge_ranking": bge_ranking,
                        "bge_top5": top5,
                    }
                )
        finally:
            backend.close()

        tmp = FULL_SCORES.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, FULL_SCORES)
        print(f"[WRITE] {FULL_SCORES}", flush=True)

    base_metrics = _metric(rows, "bge_top5")
    prior_rows = [{**r, "prior_top5": r["candidate_documents"][:5]} for r in rows]
    prior_metrics = _metric(prior_rows, "prior_top5")

    grid = [
        (k, round(w / 20.0, 2))
        for k in (0, 1, 5, 10, 20, 40, 60)
        for w in range(10, 21)
    ]
    tops_by_param = {}
    same_fold_results = []
    for k, weight in grid:
        tops = [_rrf_top5(r, k=k, bge_weight=weight) for r in rows]
        tops_by_param[(k, weight)] = tops
        metric_rows = [{**r, "fusion_top5": top} for r, top in zip(rows, tops)]
        m = _metric(metric_rows, "fusion_top5")
        same_fold_results.append(
            {
                "rrf_k": k,
                "bge_weight": weight,
                "prior_weight": round(1.0 - weight, 2),
                **m,
            }
        )

    same_fold_results.sort(
        key=lambda x: (
            x["official_macro_recall"],
            x["official_macro_precision"],
            x["multi_gold_recall"] if x["multi_gold_recall"] is not None else -1.0,
        ),
        reverse=True,
    )
    best = same_fold_results[0]

    # Deterministic 5-way inner CV on fold0. This is still diagnostic, but avoids
    # reporting only the optimistic score from tuning and evaluating on the same queries.
    split_indices = [[] for _ in range(5)]
    for i, row in enumerate(rows):
        bucket = int(hashlib.sha256(row["query_id"].encode()).hexdigest(), 16) % 5
        split_indices[bucket].append(i)

    cv_rows = []
    weighted_delta = 0.0
    weighted_n = 0
    for holdout in range(5):
        train_idx = [
            i for bucket in range(5) if bucket != holdout
            for i in split_indices[bucket]
        ]
        test_idx = split_indices[holdout]

        def train_recall(param):
            return _recall_for_indices(rows, tops_by_param[param], train_idx)

        selected = max(
            grid,
            key=lambda param: (
                train_recall(param),
                param[1],     # conservative tie-break: prefer more original BGE
                -param[0],
            ),
        )
        test_recall = _recall_for_indices(rows, tops_by_param[selected], test_idx)
        base_test = _recall_for_indices(
            rows, [r["bge_top5"] for r in rows], test_idx
        )
        delta = test_recall - base_test
        weighted_delta += delta * len(test_idx)
        weighted_n += len(test_idx)
        cv_rows.append(
            {
                "holdout": holdout,
                "query_count": len(test_idx),
                "selected_rrf_k": selected[0],
                "selected_bge_weight": selected[1],
                "base_recall": base_test,
                "fusion_recall": test_recall,
                "delta": delta,
            }
        )

    report = {
        "schema_version": "p13-recovery-v2-fusion-fold0-v1",
        "status": "COMPLETE",
        "note": "No model training was performed.",
        "candidate_depth": 100,
        "evidence_limit": 2,
        "original_bge": base_metrics,
        "candidate_prior": prior_metrics,
        "same_fold_best": best,
        "same_fold_delta": best["official_macro_recall"] - base_metrics["official_macro_recall"],
        "inner_cv": {
            "folds": cv_rows,
            "weighted_recall_delta": weighted_delta / weighted_n,
        },
        "top_10_same_fold_configs": same_fold_results[:10],
        "full_scores": str(FULL_SCORES),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== RESULT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={REPORT}", flush=True)
    return report


if __name__ == "__main__":
    print("Enqueuing P13 Recovery V2 fusion diagnostic...", flush=True)
    result = run.remote()
    print("REMOTE RESULT:", result, flush=True)
