from __future__ import annotations

from beam import function, Image, Volume

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
FOLDS = RUNTIME / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
RAW_K200 = RUNTIME / "artifacts/task1/raw_k200.jsonl"
DATA_ZIP = VOLUME_ROOT / "udsc_p13_data.zip"
MODEL_ZIP = VOLUME_ROOT / "udsc_p13_reranker.zip"

OUT = RUNTIME / "artifacts/task1/diagnostics/p13_recovery_v3_chunkagg_fold0"
CACHE = OUT / "fold0_original_bge_chunk200_compact.jsonl"
REPORT = OUT / "chunk_aggregation_report.json"

image = Image(
    python_version="python3.11",
    python_packages=[
        "torch",
        "transformers==5.0.0",
        "accelerate>=1.1,<2",
        "huggingface-hub>=1.3,<2",
        "tqdm>=4.66,<5",
    ],
)


def _ensure_raw_k200() -> None:
    if RAW_K200.is_file():
        print(f"[RAW] found {RAW_K200}", flush=True)
        return
    if not DATA_ZIP.is_file():
        raise FileNotFoundError(
            f"{RAW_K200} is missing and archive {DATA_ZIP} is unavailable"
        )
    print(f"[RAW] {RAW_K200} missing; inspecting {DATA_ZIP.name}", flush=True)
    wanted = "artifacts/task1/raw_k200.jsonl"
    with zipfile.ZipFile(DATA_ZIP) as zf:
        names = set(zf.namelist())
        candidates = [
            name for name in names
            if name.endswith("/artifacts/task1/raw_k200.jsonl")
            or name == wanted
        ]
        if not candidates:
            raise FileNotFoundError(
                "raw_k200.jsonl is not present in udsc_p13_data.zip. "
                "Upload artifacts/task1/raw_k200.jsonl to "
                "/workspace/p13/runtime/artifacts/task1/raw_k200.jsonl and rerun."
            )
        member = sorted(candidates, key=len)[0]
        RAW_K200.parent.mkdir(parents=True, exist_ok=True)
        tmp = RAW_K200.with_suffix(".jsonl.tmp")
        with zf.open(member) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        os.replace(tmp, RAW_K200)
    print(f"[RAW] extracted {RAW_K200}", flush=True)


def _stage_model() -> Path:
    stage = Path("/tmp/p13_v3_model_stage")
    local_zip = Path("/tmp/p13_v3_reranker.zip")
    model = stage / "models/reranker"
    shutil.rmtree(stage, ignore_errors=True)
    if local_zip.exists():
        local_zip.unlink()
    if not MODEL_ZIP.is_file():
        raise FileNotFoundError(MODEL_ZIP)
    shutil.copy2(MODEL_ZIP, local_zip)
    stage.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/python3.11", "-m", "zipfile", "-e", str(local_zip), str(stage)],
        check=True,
    )
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model staging failed: {model}")
    return model


def _load_train() -> dict[str, dict[str, Any]]:
    raw = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    out = {}
    for qid, row in raw.items():
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        answers = row.get("answer")
        if question and isinstance(answers, list) and answers:
            out[str(qid)] = {
                "question": question,
                "gold": [str(x).strip() for x in answers if str(x).strip()],
            }
    return out


def _load_fold0_ids() -> list[str]:
    raw = json.loads(FOLDS.read_text(encoding="utf-8-sig"))
    ids = []
    for item in raw.get("folds", []):
        if int(item.get("fold", -1)) == 0:
            ids.extend(str(x) for x in item.get("validation_ids", []))
    return sorted(ids)


class BGEReranker:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.torch = torch
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.max_length = 512

    def _logits(self, encoded):
        logits = self.model(**encoded).logits
        if logits.ndim == 2 and logits.shape[-1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits.reshape(-1)

    def score(self, query: str, texts: list[str]) -> list[float]:
        torch = self.torch
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    vals = self._logits(encoded).detach().float().cpu().tolist()
                scores.extend(float(x) for x in vals)
        return scores

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def _rank_aggregate(hits: list[dict], *, source: str, cap: int, rank_k: int) -> list[str]:
    if source == "dense":
        ordered = sorted(hits, key=lambda h: (h["dense_rank"], h["doc_id"], h["chunk_id"]))
    elif source == "bge":
        ordered = sorted(
            hits,
            key=lambda h: (-h["bge_score"], h["dense_rank"], h["doc_id"], h["chunk_id"]),
        )
    else:
        raise ValueError(source)

    scores = {}
    counts = {}
    first_rank = {}
    for rank, hit in enumerate(ordered, 1):
        doc = hit["doc_id"]
        first_rank.setdefault(doc, rank)
        count = counts.get(doc, 0)
        if count >= cap:
            continue
        counts[doc] = count + 1
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rank_k + rank)
    return sorted(scores, key=lambda d: (-scores[d], first_rank[d], d))


def _weighted_rrf(rankings: list[list[str]], weights: list[float], rank_k: int) -> list[str]:
    scores = {}
    maps = []
    for ranking, weight in zip(rankings, weights):
        rank_map = {doc: i + 1 for i, doc in enumerate(ranking)}
        maps.append(rank_map)
        for doc, rank in rank_map.items():
            scores[doc] = scores.get(doc, 0.0) + weight / (rank_k + rank)
    return sorted(
        scores,
        key=lambda d: (-scores[d], *(m.get(d, 10**9) for m in maps), d),
    )


def _recall(gold: list[str], top5: list[str]) -> float:
    truth = set(gold)
    return len(truth & set(top5[:5])) / len(truth)


def _metrics(rows: list[dict], tops: list[list[str]]) -> dict:
    recalls = []
    precisions = []
    multi = []
    full = 0
    for row, top in zip(rows, tops):
        gold = set(row["gold_documents"])
        pred = set(top[:5])
        hit = gold & pred
        rec = len(hit) / len(gold)
        recalls.append(rec)
        precisions.append(len(hit) / len(pred) if pred else 0.0)
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


def _all_source_configs():
    return [
        (cap, rank_k)
        for cap in (2, 3, 5, 10)
        for rank_k in (0, 5, 10, 20, 40, 60)
    ]


@function(
    name="udsc-p13-recovery-v3-chunkagg-fold0",
    cpu=4,
    memory="32Gi",
    gpu="RTX4090",
    image=image,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    print("=== P13 RECOVERY V3: ORIGINAL BGE CHUNK AGGREGATION, FOLD0 ===", flush=True)
    print("No training. Scores raw top200 chunks with the original reranker.", flush=True)

    for p in (TRAIN, FOLDS, MODEL_ZIP):
        print(f"[CHECK] {p} exists={p.exists()}", flush=True)
        if not p.exists():
            raise FileNotFoundError(p)
    _ensure_raw_k200()
    OUT.mkdir(parents=True, exist_ok=True)

    train = _load_train()
    fold0 = _load_fold0_ids()
    if len(fold0) != 1400:
        raise RuntimeError(f"expected 1400 fold0 IDs, got {len(fold0)}")
    fold0_set = set(fold0)

    rows = []
    if CACHE.is_file():
        try:
            with CACHE.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if [r["query_id"] for r in rows] != fold0:
                print("[CACHE] mismatch -> recompute", flush=True)
                rows = []
            else:
                print(f"[CACHE] reuse {len(rows)} scored fold0 queries", flush=True)
        except Exception:
            rows = []

    if not rows:
        raw_rows = {}
        with RAW_K200.open(encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("question_id", row.get("id", "")))
                if qid in fold0_set:
                    raw_rows[qid] = row
        missing = sorted(fold0_set - set(raw_rows))
        if missing:
            raise RuntimeError(f"raw_k200 missing fold0 IDs: {missing[:10]}")

        model_dir = _stage_model()
        reranker = BGEReranker(model_dir)
        try:
            from tqdm import tqdm
            for qid in tqdm(fold0, desc="Original BGE chunk200 scoring", unit="q"):
                raw = raw_rows[qid]
                compact_hits = []
                texts = []
                raw_hits = raw.get("hits", [])
                for idx, hit in enumerate(raw_hits[:200], 1):
                    if not isinstance(hit, dict):
                        continue
                    doc = str(hit.get("doc_id", hit.get("document_id", ""))).strip()
                    chunk = str(hit.get("chunk_id", hit.get("evidence_id", ""))).strip()
                    text = str(hit.get("text", hit.get("evidence", ""))).strip()
                    if not doc or not text:
                        continue
                    dense_raw = hit.get("dense_score", hit.get("score", 0.0))
                    compact_hits.append(
                        {
                            "doc_id": doc,
                            "chunk_id": chunk or f"{doc}:raw:{idx}",
                            "dense_rank": idx,
                            "dense_score": float(dense_raw),
                        }
                    )
                    texts.append(text)
                if not compact_hits:
                    raise RuntimeError(f"no usable raw hits at {qid}")
                bge_scores = reranker.score(train[qid]["question"], texts)
                if len(bge_scores) != len(compact_hits):
                    raise RuntimeError(f"score length mismatch at {qid}")
                for hit, score in zip(compact_hits, bge_scores):
                    hit["bge_score"] = float(score)

                rows.append(
                    {
                        "query_id": qid,
                        "gold_documents": train[qid]["gold"],
                        "hits": compact_hits,
                    }
                )
        finally:
            reranker.close()

        tmp = CACHE.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, CACHE)
        print(f"[WRITE] {CACHE}", flush=True)

    source_configs = _all_source_configs()
    dense_tops = {}
    bge_tops = {}
    source_results = []

    for cap, rank_k in source_configs:
        name = f"rank_cap{cap}_k{rank_k}"
        dtops = [_rank_aggregate(r["hits"], source="dense", cap=cap, rank_k=rank_k)[:5] for r in rows]
        btops = [_rank_aggregate(r["hits"], source="bge", cap=cap, rank_k=rank_k)[:5] for r in rows]
        dense_tops[name] = dtops
        bge_tops[name] = btops
        source_results.append({"source": "dense", "config": name, **_metrics(rows, dtops)})
        source_results.append({"source": "bge", "config": name, **_metrics(rows, btops)})

    source_results.sort(
        key=lambda x: (x["official_macro_recall"], x["official_macro_precision"]),
        reverse=True,
    )

    # Historical robust configs are evaluated exactly, before any fold0 tuning.
    hist_defs = [
        {
            "name": "old_robust",
            "dense": "rank_cap3_k10",
            "bge": "rank_cap5_k10",
            "dense_weight": 0.3,
            "bge_weight": 0.7,
            "rrf_k": 0,
        },
        {
            "name": "old_best_pooled",
            "dense": "rank_cap5_k5",
            "bge": "rank_cap5_k10",
            "dense_weight": 0.3,
            "bge_weight": 0.7,
            "rrf_k": 5,
        },
    ]
    historical = []
    for cfg in hist_defs:
        tops = [
            _weighted_rrf(
                [dense_tops[cfg["dense"]][i], bge_tops[cfg["bge"]][i]],
                [cfg["dense_weight"], cfg["bge_weight"]],
                cfg["rrf_k"],
            )[:5]
            for i in range(len(rows))
        ]
        historical.append({**cfg, **_metrics(rows, tops)})

    # Full diagnostic grid on fold0.
    dense_short = [
        "rank_cap5_k10",
        "rank_cap10_k5",
        "rank_cap10_k10",
        "rank_cap5_k5",
        "rank_cap3_k10",
        "rank_cap3_k20",
        "rank_cap5_k20",
    ]
    bge_short = [
        "rank_cap10_k5",
        "rank_cap10_k10",
        "rank_cap5_k10",
        "rank_cap5_k20",
        "rank_cap5_k5",
        "rank_cap3_k20",
        "rank_cap3_k40",
    ]
    grid = []
    tops_cache = {}
    for dn in dense_short:
        for bn in bge_short:
            for rrf_k in (0, 5, 10, 20, 40, 60):
                for dense_tenths in range(1, 6):
                    dw = dense_tenths / 10.0
                    bw = 1.0 - dw
                    key = (dn, bn, dw, rrf_k)
                    tops = [
                        _weighted_rrf(
                            [dense_tops[dn][i], bge_tops[bn][i]],
                            [dw, bw],
                            rrf_k,
                        )[:5]
                        for i in range(len(rows))
                    ]
                    tops_cache[key] = tops
                    grid.append(
                        {
                            "dense": dn,
                            "bge": bn,
                            "dense_weight": dw,
                            "bge_weight": bw,
                            "rrf_k": rrf_k,
                            **_metrics(rows, tops),
                        }
                    )
    grid.sort(
        key=lambda x: (x["official_macro_recall"], x["official_macro_precision"]),
        reverse=True,
    )
    best = grid[0]

    # Deterministic inner CV: choose config on 4 buckets, evaluate held-out bucket.
    buckets = [[] for _ in range(5)]
    for i, row in enumerate(rows):
        bucket = int(hashlib.sha256(row["query_id"].encode()).hexdigest(), 16) % 5
        buckets[bucket].append(i)

    def avg_recall(indices, tops):
        return sum(_recall(rows[i]["gold_documents"], tops[i]) for i in indices) / len(indices)

    cv = []
    weighted = 0.0
    total_n = 0
    for holdout in range(5):
        train_idx = [i for b in range(5) if b != holdout for i in buckets[b]]
        test_idx = buckets[holdout]

        def train_score(cfg):
            key = (cfg["dense"], cfg["bge"], cfg["dense_weight"], cfg["rrf_k"])
            return avg_recall(train_idx, tops_cache[key])

        chosen = max(
            grid,
            key=lambda c: (
                train_score(c),
                c["bge_weight"],  # conservative tie-break
                -c["rrf_k"],
            ),
        )
        key = (chosen["dense"], chosen["bge"], chosen["dense_weight"], chosen["rrf_k"])
        test_rec = avg_recall(test_idx, tops_cache[key])
        cv.append(
            {
                "holdout": holdout,
                "query_count": len(test_idx),
                "selected": {
                    k: chosen[k]
                    for k in ("dense", "bge", "dense_weight", "bge_weight", "rrf_k")
                },
                "recall": test_rec,
            }
        )
        weighted += test_rec * len(test_idx)
        total_n += len(test_idx)

    report = {
        "schema_version": "p13-recovery-v3-chunkagg-fold0-v1",
        "status": "COMPLETE",
        "note": "No model training. Original BGE scores raw top200 chunks.",
        "historical_precommitted_configs": historical,
        "best_source_configs": source_results[:12],
        "same_fold_best_grid": best,
        "inner_cv": {
            "folds": cv,
            "weighted_recall": weighted / total_n,
        },
        "top_20_grid": grid[:20],
        "cache": str(CACHE),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={REPORT}", flush=True)
    return report


if __name__ == "__main__":
    print("Enqueuing P13 Recovery V3 chunk aggregation fold0...", flush=True)
    result = run.remote()
    print("REMOTE RESULT:", result, flush=True)
