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
    name="udsc-p13-recovery-v3-confirm-f1to4",
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
    print("=== P13 RECOVERY V3 CONFIRM: LOCKED CONFIG, FOLDS 1-4 ===", flush=True)
    print("No training. No grid search on folds 1-4.", flush=True)

    for p in (TRAIN, FOLDS, MODEL_ZIP):
        print(f"[CHECK] {p} exists={p.exists()}", flush=True)
        if not p.exists():
            raise FileNotFoundError(p)
    _ensure_raw_k200()

    out_dir = RUNTIME / "artifacts/task1/diagnostics/p13_recovery_v3_confirm_f1to4"
    cache = out_dir / "f1to4_original_bge_chunk200_compact.jsonl"
    report_path = out_dir / "confirm_report.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_train()
    fold_payload = json.loads(FOLDS.read_text(encoding="utf-8-sig"))
    qid_to_fold: dict[str, int] = {}
    confirm_ids: list[str] = []
    for item in fold_payload.get("folds", []):
        fold = int(item.get("fold", -1))
        if fold not in (1, 2, 3, 4):
            continue
        for qid in item.get("validation_ids", []):
            qid = str(qid)
            qid_to_fold[qid] = fold
            confirm_ids.append(qid)
    confirm_ids = sorted(confirm_ids)
    if len(confirm_ids) != 5600:
        raise RuntimeError(f"expected 5600 confirmation IDs, got {len(confirm_ids)}")
    confirm_set = set(confirm_ids)

    rows: list[dict] = []
    if cache.is_file():
        try:
            with cache.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if [r.get("query_id") for r in rows] != confirm_ids:
                print("[CACHE] mismatch -> recompute", flush=True)
                rows = []
            else:
                print(f"[CACHE] reuse {len(rows)} scored queries", flush=True)
        except Exception:
            rows = []

    if not rows:
        raw_rows = {}
        with RAW_K200.open(encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                qid = str(raw.get("question_id", raw.get("id", "")))
                if qid in confirm_set:
                    raw_rows[qid] = raw
        missing = sorted(confirm_set - set(raw_rows))
        if missing:
            raise RuntimeError(f"raw_k200 missing confirmation IDs: {missing[:10]}")

        model_dir = _stage_model()
        reranker = BGEReranker(model_dir)
        try:
            from tqdm import tqdm
            for qid in tqdm(confirm_ids, desc="Confirm BGE chunk200 scoring", unit="q"):
                raw = raw_rows[qid]
                compact_hits = []
                texts = []
                for idx, hit in enumerate(raw.get("hits", [])[:200], 1):
                    if not isinstance(hit, dict):
                        continue
                    doc = str(hit.get("doc_id", hit.get("document_id", ""))).strip()
                    chunk = str(hit.get("chunk_id", hit.get("evidence_id", ""))).strip()
                    chunk_text = str(hit.get("text", hit.get("evidence", ""))).strip()
                    if not doc or not chunk_text:
                        continue
                    dense_raw = hit.get("dense_score", hit.get("score", 0.0))
                    compact_hits.append({
                        "doc_id": doc,
                        "chunk_id": chunk or f"{doc}:raw:{idx}",
                        "dense_rank": idx,
                        "dense_score": float(dense_raw),
                    })
                    texts.append(chunk_text)
                if not compact_hits:
                    raise RuntimeError(f"no usable raw hits at {qid}")
                scores = reranker.score(train[qid]["question"], texts)
                if len(scores) != len(compact_hits):
                    raise RuntimeError(f"score length mismatch at {qid}")
                for hit, score in zip(compact_hits, scores):
                    hit["bge_score"] = float(score)
                rows.append({
                    "query_id": qid,
                    "fold": qid_to_fold[qid],
                    "gold_documents": train[qid]["gold"],
                    "hits": compact_hits,
                })
        finally:
            reranker.close()

        tmp = cache.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\\n")
        os.replace(tmp, cache)
        print(f"[WRITE] {cache}", flush=True)

    # PRECOMMITTED BEFORE READING F1-F4 METRICS.
    locked_configs = [
        {
            "name": "fold0_best_locked",
            "dense": {"cap": 5, "k": 5},
            "bge": {"cap": 10, "k": 5},
            "dense_weight": 0.3,
            "bge_weight": 0.7,
            "rrf_k": 0,
        },
        {
            "name": "historical_robust_locked",
            "dense": {"cap": 3, "k": 10},
            "bge": {"cap": 5, "k": 10},
            "dense_weight": 0.3,
            "bge_weight": 0.7,
            "rrf_k": 0,
        },
        {
            "name": "bge_chunk_only_locked",
            "bge_only": {"cap": 10, "k": 5},
        },
    ]

    results = []
    for cfg in locked_configs:
        tops: list[list[str]] = []
        for row in rows:
            if "bge_only" in cfg:
                bc = cfg["bge_only"]
                top = _rank_aggregate(
                    row["hits"], source="bge", cap=bc["cap"], rank_k=bc["k"]
                )[:5]
            else:
                dc = cfg["dense"]
                bc = cfg["bge"]
                dr = _rank_aggregate(
                    row["hits"], source="dense", cap=dc["cap"], rank_k=dc["k"]
                )
                br = _rank_aggregate(
                    row["hits"], source="bge", cap=bc["cap"], rank_k=bc["k"]
                )
                top = _weighted_rrf(
                    [dr, br], [cfg["dense_weight"], cfg["bge_weight"]], cfg["rrf_k"]
                )[:5]
            tops.append(top)

        pooled = _metrics(rows, tops)
        per_fold = {}
        for fold in (1, 2, 3, 4):
            idx = [i for i, row in enumerate(rows) if int(row["fold"]) == fold]
            fold_rows = [rows[i] for i in idx]
            fold_tops = [tops[i] for i in idx]
            per_fold[str(fold)] = _metrics(fold_rows, fold_tops)
        results.append({
            "name": cfg["name"],
            "config": {k: v for k, v in cfg.items() if k != "name"},
            "pooled_folds1to4": pooled,
            "per_fold": per_fold,
        })

    report = {
        "schema_version": "p13-recovery-v3-confirm-f1to4-v1",
        "status": "COMPLETE",
        "note": (
            "Confirmatory evaluation only. Configurations were locked before folds 1-4 "
            "metrics were read. No training and no grid search on folds 1-4."
        ),
        "query_count": len(rows),
        "results": results,
        "cache": str(cache),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={report_path}", flush=True)
    return report


if __name__ == "__main__":
    print("Enqueuing P13 V3 confirmation on folds 1-4...", flush=True)
    result = run.remote()
    print("REMOTE RESULT:", result, flush=True)
