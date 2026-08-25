"""Selective BGE scoring rank 201--500 for adaptive-K500 OOF queries only.

This is a Beam GPU job.  It never scores rank <=200 and never writes an
existing cache.  Per-query JSON checkpoints make container retries safe; final
JSONL outputs are rebuilt atomically only after every selected query is done.
"""
from __future__ import annotations

from beam import Image, Volume, function

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
RAW_K500 = RUNTIME / "artifacts/task1/raw_k500.jsonl"
DECISIONS = RUNTIME / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_decisions.jsonl"
K200_A = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl"
K200_B = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
OUT = RUNTIME / "artifacts/task1/recovery_096/selective_bge_k500_v1"
MODEL_ARCHIVES = (VOLUME_ROOT / "udsc_reranker.zip", VOLUME_ROOT / "udsc_p13_reranker.zip")

image = Image(python_version="python3.11", python_packages=["torch", "transformers==5.0.0", "accelerate>=1.1,<2"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def is_file_retry(path: Path, attempts: int = 8) -> bool:
    """Shared Beam volumes can transiently return EAGAIN for stat()."""
    for attempt in range(attempts):
        try:
            return path.is_file()
        except BlockingIOError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    return False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A retry or an accidentally overlapping Beam invocation may target the
    # same query.  Use a unique temporary name so they never contend on the
    # same ``*.tmp`` inode (the shared volume can otherwise return EAGAIN).
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        payload = json.dumps(value, ensure_ascii=False) + "\n"
        for attempt in range(6):
            try:
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, path)
                return
            except BlockingIOError:
                if attempt == 5:
                    raise
                time.sleep(0.5 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_model() -> tuple[Path, Path]:
    archive = next((p for p in MODEL_ARCHIVES if p.is_file()), None)
    if archive is None:
        raise FileNotFoundError("reranker archive missing; checked: " + ", ".join(str(p) for p in MODEL_ARCHIVES))
    stage = Path("/tmp/selective_k500_model")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    local = Path("/tmp/selective_k500_reranker.zip")
    shutil.copy2(archive, local)
    subprocess.run(["/usr/bin/python3.11", "-m", "zipfile", "-e", str(local), str(stage)], check=True)
    model = stage / "models/reranker"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model staging failed: {model}")
    return model, archive


class Reranker:
    def __init__(self, model: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for selective BGE K500")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True).to("cuda").eval()

    def score(self, question: str, texts: list[str]) -> list[float]:
        out: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                encoded = self.tokenizer([question] * len(texts[start:start + 16]), texts[start:start + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
                with self.torch.cuda.amp.autocast(dtype=self.torch.float16):
                    logits = self.model(**encoded).logits
                vals = logits[:, 1] - logits[:, 0] if logits.ndim == 2 and logits.shape[-1] == 2 else logits.reshape(-1)
                out.extend(float(x) for x in vals.float().cpu().tolist())
        return out


def rank_docs(hits: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(hits, key=lambda h: (-float(h["bge_score"]), int(h["dense_rank"]), h["doc_id"], h["chunk_id"]))
    score: dict[str, float] = defaultdict(float)
    first: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)
    for rank, hit in enumerate(ordered, 1):
        doc = hit["doc_id"]
        first.setdefault(doc, rank)
        if counts[doc] >= 10:
            continue
        counts[doc] += 1
        score[doc] += 1.0 / (2 + rank)
    return sorted(score, key=lambda d: (-score[d], first[d], d))


@function(name="udsc-selective-bge-k500-v1", cpu=4, memory="32Gi", gpu="RTX4090", image=image, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run() -> None:
    for path in (TRAIN, RAW_K500, DECISIONS, K200_A, K200_B):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    decisions = {str(x["query_id"]): x for x in load_jsonl(DECISIONS)}
    selected = sorted(q for q, x in decisions.items() if bool(x["expand_to_k500"]))
    train = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    k200 = {str(x["query_id"]): x for x in load_jsonl(K200_A) + load_jsonl(K200_B)}
    if set(k200) != set(decisions):
        raise ValueError("K200 cache and adaptive decisions coverage mismatch")
    state = OUT / "checkpoints"
    state.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict[str, Any]] = {}
    chosen = set(selected)
    with RAW_K500.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["question_id"]) in chosen:
                raw[str(row["question_id"])] = row
    missing = sorted(chosen - set(raw))
    if missing:
        raise ValueError(f"raw K500 missing selected query IDs: {missing[:10]}")
    model, archive = stage_model()
    scorer = Reranker(model)
    started = time.monotonic()
    try:
        for n, qid in enumerate(selected, 1):
            checkpoint = state / f"{qid}.json"
            if is_file_retry(checkpoint):
                continue
            old_chunks = {str(h["chunk_id"]) for h in k200[qid]["hits"]}
            candidate: list[dict[str, Any]] = []
            for rank, hit in enumerate(raw[qid].get("hits", []), 1):
                if rank <= 200 or rank > 500 or not isinstance(hit, dict):
                    continue
                doc, chunk, text = str(hit.get("doc_id", "")).strip(), str(hit.get("chunk_id", "")).strip(), str(hit.get("text", "")).strip()
                if not doc or not chunk or not text or chunk in old_chunks:
                    continue
                candidate.append({"doc_id": doc, "chunk_id": chunk, "dense_rank": rank, "dense_score": float(hit.get("dense_score", hit.get("score", 0.0))), "text": text})
            scores = scorer.score(str(train[qid]["question"]), [x["text"] for x in candidate]) if candidate else []
            rows = [{k: v for k, v in x.items() if k != "text"} | {"bge_score": score} for x, score in zip(candidate, scores)]
            atomic_json(checkpoint, {"query_id": qid, "new_hits": rows})
            if n % 25 == 0:
                print(f"[CHECKPOINT] {n}/{len(selected)}", flush=True)
    finally:
        scorer.torch.cuda.synchronize()
        gpu_seconds = time.monotonic() - started
        del scorer.model
        scorer.torch.cuda.empty_cache()
    scored = [json.loads((state / f"{qid}.json").read_text(encoding="utf-8")) for qid in selected]
    if len(scored) != len(selected):
        raise RuntimeError("incomplete checkpoints; final outputs withheld")
    atomic_jsonl(OUT / "new_chunk_scores.jsonl", scored)
    combined = []
    for row in scored:
        qid = row["query_id"]
        hits = k200[qid]["hits"] + row["new_hits"]
        combined.append({"query_id": qid, "fold": decisions[qid]["fold"], "document_ranking": rank_docs(hits), "new_chunk_count": len(row["new_hits"])})
    atomic_jsonl(OUT / "combined_k500_document_rankings.jsonl", combined)
    report = {"schema_version": "selective-bge-k500-v1", "selected_query_count": len(selected), "new_chunks_scored": sum(len(x["new_hits"]) for x in scored), "average_new_chunks_per_selected_query": sum(len(x["new_hits"]) for x in scored) / len(selected), "gpu_seconds": gpu_seconds, "original_bge_revision": sha256(archive), "input_sha256": {"decisions": sha256(DECISIONS), "k200_fold0": sha256(K200_A), "k200_fold1to4": sha256(K200_B), "raw_k500": sha256(RAW_K500)}, "constraints": {"rank_le_200_rescored": False, "neural_training": False, "public_submission_created": False}}
    atomic_json(OUT / "cost_report.json", report)
    atomic_json(OUT / "manifest.json", report)


if __name__ == "__main__":
    print("Enqueuing selective BGE K500 V1 GPU job...", flush=True)
    print(run.remote())
