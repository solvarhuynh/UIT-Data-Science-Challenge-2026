"""Checkpointed GPU diagnostic for the hash-locked BM25 B4 worklist.

The function stages exactly one small worklist from deployed source into the
``udsc-p13`` volume, then scores only its S2-selected raw chunks.  It never
trains, submits, overwrites source caches, or evaluates top-5 quality.
"""
from __future__ import annotations

from beam import Image, Volume, function

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
PAYLOADS = RUNTIME / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
WORKLIST = RUNTIME / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
K200_A = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl"
K200_B = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
K500 = RUNTIME / "artifacts/task1/recovery_096/selective_bge_k500_v1/new_chunk_scores.jsonl"
OUT = RUNTIME / "artifacts/task1/recovery_096/bm25_grounded_bge_b4_v1"
MODEL_ARCHIVES = (VOLUME_ROOT / "udsc_p13_reranker.zip", VOLUME_ROOT / "udsc_reranker.zip")
EXPECTED_WORKLIST_SHA256 = "34d3ba1f0f9e202222c19a921f10725faace86af8869809d3c947f66b035ba32"
EXPECTED_CHUNKS = 161_763
SELECTOR_NAME = "S2_bm25_within_document_v1"
INFERENCE_CONTRACT = "pair(question,raw_chunk_text);truncation=True;max_length=512;batch=16;fp16;transformers=5.0.0"

image = Image(python_version="python3.11", python_packages=["torch", "transformers==5.0.0", "accelerate>=1.1,<2", "pyvi"])

_PROTECTED = re.compile(r"(?:\b(?:(?:Điều|Khoản)\s+\d+[a-zđ]?|Điểm\s+[a-zđ])\b|\b\d{1,4}/\d{4}/[a-zđ]{1,12}\d{0,4}\b)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_payloads(path: Path):
    """Stream payloads using the parser proven by the full payload audit.

    Refill after every raw_decode boundary: JSON punctuation (``:``/``,``)
    may begin in the next 1 MiB block.
    """
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""; eof = False
        def refill() -> None:
            nonlocal buffer, eof
            block = handle.read(1 << 20)
            if block: buffer += block
            else: eof = True
        while '"payloads"' not in buffer and not eof: refill()
        marker = buffer.find('"payloads"')
        if marker < 0: raise ValueError("payloads.json has no top-level payloads field")
        buffer = buffer[marker + len('"payloads"'):]
        while "{" not in buffer and not eof: refill()
        opening = buffer.find("{")
        if opening < 0: raise ValueError("payloads field is not an object")
        buffer = buffer[opening + 1:]
        while True:
            while True:
                buffer = buffer.lstrip()
                if buffer or eof:
                    break
                refill()
            if not buffer: raise ValueError("unexpected EOF in payloads object")
            if buffer[0] == "}": return
            while True:
                try: _, pos = decoder.raw_decode(buffer); break
                except json.JSONDecodeError:
                    if eof: raise
                    refill()
            buffer = buffer[pos:].lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if not buffer.startswith(":"): raise ValueError("invalid payload separator")
            buffer = buffer[1:].lstrip()
            while True:
                try: value, pos = decoder.raw_decode(buffer); break
                except json.JSONDecodeError:
                    if eof: raise
                    refill()
            if not isinstance(value, dict): raise ValueError("payload row is not an object")
            yield value
            buffer = buffer[pos:].lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if buffer.startswith(","): buffer = buffer[1:]


def load_payload_docs(path: Path, wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_payloads(path):
        doc = str(payload.get("doc_id", ""))
        if doc in wanted:
            result[doc].append(payload)
    return result


def stage_worklist() -> None:
    """Require the immutable worklist to be uploaded to the mounted Volume."""
    if not WORKLIST.is_file():
        raise FileNotFoundError(
            "B4 worklist missing on Volume; upload "
            "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/"
            "B4_bm25_rank_le_20.json to udsc-p13 before launching"
        )


def tokenize_vi(text: str) -> list[str]:
    """Exact S2 tokenizer logic used by the reconciled local audit."""
    from pyvi import ViTokenizer
    if text is None or not text.strip():
        return []
    output: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        output.extend(token.lower() for token in ViTokenizer.tokenize(text[cursor:match.start()]).split() if token)
        output.append(match.group(0))
        cursor = match.end()
    output.extend(token.lower() for token in ViTokenizer.tokenize(text[cursor:]).split() if token)
    return output


def bm25_scores(corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = .75) -> list[float]:
    lengths = [len(document) for document in corpus]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if average <= 0:
        return [0.0] * len(corpus)
    frequencies = Counter(token for document in corpus for token in set(document))
    idf = {token: math.log(1 + (len(corpus) - count + .5) / (count + .5)) for token, count in frequencies.items()}
    result = []
    for document, length in zip(corpus, lengths):
        tf = Counter(document); score = 0.0; norm = k1 * (1 - b + b * length / average)
        for token in query:
            count = tf.get(token, 0)
            if count:
                score += idf.get(token, 0.0) * count * (k1 + 1) / (count + norm)
        result.append(score)
    return result


def select_s2(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = tokenize_vi(question)
    corpus = [tokenize_vi(str(row.get("text", ""))) for row in chunks]
    scores = bm25_scores(corpus, query) if query else [0.0] * len(chunks)
    ordered = sorted(range(len(chunks)), key=lambda i: (-scores[i], str(chunks[i].get("chunk_id", ""))))
    return [{"chunk_id": str(chunks[i]["chunk_id"]), "text": str(chunks[i]["text"]), "bm25_within_doc_score": float(scores[i]), "bm25_within_doc_rank": rank}
            for rank, i in enumerate(ordered[:3], 1) if str(chunks[i].get("text", "")).strip()]


def stage_model() -> tuple[Path, Path]:
    archive = next((path for path in MODEL_ARCHIVES if path.is_file()), None)
    if archive is None:
        raise FileNotFoundError("reranker archive missing")
    stage = Path("/tmp/bm25_b4_model")
    shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
    local_archive = Path("/tmp/bm25_b4_reranker.zip")
    shutil.copy2(archive, local_archive)
    subprocess.run(["/usr/bin/python3.11", "-m", "zipfile", "-e", str(local_archive), str(stage)], check=True)
    model = stage / "models/reranker"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model staging failed: {model}")
    return model, archive


class Reranker:
    def __init__(self, model: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True).to("cuda").eval()

    def score(self, question: str, texts: list[str]) -> list[float]:
        output: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start:start + 16]
                encoded = self.tokenizer([question] * len(batch), batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                encoded = {name: value.to("cuda") for name, value in encoded.items()}
                with self.torch.cuda.amp.autocast(dtype=self.torch.float16):
                    logits = self.model(**encoded).logits
                values = logits[:, 1] - logits[:, 0] if logits.ndim == 2 and logits.shape[-1] == 2 else logits.reshape(-1)
                output.extend(float(value) for value in values.float().cpu().tolist())
        return output


def _run_impl() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    required = (TRAIN, PAYLOADS, K200_A, K200_B, K500)
    missing = [str(path) for path in required if not path.exists()]
    if not any(path.is_file() for path in MODEL_ARCHIVES):
        missing.append("one of: " + ", ".join(str(path) for path in MODEL_ARCHIVES))
    try:
        stage_worklist()
    except Exception as exc:
        atomic_json(OUT / "failure_report.json", {"status": "PRECHECK_FAILED", "error": str(exc), "missing_inputs": missing})
        raise
    if missing:
        atomic_json(OUT / "failure_report.json", {"status": "PRECHECK_FAILED", "error": "required input missing", "missing_inputs": missing})
        raise FileNotFoundError("required input missing: " + ", ".join(missing))
    payload = json.loads(WORKLIST.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    canonical = hashlib.sha256("".join(f"{r['query_id']}\t{r['doc_id']}\n" for r in sorted(rows, key=lambda r: (str(r['query_id']), str(r['doc_id'])))).encode()).hexdigest()
    if canonical != EXPECTED_WORKLIST_SHA256 or len(rows) != 53_924:
        raise ValueError("B4 worklist SHA/count mismatch; refusing to score a changed worklist")
    known_chunks = defaultdict(set)
    for record in load_jsonl(K200_A) + load_jsonl(K200_B):
        known_chunks[str(record["query_id"])].update(str(hit["chunk_id"]) for hit in record["hits"])
    for record in load_jsonl(K500):
        known_chunks[str(record["query_id"])].update(str(hit["chunk_id"]) for hit in record["new_hits"])
    train = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qid, doc = str(row["query_id"]), str(row["doc_id"])
        if row.get("bm25_rank") is None or int(row["bm25_rank"]) > 20:
            raise ValueError(f"non-B4 row: {qid}/{doc}")
        if qid not in train:
            raise KeyError(f"worklist query absent from train: {qid}")
        by_query[qid].append(row)
    payload_docs = load_payload_docs(PAYLOADS, {str(row["doc_id"]) for row in rows})
    if len(payload_docs) != len({str(row["doc_id"]) for row in rows}):
        raise ValueError("payload backend does not cover every B4 candidate document")
    state = OUT / "checkpoints"; state.mkdir(parents=True, exist_ok=True)
    expected_qids = sorted(by_query)
    # Preflight evidence is deterministic and validates the expected 161,763
    # S2 pairs before a single BGE forward pass is made.
    evidence_by_query: dict[str, list[dict[str, Any]]] = {}
    for n, qid in enumerate(expected_qids, 1):
        evidence: list[dict[str, Any]] = []
        for row in sorted(by_query[qid], key=lambda r: (int(r["bm25_rank"]), int(r["union_rank"]), str(r["doc_id"]))):
            doc = str(row["doc_id"]); source = PAYLOADS
            chunks = payload_docs.get(doc, [])
            if not chunks:
                raise FileNotFoundError(f"payload chunks missing for doc: {doc}")
            selected = select_s2(str(train[qid]["question"]), chunks)
            if not selected:
                raise ValueError(f"S2 selected no raw chunks: {qid}/{doc}")
            for item in selected:
                if item["chunk_id"] in known_chunks[qid]:
                    raise ValueError(f"attempted K200/K500 re-score: {qid}/{item['chunk_id']}")
                evidence.append({"doc_id": doc, "union_rank": int(row["union_rank"]), "bm25_rank": int(row["bm25_rank"]), "source_support": int(row["source_support"]), "payload_source": str(source.relative_to(RUNTIME)), **item})
        if len({item["chunk_id"] for item in evidence}) != len(evidence):
            raise ValueError(f"duplicate (query,chunk) in S2 evidence: {qid}")
        evidence_by_query[qid] = evidence
        if n % 500 == 0: print(f"[PREFLIGHT] {n}/{len(expected_qids)}", flush=True)
    if sum(len(items) for items in evidence_by_query.values()) != EXPECTED_CHUNKS:
        raise ValueError("S2 evidence count does not reconcile to locked B4 expected chunks")
    model, archive = stage_model(); scorer = Reranker(model); started = time.monotonic()
    try:
        for n, qid in enumerate(expected_qids, 1):
            checkpoint = state / f"{qid}.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("worklist_sha256") != EXPECTED_WORKLIST_SHA256 or saved.get("inference_contract") != INFERENCE_CONTRACT:
                    raise ValueError(f"incompatible checkpoint: {checkpoint}")
                continue
            evidence = evidence_by_query[qid]
            scores = scorer.score(str(train[qid]["question"]), [item["text"] for item in evidence])
            hits = [{key: value for key, value in item.items() if key != "text"} | {"bge_score": score, "selector_name": SELECTOR_NAME, "model_dedup_key": f"{qid}:{item['chunk_id']}:{EXPECTED_WORKLIST_SHA256}:{INFERENCE_CONTRACT}"}
                    for item, score in zip(evidence, scores)]
            atomic_json(checkpoint, {"query_id": qid, "worklist_sha256": EXPECTED_WORKLIST_SHA256, "inference_contract": INFERENCE_CONTRACT, "new_hits": hits})
            if n % 100 == 0: print(f"[CHECKPOINT] {n}/{len(expected_qids)}", flush=True)
    finally:
        scorer.torch.cuda.synchronize(); gpu_seconds = time.monotonic() - started
        del scorer.model; scorer.torch.cuda.empty_cache()
    completed = [json.loads((state / f"{qid}.json").read_text(encoding="utf-8")) for qid in expected_qids]
    if len(completed) != len(expected_qids) or sum(len(row["new_hits"]) for row in completed) != EXPECTED_CHUNKS:
        raise RuntimeError("incomplete checkpoints; final output withheld")
    final_rows = [{"query_id": row["query_id"], "new_hits": row["new_hits"]} for row in completed]
    atomic_jsonl(OUT / "new_chunk_scores.jsonl", final_rows)
    result = {"schema_version": "bm25-grounded-bge-b4-v1", "status": "COMPLETE_NO_TOP5_EVALUATION", "candidate_doc_occurrences": len(rows), "unique_docs": len({str(row['doc_id']) for row in rows}), "query_count": len(expected_qids), "chunks_requested": EXPECTED_CHUNKS, "chunks_scored": EXPECTED_CHUNKS, "chunks_reused": 0, "missing_error_count": 0, "gpu_seconds": gpu_seconds, "model_revision_sha256": sha256(archive), "inference_settings": INFERENCE_CONTRACT, "selector": SELECTOR_NAME, "worklist_sha256": EXPECTED_WORKLIST_SHA256, "payload_backend": {"path": str(PAYLOADS), "payload_sha256": sha256(PAYLOADS), "payload_manifest": str(PAYLOADS.with_name("manifest.json"))}, "input_sha256": {"worklist": sha256(WORKLIST), "train": sha256(TRAIN), "k200_fold0": sha256(K200_A), "k200_fold1to4": sha256(K200_B), "k500_v1": sha256(K500)}, "new_chunk_scores_sha256": sha256(OUT / "new_chunk_scores.jsonl"), "checkpoint_completeness": {"completed_queries": len(completed), "expected_queries": len(expected_qids), "complete": True}, "constraints": {"no_training": True, "no_submission": True, "no_k200_k500_rescore": True, "no_top5_evaluation": True}}
    atomic_json(OUT / "report.json", result)
    return {"status": result["status"], "chunks_scored": result["chunks_scored"], "output": str(OUT)}


@function(name="udsc-bm25-grounded-bge-b4-v1", cpu=4, memory="32Gi", gpu="RTX4090", image=image, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run() -> None:
    print("[ENTRY] bm25_grounded_bge_b4_v1 run() reached", flush=True)
    try:
        result = _run_impl()
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print("[FAILURE] " + repr(exc), flush=True)
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            atomic_json(
                OUT / "failure_report.json",
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                },
            )
        except Exception as write_exc:
            print("[FAILURE_REPORT_WRITE_FAILED] " + repr(write_exc), flush=True)
        raise


def smoke() -> dict[str, Any]:
    """Local no-GPU assertion for the immutable candidate contract."""
    local = Path(__file__).resolve().parents[2] / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
    payload = json.loads(local.read_text(encoding="utf-8")); rows = payload["rows"]
    canonical = hashlib.sha256("".join(f"{r['query_id']}\t{r['doc_id']}\n" for r in sorted(rows, key=lambda r: (str(r['query_id']), str(r['doc_id'])))).encode()).hexdigest()
    if canonical != EXPECTED_WORKLIST_SHA256 or len(rows) != 53_924:
        raise ValueError("local B4 worklist lock mismatch")
    return {"smoke": "PASS_NO_GPU", "worklist_sha256": canonical, "doc_occurrences": len(rows), "expected_chunks": EXPECTED_CHUNKS, "volume": "udsc-p13", "mount": str(VOLUME_ROOT)}


if __name__ == "__main__":
    import sys
    print(json.dumps(smoke(), ensure_ascii=False) if "--smoke" in sys.argv else run.remote())
