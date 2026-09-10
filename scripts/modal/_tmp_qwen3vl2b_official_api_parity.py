"""Tiny, independent official-API parity diagnostic for Qwen3-VL-Reranker-2B.

This file is deliberately not a Modal entrypoint and is not run by this audit.
It is restricted to the frozen 24-qdoc / 72-chunk manifest.  The canonical
path uses the canonical runner only; the official path uses the model-provided
SentenceTransformers CrossEncoder pipeline and never calls a canonical scoring
or formatting helper.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import modal


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path.cwd()
MANIFEST = ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl"
CHUNKS = ROOT / "data/processed_v3/chunks"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CANONICAL_PATH = ROOT / "scripts/modal/task1_b2a_qwen3vl2b.py"
CANONICAL_REMOTE_PATH = Path("/root/task1_b2a_qwen3vl2b.py")
MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EXPECTED_QDOCS = 24
EXPECTED_CHUNKS = 72
INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the Document "
    "contains legal provisions relevant to answering the Query."
)
QWEN_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
MOUNT = Path("/workspace/p13")
RUNTIME = MOUNT / "runtime"
REMOTE_MANIFEST = "/root/qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl"
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0", "torchvision==0.23.0", "transformers==5.0.0", "sentence-transformers==5.4.0",
    "accelerate>=1.1,<2.0", "qwen-vl-utils>=0.0.14", "huggingface-hub>=0.30", "safetensors>=0.5,<1.0",
).add_local_file(
    "scripts/modal/task1_b2a_qwen3vl2b.py", "/root/task1_b2a_qwen3vl2b.py"
).add_local_file(
    "reports/task1/workflow_b/tv2/b2a/manifests/qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl",
    str(REMOTE_MANIFEST),
)
volume = modal.Volume.from_name("udsc-p13", create_if_missing=False)
app = modal.App("task1-b2a-qwen3vl2b-official-api-parity")


def require_runtime() -> tuple[Any, Any]:
    import sentence_transformers
    import torch
    import transformers

    if sentence_transformers.__version__ != "5.4.0":
        raise RuntimeError("requires the model-declared sentence-transformers==5.4.0")
    if int(transformers.__version__.split(".", 1)[0]) < 5:
        raise RuntimeError("requires transformers>=5 for sentence_bert_config any-to-any")
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError("requires the official model-card runtime torch==2.8.0")
    if not torch.cuda.is_available():
        raise RuntimeError("this future numerical parity diagnostic requires CUDA")
    return torch, transformers


def load_canonical() -> Any:
    source = CANONICAL_PATH if CANONICAL_PATH.is_file() else CANONICAL_REMOTE_PATH
    spec = importlib.util.spec_from_file_location("b2a_canonical", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical runner: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.TRAIN = TRAIN
    module.CHUNKS = CHUNKS
    return module


def configure_remote_paths() -> None:
    """Point the immutable diagnostic at the mounted canonical data only."""
    global ROOT, MANIFEST, CHUNKS, TRAIN
    ROOT = RUNTIME
    MANIFEST = Path(REMOTE_MANIFEST)
    CHUNKS = ROOT / "data/processed_v3/chunks"
    TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"


def load_work() -> tuple[list[dict[str, Any]], set[str]]:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_QDOCS:
        raise RuntimeError(f"manifest q-doc count is {len(rows)}, expected {EXPECTED_QDOCS}")
    seen_pairs: set[tuple[str, str]] = set()
    work: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for row in rows:
        query_id, doc_id = str(row["query_id"]), str(row["doc_id"])
        if (query_id, doc_id) in seen_pairs or len(row["selected_chunk_ids"]) != 3:
            raise RuntimeError("frozen manifest is not 24 unique q-docs with exactly three chunks each")
        seen_pairs.add((query_id, doc_id)); query_ids.add(query_id)
        raw = {entry["chunk_id"]: entry["text"] for entry in (json.loads(x) for x in (CHUNKS / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines())}
        for chunk_id in row["selected_chunk_ids"]:
            text = raw.get(chunk_id)
            if not text:
                raise RuntimeError(f"missing frozen chunk {chunk_id}")
            work.append({"query_id": query_id, "doc_id": doc_id, "chunk_id": chunk_id, "text": text})
    if len(work) != EXPECTED_CHUNKS:
        raise RuntimeError(f"frozen chunk count is {len(work)}, expected {EXPECTED_CHUNKS}")
    return work, query_ids


def official_messages(query: str, document: str) -> list[dict[str, Any]]:
    """Exact v5.4 InputFormatter pair-to-message result plus prompt injection."""
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "query", "content": [{"type": "text", "text": query}]},
        {"role": "document", "content": [{"type": "text", "text": document}]},
    ]


def official_render(processor: Any, query: str, document: str) -> str:
    return processor.apply_chat_template(
        official_messages(query, document), chat_template="reranker", tokenize=False, add_generation_prompt=True
    )


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values); start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]: end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]: result[index] = rank
        start = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    a, b = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - a) * (y - b) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - a) ** 2 for x in left) * sum((y - b) ** 2 for y in right))
    return numerator / denominator if denominator else float("nan")


def official_preflight() -> dict[str, Any]:
    """Import/config-only proof; it never instantiates a 2B model or runs forward."""
    torch, transformers = require_runtime()
    import sentence_transformers
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.modules.logit_score import LogitScore

    canonical = load_canonical()
    work, query_ids = load_work()
    questions = canonical.load_questions(query_ids)
    example = next(item for item in work if item["query_id"] == "13844")
    if not questions[example["query_id"]].strip() or not example["text"].strip():
        raise RuntimeError("frozen query/chunk resolution failed")
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir="/tmp/hf-parity-preflight", local_files_only=False,
        allow_patterns=["modules.json", "1_LogitScore/config.json", "additional_chat_templates/reranker.jinja"],
    ))
    modules = json.loads((snapshot / "modules.json").read_text(encoding="utf-8"))
    logit = json.loads((snapshot / "1_LogitScore/config.json").read_text(encoding="utf-8"))
    template = snapshot / "additional_chat_templates/reranker.jinja"
    if not template.is_file() or (logit.get("true_token_id"), logit.get("false_token_id")) != (9693, 2152):
        raise RuntimeError("pinned model-provided reranker module contract is incomplete")
    return {
        "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
        "torch": torch.__version__, "cross_encoder_class": CrossEncoder.__module__ + ".CrossEncoder",
        "logit_score_class": LogitScore.__module__ + ".LogitScore", "modules": modules,
        "revision": REVISION, "example": {key: example[key] for key in ("query_id", "doc_id", "chunk_id")},
    }


def run_official_smoke() -> dict[str, Any]:
    """Exactly one official CrossEncoder score; no canonical model/scorer is called."""
    torch, _ = require_runtime()
    canonical = load_canonical()
    work, query_ids = load_work()
    questions = canonical.load_questions(query_ids)
    item = next(item for item in work if item["query_id"] == "13844")
    snapshot, revision, _, _, _ = canonical.download_locked_snapshot()
    if revision != REVISION:
        raise RuntimeError("pinned revision mismatch")
    from sentence_transformers import CrossEncoder
    print("OFFICIAL_SMOKE_START", flush=True)
    print(f"query_id={item['query_id']} doc_id={item['doc_id']} chunk_id={item['chunk_id']}", flush=True)
    official = CrossEncoder(str(snapshot), local_files_only=True, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    raw = float(official.predict([(questions[item["query_id"]], item["text"])], prompt=INSTRUCTION, batch_size=1, activation_fn=torch.nn.Identity())[0])
    probability = float(torch.sigmoid(torch.tensor(raw)).item())
    print(f"official_raw_score={raw}", flush=True)
    print(f"official_sigmoid_score={probability}", flush=True)
    print("OFFICIAL_SMOKE_PASS", flush=True)
    return {"query_id": item["query_id"], "doc_id": item["doc_id"], "chunk_id": item["chunk_id"], "official_raw_score": raw, "official_probability": probability}


def run_diagnostic() -> dict[str, Any]:
    torch, transformers = require_runtime()
    canonical = load_canonical()
    work, query_ids = load_work()
    questions = canonical.load_questions(query_ids)  # reads question text only; never reads answer labels.
    if set(questions) != query_ids or any(not questions[key].strip() for key in query_ids):
        raise RuntimeError("canonical question-text guard failed")

    snapshot, revision, _, yes_id, no_id = canonical.download_locked_snapshot()
    if revision != REVISION or (yes_id, no_id) != (9693, 2152):
        raise RuntimeError("canonical locked model identity differs from audited official contract")
    canonical_processor, canonical_model, *_ = canonical.load_model(torch, transformers, snapshot, yes_id, no_id)

    from sentence_transformers import CrossEncoder
    # This is the independent author-supported CrossEncoder route.  It receives
    # the same immutable snapshot directory as the canonical path, never a
    # branch name or an unpinned latest model.
    official = CrossEncoder(str(snapshot), local_files_only=True, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    if official[1].true_token_id != 9693 or official[1].false_token_id != 2152:
        raise RuntimeError("official model-provided LogitScore IDs are not 9693/2152")

    canonical_scores: list[float] = []
    official_raw: list[float] = []
    official_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    for item in work:  # batch 1 is intentional: exact B2A operational condition.
        query, document = questions[item["query_id"]], item["text"]
        canonical_score = canonical.score_batch(canonical_processor, canonical_model, torch, [query], [document])[0][0]
        raw = float(official.predict([(query, document)], prompt=INSTRUCTION, batch_size=1, activation_fn=torch.nn.Identity())[0])
        probability = float(torch.sigmoid(torch.tensor(raw)).item())
        rendered_canonical = canonical_processor.apply_chat_template(canonical.format_pair(query, document), tokenize=False, add_generation_prompt=True)
        rendered_official = official_render(official.processor, query, document)
        if rendered_canonical != rendered_official:
            raise RuntimeError(f"rendered prompt mismatch for {item['query_id']}/{item['doc_id']}/{item['chunk_id']}")
        canonical_scores.append(canonical_score); official_raw.append(raw); official_probabilities.append(probability)
        rows.append({
            "stage": item.get("stage"), "query_id": item["query_id"], "doc_id": item["doc_id"], "chunk_id": item["chunk_id"],
            "canonical_probability": canonical_score, "official_raw_logit_difference": raw, "official_probability": probability,
        })

    deltas = [abs(a - b) for a, b in zip(canonical_scores, official_probabilities)]
    qdoc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows: qdoc.setdefault((row["query_id"], row["doc_id"]), []).append(row)
    top_agreement = sum(
        max(group, key=lambda row: row["canonical_probability"])["chunk_id"] == max(group, key=lambda row: row["official_probability"])["chunk_id"]
        for group in qdoc.values()
    )
    canonical_doc = [max(group, key=lambda row: row["canonical_probability"])["canonical_probability"] for group in qdoc.values()]
    official_doc = [max(group, key=lambda row: row["official_probability"])["official_probability"] for group in qdoc.values()]
    doc_deltas = [abs(a - b) for a, b in zip(canonical_doc, official_doc)]
    report = {
        "joined_chunks": len(rows), "qdocs": len(qdoc), "max_abs_score_delta": max(deltas),
        "mean_abs_score_delta": statistics.fmean(deltas), "median_abs_score_delta": statistics.median(deltas),
        "p95_abs_score_delta": sorted(deltas)[math.ceil(.95 * len(deltas)) - 1],
        "pearson": pearson(canonical_scores, official_probabilities),
        "spearman": pearson(ranks(canonical_scores), ranks(official_probabilities)),
        "document_max_score_delta": max(doc_deltas), "document_mean_score_delta": statistics.fmean(doc_deltas),
        "document_max_score_pearson": pearson(canonical_doc, official_doc),
        "document_max_score_spearman": pearson(ranks(canonical_doc), ranks(official_doc)),
        "top_ranked_chunk_agreement": f"{top_agreement}/{len(qdoc)}", "rows": rows,
    }
    return report


@app.function(
    image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=2 * 60 * 60, cpu=8, memory=32768,
)
def parity_remote() -> dict[str, Any]:
    configure_remote_paths()
    return run_diagnostic()


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=20 * 60, cpu=2, memory=4096)
def preflight_remote() -> dict[str, Any]:
    configure_remote_paths()
    return official_preflight()


@app.function(image=image, gpu="A10", volumes={str(MOUNT): volume}, timeout=60 * 60, cpu=4, memory=24576)
def official_smoke_remote() -> dict[str, Any]:
    configure_remote_paths()
    return run_official_smoke()


@app.local_entrypoint()
def main(mode: str = "preflight") -> None:
    if mode == "preflight":
        result = preflight_remote.remote()
    elif mode == "official_smoke":
        result = official_smoke_remote.remote()
    elif mode == "parity":
        result = parity_remote.remote()
    else:
        raise ValueError("mode must be preflight, official_smoke, or parity")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
