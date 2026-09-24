"""Score a frozen PV1 q-doc worklist with a SAME-BGE FT V2 checkpoint.

The script is deliberately local-GPU-only and opt-in. It loads no model when
imported, uses the selected chunk IDs already frozen in the worklist, and emits
the current score schema needed by the PV1 offline evaluator. It never rebuilds
retrieval, runs a selector, reads Private answers, or changes corpus content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "BAAI/bge-reranker-v2-m3"
BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
SELECTOR = "true_s2_bm25_within_document_v2"
MAX_LENGTH = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("query_id", "")), str(row.get("document_id", "")))
            if not all(key) or key in seen:
                raise ValueError(f"invalid or duplicate worklist identity at {line_no}")
            ids = [str(value) for value in row.get("selected_chunk_ids", [])]
            if not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
                raise ValueError(f"invalid frozen selected IDs at {key}")
            if row.get("selector") != SELECTOR or row.get("aggregation") != "MAX":
                raise ValueError(f"inference contract mismatch at {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("empty worklist")
    return rows


def load_questions(path: Path, ids: set[str]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for qid in ids:
        item = payload.get(qid)
        question = str(item.get("question", "")) if isinstance(item, dict) else ""
        if not question.strip():
            raise ValueError(f"canonical question missing: {qid}")
        result[qid] = question
    return result


def resolve_pairs(rows: list[dict[str, Any]], questions: dict[str, str], chunks_root: Path) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    required: dict[str, set[str]] = {}
    for row in rows:
        doc = str(row["document_id"])
        required.setdefault(doc, set()).update(map(str, row["selected_chunk_ids"]))
    text_by_id: dict[str, str] = {}
    for doc, ids in required.items():
        path = chunks_root / f"{doc}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"PV1 chunks missing: {path}")
        found: dict[str, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                chunk_id = str(item.get("chunk_id", ""))
                if chunk_id in ids:
                    text = str(item.get("raw_chunk_text", item.get("text", ""))).strip()
                    if not chunk_id.startswith(doc + "_") or not text:
                        raise ValueError(f"invalid selected chunk: {doc}/{chunk_id}")
                    found[chunk_id] = text
        if set(found) != ids:
            raise ValueError(f"frozen chunk resolution mismatch: {doc}")
        text_by_id.update(found)
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        begin = len(pairs)
        question = questions[str(row["query_id"])]
        pairs.extend((question, text_by_id[str(chunk_id)]) for chunk_id in row["selected_chunk_ids"])
        spans.append((begin, len(pairs)))
    return pairs, spans


def validate_model(path: Path) -> dict[str, Any]:
    if "qwen" in str(path).casefold():
        raise RuntimeError("UNEXPECTED_QWEN_RUNTIME_PATH")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = int(config.get("num_labels", len(config.get("id2label", {})) or -1))
    if config.get("model_type") != "xlm-roberta" or labels != 1:
        raise RuntimeError("UNEXPECTED_NON_BGE_MODEL_CONFIG")
    weight = path / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(f"model weights missing: {weight}")
    return {"weight_sha256": sha256(weight), "config_sha256": sha256(config_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--chunks-root", type=Path, default=ROOT / "data/processed_pv1/chunks")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.device != "cuda" or args.batch_size <= 0:
        raise ValueError("this production scorer requires --device cuda and a positive batch size")
    rows = load_rows(args.worklist)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:args.limit]
    references = {(str(row["query_id"]), str(row["document_id"])): row for row in load_rows(args.reference_scores)}
    if len(references) != sum(1 for _ in references):
        raise ValueError("duplicate reference score identity")
    for row in rows:
        ref = references.get((str(row["query_id"]), str(row["document_id"])))
        if ref is None or list(ref.get("selected_chunk_ids", [])) != list(row["selected_chunk_ids"]):
            raise ValueError("reference score provenance/selected-chunk mismatch")
    model_meta = validate_model(args.model)
    questions = load_questions(args.questions, {str(row["query_id"]) for row in rows})
    pairs, spans = resolve_pairs(rows, questions, args.chunks_root)
    import torch
    from sentence_transformers import CrossEncoder
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this scorer")
    model = CrossEncoder(str(args.model), device="cuda", max_length=MAX_LENGTH, local_files_only=True)
    model.model.eval()
    with torch.inference_mode():
        values = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    scores = [float(value) for value in values]
    if len(scores) != len(pairs) or any(not math.isfinite(value) for value in scores):
        raise RuntimeError("non-finite or misaligned BGE scores")
    output: list[dict[str, Any]] = []
    for row, (begin, end) in zip(rows, spans):
        key = (str(row["query_id"]), str(row["document_id"]))
        reference = dict(references[key])
        chunk_scores = scores[begin:end]
        reference.update({
            "query_id": key[0], "document_id": key[1], "bge_ft_score": max(chunk_scores),
            "ft_chunk_scores": chunk_scores, "selected_chunk_ids": list(row["selected_chunk_ids"]),
            "ft_artifact_id": f"sha256:{model_meta['weight_sha256']}", "ft_weight_sha256": model_meta["weight_sha256"],
            "ft_config_sha256": model_meta["config_sha256"], "model_id": MODEL_ID,
            "base_model_revision": BASE_REVISION, "selector": SELECTOR, "aggregation": "MAX",
        })
        output.append(reference)
    output_sha = atomic_jsonl(args.output, output)
    metrics = {
        "status": "COMPLETE", "gpu": torch.cuda.get_device_name(0), "model": str(args.model), **model_meta,
        "worklist": str(args.worklist), "worklist_sha256": sha256(args.worklist), "reference_scores": str(args.reference_scores),
        "reference_scores_sha256": sha256(args.reference_scores), "qdocs": len(output), "units": len(pairs),
        "batch_size": args.batch_size, "selector": SELECTOR, "aggregation": "MAX", "max_length": MAX_LENGTH,
        "output": str(args.output), "output_sha256": output_sha, "all_scores_finite": True, "duplicate_qdocs": 0,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
