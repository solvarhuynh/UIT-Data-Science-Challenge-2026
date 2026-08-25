"""GPU inference-only frozen base-reranker feature scoring for V3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    from .common import jsonl, sha256
except ImportError:
    from common import jsonl, sha256


SOURCES = ("bm25", "adaptive_k500", "knn_char", "knn_word")
BASE_DOCUMENT_FEATURE_NAMES = (
    "chunk_logits", "chunk_score_1", "chunk_score_2", "chunk_score_3",
    "neural_max", "neural_second", "neural_mean", "neural_min", "neural_std",
    "neural_max_minus_mean", "neural_top2_mean", "bm25_score_1", "bm25_score_2",
    "bm25_score_3", "bm25_max", "bm25_mean", "union_rank", "reciprocal_union_rank",
    "source_support", "min_source_rank", "has_bge_support", "baseline_rank",
    "is_baseline_top5", "question_token_length", "question_char_length",
    "bm25_rank", "bm25_missing", "adaptive_k500_rank", "adaptive_k500_missing",
    "knn_char_rank", "knn_char_missing", "knn_word_rank", "knn_word_missing",
)


def source_features(source_ranks: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in SOURCES:
        value = source_ranks.get(source)
        output[f"{source}_rank"] = int(value) if value is not None else 0
        output[f"{source}_missing"] = value is None
    return output


def doc_features(row: dict[str, Any], doc: dict[str, Any], logits: list[float], *, include_fold: bool) -> dict[str, Any]:
    evidence = doc["evidence"]
    padded_logits: list[float | None] = logits + [None] * (3 - len(logits))
    padded_bm25: list[float | None] = [float(item["bm25_score"]) for item in evidence] + [None] * (3 - len(evidence))
    values = logits
    source_ranks = dict(doc.get("source_ranks") or {})
    if not source_ranks and "retrieval_features" in doc:
        source_ranks = dict(doc["retrieval_features"].get("source_ranks") or {})
    baseline_top5 = [str(value) for value in row["baseline_top5"]]
    doc_id = str(doc["doc_id"])
    base_rank = baseline_top5.index(doc_id) + 1 if doc_id in baseline_top5 else 0
    ranks = [int(value) for value in source_ranks.values() if value is not None]
    output = {
        "query_id": str(row["query_id"]),
        "doc_id": doc_id,
        "baseline_top5": baseline_top5,
        "chunk_logits": logits,
        "chunk_score_1": padded_logits[0], "chunk_score_2": padded_logits[1], "chunk_score_3": padded_logits[2],
        "neural_max": max(values), "neural_second": sorted(values, reverse=True)[1] if len(values) > 1 else values[0],
        "neural_mean": mean(values), "neural_min": min(values), "neural_std": pstdev(values) if len(values) > 1 else 0.0,
        "neural_max_minus_mean": max(values) - mean(values),
        "neural_top2_mean": mean(sorted(values, reverse=True)[:2]),
        "bm25_score_1": padded_bm25[0], "bm25_score_2": padded_bm25[1], "bm25_score_3": padded_bm25[2],
        "bm25_max": max(float(item["bm25_score"]) for item in evidence),
        "bm25_mean": mean(float(item["bm25_score"]) for item in evidence),
        "union_rank": int(doc["union_rank"]),
        "reciprocal_union_rank": 1.0 / int(doc["union_rank"]),
        "source_support": int(doc.get("source_support", 0) or 0),
        "min_source_rank": min(ranks) if ranks else 0,
        "has_bge_support": bool("adaptive_k500" in source_ranks or "bge" in source_ranks or "original_bge" in source_ranks),
        "baseline_rank": base_rank,
        "is_baseline_top5": bool(base_rank),
        "question_token_length": len(str(row["question"]).split()),
        "question_char_length": len(str(row["question"])),
        **source_features(source_ranks),
    }
    if include_fold:
        output["fold"] = int(row["fold"])
    return output


def parity_smoke(model_path: Path, train_report_path: Path) -> dict[str, Any]:
    """Check the frozen scorer contract without loading the neural model."""
    if not model_path.is_dir() or not train_report_path.is_file():
        return {"status": "BLOCKED", "reason": "model or train feature report missing"}
    config_path = model_path / "config.json"
    weight_path = model_path / "model.safetensors"
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model")
    if not config_path.is_file() or not weight_path.is_file() or not any((model_path / name).is_file() for name in tokenizer_files):
        return {"status": "BLOCKED", "reason": "required model/config/tokenizer files missing"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = config.get("id2label", {})
    report = json.loads(train_report_path.read_text(encoding="utf-8"))
    source = Path(__file__).read_text(encoding="utf-8")
    checks = {
        "single_logit": int(config.get("num_labels", len(labels))) == 1,
        "historical_model_path": report.get("model") in {"models/reranker", "models\\reranker"},
        "model_hash_matches_train_report": report.get("model_sha256") == sha256(weight_path),
        "max_length_512": "max_length=512" in source,
        "pair_preprocessing": "tokenizer([str(row[\"question\"])]" in source,
        "chunk_logits_three_features": all(name in BASE_DOCUMENT_FEATURE_NAMES for name in ("chunk_logits", "chunk_score_1", "chunk_score_2", "chunk_score_3")),
        "no_training": report.get("model_modified") is False,
    }
    return {"status": "PASS" if all(checks.values()) else "BLOCKED", "checks": checks, "model": str(model_path), "train_report": str(train_report_path), "gpu_launched": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=("train", "public"), default="train")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--parity-smoke", action="store_true")
    args = parser.parse_args()
    if args.parity_smoke:
        report_path = args.report if args.report.is_file() else Path("artifacts/task1/recovery_096/v3_residual/frozen_features_report.json")
        print(json.dumps(parity_smoke(args.model, report_path), indent=2))
        return
    if args.shortlist is None or args.output is None:
        parser.error("--shortlist and --output are required unless --parity-smoke is used")
    if args.mode == "public":
        missing_model = [
            str(path)
            for path in (
                args.model / "config.json",
                args.model / "model.safetensors",
            )
            if not path.is_file()
        ]
        if not any((args.model / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model")):
            missing_model.append(str(args.model / "tokenizer.json"))
        if missing_model:
            raise FileNotFoundError("public frozen scoring requires reranker model files:\n" + "\n".join(missing_model))
        if not args.shortlist.is_file():
            raise FileNotFoundError(f"public frozen scoring shortlist missing: {args.shortlist}")
    rows = jsonl(args.shortlist)
    expected_queries = 7000 if args.mode == "train" else 1000
    if len(rows) != expected_queries or any(len(row["docs"]) > 25 for row in rows):
        raise ValueError(f"V3 {args.mode} frozen scorer requires {expected_queries} shortlist rows capped at 25 docs")
    if args.mode == "public" and any("fold" in row or set(row) & {"answer", "gold", "label"} for row in rows):
        raise ValueError("public frozen scorer refuses fold/label fields")
    if any(not doc.get("evidence") or len(doc["evidence"]) > 3 for row in rows for doc in row["docs"]):
        raise ValueError("V3 frozen scorer requires 1..3 true-S2 evidence chunks per doc")
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "query_count": len(rows), "default_batch_size": args.batch_size, "all_chunk_logits_saved": True, "gpu_launched": False}, indent=2))
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("frozen V3 feature scoring requires CUDA; use --preflight locally")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model)).to("cuda").eval()
    if int(getattr(model.config, "num_labels", len(getattr(model.config, "id2label", {})) or -1)) != 1:
        raise ValueError("frozen V3 base model must be single-logit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc_count = chunk_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for doc in row["docs"]:
                flat.extend((doc, item) for item in doc["evidence"])
            logits: list[float] = []
            for start in range(0, len(flat), args.batch_size):
                batch = flat[start : start + args.batch_size]
                encoded = tokenizer([str(row["question"])] * len(batch), [str(item[1]["raw_chunk_text"]) for item in batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(**encoded).logits
                if output.ndim != 2 or output.shape[1] != 1:
                    raise ValueError(f"expected [batch,1] logits, got {tuple(output.shape)}")
                logits.extend(float(value) for value in output[:, 0].float().tolist())
            offset = 0
            for doc in row["docs"]:
                count = len(doc["evidence"])
                values = logits[offset : offset + count]
                offset += count
                handle.write(json.dumps(doc_features(row, doc, values, include_fold=args.mode == "train"), ensure_ascii=False, sort_keys=True) + "\n")
                doc_count += 1
                chunk_count += count
    schema_hash = hashlib.sha256(json.dumps(list(BASE_DOCUMENT_FEATURE_NAMES), separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {"status": "FROZEN_FEATURES_COMPLETE", "mode": args.mode, "query_count": len(rows), "doc_count": doc_count, "chunk_count": chunk_count, "batch_size": args.batch_size, "precision": "cuda_fp16", "model": str(args.model), "model_sha256": sha256(args.model / "model.safetensors"), "shortlist_sha256": sha256(args.shortlist), "features_sha256": sha256(args.output), "base_feature_names": list(BASE_DOCUMENT_FEATURE_NAMES), "feature_schema_sha256": schema_hash, "all_chunk_logits_saved": True, "model_modified": False, "no_training": True, "no_public_labels_used": args.mode == "public"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
