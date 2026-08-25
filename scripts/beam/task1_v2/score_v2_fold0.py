"""Matched V2 fold0 inference: true-S2 evidence, batched fp16, max aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .evidence import select_true_s2
except ImportError:
    from evidence import select_true_s2


def load_groups(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def validate_groups(groups: list[dict[str, Any]]) -> None:
    if not groups:
        raise ValueError("empty V2 fold0 evaluation groups")
    for group in groups:
        if select_true_s2.__name__ != "select_true_s2":
            raise ValueError("V2 scorer is not importing the canonical true-S2 selector")
        seen: set[str] = set()
        for doc in group.get("docs", []):
            doc_id = str(doc["doc_id"])
            if doc_id in seen:
                raise ValueError(f"duplicate candidate doc {group['query_id']}:{doc_id}")
            seen.add(doc_id)
            if not doc.get("evidence"):
                raise ValueError(f"missing evidence {group['query_id']}:{doc_id}")
            if any(item.get("selector_name") != "true_s2_bm25_within_document_v2" for item in doc["evidence"]):
                raise ValueError("V2 score input contains non-true-S2 evidence")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docgroups", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--doc-scores", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.max_length != 512:
        raise ValueError("V2 inference contract requires max_length=512")
    groups = load_groups(args.docgroups)
    validate_groups(groups)
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "query_count": len(groups), "gpu_launched": False}, indent=2))
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("V2 scoring requires CUDA; use --preflight for local checks")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model)).to("cuda").eval()
    if int(getattr(model.config, "num_labels", len(getattr(model.config, "id2label", {})) or -1)) != 1:
        raise ValueError("V2 checkpoint is not single-logit")

    score_rows: list[dict[str, Any]] = []
    for group in groups:
        flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for doc in group["docs"]:
            flat.extend((doc, evidence) for evidence in doc["evidence"])
        chunk_scores: list[float] = []
        for start in range(0, len(flat), args.batch_size):
            batch = flat[start : start + args.batch_size]
            encoded = tokenizer(
                [str(group["question"])] * len(batch),
                [str(item[1]["raw_chunk_text"]) for item in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError(f"expected [batch,1] logits, got {tuple(logits.shape)}")
            chunk_scores.extend(float(value) for value in logits[:, 0].float().tolist())
        offset = 0
        docs: list[dict[str, Any]] = []
        for doc in group["docs"]:
            count = len(doc["evidence"])
            values = chunk_scores[offset : offset + count]
            offset += count
            features = dict(doc.get("retrieval_features") or {})
            source_ranks = dict(features.get("source_ranks") or {})
            docs.append(
                {
                    "query_id": str(group["query_id"]),
                    "doc_id": str(doc["doc_id"]),
                    "neural_score": max(values),
                    "union_rank": int(doc["union_rank"]),
                    "source_support": int(features.get("source_support", 0) or 0),
                    "source_ranks": source_ranks,
                    "min_source_rank": features.get("min_source_rank"),
                    "has_bge_support": bool(features.get("has_bge_support", False)),
                }
            )
        docs.sort(key=lambda row: (-row["neural_score"], row["union_rank"], row["doc_id"]))
        for rank, doc in enumerate(docs, 1):
            doc["neural_rank"] = rank
            score_rows.append(doc)

    predictions: list[dict[str, Any]] = []
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        by_query.setdefault(row["query_id"], []).append(row)
    for query_id in sorted(by_query, key=lambda value: (int(value) if value.isdigit() else value)):
        docs = sorted(by_query[query_id], key=lambda row: (-row["neural_score"], row["union_rank"], row["doc_id"]))
        predictions.append({"query_id": query_id, "fold": 0, "top5": [row["doc_id"] for row in docs[:5]]})
    write_jsonl(args.doc_scores, score_rows)
    write_jsonl(args.predictions, predictions)


if __name__ == "__main__":
    main()
