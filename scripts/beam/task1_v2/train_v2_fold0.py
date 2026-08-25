"""Train one fold0 V2 document-level pairwise reranker.

Training is deliberately opt-in via ``--allow-training``.  ``--preflight``
only validates the groups and never imports CUDA or launches Beam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

try:
    from .losses import query_group_pairwise_loss
    from .evidence import select_true_s2
except ImportError:
    from losses import query_group_pairwise_loss
    from evidence import select_true_s2


def load_groups(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def validate_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if select_true_s2.__name__ != "select_true_s2":
        raise ValueError("V2 trainer is not importing the canonical true-S2 selector")
    query_ids: set[str] = set()
    folds: set[int] = set()
    docs = positives = negatives = evidence = missing = 0
    for group in groups:
        qid = str(group["query_id"])
        if qid in query_ids:
            raise ValueError(f"duplicate training query: {qid}")
        query_ids.add(qid)
        fold = int(group["fold"])
        folds.add(fold)
        pos = neg = 0
        for doc in group.get("docs", []):
            docs += 1
            if bool(doc.get("is_positive_doc")):
                pos += 1
                positives += 1
            else:
                neg += 1
                negatives += 1
            selected = doc.get("evidence", [])
            evidence += len(selected)
            if not selected:
                missing += 1
            if any(item.get("selector_name") != "true_s2_bm25_within_document_v2" for item in selected):
                raise ValueError(f"non-true-S2 evidence in query {qid}")
        if not pos or not neg:
            raise ValueError(f"query {qid} needs at least one positive and one hard negative")
    if 0 in folds:
        raise ValueError("fold0 leakage in V2 training groups")
    if folds != {1, 2, 3, 4}:
        raise ValueError(f"expected training folds {{1,2,3,4}}, got {sorted(folds)}")
    return {
        "train_queries": len(groups),
        "train_docs": docs,
        "positive_docs": positives,
        "negative_docs": negatives,
        "evidence_chunks": evidence,
        "missing_evidence_docs": missing,
        "folds": sorted(folds),
        "true_s2_selector_share": 1.0 if evidence and not missing else 0.0,
    }


def score_pairs(model, tokenizer, pairs, batch_size: int, max_length: int, device, torch):
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**encoded).logits.reshape(-1).float()
        scores.extend(logits.unbind())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docgroups", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.max_length != 512:
        raise ValueError("V2 contract requires max_length=512")
    groups = load_groups(args.docgroups)
    stats = validate_groups(groups)
    if stats["missing_evidence_docs"]:
        raise ValueError("V2 training groups contain missing evidence")
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", **stats, "gpu_launched": False}, indent=2))
        return
    if not args.allow_training:
        raise SystemExit("training is guarded; pass --allow-training on the remote Beam step")
    if args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs and gradient accumulation must be positive")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("V2 training requires CUDA; local CPU training is intentionally disabled")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.base_model)).to(device)
    if int(getattr(model.config, "num_labels", len(getattr(model.config, "id2label", {})) or -1)) != 1:
        raise ValueError("base model must be single-logit sequence classification")
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError("requested gradient checkpointing but model does not support it")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    micro_steps = 0
    losses: list[float] = []
    groups_per_step = max(1, int(args.batch_queries))
    accumulation = max(1, int(args.gradient_accumulation))
    for epoch in range(args.epochs):
        epoch_groups = list(groups)
        random.Random(args.seed + epoch).shuffle(epoch_groups)
        for group_index in range(0, len(epoch_groups), groups_per_step):
            mini = epoch_groups[group_index : group_index + groups_per_step]
            mini_loss = None
            for group in mini:
                pairs: list[tuple[str, str]] = []
                spans: list[tuple[bool, int, int]] = []
                for doc in group["docs"]:
                    begin = len(pairs)
                    pairs.extend((str(group["question"]), str(item["raw_chunk_text"])) for item in doc["evidence"])
                    spans.append((bool(doc["is_positive_doc"]), begin, len(pairs)))
                chunk_scores = score_pairs(model, tokenizer, pairs, args.batch_size, args.max_length, device, torch)
                doc_scores = [torch.stack(chunk_scores[begin:end]).max() for _, begin, end in spans]
                positive_scores = torch.stack([score for (is_pos, _, _), score in zip(spans, doc_scores) if is_pos])
                negative_scores = torch.stack([score for (is_pos, _, _), score in zip(spans, doc_scores) if not is_pos])
                group_loss = query_group_pairwise_loss(positive_scores, negative_scores, args.margin)
                mini_loss = group_loss if mini_loss is None else mini_loss + group_loss
            mini_loss = mini_loss / len(mini) / accumulation
            scaler.scale(mini_loss).backward()
            losses.append(float(mini_loss.detach().float().cpu()) * accumulation)
            micro_steps += 1
            if micro_steps % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
    if micro_steps and micro_steps % accumulation:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "status": "TRAINING_COMPLETE",
        "train_sha256": hashlib.sha256(args.docgroups.read_bytes()).hexdigest(),
        "base_model": str(args.base_model),
        "max_length": 512,
        "learning_rate": args.learning_rate,
        "learning_rate_candidates": [5e-7, 1e-6, 2e-6],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batch_queries": args.batch_queries,
        "gradient_accumulation": args.gradient_accumulation,
        "margin": args.margin,
        "seed": args.seed,
        "objective": "multi-positive document-level pairwise softplus",
        "aggregation": "max_chunk_score_per_doc",
        "selector": "true_s2_bm25_within_document_v2",
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "optimizer_updates": updates,
        "mean_logged_loss": sum(losses) / len(losses) if losses else None,
        "train_runtime_seconds": time.time() - start_time,
        **stats,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
