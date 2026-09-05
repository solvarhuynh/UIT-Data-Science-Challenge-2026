"""Fail-fast CUDA probe for the exact Task 2 parent reranker training path."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence


def _emit(phase: str, **fields: object) -> None:
    print(
        json.dumps(
            {"event": "reranker_cuda_preflight", "phase": phase, **fields},
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--freeze-layers", type=int, default=6)
    return parser.parse_args(argv)


def _largest_group_sizes(path: Path, groups_per_batch: int) -> list[int]:
    """Return the real group sizes forming the worst listwise microbatch."""

    if groups_per_batch < 1:
        raise ValueError("groups-per-batch must be positive")

    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"invalid question_id at {path}:{line_number}")
            counts[question_id] += 1
    if not counts:
        raise ValueError(f"parent training data are empty: {path}")
    return sorted(counts.values(), reverse=True)[:groups_per_batch]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import PhobertTokenizer, RobertaForSequenceClassification

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _emit("torch_ready", torch_version=str(torch.__version__))

    model_dir = args.model_dir.resolve()
    tokenizer = PhobertTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    _emit("tokenizer_ready")
    model = RobertaForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    _emit("model_cpu_ready")

    device = torch.device(args.device)
    model.to(device)
    _emit(
        "model_cuda_ready",
        allocated_bytes=int(torch.cuda.memory_allocated(0))
        if device.type == "cuda"
        else 0,
    )

    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if args.freeze_layers < 0 or args.freeze_layers > len(model.roberta.encoder.layer):
        raise ValueError("freeze-layers is outside the encoder depth")
    group_sizes = _largest_group_sizes(args.train_data, args.groups_per_batch)
    sequence_batch_size = sum(group_sizes)
    questions = [
        "Quyền và nghĩa vụ của người lao động là gì?"
    ] * sequence_batch_size
    passages = [
        "Người lao động có quyền làm việc và hưởng lương theo thỏa thuận."
    ] * sequence_batch_size
    batch = tokenizer(
        questions,
        passages,
        padding="max_length",
        truncation="only_second",
        max_length=args.max_length,
        return_overflowing_tokens=False,
        return_tensors="pt",
        verbose=False,
    )
    inputs = {key: value.to(device) for key, value in batch.items()}
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[: args.freeze_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(**inputs).logits.squeeze(-1).float()
        losses = []
        offset = 0
        for group_size in group_sizes:
            group_logits = logits[offset : offset + group_size]
            target_scores = torch.linspace(
                0.0,
                1.0,
                group_size,
                device=device,
                dtype=torch.float32,
            )
            target_distribution = torch.softmax(target_scores / 0.10, dim=0)
            losses.append(
                -(target_distribution * torch.log_softmax(group_logits, dim=0)).sum()
            )
            offset += group_size
        loss = torch.stack(losses).mean()
        scaled_loss = loss / args.gradient_accumulation
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    if optimizer_state_bytes <= 0:
        raise RuntimeError("AdamW optimizer state was not allocated")
    _emit(
        "optimizer_step_pass",
        loss=float(loss.detach().cpu()),
        objective="listwise",
        group_sizes=group_sizes,
        sequence_batch_size=sequence_batch_size,
        max_length=args.max_length,
        optimizer_state_bytes=optimizer_state_bytes,
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(0)) if device.type == "cuda" else 0
        ),
    )
    print("RERANKER_CUDA_PREFLIGHT_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
