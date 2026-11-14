"""Preference-optimize the registered Task 2 Qwen adapter without a ref clone.

Reference chosen/rejected log-probability margins are cached before the first
optimizer step.  Training then applies length-normalized DPO plus a small
chosen-answer SFT anchor, avoiding both a second 1.7B model and the long-answer
bias of raw summed sequence log probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.training import finetune_task2_qwen_lora as qwen  # noqa: E402

SCHEMA_VERSION = "task2-p18-length-normalized-dpo-v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--reference-batch-size", type=_positive_int, default=4)
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=_positive_float, default=5e-6)
    parser.add_argument("--weight-decay", type=_unit_float, default=0.01)
    parser.add_argument("--warmup-ratio", type=_unit_float, default=0.05)
    parser.add_argument("--beta", type=_positive_float, default=0.2)
    parser.add_argument("--sft-weight", type=_unit_float, default=0.1)
    parser.add_argument("--max-length", type=_positive_int, default=2560)
    parser.add_argument("--max-context-tokens", type=_positive_int, default=1500)
    parser.add_argument("--max-answer-tokens", type=_positive_int, default=768)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=_positive_int, default=10)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_preferences(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_ties = 0
    required = {"id", "question", "contexts", "chosen", "rejected"}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required <= set(row):
                raise ValueError(f"{path}:{line_number} is not a preference row")
            question_id = str(row["id"])
            if question_id in seen:
                raise ValueError(f"duplicate preference ID {question_id!r}")
            seen.add(question_id)
            for field in ("question", "chosen", "rejected"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"blank {field} for preference {question_id!r}")
            if not isinstance(row["contexts"], list) or not row["contexts"]:
                raise ValueError(f"blank contexts for preference {question_id!r}")
            if row["chosen"].strip() == row["rejected"].strip():
                skipped_ties += 1
                continue
            rows.append(row)
    if not rows:
        raise ValueError("preference data are empty")
    if skipped_ties:
        print(
            json.dumps(
                {
                    "event": "p18_preference_ties_skipped",
                    "count": skipped_ties,
                }
            ),
            flush=True,
        )
    return rows


def _chat_prompt_ids(tokenizer: Any, question: str, context: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": qwen.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": qwen._user_prompt(question, context),
            },
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not encoded:
        raise RuntimeError("chat template returned no token IDs")
    return [int(value) for value in encoded]


def encode_preference(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    max_context_tokens: int,
    max_answer_tokens: int,
) -> dict[str, list[int]]:
    """Encode chosen/rejected with one identical, jointly bounded prompt."""

    chosen = tokenizer.encode(row["chosen"], add_special_tokens=False)[
        :max_answer_tokens
    ]
    rejected = tokenizer.encode(row["rejected"], add_special_tokens=False)[
        :max_answer_tokens
    ]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        chosen.append(eos_id)
        rejected.append(eos_id)
    longest_answer = max(len(chosen), len(rejected))
    if longest_answer >= max_length:
        raise ValueError(f"preference {row['id']!r} exhausts max-length")
    context = qwen._truncate_context(
        tokenizer,
        qwen._context_block(row["contexts"]),
        max_context_tokens,
    )
    prompt = _chat_prompt_ids(tokenizer, row["question"], context)
    available = max_length - longest_answer
    if len(prompt) > available:
        overflow = len(prompt) - available
        reduced_limit = max(0, max_context_tokens - overflow - 16)
        context = qwen._truncate_context(tokenizer, context, reduced_limit)
        prompt = _chat_prompt_ids(tokenizer, row["question"], context)
    if len(prompt) > available:
        raise ValueError(f"prompt for preference {row['id']!r} exhausts max-length")
    return {
        "chosen_ids": prompt + chosen,
        "chosen_labels": [-100] * len(prompt) + chosen,
        "rejected_ids": prompt + rejected,
        "rejected_labels": [-100] * len(prompt) + rejected,
    }


def _collate(
    torch: Any,
    tokenizer: Any,
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sequences = [row["chosen_ids"] for row in batch] + [
        row["rejected_ids"] for row in batch
    ]
    labels = [row["chosen_labels"] for row in batch] + [
        row["rejected_labels"] for row in batch
    ]
    max_size = max(len(value) for value in sequences)
    padded_ids: list[list[int]] = []
    padded_labels: list[list[int]] = []
    masks: list[list[int]] = []
    for ids, label in zip(sequences, labels):
        padding = max_size - len(ids)
        padded_ids.append(ids + [tokenizer.pad_token_id] * padding)
        padded_labels.append(label + [-100] * padding)
        masks.append([1] * len(ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "indices": torch.tensor([int(row["index"]) for row in batch]),
        "pair_count": len(batch),
    }


def _sequence_average_logps(torch: Any, logits: Any, labels: Any) -> Any:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    counts = mask.sum(dim=-1)
    if bool(counts.eq(0).any()):
        raise ValueError("a preference completion contains no scored tokens")
    return (token_logps * mask).sum(dim=-1) / counts


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "preference_sha256": _sha256(args.preference_data),
        "base_fingerprint": qwen._base_model_fingerprint(args.model_dir),
        "adapter_fingerprint": qwen._base_model_fingerprint(args.initial_adapter),
        "epochs": args.epochs,
        "effective_batch": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "sft_weight": args.sft_weight,
        "max_length": args.max_length,
        "max_context_tokens": args.max_context_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "dtype": args.dtype,
        "seed": args.seed,
        "length_normalized_logps": True,
    }


def run(args: argparse.Namespace) -> list[Path]:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.max_answer_tokens >= args.max_length:
        raise ValueError("max-answer-tokens must be smaller than max-length")
    for path, label in (
        (args.model_dir, "base model"),
        (args.initial_adapter, "initial adapter"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {path}")
    contract = _contract(args)
    manifest = args.output_dir / "training_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("training_contract") != contract:
            raise ValueError("completed P18 output has a different contract")
        print(f"P18_DPO_ALREADY_COMPLETE {manifest}", flush=True)
        return [args.output_dir, manifest]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"P18 output directory is not empty: {args.output_dir}")

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = _load_preferences(args.preference_data)
    encoded = [
        {
            **encode_preference(
                tokenizer,
                row,
                max_length=args.max_length,
                max_context_tokens=args.max_context_tokens,
                max_answer_tokens=args.max_answer_tokens,
            ),
            "index": index,
        }
        for index, row in enumerate(rows)
    ]

    class PreferenceDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(encoded)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return encoded[index]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=qwen._dtype(torch, args.dtype),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.initial_adapter,
        is_trainable=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.to(args.device)
    dataset = PreferenceDataset()
    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _collate(torch, tokenizer, batch)
    reference_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.reference_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device == "cuda",
    )
    reference_deltas = torch.empty(len(dataset), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        total = len(reference_loader)
        interval = max(1, total // 20)
        for step, batch in enumerate(reference_loader, 1):
            indices = batch.pop("indices")
            pair_count = int(batch.pop("pair_count"))
            batch = {
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
            }
            labels = batch.pop("labels")
            with torch.autocast(
                device_type="cuda" if args.device == "cuda" else "cpu",
                dtype=qwen._dtype(torch, args.dtype),
                enabled=args.dtype != "float32",
            ):
                logits = model(**batch).logits
            logps = _sequence_average_logps(torch, logits, labels)
            deltas = logps[:pair_count] - logps[pair_count:]
            reference_deltas[indices] = deltas.float().cpu()
            if step == 1 or step % interval == 0 or step == total:
                print(
                    json.dumps(
                        {
                            "event": "p18_reference_cache",
                            "batch": step,
                            "batches": total,
                            "percent": round(100 * step / total, 1),
                        }
                    ),
                    flush=True,
                )
    if not bool(torch.isfinite(reference_deltas).all()):
        raise RuntimeError("reference cache contains non-finite values")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        generator=generator,
        pin_memory=args.device == "cuda",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.dtype == "float16" and args.device == "cuda"
    )
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        rewards: list[float] = []
        chosen_logps: list[float] = []
        for step, batch in enumerate(train_loader, 1):
            indices = batch.pop("indices")
            pair_count = int(batch.pop("pair_count"))
            batch = {
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
            }
            labels = batch.pop("labels")
            reference = reference_deltas[indices].to(args.device)
            with torch.autocast(
                device_type="cuda" if args.device == "cuda" else "cpu",
                dtype=qwen._dtype(torch, args.dtype),
                enabled=args.dtype != "float32",
            ):
                logits = model(**batch).logits
                logps = _sequence_average_logps(torch, logits, labels)
                chosen = logps[:pair_count]
                rejected = logps[pair_count:]
                policy_delta = chosen - rejected
                preference_reward = policy_delta - reference
                dpo_loss = -torch.nn.functional.logsigmoid(
                    args.beta * preference_reward
                ).mean()
                chosen_nll = -chosen.mean()
                loss = dpo_loss + args.sft_weight * chosen_nll
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            rewards.append(float(preference_reward.detach().mean().cpu()))
            chosen_logps.append(float(chosen.detach().mean().cpu()))
            update = step % args.gradient_accumulation == 0 or step == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if (
                    global_update % args.log_every == 0
                    or global_update == total_updates
                ):
                    recent = min(args.log_every, len(losses))
                    print(
                        json.dumps(
                            {
                                "event": "p18_training_progress",
                                "epoch": epoch,
                                "epochs": args.epochs,
                                "update": global_update,
                                "updates": total_updates,
                                "loss": sum(losses[-recent:]) / recent,
                                "reward_margin": sum(rewards[-recent:]) / recent,
                                "chosen_logp": sum(chosen_logps[-recent:]) / recent,
                            }
                        ),
                        flush=True,
                    )
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "reward_margin": sum(rewards) / len(rewards),
                "chosen_logp": sum(chosen_logps) / len(chosen_logps),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    _atomic_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "records": len(rows),
            "training_contract": contract,
            "global_updates": global_update,
            "history": history,
            "reference_margin_mean": float(reference_deltas.mean()),
            "reference_margin_min": float(reference_deltas.min()),
            "reference_margin_max": float(reference_deltas.max()),
            "trainable_parameters": sum(value.numel() for value in trainable),
        },
    )
    print(f"P18_DPO_COMPLETE {args.output_dir}", flush=True)
    return [args.output_dir, manifest]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Task 2 P18 DPO error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
