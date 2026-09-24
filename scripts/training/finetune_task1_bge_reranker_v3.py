"""Gated low-LR BGE chunk trainer with teacher anchoring (experimental V3).

This entry point never runs training unless ``--allow-training`` is supplied.
It writes only to a new output directory and intentionally does not promote a
checkpoint or modify the production reranker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def _load_examples(path: Path, max_queries: int | None) -> list[tuple[str, str, float]]:
    examples: list[tuple[str, str, float]] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if max_queries is not None and index >= max_queries:
                break
            row = json.loads(line)
            question = str(row["question"])
            positives = row.get("positive_chunks", [])
            bands = row.get("negative_chunks", {})
            negatives = [chunk for values in bands.values() for chunk in values]
            # Per-query balance prevents a query with many candidates dominating.
            limit = min(len(positives), len(negatives))
            for chunk in positives[:limit]:
                examples.append((question, str(chunk["text"]), 1.0))
            for chunk in negatives[:limit]:
                examples.append((question, str(chunk["text"]), 0.0))
    if not examples:
        raise ValueError("training groups produced no balanced examples")
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logits(value: Any) -> Any:
    logits = value.logits
    if logits.ndim == 2 and logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits.reshape(-1)


def _freeze_encoder(model: Any, freeze_layers: int) -> int:
    base = getattr(model, "roberta", getattr(model, "xlm_roberta", None))
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", ())
    frozen = 0
    for layer in list(layers)[:freeze_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
            frozen += parameter.numel()
    embeddings = getattr(base, "embeddings", None)
    if embeddings is not None and freeze_layers > 0:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-groups", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("models/reranker"))
    parser.add_argument(
        "--teacher-model",
        type=Path,
        help="Frozen teacher checkpoint; defaults to --base-model for SAME-BGE runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Return a verified completed run; partial optimizer-state resume is intentionally unsupported.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--freeze-layers", type=int, default=18)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument(
        "--teacher-anchor-mode",
        choices=("legacy", "conflict-aware"),
        default="legacy",
        help="legacy reproduces V2; conflict-aware removes MSE only when the teacher disagrees with the target at p=0.5.",
    )
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--strict-oof-f1-f4",
        action="store_true",
        help="Mark this run for the frozen F1--F4 outer-OOF gate, never Fold0.",
    )
    return parser


def compute_teacher_anchor_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    teacher_weight: float,
    mode: str,
) -> tuple[Any, dict[str, int]]:
    """Compute the frozen V2 loss or the single V3 conflict-aware variant.

    The function is intentionally torch-agnostic at import time so CPU unit
    tests can import this trainer without loading a model. ``legacy`` keeps
    the exact batch-mean formula used by V2. ``conflict-aware`` applies the
    fixed p=0.5 consistency rule per example and uses BCE-only on conflicts.
    """
    import torch
    from torch.nn import functional

    if mode not in {"legacy", "conflict-aware"}:
        raise ValueError(f"unsupported teacher anchor mode: {mode}")
    bce_per_example = functional.binary_cross_entropy_with_logits(
        student_logits, labels, reduction="none"
    )
    mse_per_example = (student_logits - teacher_logits).square()
    if mode == "legacy":
        supervised = bce_per_example.mean()
        anchoring = mse_per_example.mean()
        loss = (1.0 - teacher_weight) * supervised + teacher_weight * anchoring
        diagnostics = {
            "teacher_consistent_examples": int(labels.numel()),
            "teacher_conflict_examples": 0,
            "anchored_examples": int(labels.numel()),
        }
        return loss, diagnostics

    teacher_probability = teacher_logits.sigmoid()
    teacher_consistent = torch.where(
        labels >= 0.5,
        teacher_probability >= 0.5,
        teacher_probability < 0.5,
    )
    per_example = torch.where(
        teacher_consistent,
        (1.0 - teacher_weight) * bce_per_example + teacher_weight * mse_per_example,
        bce_per_example,
    )
    return per_example.mean(), {
        "teacher_consistent_examples": int(teacher_consistent.sum().item()),
        "teacher_conflict_examples": int((~teacher_consistent).sum().item()),
        "anchored_examples": int(teacher_consistent.sum().item()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    teacher_path = args.teacher_model or args.base_model
    if not args.train_groups.is_file() or not args.base_model.is_dir() or not teacher_path.is_dir():
        raise FileNotFoundError("training groups/base model are required locally")
    base = args.base_model.resolve()
    output = args.output_dir.resolve()
    canonical_corpus = Path("data/processed_v3").resolve()
    if (
        output == base
        or base in output.parents
        or output == canonical_corpus
        or canonical_corpus in output.parents
    ):
        raise ValueError(
            "experimental output must not be inside base model/canonical corpus"
        )
    if not (0 < args.learning_rate <= 1e-6):
        raise ValueError("V3 conservative gate requires 0 < learning rate <= 1e-6")
    if (
        args.epochs < 1
        or args.epochs > 2
        or args.gradient_accumulation < 1
        or not (0 <= args.teacher_weight <= 1)
    ):
        raise ValueError("V3 gate permits only 1-2 epochs and teacher weight in [0,1]")
    if args.teacher_anchor_mode not in {"legacy", "conflict-aware"}:
        raise ValueError("teacher_anchor_mode must be legacy or conflict-aware")
    examples = _load_examples(args.train_groups, args.max_queries)
    preflight = {
        "schema_version": "task1-bge-reranker-v3-teacher-anchor-v1",
        "status": "preflight_only" if not args.allow_training else "training_started",
        "training_example_count": len(examples),
        "train_groups_sha256": _sha256(args.train_groups),
        "representation": "retrieved raw chunks matching inference",
        "base_model": str(args.base_model),
        "teacher_model": str(teacher_path),
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "freeze_layers": args.freeze_layers,
        "teacher_weight": args.teacher_weight,
        "teacher_anchor_mode": args.teacher_anchor_mode,
        "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed,
        "validation_gate": "STRICT_OOF_F1_F4" if args.strict_oof_f1_f4 else "LEGACY_FOLD0_GATE",
        "promotable": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.allow_training:
        return preflight

    completed = args.output_dir / "training_report.json"
    checkpoint = args.output_dir / "checkpoint"
    if args.resume and completed.is_file():
        report = json.loads(completed.read_text(encoding="utf-8"))
        if (
            report.get("status") == (
                "trained_requires_strict_oof_gate"
                if args.strict_oof_f1_f4
                else "trained_requires_fold0_gate"
            )
            and report.get("train_groups_sha256") == preflight["train_groups_sha256"]
            and report.get("base_model") == preflight["base_model"]
            and report.get("teacher_model") == preflight["teacher_model"]
            and (checkpoint / "config.json").is_file()
            and (checkpoint / "model.safetensors").is_file()
        ):
            return {**report, "status": "resume_complete_no_model_loaded"}
        raise ValueError("--resume found an incompatible or incomplete checkpoint")

    import torch
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True
    ).to(device)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_path, local_files_only=True
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    frozen_parameters = _freeze_encoder(student, args.freeze_layers)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        examples, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    losses = []
    teacher_consistent_examples = 0
    teacher_conflict_examples = 0
    anchored_examples = 0
    student.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    for _ in range(args.epochs):
        for batch_index, batch in enumerate(loader, 1):
            questions, texts, raw_labels = batch
            encoded = tokenizer(
                list(questions),
                list(texts),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = raw_labels.to(device=device, dtype=torch.float32)
            student_logits = _logits(student(**encoded))
            with torch.no_grad():
                teacher_logits = _logits(teacher(**encoded))
            loss, anchor_diagnostics = compute_teacher_anchor_loss(
                student_logits,
                teacher_logits,
                labels,
                args.teacher_weight,
                args.teacher_anchor_mode,
            )
            teacher_consistent_examples += anchor_diagnostics["teacher_consistent_examples"]
            teacher_conflict_examples += anchor_diagnostics["teacher_conflict_examples"]
            anchored_examples += anchor_diagnostics["anchored_examples"]
            if not math.isfinite(float(loss.detach().cpu())):
                raise RuntimeError("non-finite training loss")
            (loss / args.gradient_accumulation).backward()
            micro_steps += 1
            if (
                micro_steps % args.gradient_accumulation == 0
                or batch_index == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
    checkpoint.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    # A reload is part of the GPU smoke/production checkpoint contract. This
    # occurs only after opt-in training and never in CPU-only preflight mode.
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    if int(getattr(reloaded.config, "num_labels", 1)) != int(
        getattr(student.config, "num_labels", 1)
    ):
        raise RuntimeError("checkpoint reload model-label contract mismatch")
    del reloaded
    report = {
        **preflight,
        "status": (
            "trained_requires_strict_oof_gate"
            if args.strict_oof_f1_f4
            else "trained_requires_fold0_gate"
        ),
        "mean_training_loss": sum(losses) / len(losses),
        "teacher_consistent_examples": teacher_consistent_examples,
        "teacher_conflict_examples": teacher_conflict_examples,
        "anchored_examples": anchored_examples,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "frozen_parameter_count": frozen_parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_reload_verified": True,
        "promotable": False,
        "next_gate": (
            "score only the held-out F1--F4 fold and apply the frozen strict OOF gate"
            if args.strict_oof_f1_f4
            else "evaluate fold0 once against original BGE full-stack OOF baseline"
        ),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Task1 V3 trainer error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
