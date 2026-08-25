"""Gated low-LR BGE chunk trainer with teacher anchoring (experimental V3).

This entry point never runs training unless ``--allow-training`` is supplied.
It writes only to a new output directory and intentionally does not promote a
checkpoint or modify the production reranker.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--freeze-layers", type=int, default=18)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--max-queries", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.train_groups.is_file() or not args.base_model.is_dir():
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
    if args.epochs < 1 or args.epochs > 2 or not (0 <= args.teacher_weight <= 1):
        raise ValueError("V3 gate permits only 1-2 epochs and teacher weight in [0,1]")
    examples = _load_examples(args.train_groups, args.max_queries)
    preflight = {
        "schema_version": "task1-bge-reranker-v3-teacher-anchor-v1",
        "status": "preflight_only" if not args.allow_training else "training_started",
        "training_example_count": len(examples),
        "representation": "retrieved raw chunks matching inference",
        "base_model": str(args.base_model),
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "freeze_layers": args.freeze_layers,
        "teacher_weight": args.teacher_weight,
        "promotable": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.allow_training:
        return preflight

    import torch
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True
    ).to(device)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True
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
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    losses = []
    student.train()
    for _ in range(args.epochs):
        for batch in loader:
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
            supervised = functional.binary_cross_entropy_with_logits(
                student_logits, labels
            )
            anchoring = functional.mse_loss(student_logits, teacher_logits)
            loss = (
                1.0 - args.teacher_weight
            ) * supervised + args.teacher_weight * anchoring
            if not math.isfinite(float(loss.detach().cpu())):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    checkpoint = args.output_dir / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    report = {
        **preflight,
        "status": "trained_requires_fold0_gate",
        "mean_training_loss": sum(losses) / len(losses),
        "optimizer_steps": len(losses),
        "frozen_parameter_count": frozen_parameters,
        "checkpoint": str(checkpoint),
        "promotable": False,
        "next_gate": "evaluate fold0 once against original BGE full-stack OOF baseline",
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
