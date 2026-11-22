"""Train a cross-encoder with grouped softmax or balanced BCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from udsc2026.evaluation.reporting import write_run_manifest  # noqa: E402
from udsc2026.training.provenance import (  # noqa: E402
    checkpoint_files,
    commit_training_checkpoint,
    identity_inputs,
    read_json,
    verify_checkpoint,
)

MODEL_ID = "AITeamVN/Vietnamese_Reranker"
MODEL_PATH = Path("models/aiteam-vietnamese-reranker")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--fold", type=int, choices=range(5))
    scope.add_argument(
        "--all-queries",
        action="store_true",
        help="Train the final checkpoint on every usable grouped query.",
    )
    parser.add_argument("--loss", choices=("softmax", "bce"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--modal-volume")
    return parser


def load_training(args: argparse.Namespace) -> tuple[list[dict], dict]:
    groups = args.groups_dir / "groups.jsonl"
    manifest_path = args.groups_dir / "manifest.json"
    manifest = read_json(manifest_path)
    model_manifest = Path("models/download_manifest.json")
    model = read_json(model_manifest)["models"]["reranker"]
    inputs = identity_inputs({
        "folds": args.folds,
        "groups": groups,
        "groups_manifest": manifest_path,
        "model_manifest": model_manifest,
        "trainer": Path(__file__),
    })
    if (
        manifest.get("schema_version") != "retrieval-training-groups-v2"
        or manifest.get("groups_sha256") != inputs["groups"]
        or manifest.get("folds_sha256") != inputs["folds"]
        or model.get("repo_id") != MODEL_ID
        or model.get("resolved_revision") != manifest.get("base_model_revision")
    ):
        raise ValueError("training groups, folds or model fail provenance checks")
    scope = "all" if args.all_queries else "fold"
    if args.all_queries:
        train_ids: set[str] | None = None
        held_out_ids: set[str] = set()
    else:
        fold = next(
            row for row in read_json(args.folds)["folds"] if row["fold"] == args.fold
        )
        train_ids = set(fold["training_ids"])
        held_out_ids = set(fold["validation_ids"])
        if train_ids & held_out_ids:
            raise ValueError("training and validation IDs overlap")
    examples = []
    seen = set()
    with groups.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            qid = row["query_id"]
            allowed = train_ids is None or qid in train_ids | held_out_ids
            if qid in seen or not allowed:
                raise ValueError(f"unexpected or duplicate training ID {qid}")
            seen.add(qid)
            if train_ids is None or qid in train_ids:
                examples.append(row)
    if not examples:
        raise ValueError("training fold has no usable query groups")
    return examples, {
        "schema_version": "retrieval-reranker-training-v2",
        "inputs": inputs,
        "source_sha256": manifest["source_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "questions_sha256": manifest["questions_sha256"],
        "base_model": MODEL_ID,
        "base_model_revision": model["resolved_revision"],
        "scope": scope,
        "fold": args.fold if not args.all_queries else None,
        "loss": args.loss,
        "max_length": manifest["max_length"],
        "rerank_k": manifest["rerank_k"],
        "train_queries": len(examples),
        "held_out_queries": len(held_out_ids),
        "configuration": {
            name: getattr(args, name)
            for name in (
                "learning_rate", "epochs", "batch_size", "gradient_accumulation",
                "save_steps", "seed",
            )
        },
    }


def grouped_loss(logits, labels, sizes, loss_name):
    import torch
    from torch.nn import functional as F

    losses = []
    for scores, targets in zip(logits.float().split(sizes), labels.split(sizes)):
        positive = scores[targets.bool()]
        if loss_name == "softmax":
            losses.append(torch.logsumexp(scores, dim=0) - positive.mean())
        else:
            negative = scores[~targets.bool()]
            losses.append((F.softplus(-positive).mean() + F.softplus(negative).mean()) / 2)
    return torch.stack(losses).mean()


def collate_groups(rows, tokenizer, max_length):
    import torch

    questions, texts, labels, sizes = [], [], [], []
    for row in rows:
        positives = row["positive_chunks"]
        negatives = [hit for band in row["negative_chunks"].values() for hit in band]
        sizes.append(len(positives) + len(negatives))
        for chunk in positives + negatives:
            questions.append(row["question"])
            texts.append(chunk["text"])
        labels.extend([1.0] * len(positives) + [0.0] * len(negatives))
    batch = tokenizer(
        questions, texts, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )
    return {**batch, "labels": torch.tensor(labels), "sizes": sizes}


def run(args: argparse.Namespace) -> dict:
    examples, identity = load_training(args)
    if args.preflight:
        return identity
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if args.output_dir.resolve() == MODEL_PATH.resolve() or (
        MODEL_PATH.resolve() in args.output_dir.resolve().parents
    ):
        raise ValueError("output would overwrite the base model")
    identity["base_files"] = checkpoint_files(MODEL_PATH)
    identity["dependencies"] = {
        name: version(name)
        for name in ("accelerate", "sentencepiece", "torch", "transformers")
    }
    has_cuda = torch.cuda.is_available()
    identity["device"] = torch.cuda.get_device_name() if has_cuda else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "training_report.json"
    checkpoint = args.output_dir / "checkpoint"
    if report_path.exists():
        previous = read_json(report_path)
        if not args.resume or any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("output contains a different run or requires --resume")
        if previous.get("complete"):
            verify_checkpoint(checkpoint, previous)
            return previous
    elif any(args.output_dir.iterdir()):
        raise ValueError("training output directory already contains untracked outputs")
    write_run_manifest({**identity, "complete": False}, report_path)

    class GroupTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            sizes = inputs.pop("sizes")
            outputs = model(**inputs)
            logits = outputs.logits.reshape(-1)
            loss = grouped_loss(logits, labels, sizes, args.loss)
            return (loss, outputs) if return_outputs else loss

    class CommitCheckpoint(TrainerCallback):
        def on_save(self, *unused_args, **unused_kwargs):
            commit_training_checkpoint(args.modal_volume)

    started = time.monotonic()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    if model.config.num_labels != 1:
        raise ValueError("group training requires a single reranker logit")
    trainer = GroupTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "state"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            gradient_checkpointing=has_cuda,
            bf16=has_cuda and torch.cuda.is_bf16_supported(),
            warmup_steps=0.05,
            save_steps=args.save_steps,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=examples,
        data_collator=lambda rows: collate_groups(rows, tokenizer, identity["max_length"]),
        callbacks=[CommitCheckpoint()],
    )
    trainer.model_accepts_loss_kwargs = False
    last = get_last_checkpoint(str(args.output_dir / "state")) if (
        (args.output_dir / "state").is_dir()
    ) else None
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    report = {
        **identity,
        "complete": True,
        "files": checkpoint_files(checkpoint),
        "runtime_seconds": time.monotonic() - started,
        "metrics": result.metrics,
        "stopped_after_optimizer_steps": result.global_step,
    }
    write_run_manifest(report, report_path)
    commit_training_checkpoint(args.modal_volume)
    return report


def main() -> int:
    print(json.dumps(run(build_parser().parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
