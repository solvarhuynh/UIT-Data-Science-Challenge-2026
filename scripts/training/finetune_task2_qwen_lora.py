"""Prepare, fine-tune, and run a Task 2 Qwen LoRA model.

The preparation stage may use LegalQA references only for organizer training
IDs. Generation consumes source-only parent rankings. This keeps strict
held-out evaluation and public inference free of answer leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SYSTEM_PROMPT = (
    "Bạn là chuyên gia pháp luật Việt Nam. Chỉ dùng dữ liệu pháp lý được cung "
    "cấp. Trả lời trực tiếp, chính xác, giữ nguyên số tiền, thời hạn, điều, "
    "khoản và tên văn bản. Không chào hỏi, không bịa và không viết phần suy nghĩ."
)
SCHEMA_VERSION = 1
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
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
    """Build the command line interface without importing GPU libraries."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Create leakage-scoped SFT records from labeled parent candidates.",
    )
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--labels", type=Path, action="append", required=True)
    prepare.add_argument("--question-ids", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--contexts-per-question", type=_positive_int, default=3)

    train = commands.add_parser(
        "train",
        help="Fine-tune Qwen with LoRA and optionally write merged weights.",
    )
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument(
        "--initial-adapter",
        type=Path,
        help="Continue training from an existing LoRA adapter on model-dir.",
    )
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--merge-output", type=Path)
    train.add_argument("--device", default="cuda")
    train.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    train.add_argument("--epochs", type=_positive_int, default=2)
    train.add_argument("--batch-size", type=_positive_int, default=1)
    train.add_argument("--gradient-accumulation", type=_positive_int, default=16)
    train.add_argument("--learning-rate", type=_positive_float, default=2e-4)
    train.add_argument("--weight-decay", type=_unit_float, default=0.01)
    train.add_argument("--warmup-ratio", type=_unit_float, default=0.05)
    train.add_argument("--max-length", type=_positive_int, default=3072)
    train.add_argument("--max-context-tokens", type=_positive_int, default=1800)
    train.add_argument("--max-answer-tokens", type=_positive_int, default=1024)
    train.add_argument("--lora-rank", type=_positive_int, default=32)
    train.add_argument("--lora-alpha", type=_positive_int, default=64)
    train.add_argument("--lora-dropout", type=_unit_float, default=0.05)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--log-every", type=_positive_int, default=50)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest completed epoch checkpoint in output-dir.",
    )
    train.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Run one worst-shape LoRA forward/backward/optimizer step and exit "
            "without writing a training checkpoint."
        ),
    )
    train.add_argument(
        "--data-parallel",
        action="store_true",
        help=(
            "Use every visible CUDA GPU through torch DataParallel. The batch "
            "size must be at least the number of GPUs."
        ),
    )

    generate = commands.add_parser(
        "generate",
        help="Generate deterministic answers from source-only parent rankings.",
    )
    generate.add_argument("--questions", type=Path, required=True)
    generate.add_argument("--rankings", type=Path, required=True)
    generate.add_argument("--model-dir", type=Path, required=True)
    generate.add_argument(
        "--base-model-dir",
        type=Path,
        help=(
            "Local base-model directory to use when model-dir is a PEFT adapter. "
            "This prevents offline inference from resolving the adapter's "
            "original Hugging Face repository."
        ),
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--diagnostics", type=Path, required=True)
    generate.add_argument("--question-ids", type=Path)
    generate.add_argument("--known-answers", type=Path)
    generate.add_argument("--known-answer-ids", type=Path)
    generate.add_argument("--device", default="cuda")
    generate.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    generate.add_argument("--batch-size", type=_positive_int, default=4)
    generate.add_argument("--top-parents", type=_positive_int, default=3)
    generate.add_argument("--max-parent-words", type=_positive_int, default=768)
    generate.add_argument("--max-context-tokens", type=_positive_int, default=2400)
    generate.add_argument("--max-new-tokens", type=_positive_int, default=1024)
    generate.add_argument("--ce-weight", type=_unit_float, default=0.6)
    generate.add_argument("--retrieval-weight", type=_unit_float, default=0.4)
    generate.add_argument("--rrf-k", type=_nonnegative_int, default=20)
    generate.add_argument("--resume", action="store_true")

    evaluate = commands.add_parser(
        "evaluate",
        help="Score a strict held-out prediction and enforce a promotion gate.",
    )
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--question-ids", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--minimum-meteor", type=_unit_float, default=0.6)
    return parser


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected a non-empty JSON object: {path}")
    return payload


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty ID array: {path}")
    ids = [str(value).strip() for value in payload]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid or duplicate IDs: {path}")
    return ids


def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                yield row


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question_key(question: str) -> str:
    normalized = unicodedata.normalize("NFC", question).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def prepare_records(
    questions: dict[str, Any],
    question_ids: Sequence[str],
    label_rows: Iterable[dict[str, Any]],
    *,
    contexts_per_question: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build SFT records while enforcing the requested training-ID boundary."""

    requested = set(question_ids)
    unknown = requested - set(questions)
    if unknown:
        raise ValueError(
            f"question IDs are missing from questions: {sorted(unknown)[:5]}"
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    ignored_rows = 0
    for row in label_rows:
        question_id = str(row.get("question_id", "")).strip()
        if question_id not in requested:
            ignored_rows += 1
            continue
        parent_text = row.get("parent_text")
        overlap = row.get("reference_overlap")
        if not isinstance(parent_text, str) or not parent_text.strip():
            raise ValueError(f"labeled row {question_id!r} has no parent text")
        if not isinstance(overlap, (int, float)) or not math.isfinite(float(overlap)):
            raise ValueError(f"labeled row {question_id!r} has invalid overlap")
        grouped.setdefault(question_id, []).append(row)

    records: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for question_id in question_ids:
        source = questions[question_id]
        question = source.get("question") if isinstance(source, dict) else None
        answer = source.get("answer") if isinstance(source, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no text")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"training question {question_id!r} has no answer")
        candidates = grouped.get(question_id, [])
        if not candidates:
            missing_labels.append(question_id)
            continue
        candidates.sort(
            key=lambda row: (
                -float(row["reference_overlap"]),
                int(row.get("candidate_rank", 10**9)),
                str(row.get("doc_id", "")),
                str(row.get("parent_id", "")),
            )
        )
        selected_contexts: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in candidates:
            key = (str(row.get("doc_id", "")), str(row.get("parent_id", "")))
            if key in seen:
                continue
            seen.add(key)
            selected_contexts.append(
                {
                    "doc_id": key[0],
                    "parent_id": key[1],
                    "law_name": str(row.get("law_name") or ""),
                    "article": str(row.get("article") or ""),
                    "text": row["parent_text"].strip(),
                    "reference_overlap": float(row["reference_overlap"]),
                    "candidate_rank": int(row.get("candidate_rank", 10**9)),
                }
            )
            if len(selected_contexts) == contexts_per_question:
                break
        # Keep the useful teacher-selected set, but restore label-free retrieval
        # order so the generator cannot learn that the first context is gold.
        contexts = sorted(
            selected_contexts,
            key=lambda context: (
                context["candidate_rank"],
                context["doc_id"],
                context["parent_id"],
            ),
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": question_id,
                "question": question.strip(),
                "contexts": contexts,
                "answer": answer.strip(),
            }
        )
    return records, {
        "schema_version": SCHEMA_VERSION,
        "requested_questions": len(question_ids),
        "emitted_questions": len(records),
        "missing_label_questions": missing_labels,
        "ignored_out_of_scope_label_rows": ignored_rows,
        "contexts_per_question": contexts_per_question,
        "leakage_policy": "answers restricted to explicit training IDs",
    }


def run_prepare(args: argparse.Namespace) -> list[Path]:
    if args.output in args.labels or args.output in {args.questions, args.question_ids}:
        raise ValueError("output must not overwrite an input")
    questions = _load_json_object(args.questions)
    question_ids = _load_ids(args.question_ids)
    records, report = prepare_records(
        questions,
        question_ids,
        _read_jsonl(args.labels),
        contexts_per_question=args.contexts_per_question,
    )
    if not records:
        raise ValueError("preparation produced no SFT records")
    _write_jsonl(args.output, records)
    manifest = args.output.with_name(f"{args.output.stem}.manifest.json")
    report["output"] = str(args.output)
    report["output_sha256"] = _sha256(args.output)
    report["inputs"] = {
        "questions": str(args.questions),
        "question_ids": str(args.question_ids),
        "labels": [str(path) for path in args.labels],
    }
    _write_json(manifest, report)
    return [args.output, manifest]


def _context_block(contexts: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, context in enumerate(contexts, 1):
        heading = " - ".join(
            value
            for value in (
                str(context.get("law_name") or "").strip(),
                str(context.get("article") or "").strip(),
            )
            if value
        )
        text = str(context.get("text") or "").strip()
        block = f"[{index}] {heading}\n{text}" if heading else f"[{index}]\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _user_prompt(question: str, context: str) -> str:
    return f"DỮ LIỆU PHÁP LÝ:\n{context}\n\nCÂU HỎI:\n{question}\n\nTRẢ LỜI:"


def _chat_prompt_ids(tokenizer: Any, question: str, context: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(question, context)},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, list) or not encoded:
        raise RuntimeError("chat template returned no token IDs")
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError("chat template unexpectedly returned a token batch")
        encoded = encoded[0]
    if not all(isinstance(value, int) for value in encoded):
        raise RuntimeError("chat template returned invalid token IDs")
    return encoded


def _truncate_context(tokenizer: Any, context: str, max_tokens: int) -> str:
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def _encode_training_record(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    max_context_tokens: int,
    max_answer_tokens: int,
) -> tuple[list[int], list[int]]:
    context = _truncate_context(
        tokenizer,
        _context_block(row["contexts"]),
        max_context_tokens,
    )
    prompt_ids = _chat_prompt_ids(tokenizer, row["question"], context)
    answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    answer_ids = answer_ids[:max_answer_tokens]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids.append(eos_id)
    if len(answer_ids) >= max_length:
        raise ValueError(f"answer for {row['id']!r} exhausts max-length")
    available_prompt = max_length - len(answer_ids)
    if len(prompt_ids) > available_prompt:
        overflow = len(prompt_ids) - available_prompt
        reduced_limit = max(0, max_context_tokens - overflow - 16)
        context = _truncate_context(tokenizer, context, reduced_limit)
        prompt_ids = _chat_prompt_ids(tokenizer, row["question"], context)
    if len(prompt_ids) > available_prompt:
        raise ValueError(f"prompt overhead for {row['id']!r} exhausts max-length")
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def _load_sft_records(path: Path) -> list[dict[str, Any]]:
    rows = list(_read_jsonl([path]))
    required = {"id", "question", "contexts", "answer"}
    seen: set[str] = set()
    for row in rows:
        if not required <= set(row):
            raise ValueError(f"SFT row is missing fields: {required - set(row)}")
        question_id = str(row["id"])
        if question_id in seen:
            raise ValueError(f"duplicate SFT ID {question_id!r}")
        seen.add(question_id)
    if not rows:
        raise ValueError("SFT data are empty")
    return rows


def _dtype(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _disable_unused_torchao_dispatcher() -> None:
    """Skip Kaggle's optional, incompatible TorchAO LoRA dispatcher.

    P14 trains ordinary FP16/BF16 ``torch.nn.Linear`` weights and never uses
    TorchAO quantization.  Some Kaggle images nevertheless preinstall an old
    TorchAO release which recent PEFT detects and rejects before reaching its
    normal Linear dispatcher.
    """

    try:
        from peft.tuners.lora import torchao as torchao_dispatch
    except ImportError:
        return
    torchao_dispatch.is_torchao_available = lambda: False
    print("PEFT_TORCHAO_OPTIONAL_DISPATCH_DISABLED", flush=True)


def _latest_epoch_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.glob("checkpoint-epoch-*"):
        match = re.fullmatch(r"checkpoint-epoch-(\d+)", path.name)
        if (
            match is not None
            and path.is_dir()
            and (path / "adapter_model.safetensors").is_file()
            and (path / "training_state.pt").is_file()
        ):
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=None, key=lambda item: item[0])


def _base_model_fingerprint(model_dir: Path) -> str:
    files = sorted(
        {
            *model_dir.glob("*.safetensors"),
            *model_dir.glob("*.json"),
            *model_dir.glob("tokenizer*"),
        }
    )
    files = [path for path in files if not path.name.startswith(".")]
    if not files:
        raise FileNotFoundError(f"base model has no fingerprintable files: {model_dir}")
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _training_contract(
    args: argparse.Namespace,
    train_data_sha256: str,
    base_model_fingerprint: str,
    initial_adapter_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "train_data_sha256": train_data_sha256,
        "base_model_fingerprint": base_model_fingerprint,
        "initial_adapter_fingerprint": initial_adapter_fingerprint,
        "epochs": args.epochs,
        # Physical microbatch size may change when a retry lands on another
        # Beam GPU. The optimization contract is the effective batch, which
        # stays at 16 across the 24/32/80 GiB runtime profiles.
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "max_context_tokens": args.max_context_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dtype": args.dtype,
        "seed": args.seed,
        "data_parallel": args.data_parallel,
    }


def run_train(args: argparse.Namespace) -> list[Path]:
    """Train LoRA weights with completion-only loss."""

    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.max_answer_tokens >= args.max_length:
        raise ValueError("max-answer-tokens must be smaller than max-length")
    if args.smoke_only and args.resume:
        raise ValueError("--smoke-only cannot be combined with --resume")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    if args.initial_adapter is not None:
        if not args.initial_adapter.is_dir():
            raise FileNotFoundError(
                f"initial adapter does not exist: {args.initial_adapter}"
            )
        required_adapter_files = (
            args.initial_adapter / "adapter_config.json",
            args.initial_adapter / "adapter_model.safetensors",
        )
        missing_adapter_files = [
            str(path) for path in required_adapter_files if not path.is_file()
        ]
        if missing_adapter_files:
            raise FileNotFoundError(
                "initial adapter is incomplete: " + ", ".join(missing_adapter_files)
            )
    train_data_sha256 = _sha256(args.train_data)
    base_model_fingerprint = _base_model_fingerprint(args.model_dir)
    initial_adapter_fingerprint = (
        _base_model_fingerprint(args.initial_adapter)
        if args.initial_adapter is not None
        else None
    )
    contract = _training_contract(
        args,
        train_data_sha256,
        base_model_fingerprint,
        initial_adapter_fingerprint,
    )
    manifest = args.output_dir / "training_manifest.json"
    if manifest.is_file():
        if not args.resume:
            raise ValueError(f"completed output already exists: {args.output_dir}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("training_contract") != contract:
            raise ValueError(
                "completed output contract differs from the requested training run"
            )
        print(f"QWEN_TRAIN_ALREADY_COMPLETE {manifest}", flush=True)
        written = [args.output_dir, manifest]
        if args.merge_output is not None and args.merge_output.is_dir():
            written.append(args.merge_output)
        return written
    resume_checkpoint = (
        _latest_epoch_checkpoint(args.output_dir) if args.resume else None
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if resume_checkpoint is None:
            raise ValueError(
                f"output directory is incomplete and has no resumable epoch: "
                f"{args.output_dir}"
            )
    if (
        args.merge_output is not None
        and args.merge_output.exists()
        and any(args.merge_output.iterdir())
    ):
        raise ValueError(f"merge output directory is not empty: {args.merge_output}")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
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
    rows = _load_sft_records(args.train_data)

    class SFTDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            input_ids, labels = _encode_training_record(
                tokenizer,
                rows[index],
                max_length=args.max_length,
                max_context_tokens=args.max_context_tokens,
                max_answer_tokens=args.max_answer_tokens,
            )
            return {"input_ids": input_ids, "labels": labels}

    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_size = max(len(row["input_ids"]) for row in batch)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for row in batch:
            padding = max_size - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention_masks.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=_dtype(torch, args.dtype),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    _disable_unused_torchao_dispatcher()
    if resume_checkpoint is not None:
        print(f"QWEN_RESUME_ADAPTER {resume_checkpoint[1]}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            resume_checkpoint[1],
            is_trainable=True,
            local_files_only=True,
        )
    elif args.initial_adapter is not None:
        print(f"QWEN_INITIAL_ADAPTER {args.initial_adapter}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            args.initial_adapter,
            is_trainable=True,
            local_files_only=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    model.to(args.device)
    data_parallel_devices = 1
    if args.data_parallel:
        if args.device != "cuda":
            raise ValueError("--data-parallel requires --device cuda")
        data_parallel_devices = torch.cuda.device_count()
        if data_parallel_devices < 2:
            raise RuntimeError(
                "--data-parallel requested but fewer than two CUDA GPUs are visible"
            )
        if args.batch_size < data_parallel_devices:
            raise ValueError(
                "--batch-size must be at least the visible GPU count when "
                "--data-parallel is enabled"
            )
        model = torch.nn.DataParallel(model)
        print(
            f"QWEN_DATA_PARALLEL enabled devices={data_parallel_devices} "
            f"batch_size={args.batch_size}",
            flush=True,
        )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataset = SFTDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        generator=loader_generator,
        pin_memory=args.device == "cuda",
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_scaler = args.dtype == "float16" and args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimizer.zero_grad(set_to_none=True)
    if args.smoke_only:
        # Exercise the worst configured tensor shape, not merely the first
        # real record's length. This catches the same activation/optimizer
        # memory pressure as full training before an expensive Beam run starts.
        sample = dataset[0]
        padding = args.max_length - len(sample["input_ids"])
        if padding < 0:
            raise RuntimeError("encoded smoke sample exceeds max-length")
        padded_sample = {
            "input_ids": sample["input_ids"]
            + [tokenizer.pad_token_id] * padding,
            "labels": sample["labels"] + [-100] * padding,
        }
        smoke_batch = collate([padded_sample] * args.batch_size)
        smoke_batch = {
            key: value.to(args.device, non_blocking=True)
            for key, value in smoke_batch.items()
        }
        model.train()
        with torch.autocast(
            device_type="cuda" if args.device == "cuda" else "cpu",
            dtype=_dtype(torch, args.dtype),
            enabled=args.dtype != "float32",
        ):
            smoke_loss = model(**smoke_batch).loss
            if smoke_loss.ndim:
                smoke_loss = smoke_loss.mean()
        scaler.scale(smoke_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if args.device == "cuda":
            torch.cuda.synchronize()
        print(
            "QWEN_TRAINING_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "loss": float(smoke_loss.detach().cpu()),
                    "allocated_bytes": (
                        int(torch.cuda.memory_allocated(0))
                        if args.device == "cuda"
                        else 0
                    ),
                    "reserved_bytes": (
                        int(torch.cuda.memory_reserved(0))
                        if args.device == "cuda"
                        else 0
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if args.data_parallel else model
        model_to_save.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"QWEN_SMOKE_ADAPTER_READY {args.output_dir}", flush=True)
        return [args.output_dir]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        saved_epoch, checkpoint_dir = resume_checkpoint
        state = torch.load(
            checkpoint_dir / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        if state.get("contract") != contract:
            raise ValueError(
                "resume checkpoint contract differs from the requested training run"
            )
        if int(state.get("epoch", -1)) != saved_epoch:
            raise ValueError("resume checkpoint epoch metadata is inconsistent")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        loader_generator.set_state(state["loader_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        history = list(state.get("history", []))
        global_step = int(state["global_step"])
        start_epoch = saved_epoch + 1
        print(
            f"QWEN_RESUME_READY epoch={saved_epoch} "
            f"global_update={global_step}/{total_updates}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(loader, 1):
            batch = {
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type="cuda" if args.device == "cuda" else "cpu",
                dtype=_dtype(torch, args.dtype),
                enabled=args.dtype != "float32",
            ):
                loss = model(**batch).loss
                # DataParallel gathers the scalar loss from every device into a
                # vector. Average it so optimization is equivalent to a single
                # global batch and backward() receives a scalar.
                if loss.ndim:
                    loss = loss.mean()
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            update = step % args.gradient_accumulation == 0 or step == len(loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    recent = losses[-args.log_every :]
                    recent_loss = sum(recent) / len(recent)
                    print(
                        f"epoch={epoch}/{args.epochs} update={global_step}/"
                        f"{total_updates} loss={recent_loss:.6f}",
                        flush=True,
                    )
        history.append(
            {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
        )
        checkpoint = args.output_dir / f"checkpoint-epoch-{epoch}"
        model_to_save = model.module if args.data_parallel else model
        model_to_save.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        state_path = checkpoint / "training_state.pt"
        temporary_state = checkpoint / ".training_state.pt.tmp"
        torch.save(
            {
                "schema_version": 1,
                "contract": contract,
                "epoch": epoch,
                "global_step": global_step,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "loader_generator_state": loader_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
            temporary_state,
        )
        os.replace(temporary_state, state_path)

    model_to_save = model.module if args.data_parallel else model
    model_to_save.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    written = [args.output_dir]
    if args.merge_output is not None:
        args.merge_output.mkdir(parents=True, exist_ok=True)
        merged = model_to_save.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(args.merge_output, safe_serialization=True)
        tokenizer.save_pretrained(args.merge_output)
        written.append(args.merge_output)
    _write_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "base_model": str(args.model_dir),
            "initial_adapter": (
                str(args.initial_adapter) if args.initial_adapter else None
            ),
            "initial_adapter_fingerprint": initial_adapter_fingerprint,
            "train_data": str(args.train_data),
            "train_data_sha256": train_data_sha256,
            "records": len(rows),
            "epochs": args.epochs,
            "global_updates": global_step,
            "history": history,
            "completion_only_loss": True,
            "enable_thinking": False,
            "max_length": args.max_length,
            "max_context_tokens": args.max_context_tokens,
            "max_answer_tokens": args.max_answer_tokens,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "data_parallel_devices": data_parallel_devices,
            "training_contract": contract,
            "resumed_from_epoch": (
                resume_checkpoint[0] if resume_checkpoint is not None else None
            ),
            "merged_output": str(args.merge_output) if args.merge_output else None,
        },
    )
    written.append(manifest)
    return written


def fuse_parent_rankings(
    parents: Sequence[dict[str, Any]],
    *,
    ce_weight: float,
    retrieval_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Fuse CE and original retrieval ranks without using references."""

    if ce_weight + retrieval_weight <= 0:
        raise ValueError("at least one ranking weight must be positive")
    scored: list[tuple[float, dict[str, Any]]] = []
    for parent in parents:
        ce_rank = int(parent.get("rank", 10**9))
        retrieval_rank = int(parent.get("candidate_rank", 10**9))
        score = ce_weight / (rrf_k + ce_rank) + retrieval_weight / (
            rrf_k + retrieval_rank
        )
        scored.append((score, parent))
    return [
        parent
        for _, parent in sorted(
            scored,
            key=lambda item: (
                -item[0],
                int(item[1].get("candidate_rank", 10**9)),
                str(item[1].get("doc_id", "")),
                str(item[1].get("parent_id", "")),
            ),
        )
    ]


def _bounded_parent_words(parent: dict[str, Any], limit: int) -> str:
    words = str(parent.get("parent_text") or "").split()
    anchor = str(parent.get("anchor_text") or "").split()
    if len(words) <= limit:
        return " ".join(words)
    start = 0
    if anchor:
        anchor_size = min(len(anchor), len(words))
        for index in range(len(words) - anchor_size + 1):
            if words[index : index + anchor_size] == anchor[:anchor_size]:
                start = max(0, index - (limit - anchor_size) // 2)
                break
    start = min(start, len(words) - limit)
    return " ".join(words[start : start + limit])


def _load_rankings(path: Path) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl([path]):
        question_id = str(row.get("question_id", "")).strip()
        parents = row.get("parents")
        if not question_id or not isinstance(parents, list) or not parents:
            raise ValueError(f"invalid ranking row for {question_id!r}")
        if question_id in rankings:
            raise ValueError(f"duplicate ranking question {question_id!r}")
        rankings[question_id] = parents
    return rankings


def exact_known_answers(
    questions: dict[str, Any],
    known_questions: dict[str, Any],
    *,
    allowed_ids: set[str] | None = None,
) -> dict[str, str]:
    """Map normalized exact organizer questions to known training answers."""

    known: dict[str, str] = {}
    ambiguous: set[str] = set()
    for question_id, row in known_questions.items():
        if allowed_ids is not None and question_id not in allowed_ids:
            continue
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        key = _question_key(question)
        prior = known.get(key)
        if prior is not None and prior != answer.strip():
            ambiguous.add(key)
        else:
            known[key] = answer.strip()
    for key in ambiguous:
        known.pop(key, None)
    return {
        question_id: known[key]
        for question_id, row in questions.items()
        if isinstance(row, dict)
        and isinstance(row.get("question"), str)
        and (key := _question_key(row["question"])) in known
    }


def _load_existing_predictions(path: Path, resume: bool) -> list[dict[str, str]]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"output exists; use --resume: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("existing output must be an array")
    return payload


def run_generate(args: argparse.Namespace) -> list[Path]:
    """Run deterministic batched generation and checkpoint every batch."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    questions = _load_json_object(args.questions)
    selected_ids = (
        _load_ids(args.question_ids) if args.question_ids else list(questions)
    )
    if missing := set(selected_ids) - set(questions):
        raise ValueError(f"selected questions are missing: {sorted(missing)[:5]}")
    rankings = _load_rankings(args.rankings)
    if missing := set(selected_ids) - set(rankings):
        raise ValueError(f"rankings are missing questions: {sorted(missing)[:5]}")

    overlays: dict[str, str] = {}
    if args.known_answers is not None:
        known_ids = (
            set(_load_ids(args.known_answer_ids))
            if args.known_answer_ids is not None
            else None
        )
        overlays = exact_known_answers(
            {question_id: questions[question_id] for question_id in selected_ids},
            _load_json_object(args.known_answers),
            allowed_ids=known_ids,
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer_source = args.base_model_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "dtype": _dtype(torch, args.dtype),
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if (args.model_dir / "adapter_config.json").is_file():
        if args.base_model_dir is not None:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_dir,
                **model_kwargs,
            )
            model = PeftModel.from_pretrained(
                base_model,
                args.model_dir,
                local_files_only=True,
            )
            print(
                f"QWEN_ADAPTER_LOCAL_BASE {args.base_model_dir}",
                flush=True,
            )
        else:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_dir,
                **model_kwargs,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            **model_kwargs,
        )
    model = model.to(args.device)
    # The released legal Qwen config disables cache for training. Re-enable
    # the KV cache for autoregressive inference; otherwise every generated
    # token recomputes the full legal context and generation becomes many
    # times slower.
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    model.eval()

    predictions = _load_existing_predictions(args.output, args.resume)
    completed = {str(row["id"]) for row in predictions}
    diagnostics: list[dict[str, Any]] = []
    if args.diagnostics.exists() and args.resume:
        loaded = json.loads(args.diagnostics.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            diagnostics = loaded
    pending = [
        question_id for question_id in selected_ids if question_id not in completed
    ]
    for batch_start in range(0, len(pending), args.batch_size):
        batch_ids = pending[batch_start : batch_start + args.batch_size]
        generated_ids: list[str] = []
        prompts: list[list[int]] = []
        for question_id in batch_ids:
            if question_id in overlays:
                predictions.append({"id": question_id, "answer": overlays[question_id]})
                diagnostics.append({"id": question_id, "source": "exact_known_answer"})
                continue
            fused = fuse_parent_rankings(
                rankings[question_id],
                ce_weight=args.ce_weight,
                retrieval_weight=args.retrieval_weight,
                rrf_k=args.rrf_k,
            )[: args.top_parents]
            contexts = [
                {
                    "law_name": parent.get("law_name", ""),
                    "article": parent.get("article", ""),
                    "text": _bounded_parent_words(parent, args.max_parent_words),
                }
                for parent in fused
            ]
            context = _truncate_context(
                tokenizer,
                _context_block(contexts),
                args.max_context_tokens,
            )
            question = str(questions[question_id]["question"])
            prompts.append(_chat_prompt_ids(tokenizer, question, context))
            generated_ids.append(question_id)
        if prompts:
            max_prompt = max(len(values) for values in prompts)
            padded = [
                [tokenizer.pad_token_id] * (max_prompt - len(values)) + values
                for values in prompts
            ]
            attention = [
                [0] * (max_prompt - len(values)) + [1] * len(values)
                for values in prompts
            ]
            input_ids = torch.tensor(padded, dtype=torch.long, device=args.device)
            attention_mask = torch.tensor(
                attention,
                dtype=torch.long,
                device=args.device,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            for question_id, output in zip(generated_ids, outputs):
                answer = tokenizer.decode(
                    output[max_prompt:],
                    skip_special_tokens=True,
                ).strip()
                if not answer:
                    raise ValueError(
                        f"model generated an empty answer for {question_id}"
                    )
                predictions.append({"id": question_id, "answer": answer})
                diagnostics.append(
                    {
                        "id": question_id,
                        "source": "qwen_lora",
                        "answer_characters": len(answer),
                        "batch_seconds": elapsed,
                    }
                )
        order = {question_id: index for index, question_id in enumerate(selected_ids)}
        predictions.sort(key=lambda row: order[str(row["id"])])
        diagnostics.sort(key=lambda row: order[str(row["id"])])
        _write_json(args.output, predictions)
        _write_json(args.diagnostics, diagnostics)
        print(
            f"generated={len(predictions)}/{len(selected_ids)} "
            f"last_batch_seconds={elapsed if prompts else 0:.3f}",
            flush=True,
        )
    predictions = align_predictions(predictions, selected_ids)
    diagnosed = {str(row.get("id", "")) for row in diagnostics}
    for prediction in predictions:
        question_id = str(prediction["id"])
        if question_id not in diagnosed:
            diagnostics.append(
                {
                    "id": question_id,
                    "source": "resumed_existing_prediction",
                    "answer_characters": len(str(prediction["answer"])),
                }
            )
    order = {question_id: index for index, question_id in enumerate(selected_ids)}
    diagnostics.sort(key=lambda row: order[str(row["id"])])
    diagnostic_ids = [str(row.get("id", "")) for row in diagnostics]
    if len(diagnostic_ids) != len(set(diagnostic_ids)):
        raise ValueError("generation diagnostics contain duplicate IDs")
    _write_json(args.output, predictions)
    _write_json(args.diagnostics, diagnostics)
    return [args.output, args.diagnostics]


def align_predictions(
    predictions: Any,
    question_ids: Sequence[str],
) -> list[dict[str, str]]:
    """Validate internal predictions and align them to the explicit ID order."""

    if not isinstance(predictions, list):
        raise ValueError("predictions must be an internal JSON array")
    mapped: dict[str, dict[str, str]] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError("predictions must contain exactly id and answer")
        question_id = row["id"]
        answer = row["answer"]
        if not isinstance(question_id, str) or not isinstance(answer, str):
            raise ValueError("prediction id and answer must be strings")
        if not answer.strip() or question_id in mapped:
            raise ValueError("predictions contain an empty answer or duplicate ID")
        mapped[question_id] = {"id": question_id, "answer": answer}
    expected = set(question_ids)
    if set(mapped) != expected:
        raise ValueError(
            "prediction coverage mismatch: "
            f"missing={sorted(expected - set(mapped))[:5]} "
            f"extra={sorted(set(mapped) - expected)[:5]}"
        )
    return [mapped[question_id] for question_id in question_ids]


def run_evaluate(args: argparse.Namespace) -> list[Path]:
    """Evaluate with the organizer-compatible local METEOR and ROUGE-L code."""

    from scripts.submission.supervised_legal_qa_third_parent import (
        _OFFICIAL_STEMMER,
        _official_meteor,
        _official_rouge_l,
    )

    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.data.find("corpora/wordnet.zip")
        meteor_scorer = _official_meteor
        meteor_profile = "organizer-compatible-wordnet"
    except LookupError:
        from nltk.translate.meteor_score import meteor_score

        class EmptyWordNet:
            @staticmethod
            def synsets(word: str) -> list[Any]:
                del word
                return []

        def meteor_scorer(reference: str, prediction: str) -> float:
            return float(
                meteor_score(
                    [reference.split()],
                    prediction.split(),
                    stemmer=_OFFICIAL_STEMMER,
                    wordnet=EmptyWordNet(),
                )
            )

        meteor_profile = "exact-stem-fallback-no-wordnet"

    questions = _load_json_object(args.questions)
    question_ids = _load_ids(args.question_ids)
    aligned = align_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8-sig")),
        question_ids,
    )
    meteor: list[float] = []
    rouge_l: list[float] = []
    for index, row in enumerate(aligned, 1):
        reference_row = questions.get(row["id"])
        reference = (
            reference_row.get("answer")
            if isinstance(reference_row, dict)
            else None
        )
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"question {row['id']!r} has no reference answer")
        meteor.append(meteor_scorer(reference, row["answer"]))
        rouge_l.append(_official_rouge_l(reference, row["answer"]))
        if index % 100 == 0 or index == len(aligned):
            print(f"scored={index}/{len(aligned)}", flush=True)
    meteor_mean = sum(meteor) / len(meteor)
    rouge_mean = sum(rouge_l) / len(rouge_l)
    passed = meteor_mean >= args.minimum_meteor
    _write_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PROMOTED" if passed else "REJECTED",
            "count": len(aligned),
            "meteor": meteor_mean,
            "rouge_l": rouge_mean,
            "minimum_meteor": args.minimum_meteor,
            "predictions": str(args.predictions),
            "predictions_sha256": _sha256(args.predictions),
            "question_ids": str(args.question_ids),
            "source": "organizer-compatible local scorer",
            "meteor_profile": meteor_profile,
        },
    )
    print(
        f"status={'PROMOTED' if passed else 'REJECTED'} "
        f"meteor={meteor_mean:.6f} rouge_l={rouge_mean:.6f}",
        flush=True,
    )
    if not passed:
        raise RuntimeError(
            f"held-out METEOR {meteor_mean:.6f} is below gate "
            f"{args.minimum_meteor:.6f}; do not submit"
        )
    return [args.output]


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            written = run_prepare(args)
        elif args.command == "train":
            written = run_train(args)
        elif args.command == "generate":
            written = run_generate(args)
        else:
            written = run_evaluate(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        traceback.print_exc()
        print(f"Task 2 Qwen LoRA error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
