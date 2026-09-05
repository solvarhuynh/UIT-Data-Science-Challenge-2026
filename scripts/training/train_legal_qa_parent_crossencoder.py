"""Prepare, train, audit, and score a LegalQA parent cross-encoder.

The command is intentionally isolated from the submission pipeline.  It uses
only organizer questions, reference answers, retrieved candidate hits, and the
verbatim parents persisted under ``data/processed_v3/parents``.  Model loading
is local-only; this script never downloads a tokenizer or checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.retrieval.parent_context import (  # noqa: E402
    JsonlParentStore,
    ParentStoreError,
)

SCHEMA_VERSION = 1
LABEL_PROFILE = "legalqa-parent-reference-overlap-v1"
MAX_MODEL_LENGTH = 256
TRAINING_CONTRACT_VERSION = "legalqa-parent-crossencoder-resume-v1"
TRAINING_STATE_SCHEMA_VERSION = 1
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}", flags=re.IGNORECASE)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _sha256_hex(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing torch or transformers."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split_ids = commands.add_parser(
        "split-ids",
        help="Create deterministic first-N eval and remaining train ID files.",
    )
    split_ids.add_argument("--questions", type=Path, required=True)
    split_ids.add_argument("--eval-count", type=_positive_int, default=1000)
    split_ids.add_argument(
        "--train-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_train_ids.json"),
    )
    split_ids.add_argument(
        "--eval-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_eval_ids.json"),
    )
    split_ids.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_split_audit.json"),
    )

    prepare = commands.add_parser(
        "prepare",
        help="Derive strict train/eval parent labels from BTC references.",
    )
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument(
        "--candidates",
        type=Path,
        action="append",
        required=True,
        help="Candidate JSONL; repeat when train and eval are separate files.",
    )
    prepare.add_argument("--train-ids", type=Path, required=True)
    prepare.add_argument("--eval-ids", type=Path, required=True)
    prepare.add_argument(
        "--parents-dir",
        type=Path,
        default=Path("data/processed_v3/parents"),
    )
    prepare.add_argument("--candidate-k", type=_positive_int, default=50)
    prepare.add_argument("--positives-per-query", type=_positive_int, default=1)
    prepare.add_argument("--hard-negatives", type=_positive_int, default=4)
    prepare.add_argument(
        "--min-positive-overlap",
        type=_unit_float,
        default=0.10,
    )
    prepare.add_argument("--min-label-margin", type=_unit_float, default=0.0)
    prepare.add_argument("--negative-gap", type=_unit_float, default=0.02)
    prepare.add_argument(
        "--train-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_train.jsonl"),
    )
    prepare.add_argument(
        "--eval-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_eval.jsonl"),
    )
    prepare.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/task2/training/parent_ce_label_audit.json"),
    )

    audit = commands.add_parser(
        "audit",
        help="Validate prepared labels and strict train/eval isolation.",
    )
    audit.add_argument("--train-data", type=Path, action="append", required=True)
    audit.add_argument("--eval-data", type=Path, action="append", required=True)
    audit.add_argument("--output", type=Path)

    train = commands.add_parser(
        "train",
        help="Fine-tune a local RobertaForSequenceClassification checkpoint.",
    )
    train.add_argument("--train-data", type=Path, action="append", required=True)
    train.add_argument("--eval-data", type=Path, action="append", default=[])
    train.add_argument(
        "--preverified-input-archive-sha256",
        type=_sha256_hex,
        help=(
            "SHA-256 of an enclosing archive already verified by the caller. "
            "Required when preverified member hashes are supplied."
        ),
    )
    train.add_argument(
        "--preverified-train-data-sha256",
        type=_sha256_hex,
        action="append",
        default=[],
    )
    train.add_argument(
        "--preverified-eval-data-sha256",
        type=_sha256_hex,
        action="append",
        default=[],
    )
    train.add_argument(
        "--fit-all",
        action="store_true",
        help="Train without eval; exact organizer overlays remain out of scope.",
    )
    train.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/dek21-v2"),
    )
    train.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task2/models/dek21-parent-crossencoder"),
    )
    train.add_argument("--device", default="cuda")
    train.add_argument("--max-length", type=_positive_int, default=256)
    train.add_argument("--batch-size", type=_positive_int, default=8)
    train.add_argument("--eval-batch-size", type=_positive_int, default=16)
    train.add_argument("--epochs", type=_positive_int, default=3)
    train.add_argument("--learning-rate", type=_positive_float, default=2e-5)
    train.add_argument("--weight-decay", type=_unit_float, default=0.01)
    train.add_argument("--warmup-ratio", type=_unit_float, default=0.1)
    train.add_argument("--gradient-accumulation", type=_positive_int, default=1)
    train.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    train.add_argument(
        "--objective",
        choices=("bce", "listwise"),
        default="bce",
        help=(
            "Training objective. 'listwise' compares every candidate belonging "
            "to the same question and is recommended for parent ranking."
        ),
    )
    train.add_argument(
        "--target-temperature",
        type=_positive_float,
        default=0.1,
        help=(
            "Softmax temperature applied to reference-overlap targets for the "
            "listwise objective."
        ),
    )
    train.add_argument("--freeze-layers", type=_non_negative_int, default=6)
    train.add_argument(
        "--freeze-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--num-workers", type=_non_negative_int, default=0)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output-dir/checkpoint-latest, or return immediately "
            "when a matching completed manifest already exists."
        ),
    )

    score = commands.add_parser(
        "score",
        help="Score and rank verbatim parent candidates with a local checkpoint.",
    )
    score.add_argument("--questions", type=Path, required=True)
    score.add_argument("--candidates", type=Path, action="append", required=True)
    score.add_argument(
        "--parents-dir",
        type=Path,
        default=Path("data/processed_v3/parents"),
    )
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--candidate-k", type=_positive_int, default=50)
    score.add_argument("--batch-size", type=_positive_int, default=16)
    score.add_argument("--max-length", type=_positive_int, default=256)
    score.add_argument("--device", default="cuda")
    score.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _atomic_torch_save(torch: Any, path: Path, payload: object) -> None:
    """Persist a Torch payload with the state file acting as a commit marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _model_fingerprint(model_dir: Path) -> str:
    """Hash the local model snapshot without coupling it to its absolute path."""

    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    files = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file()
        and path.name not in {"training_state.pt"}
        and not path.name.endswith(".tmp")
    )
    if not files:
        raise FileNotFoundError(
            f"model directory has no fingerprintable files: {model_dir}"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(model_dir).as_posix().encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _training_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return the semantic contract that must match an epoch resume exactly."""

    supplied_train = list(args.preverified_train_data_sha256)
    supplied_eval = list(args.preverified_eval_data_sha256)
    supplied_any = bool(supplied_train or supplied_eval)
    if supplied_any and not args.preverified_input_archive_sha256:
        raise ValueError(
            "preverified member hashes require --preverified-input-archive-sha256"
        )
    if supplied_train and len(supplied_train) != len(args.train_data):
        raise ValueError(
            "preverified train-data hash count does not match --train-data count"
        )
    if supplied_eval and len(supplied_eval) != len(args.eval_data):
        raise ValueError(
            "preverified eval-data hash count does not match --eval-data count"
        )
    training_hashes = supplied_train or [_sha256(path) for path in args.train_data]
    evaluation_hashes = supplied_eval or [_sha256(path) for path in args.eval_data]

    return {
        "schema_version": TRAINING_CONTRACT_VERSION,
        "training_data_sha256": training_hashes,
        "evaluation_data_sha256": evaluation_hashes,
        "preverified_input_archive_sha256": args.preverified_input_archive_sha256,
        "base_model_fingerprint": _model_fingerprint(args.model_dir),
        "fit_all": args.fit_all,
        "epochs": args.epochs,
        # Beam changes the physical microbatch by GPU tier while preserving
        # the same optimizer-level batch. This keeps those retries resumable.
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "objective": args.objective,
        "target_temperature": args.target_temperature,
        "freeze_layers": args.freeze_layers,
        "freeze_embeddings": args.freeze_embeddings,
        "amp": args.amp,
        "seed": args.seed,
        "label_profile": LABEL_PROFILE,
    }


def _read_json_object_strict(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is corrupt: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _checkpoint_is_complete(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    weights = any(directory.glob("*.safetensors")) or any(
        directory.glob("pytorch_model*.bin")
    )
    return (
        weights
        and (directory / "config.json").is_file()
        and (directory / "tokenizer_config.json").is_file()
        and (directory / "training_metrics.json").is_file()
        and (directory / "training_state.pt").is_file()
    )


def _recover_checkpoint_backup(directory: Path) -> None:
    """Restore the previous committed checkpoint after an interrupted swap."""

    backup = directory.with_name(f".{directory.name}.backup")
    if not directory.exists() and backup.exists():
        os.replace(backup, directory)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    body = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, body)


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty JSON object")
    output: dict[str, dict[str, Any]] = {}
    for raw_id, raw_record in payload.items():
        question_id = str(raw_id)
        if not question_id.strip() or not isinstance(raw_record, dict):
            raise ValueError("questions contain an invalid record")
        question = raw_record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} must be a non-blank string")
        output[question_id] = raw_record
    return output


def _load_ids(path: Path) -> list[str]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        raw_ids: object = list(payload)
    else:
        raw_ids = payload
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"ID file must be a non-empty JSON list or object: {path}")
    ids = [str(value) for value in raw_ids]
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"ID file contains blank or duplicate IDs: {path}")
    return ids


def _load_candidate_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be an object")
                question_id = row.get("question_id", row.get("id"))
                if not isinstance(question_id, str) or not question_id.strip():
                    raise ValueError(f"{path}:{line_number} has no question_id")
                hits = row.get("hits")
                if not isinstance(hits, list) or not hits:
                    raise ValueError(f"{path}:{line_number} has no candidate hits")
                if question_id in output:
                    raise ValueError(f"duplicate candidate question {question_id!r}")
                output[question_id] = row
    if not output:
        raise ValueError("candidate inputs are empty")
    return output


def _question_key(question: str) -> str:
    normalized = unicodedata.normalize("NFC", question).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", text).casefold())


def reference_overlap(reference: str, parent_text: str) -> float:
    """Return a deterministic METEOR-style exact multiset overlap score."""

    reference_tokens = _tokens(reference)
    parent_tokens = _tokens(parent_text)
    if not reference_tokens or not parent_tokens:
        return 0.0
    matches = sum((Counter(reference_tokens) & Counter(parent_tokens)).values())
    if not matches:
        return 0.0
    precision = matches / len(parent_tokens)
    recall = matches / len(reference_tokens)
    denominator = 0.9 * precision + 0.1 * recall
    return precision * recall / denominator if denominator else 0.0


def _resolved_parent_id(hit: dict[str, Any]) -> str | None:
    direct = hit.get("parent_id")
    metadata = hit.get("metadata")
    nested = metadata.get("parent_id") if isinstance(metadata, dict) else None
    values = [value.strip() for value in (direct, nested) if isinstance(value, str)]
    values = [value for value in values if value]
    if len(set(values)) > 1:
        return None
    return values[0] if values else None


def _retrieval_score(hit: dict[str, Any]) -> float | None:
    for name in (
        "final_score",
        "rerank_score",
        "hybrid_score",
        "dense_score",
        "score",
    ):
        value = hit.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
    return None


def collect_parent_candidates(
    row: dict[str, Any],
    store: JsonlParentStore,
    *,
    candidate_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hydrate and deduplicate verbatim parents for one candidate row."""

    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {"missing_parent_id": 0, "parent_load_error": 0}
    for rank, raw_hit in enumerate(row["hits"][:candidate_k], 1):
        if not isinstance(raw_hit, dict):
            raise ValueError("candidate hits must contain only objects")
        chunk_id = raw_hit.get("chunk_id")
        doc_id = raw_hit.get("doc_id")
        anchor_text = raw_hit.get("text")
        parent_id = _resolved_parent_id(raw_hit)
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("candidate hit has no chunk_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"candidate {chunk_id!r} has no doc_id")
        if not isinstance(anchor_text, str) or not anchor_text.strip():
            raise ValueError(f"candidate {chunk_id!r} has no anchor text")
        if parent_id is None:
            stats["missing_parent_id"] += 1
            continue
        key = (doc_id, parent_id)
        existing = hydrated.get(key)
        if existing is not None:
            existing["source_chunk_ids"].append(chunk_id)
            continue
        try:
            parent = store.get(doc_id, parent_id)
        except ParentStoreError:
            stats["parent_load_error"] += 1
            continue
        hydrated[key] = {
            "doc_id": doc_id,
            "parent_id": parent_id,
            "parent_text": parent.text,
            "law_name": parent.law_name,
            "article": parent.article,
            "source": parent.source,
            "source_chunk_ids": [chunk_id],
            "anchor_text": anchor_text.strip(),
            "candidate_rank": rank,
            "retrieval_score": _retrieval_score(raw_hit),
        }
    return list(hydrated.values()), stats


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _derive_split_rows(
    *,
    split: str,
    ids: Sequence[str],
    questions: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    store: JsonlParentStore,
    candidate_k: int,
    positives_per_query: int,
    hard_negatives: int,
    min_positive_overlap: float,
    min_label_margin: float,
    negative_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_scores: list[float] = []
    margins: list[float] = []
    discarded = Counter()
    hydration = Counter()
    for question_id in ids:
        record = questions[question_id]
        reference = record.get("answer")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"labeled question {question_id!r} has no answer")
        parent_candidates, parent_stats = collect_parent_candidates(
            candidates[question_id],
            store,
            candidate_k=candidate_k,
        )
        hydration.update(parent_stats)
        if len(parent_candidates) < 2:
            discarded["fewer_than_two_parents"] += 1
            continue
        for parent in parent_candidates:
            parent["reference_overlap"] = reference_overlap(
                reference,
                parent["parent_text"],
            )
        ranked = sorted(
            parent_candidates,
            key=lambda parent: (
                -parent["reference_overlap"],
                parent["candidate_rank"],
                parent["doc_id"],
                parent["parent_id"],
            ),
        )
        best = float(ranked[0]["reference_overlap"])
        second = float(ranked[1]["reference_overlap"])
        margin = best - second
        best_scores.append(best)
        margins.append(margin)
        if best < min_positive_overlap:
            discarded["positive_below_threshold"] += 1
            continue
        if margin < min_label_margin:
            discarded["margin_below_threshold"] += 1
            continue
        positives = ranked[:positives_per_query]
        positive_keys = {
            (parent["doc_id"], parent["parent_id"]) for parent in positives
        }
        weakest_positive = min(
            float(parent["reference_overlap"]) for parent in positives
        )
        negatives = [
            parent
            for parent in parent_candidates
            if (parent["doc_id"], parent["parent_id"]) not in positive_keys
            and float(parent["reference_overlap"]) <= weakest_positive - negative_gap
        ]
        negatives.sort(
            key=lambda parent: (
                parent["candidate_rank"],
                -float(parent["reference_overlap"]),
                parent["doc_id"],
                parent["parent_id"],
            )
        )
        negatives = negatives[:hard_negatives]
        if len(negatives) < hard_negatives:
            discarded["insufficient_unambiguous_negatives"] += 1
            continue
        common = {
            "schema_version": SCHEMA_VERSION,
            "label_profile": LABEL_PROFILE,
            "split": split,
            "question_id": question_id,
            "question": record["question"],
            "best_parent_overlap": best,
            "overlap_margin": margin,
        }
        for label, selected in ((1, positives), (0, negatives)):
            for parent in selected:
                rows.append(
                    {
                        **common,
                        "label": label,
                        **parent,
                    }
                )
    grouped = {row["question_id"] for row in rows}
    return rows, {
        "requested_questions": len(ids),
        "emitted_questions": len(grouped),
        "discarded_questions": len(ids) - len(grouped),
        "discard_rate": (len(ids) - len(grouped)) / len(ids),
        "discard_reasons": dict(sorted(discarded.items())),
        "output_rows": len(rows),
        "positive_rows": sum(row["label"] == 1 for row in rows),
        "negative_rows": sum(row["label"] == 0 for row in rows),
        "best_parent_overlap": _percentiles(best_scores),
        "best_second_margin": _percentiles(margins),
        "hydration": dict(sorted(hydration.items())),
    }


def _reject_output_collisions(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    resolved_inputs = {path.resolve() for path in inputs}
    for output in outputs:
        if output.resolve() in resolved_inputs:
            raise ValueError(f"output must not overwrite input: {output}")


def run_split_ids(args: argparse.Namespace) -> list[Path]:
    """Split organizer-ordered questions into first-N eval and remaining train."""

    questions = _load_questions(args.questions)
    ordered_ids = list(questions)
    if args.eval_count >= len(ordered_ids):
        raise ValueError("eval-count must be smaller than the question count")
    eval_ids = ordered_ids[: args.eval_count]
    train_ids = ordered_ids[args.eval_count :]
    outputs = [args.train_output, args.eval_output, args.audit_output]
    _reject_output_collisions([args.questions], outputs)
    eval_keys = {_question_key(questions[value]["question"]) for value in eval_ids}
    exact_duplicate_train_ids = [
        value
        for value in train_ids
        if _question_key(questions[value]["question"]) in eval_keys
    ]
    _write_json(args.train_output, train_ids)
    _write_json(args.eval_output, eval_ids)
    _write_json(
        args.audit_output,
        {
            "schema_version": SCHEMA_VERSION,
            "ordering": "organizer_json_insertion_order",
            "question_count": len(ordered_ids),
            "eval_count": len(eval_ids),
            "train_count": len(train_ids),
            "train_normalized_exact_eval_duplicates": len(exact_duplicate_train_ids),
            "duplicate_train_ids": exact_duplicate_train_ids,
            "note": (
                "prepare removes these normalized-exact train duplicates before "
                "deriving labels"
            ),
            "source": {
                "path": str(args.questions),
                "sha256": _sha256(args.questions),
            },
            "outputs": {
                "train": str(args.train_output),
                "eval": str(args.eval_output),
            },
        },
    )
    return outputs


def run_prepare(args: argparse.Namespace) -> list[Path]:
    """Prepare strict parent-level labels and a preflight quality report."""

    questions = _load_questions(args.questions)
    candidates = _load_candidate_rows(args.candidates)
    train_ids = _load_ids(args.train_ids)
    eval_ids = _load_ids(args.eval_ids)
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    overlap = train_set & eval_set
    if overlap:
        raise ValueError(f"train/eval IDs overlap: {sorted(overlap)[:5]}")
    requested = train_set | eval_set
    missing_questions = requested - set(questions)
    missing_candidates = requested - set(candidates)
    if missing_questions or missing_candidates:
        raise ValueError(
            "split inputs are incomplete: "
            f"missing_questions={sorted(missing_questions)[:5]} "
            f"missing_candidates={sorted(missing_candidates)[:5]}"
        )
    eval_keys = {_question_key(questions[value]["question"]) for value in eval_ids}
    effective_train_ids = [
        value
        for value in train_ids
        if _question_key(questions[value]["question"]) not in eval_keys
    ]
    removed_exact = [value for value in train_ids if value not in effective_train_ids]
    if not effective_train_ids:
        raise ValueError("all train IDs were removed by normalized eval duplicates")
    outputs = [args.train_output, args.eval_output, args.audit_output]
    _reject_output_collisions(
        [args.questions, *args.candidates, args.train_ids, args.eval_ids],
        outputs,
    )
    store = JsonlParentStore(args.parents_dir, max_cached_documents=256)
    settings = {
        "candidate_k": args.candidate_k,
        "positives_per_query": args.positives_per_query,
        "hard_negatives": args.hard_negatives,
        "min_positive_overlap": args.min_positive_overlap,
        "min_label_margin": args.min_label_margin,
        "negative_gap": args.negative_gap,
    }
    train_rows, train_report = _derive_split_rows(
        split="train",
        ids=effective_train_ids,
        questions=questions,
        candidates=candidates,
        store=store,
        **settings,
    )
    eval_rows, eval_report = _derive_split_rows(
        split="eval",
        ids=eval_ids,
        questions=questions,
        candidates=candidates,
        store=store,
        **settings,
    )
    if not train_rows or not eval_rows:
        raise ValueError("label preparation produced an empty train or eval split")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "label_profile": LABEL_PROFILE,
        "strict_isolation": {
            "input_train_ids": len(train_ids),
            "input_eval_ids": len(eval_ids),
            "id_overlap": 0,
            "normalized_question_overlap": 0,
            "removed_train_exact_eval_duplicates": len(removed_exact),
            "removed_train_ids": removed_exact,
        },
        "settings": settings,
        "train": train_report,
        "eval": eval_report,
        "organizer_exact_overlay_policy": "excluded; handled outside this model",
        "sources": {
            "questions": {
                "path": str(args.questions),
                "sha256": _sha256(args.questions),
            },
            "candidates": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.candidates
            ],
            "train_ids": {
                "path": str(args.train_ids),
                "sha256": _sha256(args.train_ids),
            },
            "eval_ids": {
                "path": str(args.eval_ids),
                "sha256": _sha256(args.eval_ids),
            },
            "parents_dir": str(args.parents_dir),
        },
        "outputs": {
            "train": str(args.train_output),
            "eval": str(args.eval_output),
        },
    }
    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.eval_output, eval_rows)
    _write_json(args.audit_output, audit)
    return outputs


def _load_training_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be an object")
                required = {
                    "schema_version",
                    "question_id",
                    "question",
                    "doc_id",
                    "parent_id",
                    "parent_text",
                    "label",
                }
                missing = required - set(row)
                if missing:
                    raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
                if row["schema_version"] != SCHEMA_VERSION:
                    raise ValueError(f"{path}:{line_number} has wrong schema version")
                if row["label"] not in (0, 1):
                    raise ValueError(f"{path}:{line_number} label must be 0 or 1")
                for field in (
                    "question_id",
                    "question",
                    "doc_id",
                    "parent_id",
                    "parent_text",
                ):
                    if not isinstance(row[field], str) or not row[field].strip():
                        raise ValueError(
                            f"{path}:{line_number} field {field!r} is blank"
                        )
                if "answer" in row or "reference_answer" in row:
                    raise ValueError(
                        f"{path}:{line_number} leaks a reference answer field"
                    )
                rows.append(row)
    if not rows:
        raise ValueError("prepared training data are empty")
    return rows


def audit_training_rows(
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate pair uniqueness, group balance, and strict split isolation."""

    def summarize(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
        pairs: set[tuple[str, str, str]] = set()
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        questions: dict[str, str] = {}
        for row in rows:
            question_id = row["question_id"]
            pair = (question_id, row["doc_id"], row["parent_id"])
            if pair in pairs:
                raise ValueError(f"{label} repeats pair {pair!r}")
            pairs.add(pair)
            prior = questions.setdefault(question_id, row["question"])
            if prior != row["question"]:
                raise ValueError(f"{label} changes question text for {question_id}")
            groups[question_id].append(row)
        for question_id, group in groups.items():
            labels = {row["label"] for row in group}
            if labels != {0, 1}:
                raise ValueError(
                    f"{label} question {question_id!r} needs positive and negative rows"
                )
        overlaps = [
            float(group[0].get("best_parent_overlap", 0.0)) for group in groups.values()
        ]
        margins = [
            float(group[0].get("overlap_margin", 0.0)) for group in groups.values()
        ]
        return {
            "rows": len(rows),
            "questions": len(groups),
            "positive_rows": sum(row["label"] == 1 for row in rows),
            "negative_rows": sum(row["label"] == 0 for row in rows),
            "best_parent_overlap": _percentiles(overlaps),
            "best_second_margin": _percentiles(margins),
            "question_text": questions,
        }

    train = summarize(train_rows, "train")
    evaluation = summarize(eval_rows, "eval")
    train_ids = set(train["question_text"])
    eval_ids = set(evaluation["question_text"])
    id_overlap = train_ids & eval_ids
    if id_overlap:
        raise ValueError(f"train/eval question IDs overlap: {sorted(id_overlap)[:5]}")
    train_keys = {
        _question_key(question) for question in train["question_text"].values()
    }
    eval_keys = {
        _question_key(question) for question in evaluation["question_text"].values()
    }
    normalized_overlap = train_keys & eval_keys
    if normalized_overlap:
        raise ValueError(
            f"train/eval normalized questions overlap: {sorted(normalized_overlap)[:3]}"
        )
    train.pop("question_text")
    evaluation.pop("question_text")
    return {
        "schema_version": SCHEMA_VERSION,
        "strict_isolation": {
            "id_overlap": 0,
            "normalized_question_overlap": 0,
        },
        "train": train,
        "eval": evaluation,
    }


def audit_fit_all_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate one fit-all dataset without inventing an evaluation split."""

    pairs: set[tuple[str, str, str]] = set()
    groups: dict[str, set[int]] = defaultdict(set)
    question_text: dict[str, str] = {}
    for row in rows:
        question_id = row["question_id"]
        pair = (question_id, row["doc_id"], row["parent_id"])
        if pair in pairs:
            raise ValueError(f"fit-all data repeat pair {pair!r}")
        pairs.add(pair)
        prior = question_text.setdefault(question_id, row["question"])
        if prior != row["question"]:
            raise ValueError(f"fit-all changes question text for {question_id}")
        groups[question_id].add(row["label"])
    invalid = [
        question_id for question_id, labels in groups.items() if labels != {0, 1}
    ]
    if invalid:
        raise ValueError(
            f"fit-all questions need positive and negative rows: {sorted(invalid)[:5]}"
        )
    return {
        "fit_all": True,
        "rows": len(rows),
        "questions": len(groups),
        "positive_rows": sum(row["label"] == 1 for row in rows),
        "negative_rows": sum(row["label"] == 0 for row in rows),
    }


def run_audit(args: argparse.Namespace) -> list[Path]:
    train_rows = _load_training_rows(args.train_data)
    eval_rows = _load_training_rows(args.eval_data)
    report = audit_training_rows(train_rows, eval_rows)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if args.output is None:
        return []
    _write_json(args.output, report)
    return [args.output]


def _validate_model_length(max_length: int) -> None:
    if max_length > MAX_MODEL_LENGTH:
        raise ValueError(
            f"max length {max_length} exceeds the hard limit {MAX_MODEL_LENGTH}"
        )


def _require_local_model(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {path}")
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        if not (resolved / name).is_file():
            raise FileNotFoundError(f"local model is missing {name}: {resolved}")
    return resolved


def _load_model_stack(model_dir: Path, device_name: str) -> tuple[Any, Any, Any]:
    """Load Roberta classification weights locally and return torch as well."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import PhobertTokenizer, RobertaForSequenceClassification

    resolved = _require_local_model(model_dir)
    print(
        json.dumps(
            {"event": "model_load_progress", "phase": "tokenizer_start"},
            sort_keys=True,
        ),
        flush=True,
    )
    tokenizer = PhobertTokenizer.from_pretrained(
        str(resolved),
        local_files_only=True,
    )
    print(
        json.dumps(
            {"event": "model_load_progress", "phase": "model_cpu_start"},
            sort_keys=True,
        ),
        flush=True,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        str(resolved),
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    print(
        json.dumps(
            {"event": "model_load_progress", "phase": "model_cpu_ready"},
            sort_keys=True,
        ),
        flush=True,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    print(
        json.dumps(
            {"event": "model_load_progress", "phase": "model_device_start"},
            sort_keys=True,
        ),
        flush=True,
    )
    model.to(device)
    print(
        json.dumps(
            {"event": "model_load_progress", "phase": "model_device_ready"},
            sort_keys=True,
        ),
        flush=True,
    )
    return torch, tokenizer, model


def freeze_lower_layers(
    model: Any,
    *,
    freeze_layers: int,
    freeze_embeddings: bool,
) -> dict[str, int]:
    """Freeze the requested lower RoBERTa layers and report parameter counts."""

    layers = model.roberta.encoder.layer
    if freeze_layers > len(layers):
        raise ValueError(
            f"freeze_layers={freeze_layers} exceeds encoder depth {len(layers)}"
        )
    if freeze_embeddings:
        for parameter in model.roberta.embeddings.parameters():
            parameter.requires_grad = False
    for layer in layers[:freeze_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def _make_loader(
    torch: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> Any:
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, values: Sequence[dict[str, Any]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["question"] for row in batch],
            [row["parent_text"] for row in batch],
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_overflowing_tokens=False,
            return_tensors="pt",
            verbose=False,
        )
        encoded["labels"] = torch.tensor(
            [float(row["label"]) for row in batch],
            dtype=torch.float32,
        )
        encoded["question_ids"] = [row["question_id"] for row in batch]
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        PairDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def group_training_rows(
    rows: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group candidate pairs by question while preserving input order."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question_id = row["question_id"]
        grouped.setdefault(question_id, []).append(row)
    for question_id, candidates in grouped.items():
        labels = [int(row["label"]) for row in candidates]
        if not any(labels) or all(labels):
            raise ValueError(
                f"listwise question {question_id!r} needs positive and negative rows"
            )
    return list(grouped.values())


def _make_group_loader(
    torch: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> Any:
    """Build batches of complete question groups for a listwise objective."""

    class GroupDataset(torch.utils.data.Dataset):
        def __init__(self, values: Sequence[Sequence[dict[str, Any]]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> Sequence[dict[str, Any]]:
            return self.values[index]

    def collate(
        groups: Sequence[Sequence[dict[str, Any]]],
    ) -> dict[str, Any]:
        flattened = [row for group in groups for row in group]
        encoded = tokenizer(
            [row["question"] for row in flattened],
            [row["parent_text"] for row in flattened],
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_overflowing_tokens=False,
            return_tensors="pt",
            verbose=False,
        )
        encoded["labels"] = torch.tensor(
            [float(row["label"]) for row in flattened],
            dtype=torch.float32,
        )
        encoded["target_scores"] = torch.tensor(
            [float(row.get("reference_overlap", row["label"])) for row in flattened],
            dtype=torch.float32,
        )
        encoded["question_ids"] = [row["question_id"] for row in flattened]
        encoded["group_sizes"] = [len(group) for group in groups]
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        GroupDataset(group_training_rows(rows)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _move_batch(torch: Any, batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"labels", "target_scores", "question_ids", "group_sizes"}
        and isinstance(value, torch.Tensor)
    }


def _listwise_loss(
    torch: Any,
    logits: Any,
    target_scores: Any,
    group_sizes: Sequence[int],
    *,
    temperature: float,
) -> Any:
    """Cross-entropy between predicted and graded target group distributions."""

    losses: list[Any] = []
    offset = 0
    for size in group_sizes:
        group_logits = logits[offset : offset + size]
        group_targets = target_scores[offset : offset + size]
        target_distribution = torch.softmax(group_targets / temperature, dim=0)
        losses.append(
            -(target_distribution * torch.log_softmax(group_logits, dim=0)).sum()
        )
        offset += size
    if offset != logits.shape[0] or not losses:
        raise ValueError("invalid listwise group sizes")
    return torch.stack(losses).mean()


def _evaluate_model(
    torch: Any,
    model: Any,
    loader: Any,
    *,
    device: Any,
    amp_enabled: bool,
    positive_weight: Any,
    objective: str,
    target_temperature: float,
) -> dict[str, float]:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    losses: list[float] = []
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    correct = 0
    count = 0
    with torch.no_grad():
        total_eval_batches = len(loader)
        eval_progress_interval = max(1, total_eval_batches // 20)
        for eval_step, batch in enumerate(loader, 1):
            labels = batch["labels"].to(device)
            inputs = _move_batch(torch, batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**inputs).logits.squeeze(-1)
                if objective == "listwise":
                    target_scores = batch["target_scores"].to(device)
                    loss = _listwise_loss(
                        torch,
                        logits,
                        target_scores,
                        batch["group_sizes"],
                        temperature=target_temperature,
                    )
                else:
                    loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            predictions = (logits >= 0).long().cpu().tolist()
            label_values = labels.long().cpu().tolist()
            scores = logits.float().cpu().tolist()
            correct += sum(a == b for a, b in zip(predictions, label_values))
            count += len(label_values)
            for question_id, score, label in zip(
                batch["question_ids"], scores, label_values
            ):
                grouped[question_id].append((float(score), int(label)))
            if (
                eval_step == 1
                or eval_step % eval_progress_interval == 0
                or eval_step == total_eval_batches
            ):
                print(
                    json.dumps(
                        {
                            "event": "evaluation_progress",
                            "batch": eval_step,
                            "batches": total_eval_batches,
                            "percent": round(100 * eval_step / total_eval_batches, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    ranking_hits = 0
    pairwise_total = 0
    pairwise_correct = 0
    for pairs in grouped.values():
        positives = [score for score, label in pairs if label == 1]
        negatives = [score for score, label in pairs if label == 0]
        if max(positives) > max(negatives):
            ranking_hits += 1
        for positive in positives:
            for negative in negatives:
                pairwise_total += 1
                pairwise_correct += positive > negative
    return {
        "loss": sum(losses) / len(losses),
        "binary_accuracy": correct / count,
        "query_ranking_accuracy": ranking_hits / len(grouped),
        "pairwise_accuracy": pairwise_correct / pairwise_total,
    }


def _save_checkpoint(
    torch: Any,
    model: Any,
    tokenizer: Any,
    directory: Path,
    state: dict[str, Any],
) -> None:
    """Write a complete checkpoint in staging, then atomically swap it in."""

    directory.parent.mkdir(parents=True, exist_ok=True)
    backup = directory.with_name(f".{directory.name}.backup")
    if backup.exists():
        if directory.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, directory)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.",
            suffix=".tmp",
            dir=str(directory.parent),
        )
    )
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        serializable_state = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "optimizer",
                "scheduler",
                "scaler",
                "python_rng_state",
                "torch_rng_state",
                "cuda_rng_states",
                "loader_generator_state",
            }
        }
        _write_json(staging / "training_metrics.json", serializable_state)
        _atomic_torch_save(torch, staging / "training_state.pt", state)
        if not _checkpoint_is_complete(staging):
            raise RuntimeError(f"staged checkpoint is incomplete: {staging}")
        # Do not guard the rename with Path.exists(). Beam Volume metadata can
        # briefly report a false negative even though the non-empty checkpoint
        # directory is still visible to rename(2). That TOCTOU race made the
        # following staging commit fail with ENOTEMPTY. Attempt the atomic move
        # directly and treat only FileNotFoundError as "no previous commit".
        previous_committed = False
        try:
            os.replace(directory, backup)
            previous_committed = True
        except FileNotFoundError:
            pass
        try:
            os.replace(staging, directory)
        except BaseException:
            if previous_committed:
                try:
                    os.replace(backup, directory)
                except OSError as rollback_error:
                    raise RuntimeError(
                        "checkpoint commit failed and rollback could not restore "
                        f"{directory}: {rollback_error}"
                    ) from rollback_error
            raise
        if previous_committed and backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_training_state(
    torch: Any,
    checkpoint: Path,
    contract: dict[str, Any],
    *,
    total_epochs: int,
) -> dict[str, Any]:
    """Load and fail closed on an incomplete, corrupt, or stale checkpoint."""

    if not _checkpoint_is_complete(checkpoint):
        raise ValueError(f"resume checkpoint is incomplete: {checkpoint}")
    state_path = checkpoint / "training_state.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"resume checkpoint state is corrupt: {state_path}: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise ValueError(f"resume checkpoint state must be a mapping: {state_path}")
    if state.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError(
            "resume checkpoint state schema is unsupported: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("training_contract") != contract:
        raise ValueError(
            "resume checkpoint contract differs from the requested training run"
        )
    try:
        epoch = int(state["epoch"])
        global_updates = int(state["global_updates"])
        updates_per_epoch = int(state["updates_per_epoch"])
        best_epoch = int(state["best_epoch"])
        best_metric = float(state["best_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint epoch metadata is corrupt") from exc
    if not 1 <= epoch <= total_epochs:
        raise ValueError(
            "resume checkpoint epoch "
            f"{epoch} is outside requested epochs={total_epochs}"
        )
    history = state.get("history")
    if not isinstance(history, list) or [
        item.get("epoch") if isinstance(item, dict) else None for item in history
    ] != list(range(1, epoch + 1)):
        raise ValueError(
            "resume checkpoint history must contain contiguous epochs 1..saved_epoch"
        )
    if not 1 <= best_epoch <= epoch or not math.isfinite(best_metric):
        raise ValueError("resume checkpoint best-epoch metadata is corrupt")
    if updates_per_epoch <= 0 or global_updates != epoch * updates_per_epoch:
        raise ValueError(
            "resume checkpoint global update count does not match its epoch"
        )
    required = {
        "optimizer",
        "scheduler",
        "scaler",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "loader_generator_state",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"resume checkpoint state is missing fields: {missing}")
    for field in ("optimizer", "scheduler", "scaler"):
        if not isinstance(state[field], dict):
            raise ValueError(f"resume checkpoint {field} state must be a mapping")
    scheduler_step = state["scheduler"].get("last_epoch")
    if scheduler_step is not None and scheduler_step != global_updates:
        raise ValueError(
            "resume checkpoint scheduler position does not match global updates"
        )
    return state


def _resume_position(
    state: dict[str, Any],
    *,
    updates_per_epoch: int,
) -> tuple[int, int]:
    """Validate the current loader shape and return next epoch/update count."""

    saved_epoch = int(state["epoch"])
    if int(state["updates_per_epoch"]) != updates_per_epoch:
        raise ValueError(
            "resume checkpoint updates-per-epoch differs from the current loader"
        )
    expected_updates = saved_epoch * updates_per_epoch
    if int(state["global_updates"]) != expected_updates:
        raise ValueError(
            "resume checkpoint global update count does not match its epoch"
        )
    return saved_epoch + 1, expected_updates


def _direct_checkpoint(output_dir: Path, raw_name: object, label: str) -> Path:
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError(f"completed training manifest has no {label} path")
    checkpoint = output_dir / raw_name
    if checkpoint.resolve().parent != output_dir.resolve():
        raise ValueError(f"completed training manifest {label} escapes output-dir")
    if not _checkpoint_is_complete(checkpoint):
        raise ValueError(
            f"completed training {label} is missing or incomplete: {checkpoint}"
        )
    return checkpoint


def _validate_completed_run(
    torch: Any,
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    contract: dict[str, Any],
    total_epochs: int,
) -> Path:
    """Validate both final and latest checkpoints before accepting a no-op."""

    if manifest.get("status") != "COMPLETE":
        raise ValueError("completed training manifest has no COMPLETE status")
    if manifest.get("training_contract") != contract:
        raise ValueError(
            "completed output contract differs from the requested training run"
        )
    history = manifest.get("history")
    if not isinstance(history, list) or [
        item.get("epoch") if isinstance(item, dict) else None for item in history
    ] != list(range(1, total_epochs + 1)):
        raise ValueError("completed training manifest history is incomplete or corrupt")
    checkpoint = _direct_checkpoint(
        output_dir,
        manifest.get("checkpoint"),
        "best checkpoint",
    )
    latest = _direct_checkpoint(
        output_dir,
        manifest.get("latest_checkpoint"),
        "latest checkpoint",
    )
    best_state = _load_training_state(
        torch,
        checkpoint,
        contract,
        total_epochs=total_epochs,
    )
    latest_state = _load_training_state(
        torch,
        latest,
        contract,
        total_epochs=total_epochs,
    )
    try:
        manifest_best_epoch = int(manifest["best_epoch"])
        manifest_best_metric = float(manifest["best_metric"])
        manifest_updates = int(manifest["global_updates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "completed training manifest metric metadata is corrupt"
        ) from exc
    if (
        int(latest_state["epoch"]) != total_epochs
        or latest_state["history"] != history
        or int(latest_state["global_updates"]) != manifest_updates
        or int(latest_state["best_epoch"]) != manifest_best_epoch
        or float(latest_state["best_metric"]) != manifest_best_metric
    ):
        raise ValueError("completed training manifest and checkpoint-latest disagree")
    if (
        int(best_state["epoch"]) != manifest_best_epoch
        or float(best_state["best_metric"]) != manifest_best_metric
    ):
        raise ValueError("completed training manifest and best checkpoint disagree")
    return checkpoint


def run_train(args: argparse.Namespace) -> list[Path]:
    """Fine-tune the local DEk21 encoder as a binary parent cross-encoder."""

    _validate_model_length(args.max_length)
    if args.fit_all and args.eval_data:
        raise ValueError("--fit-all cannot be combined with --eval-data")
    if not args.fit_all and not args.eval_data:
        raise ValueError("--eval-data is required unless --fit-all is used")
    if args.output_dir.resolve() == args.model_dir.resolve():
        raise ValueError("output-dir must be separate from the base model")
    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "contract_hash_start",
                "train_bytes": sum(path.stat().st_size for path in args.train_data),
                "eval_bytes": sum(path.stat().st_size for path in args.eval_data),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    contract = _training_contract(args)
    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "contract_hash_complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    manifest_path = args.output_dir / "training_manifest.json"
    best_path = (
        args.output_dir / "checkpoint-final"
        if args.fit_all
        else args.output_dir / "checkpoint-best"
    )
    latest_path = args.output_dir / "checkpoint-latest"
    for checkpoint_path in (best_path, latest_path):
        _recover_checkpoint_backup(checkpoint_path)

    if manifest_path.is_file():
        if not args.resume:
            raise ValueError(
                f"completed output already exists; use --resume: {args.output_dir}"
            )
        completed = _read_json_object_strict(
            manifest_path,
            "completed training manifest",
        )
        import torch as torch_module

        checkpoint = _validate_completed_run(
            torch_module,
            completed,
            output_dir=args.output_dir,
            contract=contract,
            total_epochs=args.epochs,
        )
        print(f"PARENT_TRAIN_ALREADY_COMPLETE {manifest_path}", flush=True)
        return [checkpoint, manifest_path]

    output_has_files = args.output_dir.is_dir() and any(args.output_dir.iterdir())
    if output_has_files and not args.resume:
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    resume_checkpoint = latest_path if output_has_files and args.resume else None
    if resume_checkpoint is not None and not _checkpoint_is_complete(resume_checkpoint):
        raise ValueError(
            "output directory is incomplete and has no complete "
            f"checkpoint-latest: {args.output_dir}"
        )

    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "jsonl_load_start",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_rows = _load_training_rows(args.train_data)
    eval_rows = _load_training_rows(args.eval_data) if args.eval_data else []
    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "jsonl_load_complete",
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "audit_start",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    audit = (
        audit_training_rows(train_rows, eval_rows)
        if eval_rows
        else audit_fit_all_rows(train_rows)
    )
    print(
        json.dumps(
            {
                "event": "training_preprocess_progress",
                "phase": "audit_complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    import torch as torch_module

    random.seed(args.seed)
    torch_module.manual_seed(args.seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(args.seed)
    resume_state = (
        _load_training_state(
            torch_module,
            resume_checkpoint,
            contract,
            total_epochs=args.epochs,
        )
        if resume_checkpoint is not None
        else None
    )
    model_source = resume_checkpoint or args.model_dir
    torch, tokenizer, model = _load_model_stack(model_source, args.device)
    device = next(model.parameters()).device
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        raise ValueError("AMP is supported only with a CUDA device")
    freeze_report = freeze_lower_layers(
        model,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
    )
    loader_factory = (
        _make_group_loader if args.objective == "listwise" else _make_loader
    )
    train_loader = loader_factory(
        torch,
        tokenizer,
        train_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    eval_loader = (
        loader_factory(
            torch,
            tokenizer,
            eval_rows,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        if eval_rows
        else None
    )
    positives = sum(row["label"] == 1 for row in train_rows)
    negatives = len(train_rows) - positives
    if not positives or not negatives:
        raise ValueError("training data need positive and negative rows")
    positive_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "training_setup_ready",
                "train_batches": len(train_loader),
                "eval_batches": len(eval_loader) if eval_loader is not None else 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = 0
    global_updates = 0
    start_epoch = 1
    if resume_state is not None:
        saved_epoch = int(resume_state["epoch"])
        start_epoch, expected_updates = _resume_position(
            resume_state,
            updates_per_epoch=updates_per_epoch,
        )
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])
            train_loader.generator.set_state(resume_state["loader_generator_state"])
            random.setstate(resume_state["python_rng_state"])
            torch.set_rng_state(resume_state["torch_rng_state"])
            if torch.cuda.is_available() and resume_state["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
        except Exception as exc:
            raise ValueError(
                "resume checkpoint optimizer, scheduler, scaler, or RNG state "
                f"is corrupt or incompatible: {exc}"
            ) from exc
        history = list(resume_state["history"])
        best_metric = float(resume_state["best_metric"])
        best_epoch = int(resume_state["best_epoch"])
        global_updates = expected_updates
        if not _checkpoint_is_complete(best_path):
            raise ValueError(
                f"resume best checkpoint is missing or incomplete: {best_path}"
            )
        best_state = _load_training_state(
            torch,
            best_path,
            contract,
            total_epochs=args.epochs,
        )
        if (
            int(best_state["epoch"]) != best_epoch
            or float(best_state["best_metric"]) != best_metric
        ):
            raise ValueError("resume checkpoint-latest and best checkpoint disagree")
        print(
            json.dumps(
                {
                    "event": "training_resume_ready",
                    "saved_epoch": saved_epoch,
                    "start_epoch": start_epoch,
                    "global_updates": global_updates,
                    "total_updates": total_updates,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses: list[float] = []
        total_train_batches = len(train_loader)
        train_progress_interval = max(1, total_train_batches // 20)
        for step, batch in enumerate(train_loader, 1):
            labels = batch["labels"].to(device)
            inputs = _move_batch(torch, batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**inputs).logits.squeeze(-1)
                if args.objective == "listwise":
                    target_scores = batch["target_scores"].to(device)
                    loss = _listwise_loss(
                        torch,
                        logits,
                        target_scores,
                        batch["group_sizes"],
                        temperature=args.target_temperature,
                    )
                else:
                    loss = criterion(logits, labels)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            update = step % args.gradient_accumulation == 0 or step == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_updates += 1
            if (
                step == 1
                or step % train_progress_interval == 0
                or step == total_train_batches
            ):
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "epoch": epoch,
                            "epochs": args.epochs,
                            "batch": step,
                            "batches": total_train_batches,
                            "percent": round(100 * step / total_train_batches, 1),
                            "latest_loss": losses[-1],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
        }
        if eval_loader is not None:
            evaluation = _evaluate_model(
                torch,
                model,
                eval_loader,
                device=device,
                amp_enabled=amp_enabled,
                positive_weight=positive_weight,
                objective=args.objective,
                target_temperature=args.target_temperature,
            )
            epoch_report["eval"] = evaluation
            selection_metric = evaluation["query_ranking_accuracy"]
        else:
            selection_metric = -epoch_report["train_loss"]
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            is_new_best = True
        else:
            is_new_best = False
        state = {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "training_contract": contract,
            "epoch": epoch,
            "global_updates": global_updates,
            "updates_per_epoch": updates_per_epoch,
            "history": history,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metrics": epoch_report,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "loader_generator_state": train_loader.generator.get_state(),
        }
        if is_new_best:
            _save_checkpoint(
                torch,
                model,
                tokenizer,
                best_path,
                state,
            )
        _save_checkpoint(
            torch,
            model,
            tokenizer,
            latest_path,
            state,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "base_model": str(args.model_dir),
        "local_files_only": True,
        "model_class": "RobertaForSequenceClassification",
        "tokenizer_class": type(tokenizer).__name__,
        "max_length": args.max_length,
        "objective": args.objective,
        "target_temperature": args.target_temperature,
        "device": str(device),
        "amp": amp_enabled,
        "freeze_layers": args.freeze_layers,
        "freeze_embeddings": args.freeze_embeddings,
        "parameter_counts": freeze_report,
        "fit_all": args.fit_all,
        "audit": audit,
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "global_updates": global_updates,
        "checkpoint": best_path.name,
        "latest_checkpoint": latest_path.name,
        "training_contract": contract,
        "organizer_exact_overlay_policy": "excluded; handled separately",
        "inputs": {
            "train": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.train_data
            ],
            "eval": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.eval_data
            ],
        },
    }
    _write_json(manifest_path, manifest)
    return [best_path, manifest_path]


def _score_pairs(
    torch: Any,
    tokenizer: Any,
    model: Any,
    pairs: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    amp_enabled: bool,
) -> list[float]:
    device = next(model.parameters()).device
    model.eval()
    scores: list[float] = []
    total_batches = math.ceil(len(pairs) / batch_size)
    progress_interval = max(1, total_batches // 20)
    with torch.no_grad():
        for batch_index, offset in enumerate(range(0, len(pairs), batch_size), 1):
            batch = pairs[offset : offset + batch_size]
            encoded = tokenizer(
                [row["question"] for row in batch],
                [row["parent_text"] for row in batch],
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_overflowing_tokens=False,
                return_tensors="pt",
                verbose=False,
            )
            encoded = {
                key: value.to(device, non_blocking=True)
                for key, value in encoded.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**encoded).logits.squeeze(-1)
            scores.extend(float(value) for value in logits.float().cpu())
            if (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                print(
                    json.dumps(
                        {
                            "event": "scoring_progress",
                            "batch": batch_index,
                            "batches": total_batches,
                            "percent": round(100 * batch_index / total_batches, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return scores


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def run_score(args: argparse.Namespace) -> list[Path]:
    """Score candidate parents without applying organizer answer overlays."""

    _validate_model_length(args.max_length)
    questions = _load_questions(args.questions)
    candidates = _load_candidate_rows(args.candidates)
    _reject_output_collisions(
        [args.questions, *args.candidates],
        [args.output],
    )
    # Public scoring touches thousands of small parent files in retrieval order.
    # Keeping only 256 documents causes severe cache churn on network-backed
    # volumes (Modal/Beam) and used to leave this stage silent for many minutes.
    candidate_doc_ids = {
        hit.get("doc_id")
        for row in candidates.values()
        for hit in row.get("hits", [])[: args.candidate_k]
        if isinstance(hit, dict)
        and isinstance(hit.get("doc_id"), str)
        and hit.get("doc_id", "").strip()
    }
    store = JsonlParentStore(
        args.parents_dir,
        max_cached_documents=max(256, len(candidate_doc_ids)),
    )
    pairs: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    hydration = Counter()
    total_questions = len(candidates)
    hydration_interval = max(1, total_questions // 20)
    for question_index, (question_id, row) in enumerate(candidates.items(), 1):
        record = questions.get(question_id)
        if record is None:
            raise ValueError(f"questions are missing candidate ID {question_id!r}")
        parents, stats = collect_parent_candidates(
            row,
            store,
            candidate_k=args.candidate_k,
        )
        hydration.update(stats)
        if not parents:
            raise ValueError(f"question {question_id!r} has no loadable parents")
        grouped[question_id] = parents
        for parent in parents:
            pairs.append(
                {
                    "question_id": question_id,
                    "question": record["question"],
                    **parent,
                }
            )
        if (
            question_index == 1
            or question_index % hydration_interval == 0
            or question_index == total_questions
        ):
            print(
                json.dumps(
                    {
                        "event": "candidate_hydration_progress",
                        "question": question_index,
                        "questions": total_questions,
                        "pairs": len(pairs),
                        "cached_document_capacity": max(
                            256, len(candidate_doc_ids)
                        ),
                        "percent": round(
                            100 * question_index / total_questions, 1
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch, tokenizer, model = _load_model_stack(args.checkpoint, args.device)
    device = next(model.parameters()).device
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        raise ValueError("AMP is supported only with a CUDA device")
    scores = _score_pairs(
        torch,
        tokenizer,
        model,
        pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        amp_enabled=amp_enabled,
    )
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair, score in zip(pairs, scores):
        by_question[pair["question_id"]].append(
            {
                key: value
                for key, value in pair.items()
                if key not in {"question", "question_id"}
            }
            | {
                "crossencoder_logit": score,
                "crossencoder_probability": _sigmoid(score),
            }
        )
    output_rows: list[dict[str, Any]] = []
    for question_id in candidates:
        ranked = sorted(
            by_question[question_id],
            key=lambda parent: (
                -parent["crossencoder_logit"],
                parent["candidate_rank"],
                parent["doc_id"],
                parent["parent_id"],
            ),
        )
        for rank, parent in enumerate(ranked, 1):
            parent["rank"] = rank
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "question_id": question_id,
                "parents": ranked,
            }
        )
    _write_jsonl(args.output, output_rows)
    manifest_path = args.output.with_name(f"{args.output.stem}.manifest.json")
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(args.checkpoint),
            "local_files_only": True,
            "question_count": len(output_rows),
            "pair_count": len(pairs),
            "candidate_k": args.candidate_k,
            "max_length": args.max_length,
            "device": str(device),
            "amp": amp_enabled,
            "hydration": dict(sorted(hydration.items())),
            "organizer_exact_overlay_policy": "excluded; handled separately",
            "inputs": {
                "questions": {
                    "path": str(args.questions),
                    "sha256": _sha256(args.questions),
                },
                "candidates": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in args.candidates
                ],
            },
            "output": str(args.output),
        },
    )
    return [args.output, manifest_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "split-ids":
            written = run_split_ids(args)
        elif args.command == "prepare":
            written = run_prepare(args)
        elif args.command == "audit":
            written = run_audit(args)
        elif args.command == "train":
            written = run_train(args)
        elif args.command == "score":
            written = run_score(args)
        else:  # pragma: no cover - argparse enforces the choices.
            raise AssertionError(args.command)
    except (
        ImportError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"parent cross-encoder error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
