"""Fine-tune and strictly evaluate Task1's BGE document reranker on GPU.

This runner consumes the leakage-safe P12 records.  It deliberately keeps the
pretrained model directory read-only: every selected checkpoint is written
under the supplied artifacts directory.  Importing this module is model-free;
Torch and Transformers are imported only by the GPU backend at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir_document_candidates import (  # noqa: E402
    DocumentEvidence,
    LegalIRDocumentCandidate,
    final_document_ranking,
    final_submission_documents,
    format_document_evidence,
    union_document_candidates,
)

DEFAULT_TRAIN = Path("data/raw/btc/LegalIR/train.json")
DEFAULT_FOLDS = Path("artifacts/task1/evaluation/strict_cv_v2/folds.json")
DEFAULT_NEGATIVES = Path("artifacts/task1/training/negatives")
DEFAULT_CANDIDATES = Path("artifacts/task1/train500_dense200_predictions.jsonl")
DEFAULT_OUTPUT = Path("artifacts/task1/models/bge_reranker_finetune")
ABLATATIONS = (
    "baseline",
    "easy-random",
    "semi-hard",
    "semi-hard-plus-hard",
    "same-law-boosted",
)
NEGATIVE_TYPES = {
    "easy-random": {"easy", "easy_random"},
    "semi-hard": {"semi_hard", "lexical_confuser"},
    "semi-hard-plus-hard": {
        "semi_hard",
        "lexical_confuser",
        "hard_false_positive",
        "semantic_confuser",
    },
    "same-law-boosted": {
        "semi_hard",
        "lexical_confuser",
        "hard_false_positive",
        "semantic_confuser",
        "same_law",
    },
}


@dataclass(frozen=True)
class TrainingExample:
    """One document-relevance example produced from a P12 pair record."""

    query_id: str
    query: str
    document_id: str
    document: str
    label: float
    negative_type: str | None = None


@dataclass(frozen=True)
class FoldDataset:
    """Validated fold-specific training records and their deterministic hash."""

    fold: int
    pair_count: int
    examples: tuple[TrainingExample, ...]
    query_ids: tuple[str, ...]
    dataset_sha256: str
    negative_type_counts: dict[str, int]
    skipped_missing_evidence: int


class TrainingBackend(Protocol):
    """Runtime boundary used by the real GPU backend and local mock tests."""

    @property
    def model_revision(self) -> str:
        """Return a reproducible base/checkpoint revision description."""

    def prepare_training(
        self, examples: Sequence[TrainingExample], epochs: int
    ) -> None:
        """Prepare optimizer/schedule for a finite training run."""

    def fit_epoch(self, examples: Sequence[TrainingExample]) -> float:
        """Train exactly one epoch and return mean batch loss."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one document score per supplied document."""

    def save_checkpoint(self, directory: Path, metadata: dict[str, Any]) -> None:
        """Persist an independent checkpoint without touching the base model."""

    def close(self) -> None:
        """Release optional GPU resources."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the P13 CLI without importing model dependencies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--negatives-dir", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--base-model", default="models/reranker")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ablation",
        choices=[*ABLATATIONS, "all"],
        default="semi-hard-plus-hard",
        help="A=baseline, B=easy-random, C=semi-hard, D=semi-hard-plus-hard.",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        choices=(50, 100, 150, 200),
        default=200,
        help="Fixed document candidate depth shared by baseline and tuned BGE.",
    )
    parser.add_argument("--evidence-limit", type=int, choices=(1, 2), default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=_positive_int, default=2)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=4)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--warmup-ratio", type=_probability, default=0.1)
    parser.add_argument("--max-training-pairs", type=_nonnegative_int, default=0)
    parser.add_argument("--max-validation-queries", type=_nonnegative_int, default=0)
    parser.add_argument("--same-law-boost", type=_positive_int, default=2)
    parser.add_argument(
        "--selection-policy", choices=("fixed_epochs",), default="fixed_epochs",
        help="Score the outer validation fold once after fixed epochs.",
    )
    parser.add_argument("--early-stopping-patience", type=_nonnegative_int, default=0,
                        help="Deprecated compatibility option; never selects on outer validation.")
    parser.add_argument("--folds-to-run", type=int, nargs="+", choices=range(5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Mark output non-promotable (used by the real GPU smoke).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable CUDA FP16 mixed precision.",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--allow-partial-oof",
        action="store_true",
        help=(
            "Permit a candidate cache that does not cover all 7,000 strict-CV "
            "validation IDs. Such a run is diagnostic only and cannot promote."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate folds/P12 data and write manifests without loading a model.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Publish small completion metadata atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def load_fold_map(path: Path) -> dict[str, int]:
    """Load and validate the P1/P2 grouped five-fold artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "legal-ir-strict-cv-v2":
        raise ValueError(f"{path} is not a legal-ir-strict-cv-v2 artifact")
    mapping: dict[str, int] = {}
    for item in payload.get("folds", []):
        fold = item.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError("strict fold number must be an integer")
        for query_id in item.get("validation_ids", []):
            query_id = str(query_id)
            if query_id in mapping:
                raise ValueError(f"query ID appears in more than one fold: {query_id}")
            mapping[query_id] = fold
    if len(set(mapping.values())) != 5:
        raise ValueError("P13 requires exactly five validation folds")
    if not mapping:
        raise ValueError("strict folds contain no validation IDs")
    return mapping


def load_train(path: Path) -> dict[str, dict[str, Any]]:
    """Load the organizer mapping while retaining only valid labeled rows."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--train must be the LegalIR object mapping")
    output: dict[str, dict[str, Any]] = {}
    for query_id, item in raw.items():
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answers = item.get("answer")
        if not question or not isinstance(answers, list) or not answers:
            continue
        gold = [str(answer).strip() for answer in answers if str(answer).strip()]
        if len(gold) != len(set(gold)):
            raise ValueError(f"duplicate gold document at query {query_id}")
        if gold:
            output[str(query_id)] = {"question": question, "gold": gold}
    if not output:
        raise ValueError("--train contains no valid labeled LegalIR records")
    return output


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input artifact not found: {path}")
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records = payload.get("predictions", payload.get("records", []))
        return records if isinstance(records, list) else []
    raise ValueError(f"unsupported candidate artifact: {path}")


def _iter_json_records(path: Path):
    """Stream JSONL records for fold-local validation/building."""

    if not path.is_file():
        raise FileNotFoundError(f"input artifact not found: {path}")
    if path.suffix.casefold() != ".jsonl":
        yield from _load_json_records(path)
        return
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} must be an object")
            yield record


def _numeric_score(item: dict[str, Any]) -> float | None:
    for key in ("rerank_score", "final_score", "dense_score", "score"):
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                return float(value)
    return None


def load_document_candidates(
    path: Path,
    *,
    candidate_depth: int,
    evidence_limit: int,
) -> dict[str, list[LegalIRDocumentCandidate]]:
    """Collapse cached chunk rankings once into stable document candidates."""

    output: dict[str, list[LegalIRDocumentCandidate]] = {}
    for record in _load_json_records(path):
        query_id = str(record.get("question_id", record.get("id", ""))).strip()
        if not query_id:
            continue
        hits = record.get("hits", record.get("documents", []))
        if not isinstance(hits, list):
            continue
        evidence: list[DocumentEvidence] = []
        for index, raw in enumerate(hits, 1):
            if not isinstance(raw, dict):
                continue
            doc_id = str(
                raw.get("doc_id", raw.get("document_id", raw.get("id", "")))
            ).strip()
            nested = raw.get("evidence", [])
            nested_text = "\n\n".join(
                str(part.get("text", "")).strip()
                for part in nested
                if isinstance(part, dict) and str(part.get("text", "")).strip()
            ) if isinstance(nested, list) else ""
            text = str(raw.get("text", "")).strip() or nested_text
            if not doc_id or not text:
                continue
            metadata = raw.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            for key in ("law_name", "title", "name", "article", "clause", "point"):
                if raw.get(key) not in (None, ""):
                    metadata[key] = raw[key]
            rank = raw.get("rank", index)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                rank = index
            evidence.append(
                DocumentEvidence(
                    doc_id=doc_id,
                    evidence_id=str(raw.get("chunk_id", raw.get("id", f"{doc_id}:{index}"))),
                    text=text,
                    source="cached_candidates",
                    rank=rank,
                    score=_numeric_score(raw),
                    is_parent=bool(raw.get("is_parent", False)),
                    metadata=metadata,
                )
            )
        if evidence:
            output[query_id] = union_document_candidates(
                {"cached_candidates": evidence},
                candidate_depth=candidate_depth,
                evidence_limit=evidence_limit,
            )
    if not output:
        raise ValueError("candidate cache contains no usable document evidence")
    return output


def load_candidate_query_ids(path: Path) -> set[str]:
    """Stream only candidate query IDs for model-free dry-run validation."""

    query_ids: set[str] = set()
    for record in _iter_json_records(path):
        query_id = str(record.get("question_id", record.get("id", ""))).strip()
        if query_id:
            query_ids.add(query_id)
    if not query_ids:
        raise ValueError("candidate cache contains no query IDs")
    return query_ids


def _evidence_texts(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) == 2:
            break
    return tuple(result)


def _training_document(doc_id: str, evidence: Any, title: str | None = None) -> str:
    texts = _evidence_texts(evidence)
    candidate = LegalIRDocumentCandidate(
        doc_id=doc_id,
        source_ranks={},
        source_scores={},
        evidence_chunk_ids=tuple(
            f"{doc_id}:evidence:{index}" for index in range(1, len(texts) + 1)
        ),
        evidence_texts=texts,
        best_child_rank=None,
        best_parent_rank=None,
        first_seen_rank=1,
        metadata={"law_name": title} if title else {},
    )
    return format_document_evidence(candidate)


def _selected_negative(record: dict[str, Any], ablation: str) -> bool:
    if ablation == "baseline":
        return False
    kind = str(record.get("negative_type", "")).strip()
    return kind in NEGATIVE_TYPES[ablation]


def build_fold_dataset(
    records: Sequence[dict[str, Any]],
    *,
    fold: int,
    fold_map: dict[str, int],
    ablation: str,
    same_law_boost: int,
    max_training_pairs: int,
) -> FoldDataset:
    """Build deterministic binary document examples and reject fold leakage."""

    if ablation not in ABLATATIONS:
        raise ValueError(f"unsupported P13 ablation: {ablation}")
    selected: list[dict[str, Any]] = []
    skipped_missing_evidence = 0
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        query_id = str(record.get("query_id", "")).strip()
        if not query_id or query_id not in fold_map:
            raise ValueError("P12 record has an unknown query_id")
        record_fold = record.get("fold")
        if record_fold != fold:
            raise ValueError(
                f"negative record for {query_id} belongs to wrong output fold"
            )
        if fold_map[query_id] == fold:
            raise ValueError(
                f"leakage: validation query {query_id} appears in training fold {fold}"
            )
        if not _selected_negative(record, ablation):
            continue
        positive_doc = str(record.get("positive_doc", "")).strip()
        negative_doc = str(record.get("negative_doc", "")).strip()
        query = str(record.get("query", "")).strip()
        if not positive_doc or not negative_doc or not query:
            raise ValueError("P12 record is missing query or document IDs")
        key = (query_id, positive_doc, negative_doc)
        if key in seen:
            continue
        positive = _training_document(
            positive_doc,
            record.get("positive_evidence"),
        )
        provenance = record.get("provenance")
        title = provenance.get("law_name") if isinstance(provenance, dict) else None
        negative = _training_document(
            negative_doc,
            record.get("negative_evidence"),
            str(title) if title else None,
        )
        if not positive or not negative:
            skipped_missing_evidence += 1
            continue
        seen.add(key)
        copies = (
            same_law_boost
            if ablation == "same-law-boosted"
            and str(record.get("negative_type")) == "same_law"
            else 1
        )
        for copy_index in range(copies):
            selected.append({**record, "_copy_index": copy_index})
    selected.sort(
        key=lambda row: (
            str(row["query_id"]),
            str(row["positive_doc"]),
            str(row["negative_doc"]),
            int(row.get("_copy_index", 0)),
        )
    )
    if max_training_pairs:
        selected = selected[:max_training_pairs]
    examples: list[TrainingExample] = []
    counts: dict[str, int] = {}
    query_ids: set[str] = set()
    serializable_pairs: list[dict[str, Any]] = []
    for record in selected:
        query_id = str(record["query_id"])
        query = str(record["query"])
        positive_doc = str(record["positive_doc"])
        negative_doc = str(record["negative_doc"])
        provenance = record.get("provenance")
        title = provenance.get("law_name") if isinstance(provenance, dict) else None
        positive = _training_document(positive_doc, record.get("positive_evidence"))
        negative = _training_document(
            negative_doc,
            record.get("negative_evidence"),
            str(title) if title else None,
        )
        kind = str(record["negative_type"])
        examples.extend(
            (
                TrainingExample(query_id, query, positive_doc, positive, 1.0),
                TrainingExample(query_id, query, negative_doc, negative, 0.0, kind),
            )
        )
        counts[kind] = counts.get(kind, 0) + 1
        query_ids.add(query_id)
        serializable_pairs.append(
            {
                "query_id": query_id,
                "positive_doc": positive_doc,
                "negative_doc": negative_doc,
                "negative_type": kind,
                "copy_index": record["_copy_index"],
            }
        )
    return FoldDataset(
        fold=fold,
        pair_count=len(selected),
        examples=tuple(examples),
        query_ids=tuple(sorted(query_ids)),
        dataset_sha256=_canonical_hash(serializable_pairs),
        negative_type_counts=dict(sorted(counts.items())),
        skipped_missing_evidence=skipped_missing_evidence,
    )


def _validation_predictions(
    backend: TrainingBackend,
    validation_ids: Sequence[str],
    train: dict[str, dict[str, Any]],
    candidates: dict[str, list[LegalIRDocumentCandidate]],
    *,
    fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in tqdm(
        validation_ids,
        desc=f"Fold {fold} validation",
        unit="q",
        leave=False,
    ):
        query = train[query_id]["question"]
        pool = candidates[query_id]
        scores = backend.score(query, [format_document_evidence(item) for item in pool])
        if len(scores) != len(pool):
            raise ValueError("reranker backend returned a score count mismatch")
        scored = [
            replace(candidate, rerank_score=float(score))
            for candidate, score in zip(pool, scores)
        ]
        ranked = final_document_ranking(scored)
        top5 = final_submission_documents(ranked)
        gold = train[query_id]["gold"]
        matched = sorted(set(gold).intersection(top5))
        rows.append(
            {
                "query_id": query_id,
                "fold": fold,
                "gold_documents": gold,
                "candidate_documents": [candidate.doc_id for candidate in pool],
                "top5": top5,
                "matched_gold": matched,
            }
        )
    return rows


def summarize_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact organizer-style macro set Recall/Precision at top five."""

    if not rows:
        raise ValueError("cannot summarize an empty validation set")
    recalls: list[float] = []
    precisions: list[float] = []
    multi_recalls: list[float] = []
    full_gold = 0
    for row in rows:
        gold = set(row["gold_documents"])
        predicted = set(row["top5"][:5])
        matched = gold.intersection(predicted)
        recall = len(matched) / len(gold)
        precision = len(matched) / len(predicted) if predicted else 0.0
        recalls.append(recall)
        precisions.append(precision)
        if len(gold) > 1:
            multi_recalls.append(recall)
        if matched == gold:
            full_gold += 1
    return {
        "official_macro_recall": sum(recalls) / len(recalls),
        "official_macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": sum(multi_recalls) / len(multi_recalls)
        if multi_recalls
        else None,
        "multi_gold_query_count": len(multi_recalls),
        "full_gold_query_count": full_gold,
        "query_count": len(rows),
    }


def _better_metrics(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_key = (
        candidate["official_macro_recall"],
        candidate["official_macro_precision"],
        candidate["multi_gold_recall"]
        if candidate["multi_gold_recall"] is not None
        else -1.0,
    )
    best_key = (
        best["official_macro_recall"],
        best["official_macro_precision"],
        best["multi_gold_recall"] if best["multi_gold_recall"] is not None else -1.0,
    )
    return candidate_key > best_key


class TorchBGERerankerBackend:
    """Binary relevance trainer supporting CPU diagnostics and CUDA training."""

    def __init__(
        self,
        base_model: str,
        *,
        device: str,
        batch_size: int,
        gradient_accumulation: int,
        max_length: int,
        learning_rate: float,
        warmup_ratio: float,
        fp16: bool,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers.optimization import get_linear_schedule_with_warmup
        except ImportError as exc:  # pragma: no cover - depends on Kaggle image
            raise RuntimeError(
                "P13 GPU training requires torch and transformers. Install the "
                "project GPU extra on Kaggle."
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for --device cuda, but torch.cuda.is_available() "
                "is False. Use --device cpu for a tiny diagnostic or run on a GPU."
            )
        self._torch = torch
        self._schedule_factory = get_linear_schedule_with_warmup
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._gradient_accumulation = gradient_accumulation
        self._max_length = max_length
        self._learning_rate = learning_rate
        self._warmup_ratio = warmup_ratio
        self._fp16 = bool(fp16 and self._device.type == "cuda")
        random.seed(seed)
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            base_model, local_files_only=True
        ).to(self._device)
        self._optimizer: Any | None = None
        self._scheduler: Any | None = None
        if self._fp16:
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self._scaler = _DisabledGradScaler()
        revision = getattr(self._model.config, "_commit_hash", None)
        self._model_revision = (
            str(revision) if revision else _local_model_revision(base_model)
        )

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def prepare_training(
        self, examples: Sequence[TrainingExample], epochs: int
    ) -> None:
        torch = self._torch
        batches = math.ceil(len(examples) / self._batch_size)
        steps_per_epoch = math.ceil(batches / self._gradient_accumulation)
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(total_steps * self._warmup_ratio)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._learning_rate
        )
        self._scheduler = self._schedule_factory(
            self._optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _logits(self, encoded: dict[str, Any]) -> Any:
        logits = self._model(**encoded).logits
        if logits.ndim == 2 and logits.shape[-1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits.reshape(-1)

    def _batch(self, examples: Sequence[TrainingExample]) -> tuple[dict[str, Any], Any]:
        torch = self._torch
        encoded = self._tokenizer(
            [example.query for example in examples],
            [example.document for example in examples],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        labels = torch.tensor(
            [example.label for example in examples],
            dtype=torch.float32,
            device=self._device,
        )
        return encoded, labels

    def fit_epoch(self, examples: Sequence[TrainingExample]) -> float:
        if self._optimizer is None or self._scheduler is None:
            raise RuntimeError("prepare_training must run before fit_epoch")
        torch = self._torch
        ordered = list(examples)
        random.shuffle(ordered)
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batch_starts = range(0, len(ordered), self._batch_size)
        for start in tqdm(
            batch_starts,
            total=math.ceil(len(ordered) / self._batch_size),
            desc="Training batches",
            unit="batch",
            leave=False,
        ):
            batch = ordered[start : start + self._batch_size]
            encoded, labels = self._batch(batch)
            autocast = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if self._fp16
                else nullcontext()
            )
            with autocast:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    self._logits(encoded), labels
                )
                scaled_loss = loss / self._gradient_accumulation
            self._scaler.scale(scaled_loss).backward()
            batch_index = start // self._batch_size + 1
            is_last = start + self._batch_size >= len(ordered)
            if batch_index % self._gradient_accumulation == 0 or is_last:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._scheduler.step()
                self._optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / len(losses)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(documents), self._batch_size):
                pairs = [
                    TrainingExample("", query, "", document, 0.0)
                    for document in documents[start : start + self._batch_size]
                ]
                encoded, _ = self._batch(pairs)
                autocast = (
                    torch.cuda.amp.autocast(dtype=torch.float16)
                    if self._fp16
                    else nullcontext()
                )
                with autocast:
                    scores.extend(
                        float(value)
                        for value in self._logits(encoded)
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
        return scores

    def save_checkpoint(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        _write_json(directory / "p13_checkpoint_metadata.json", metadata)

    def reload_checkpoint(self, directory: Path) -> None:
        """Reload the saved weights so the GPU smoke tests the artifact itself."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True
        ).to(self._device)

    def close(self) -> None:
        del self._model
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()


class _DisabledGradScaler:
    """CPU-safe no-op matching the small GradScaler API used by the trainer."""

    def scale(self, loss: Any) -> Any:
        return loss

    def unscale_(self, optimizer: Any) -> None:
        del optimizer

    def step(self, optimizer: Any) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _local_model_revision(base_model: str) -> str:
    path = Path(base_model)
    config = path / "config.json"
    if config.is_file():
        return f"local-config-sha256:{_sha256(config)}"
    return "unavailable"


def _resolve_device(requested: str, *, dry_run: bool) -> str:
    """Resolve device without importing or touching CUDA during dry-run."""

    if dry_run:
        return requested
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("P13 requires torch for non-dry-run execution") from exc
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA is required for --device cuda, but torch.cuda.is_available() "
            "is False. Use --device cpu for a tiny diagnostic or run on a GPU."
        )
    return "cuda" if available else "cpu"


def _print_device_diagnostics(device: str, *, diagnostic_only: bool) -> None:
    """Print explicit runtime information for diagnostic runs."""

    print(f"diagnostic_device = {device}")
    if device != "cuda":
        return
    import torch

    print(f"cuda_available = {torch.cuda.is_available()}")
    print(f"gpu_name = {torch.cuda.get_device_name(0)}")
    print(f"cuda_version = {torch.version.cuda}")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        print(f"gpu_memory_allocated_bytes = {allocated}")
        print(f"gpu_memory_free_bytes = {free_bytes}")
        print(f"gpu_memory_total_bytes = {total_bytes}")
    except (AttributeError, RuntimeError):
        print("gpu_memory = unavailable")
    if diagnostic_only:
        print("diagnostic_subset = enabled")


def _backend_factory(args: argparse.Namespace, *, fold: int) -> TrainingBackend:
    return TorchBGERerankerBackend(
        args.base_model,
        device=args.device,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        seed=args.seed + fold,
    )


def _fold_paths(output_dir: Path, fold: int) -> dict[str, Path]:
    directory = output_dir / f"fold_{fold}"
    return {
        "directory": directory,
        "train_ids": directory / "train_ids.json",
        "validation_ids": directory / "validation_ids.json",
        "dataset": directory / "dataset_manifest.json",
        "metrics": directory / "metrics.json",
        "baseline": directory / "baseline_predictions.jsonl",
        "finetuned": directory / "finetuned_predictions.jsonl",
        "checkpoint": directory / "checkpoint",
        "completion": directory / "completion.json",
    }


def _validate_fold_negative_records(
    path: Path,
    *,
    fold: int,
    fold_map: dict[str, int],
    expected_training_ids: set[str],
    ablation: str,
    same_law_boost: int,
    retain_records: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one P12 fold, validate membership/leakage, then return that fold only."""

    records: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    positive_count = 0
    negative_count = 0
    hard_count = 0
    semi_hard_count = 0
    selected_pair_count = 0
    unresolved_evidence_count = 0
    record_count = 0
    for record in tqdm(
        _iter_json_records(path),
        desc=f"Fold {fold} negatives",
        unit="record",
        leave=False,
    ):
        query_id = str(record.get("query_id", "")).strip()
        record_count += 1
        if not query_id or query_id not in fold_map:
            raise ValueError(f"P12 record has an unknown query_id in fold {fold}")
        if record.get("fold") != fold:
            raise ValueError(f"negative record for {query_id} belongs to wrong fold")
        if query_id not in expected_training_ids:
            raise ValueError(
                f"leakage: validation query {query_id} appears in training fold {fold}"
            )
        if not str(record.get("query", "")).strip():
            raise ValueError(f"P12 record for {query_id} has no query text")
        if not str(record.get("positive_doc", "")).strip():
            raise ValueError(f"P12 record for {query_id} has no positive document")
        if not str(record.get("negative_doc", "")).strip():
            raise ValueError(f"P12 record for {query_id} has no negative document")
        query_ids.add(query_id)
        positive_count += bool(str(record.get("positive_doc", "")).strip())
        negative_count += bool(str(record.get("negative_doc", "")).strip())
        kind = str(record.get("negative_type", "")).strip()
        hard_count += kind in {"hard_false_positive", "semantic_confuser"}
        semi_hard_count += kind in {"semi_hard", "lexical_confuser"}
        if (
            not str(record.get("positive_evidence", "")).strip()
            or not str(record.get("negative_evidence", "")).strip()
        ):
            unresolved_evidence_count += 1
        if _selected_negative(record, ablation):
            selected_pair_count += (
                same_law_boost
                if ablation == "same-law-boosted" and kind == "same_law"
                else 1
            )
        if retain_records:
            records.append(record)
    missing = sorted(expected_training_ids.difference(query_ids))
    summary = {
        "negative_record_count": record_count,
        "training_query_count": len(query_ids),
        "expected_training_query_count": len(expected_training_ids),
        "missing_training_queries": missing,
        "leakage_query_count": 0,
        "positive_record_count": positive_count,
        "negative_record_with_document_count": negative_count,
        "hard_negative_count": hard_count,
        "semi_hard_negative_count": semi_hard_count,
        "selected_pair_count": selected_pair_count,
        "unresolved_evidence_count": unresolved_evidence_count,
        "status": "VALID" if not missing else "INVALID",
    }
    if missing:
        raise ValueError(
            f"fold {fold} negatives are missing {len(missing)} training queries"
        )
    if ablation != "baseline" and selected_pair_count == 0:
        raise ValueError(
            f"fold {fold} has no eligible {ablation} hard/semi-hard negatives"
        )
    return records, summary


def _run_one_ablation(
    args: argparse.Namespace,
    *,
    ablation: str,
    train: dict[str, dict[str, Any]],
    fold_map: dict[str, int],
    candidates: dict[str, list[LegalIRDocumentCandidate]],
    backend_factory: Any,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_folds = set(args.folds_to_run) if args.folds_to_run else set(range(5))
    available_folds = set(fold_map.values())
    missing_requested = sorted(requested_folds.difference(available_folds))
    if missing_requested:
        raise ValueError(f"requested folds are not resolvable: {missing_requested}")
    if args.resume and (output_dir / "training_manifest.json").is_file():
        previous_manifest = json.loads(
            (output_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        previous = previous_manifest.get("hyperparameters", {})
        current = {
            "learning_rate": args.learning_rate, "epochs": args.epochs,
            "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length, "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16, "seed": args.seed,
        }
        mismatches = [key for key, value in current.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError("resume hyperparameter mismatch: " + ", ".join(mismatches))
    all_ids = sorted(set(train).intersection(fold_map))
    if len(all_ids) != len(fold_map):
        missing_labels = sorted(set(fold_map).difference(train))
        raise ValueError(f"train labels missing strict fold IDs: {missing_labels[:5]}")
    missing_candidates = sorted(set(all_ids).difference(candidates))
    partial = bool(missing_candidates)
    if partial and not args.allow_partial_oof:
        raise ValueError(
            "candidate cache does not cover every strict-CV query "
            f"({len(missing_candidates)} missing). Provide full cached rankings, or "
            "use --allow-partial-oof only for a non-promotable diagnostic run."
        )
    fold_reports: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    base_revision: str | None = None
    for fold in sorted(requested_folds):
        paths = _fold_paths(output_dir, fold)
        validation_ids = [
            query_id
            for query_id in all_ids
            if fold_map[query_id] == fold and query_id in candidates
        ]
        if args.max_validation_queries:
            validation_ids = validation_ids[: args.max_validation_queries]
        train_ids = [query_id for query_id in all_ids if fold_map[query_id] != fold]
        expected_training_ids = set(train_ids)
        negative_path = args.negatives_dir / f"fold_{fold}.jsonl"
        negative_records, negative_summary = _validate_fold_negative_records(
            negative_path,
            fold=fold,
            fold_map=fold_map,
            expected_training_ids=expected_training_ids,
            ablation=ablation,
            same_law_boost=args.same_law_boost,
            retain_records=not args.dry_run,
        )
        if negative_summary["unresolved_evidence_count"]:
            raise ValueError(
                "P12 contains unresolved positive/negative evidence in fold "
                f"{fold}: {negative_summary['unresolved_evidence_count']} records"
            )
        dataset = None
        if not args.dry_run:
            dataset = build_fold_dataset(
                negative_records,
                fold=fold,
                fold_map=fold_map,
                ablation=ablation,
                same_law_boost=args.same_law_boost,
                max_training_pairs=args.max_training_pairs,
            )
        _write_json(paths["train_ids"], train_ids)
        _write_json(paths["validation_ids"], validation_ids)
        _write_json(
            paths["dataset"],
            {
                "schema_version": "task1-p13-fold-dataset-v1",
                "fold": fold,
                "ablation": ablation,
                "pair_count": (
                    negative_summary["selected_pair_count"]
                    if dataset is None else dataset.pair_count
                ),
                "example_count": (
                    negative_summary["selected_pair_count"] * 2
                    if dataset is None else len(dataset.examples)
                ),
                "training_query_count": (
                    negative_summary["training_query_count"]
                    if dataset is None else len(dataset.query_ids)
                ),
                "training_query_ids": sorted(
                    expected_training_ids
                    if dataset is None else dataset.query_ids
                ),
                "dataset_sha256": (
                    "dry-run-streaming-validation"
                    if dataset is None else dataset.dataset_sha256
                ),
                "negative_type_counts": negative_summary,
                "skipped_missing_evidence": 0,
                "negative_source": str(negative_path),
                "negative_source_sha256": _sha256(negative_path),
                "leakage_policy": (
                    "all fold_F training records must exclude fold F validation IDs"
                ),
                "negative_validation_summary": negative_summary,
            },
        )
        if (
            args.resume
            and paths["completion"].is_file()
            and paths["metrics"].is_file()
            and paths["baseline"].is_file()
            and paths["finetuned"].is_file()
        ):
            completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
            if completion.get("status") != "COMPLETE":
                raise ValueError(f"invalid completion marker for fold {fold}")
            previous_dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
            if previous_dataset.get("dataset_sha256") != dataset.dataset_sha256:
                raise ValueError(f"resume hash mismatch for fold {fold}: dataset changed")
            if completion.get("dataset_sha256") != dataset.dataset_sha256:
                raise ValueError(f"resume hash mismatch for fold {fold}: completion changed")
            previous_report = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            if previous_report.get("dataset_sha256") != dataset.dataset_sha256:
                raise ValueError(f"resume hash mismatch for fold {fold}: metrics changed")
            baseline_rows = _load_json_records(paths["baseline"])
            tuned_rows = _load_json_records(paths["finetuned"])
            if len(baseline_rows) != len(validation_ids) or len(tuned_rows) != len(validation_ids):
                raise ValueError(f"resume artifact row count mismatch for fold {fold}")
            fold_reports.append(previous_report)
            oof_rows.extend(
                {
                    "query_id": before["query_id"], "fold": fold,
                    "gold_documents": before["gold_documents"],
                    "candidate_documents": before["candidate_documents"],
                    "baseline_top5": before["top5"], "fine_tuned_top5": after["top5"],
                    "baseline_matched_gold": before["matched_gold"],
                    "fine_tuned_matched_gold": after["matched_gold"],
                }
                for before, after in zip(baseline_rows, tuned_rows)
            )
            continue
        if not validation_ids:
            raise ValueError(f"no covered validation candidates for fold {fold}")
        if args.dry_run:
            fold_reports.append(
                {
                    "fold": fold,
                    "status": "DRY_RUN",
                    "validation_query_count": len(validation_ids),
                    "training_pair_count": negative_summary["selected_pair_count"],
                    "dataset_sha256": "dry-run-streaming-validation",
                    **negative_summary,
                }
            )
            del negative_records, dataset
            continue
        baseline = backend_factory(args, fold=fold)
        try:
            base_revision = base_revision or baseline.model_revision
            baseline_rows = _validation_predictions(
                baseline, validation_ids, train, candidates, fold=fold
            )
            baseline_metrics = summarize_predictions(baseline_rows)
        finally:
            baseline.close()
        if ablation == "baseline":
            tuned_rows = baseline_rows
            tuned_metrics = baseline_metrics
            history: list[dict[str, Any]] = []
            selected_epoch: int | None = None
            checkpoint: str | None = None
        else:
            if not dataset.examples:
                raise ValueError(
                    f"ablation {ablation} has no eligible P12 records in fold {fold}. "
                    "For easy-random, mine explicit easy_random negatives first."
                )
            finetuned = backend_factory(args, fold=fold)
            try:
                base_revision = base_revision or finetuned.model_revision
                finetuned.prepare_training(dataset.examples, args.epochs)
                history = []
                for epoch in range(1, args.epochs + 1):
                    loss = finetuned.fit_epoch(dataset.examples)
                    history.append({"epoch": epoch, "mean_train_loss": loss})
                selected_epoch = args.epochs
                finetuned.save_checkpoint(
                    paths["checkpoint"],
                    {
                        "schema_version": "task1-p13-checkpoint-v1",
                        "base_model": args.base_model,
                        "base_model_revision": finetuned.model_revision,
                        "fold": fold,
                        "ablation": ablation,
                        "selected_epoch": selected_epoch,
                        "selection_policy": args.selection_policy,
                        "dataset_sha256": dataset.dataset_sha256,
                    },
                )
                reload_checkpoint = getattr(finetuned, "reload_checkpoint", None)
                if reload_checkpoint is not None:
                    reload_checkpoint(paths["checkpoint"])
                # Outer validation is scored only after all fixed epochs finish.
                tuned_rows = _validation_predictions(
                    finetuned, validation_ids, train, candidates, fold=fold
                )
                tuned_metrics = summarize_predictions(tuned_rows)
                checkpoint = str(paths["checkpoint"])
            finally:
                finetuned.close()
        _write_jsonl(paths["baseline"], baseline_rows)
        _write_jsonl(paths["finetuned"], tuned_rows)
        fold_report = {
            "fold": fold,
            "status": "COMPLETE",
            "validation_query_count": len(validation_ids),
            "training_pair_count": dataset.pair_count,
            "dataset_sha256": dataset.dataset_sha256,
            "baseline": baseline_metrics,
            "fine_tuned": tuned_metrics,
            "selected_epoch": selected_epoch,
            "checkpoint": checkpoint,
            "training_history": history,
        }
        _write_json(paths["metrics"], fold_report)
        _write_json_atomic(
            paths["completion"],
            {
                "schema_version": "task1-p13-fold-completion-v1",
                "status": "COMPLETE",
                "fold": fold,
                "ablation": ablation,
                "dataset_sha256": dataset.dataset_sha256,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        fold_reports.append(fold_report)
        for before, after in zip(baseline_rows, tuned_rows):
            oof_rows.append(
                {
                    "query_id": before["query_id"],
                    "fold": fold,
                    "gold_documents": before["gold_documents"],
                    "candidate_documents": before["candidate_documents"],
                    "baseline_top5": before["top5"],
                    "fine_tuned_top5": after["top5"],
                    "baseline_matched_gold": before["matched_gold"],
                    "fine_tuned_matched_gold": after["matched_gold"],
                }
            )
        del negative_records, dataset
    manifest = {
        "schema_version": "task1-p13-finetune-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRY_RUN" if args.dry_run else "COMPLETE",
        "diagnostic_only": args.diagnostic_only,
        "promotable": False if args.dry_run or args.diagnostic_only else not partial,
        "ablation": ablation,
        "base_model": {"path": args.base_model, "revision": base_revision},
        "inputs": {
            "train": str(args.train),
            "train_sha256": _sha256(args.train),
            "strict_folds": str(args.folds),
            "strict_folds_sha256": _sha256(args.folds),
            "negatives_dir": str(args.negatives_dir),
            "candidates": str(args.candidates),
            "candidates_sha256": _sha256(args.candidates),
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16,
            "seed": args.seed,
            "same_law_boost": args.same_law_boost,
            "max_training_pairs": args.max_training_pairs,
            "max_validation_queries": args.max_validation_queries,
            "selection_policy": args.selection_policy,
            "folds_to_run": sorted(requested_folds),
            "early_stopping_patience": args.early_stopping_patience,
            "requested_device": args.device,
            "resolved_device": args.device,
        },
        "document_unit": {
            "candidate_depth": args.candidate_depth,
            "evidence_limit": args.evidence_limit,
            "formatter": "legal_ir_document_candidates.format_document_evidence",
        },
        "oof": {
            "strict_fold_count": 5,
            "expected_query_count": len(all_ids),
            "evaluated_query_count": len(oof_rows),
            "missing_candidate_query_count": len(missing_candidates),
            "partial": partial,
            "allow_partial_oof": args.allow_partial_oof,
            "candidate_set_policy": (
                "pretrained and fine-tuned BGE score the same collapsed "
                "document candidates"
            ),
        },
        "folds": fold_reports,
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    if args.dry_run:
        _write_json(
            output_dir / "final_decision.json",
            {
                "schema_version": "task1-p13-decision-v1",
                "status": "NOT_RUN",
                "reason": (
                    "--dry-run validates data/folds only; no BGE model was loaded."
                ),
            },
        )
        return manifest
    oof_rows.sort(key=lambda row: row["query_id"])
    _write_jsonl(output_dir / "oof_predictions.jsonl", oof_rows)
    baseline_summary = summarize_predictions(
        [
            {
                "gold_documents": row["gold_documents"],
                "top5": row["baseline_top5"],
            }
            for row in oof_rows
        ]
    )
    tuned_summary = summarize_predictions(
        [
            {
                "gold_documents": row["gold_documents"],
                "top5": row["fine_tuned_top5"],
            }
            for row in oof_rows
        ]
    )
    completed_folds = [
        report for report in fold_reports if report["status"] == "COMPLETE"
    ]
    comparison = {
        "schema_version": "task1-p13-oof-comparison-v1",
        "ablation": ablation,
        "baseline": baseline_summary,
        "fine_tuned": tuned_summary,
        "delta": {
            key: tuned_summary[key] - baseline_summary[key]
            for key in ("official_macro_recall", "official_macro_precision")
        },
        "worst_fold_recall": {
            "baseline": min(
                report["baseline"]["official_macro_recall"]
                for report in completed_folds
            ),
            "fine_tuned": min(
                report["fine_tuned"]["official_macro_recall"]
                for report in completed_folds
            ),
        },
        "fold_count": len(completed_folds),
        "partial_oof": partial,
    }
    _write_json(output_dir / "comparison.json", comparison)
    baseline_multi = baseline_summary["multi_gold_recall"]
    tuned_multi = tuned_summary["multi_gold_recall"]
    comparison_markdown = "\n".join(
        (
            "# P13 BGE reranker strict OOF comparison",
            "",
            "| Metric | Pretrained BGE | Fine-tuned BGE |",
            "|---|---:|---:|",
            "| Official macro Recall@5 | %.6f | %.6f |"
            % (
                baseline_summary["official_macro_recall"],
                tuned_summary["official_macro_recall"],
            ),
            "| Official macro Precision@5 | %.6f | %.6f |"
            % (
                baseline_summary["official_macro_precision"],
                tuned_summary["official_macro_precision"],
            ),
            "| Worst-fold Recall | %.6f | %.6f |"
            % (
                comparison["worst_fold_recall"]["baseline"],
                comparison["worst_fold_recall"]["fine_tuned"],
            ),
            "| Multi-gold Recall | %s | %s |"
            % (
                f"{baseline_multi:.6f}" if baseline_multi is not None else "n/a",
                f"{tuned_multi:.6f}" if tuned_multi is not None else "n/a",
            ),
            "",
            (
                "The candidate document pool, evidence limit, and candidate order "
                "are fixed for both columns."
            ),
            "",
        )
    )
    (output_dir / "comparison.md").write_text(
        comparison_markdown,
        encoding="utf-8",
        newline="\n",
    )
    wins = (
        tuned_summary["official_macro_recall"]
        > baseline_summary["official_macro_recall"]
    )
    promotable = (
        wins and not partial and len(completed_folds) == 5 and not args.diagnostic_only
    )
    decision = {
        "schema_version": "task1-p13-decision-v1",
        "status": "PROMOTE_CANDIDATE" if promotable else "REJECT_CHECKPOINT",
        "selected_reranker": (
            "artifacts checkpoint candidate" if promotable else "models/reranker"
        ),
        "base_model": "BAAI/bge-reranker-v2-m3",
        "base_model_path": args.base_model,
        "diagnostic_only": args.diagnostic_only,
        "reason": (
            "Fine-tuned BGE improves complete strict OOF macro Recall@5. "
            "Production config is intentionally not changed automatically."
            if promotable
            else (
                "Fine-tuned BGE did not improve pretrained BGE macro Recall@5."
                if not wins
                else "Partial OOF diagnostics cannot promote a checkpoint."
            )
        ),
        "oof": comparison,
    }
    _write_json(output_dir / "final_decision.json", decision)
    return manifest


def run(
    args: argparse.Namespace,
    *,
    backend_factory: Any = _backend_factory,
) -> list[Path]:
    """Run one ablation, or the bounded A-E ablation suite on Kaggle."""

    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than zero")
    # Resolve explicit CUDA before touching input artifacts or model paths so a
    # local CPU-only invocation fails with the actionable device message first.
    args.device = _resolve_device(args.device, dry_run=args.dry_run)
    if args.diagnostic_only:
        _print_device_diagnostics(args.device, diagnostic_only=True)
    if not args.train.is_file():
        raise FileNotFoundError(f"train file not found: {args.train}")
    if not args.folds.is_file():
        raise FileNotFoundError(f"strict folds file not found: {args.folds}")
    if not args.candidates.is_file():
        raise FileNotFoundError(f"candidate file not found: {args.candidates}")
    if not Path(args.base_model).is_dir():
        raise FileNotFoundError(f"base model directory not found: {args.base_model}")
    if args.ablation == "all":
        ablations = ABLATATIONS
    else:
        ablations = (args.ablation,)
    train = load_train(args.train)
    fold_map = load_fold_map(args.folds)
    if args.dry_run:
        candidate_ids = load_candidate_query_ids(args.candidates)
        candidates = {query_id: [] for query_id in candidate_ids}
    else:
        candidates = load_document_candidates(
            args.candidates,
            candidate_depth=args.candidate_depth,
            evidence_limit=args.evidence_limit,
        )
    output_roots: list[Path] = []
    for ablation in ablations:
        run_args = argparse.Namespace(**vars(args))
        run_args.output_dir = (
            args.output_dir if len(ablations) == 1 else args.output_dir / ablation
        )
        _run_one_ablation(
            run_args,
            ablation=ablation,
            train=train,
            fold_map=fold_map,
            candidates=candidates,
            backend_factory=backend_factory,
        )
        output_roots.append(run_args.output_dir)
    if len(output_roots) > 1:
        _write_json(
            args.output_dir / "ablation_manifest.json",
            {
                "schema_version": "task1-p13-ablation-suite-v1",
                "ablations": list(ablations),
                "runs": [str(path) for path in output_roots],
                "note": "This is a bounded ablation suite, not a hyperparameter sweep.",
            },
        )
        output_roots.append(args.output_dir / "ablation_manifest.json")
    return output_roots


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"P13 error: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
