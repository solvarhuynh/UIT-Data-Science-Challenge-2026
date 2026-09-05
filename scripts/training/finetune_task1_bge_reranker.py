"""Fine-tune and strictly evaluate Task1's BGE document reranker on GPU.

This runner consumes the leakage-safe P12 records.  It deliberately keeps the
pretrained model directory read-only: every selected checkpoint is written
under the supplied artifacts directory.  Importing this module is model-free;
Torch and Transformers are imported only by the GPU backend at runtime.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence, cast

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
DEFAULT_CANDIDATES = Path("artifacts/task1/training/full_dense500_candidates.jsonl")
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
        choices=(50, 100, 150, 200, 300, 500),
        default=200,
        help="Fixed document candidate depth shared by baseline and tuned BGE.",
    )
    parser.add_argument("--evidence-limit", type=int, choices=(1, 2), default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=_positive_int, default=2)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument(
        "--inference-batch-size",
        type=_positive_int,
        default=16,
        help="Larger model-eval batch; lower this first if validation OOMs.",
    )
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=4)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--warmup-ratio", type=_probability, default=0.1)
    parser.add_argument("--max-training-pairs", type=_nonnegative_int, default=0)
    parser.add_argument("--max-validation-queries", type=_nonnegative_int, default=0)
    parser.add_argument("--same-law-boost", type=_positive_int, default=2)
    parser.add_argument(
        "--selection-policy",
        choices=("fixed_epochs",),
        default="fixed_epochs",
        help="Score the outer validation fold once after fixed epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=_nonnegative_int,
        default=0,
        help="Deprecated compatibility option; never selects on outer validation.",
    )
    parser.add_argument("--folds-to-run", type=int, nargs="+", choices=range(5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Mark output non-promotable (used by the real GPU smoke).",
    )
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


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input artifact not found: {path}")
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))
        return
    if isinstance(payload, dict):
        records = payload.get("predictions", payload.get("records", []))
        if isinstance(records, list):
            yield from (item for item in records if isinstance(item, dict))
        return
    raise ValueError(f"unsupported candidate artifact: {path}")


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    return list(_iter_json_records(path))


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
    for record in _iter_json_records(path):
        query_id = str(record.get("question_id", record.get("id", ""))).strip()
        if not query_id:
            continue
        hits = record.get("hits", record.get("documents", []))
        if not isinstance(hits, list):
            continue
        evidence: list[DocumentEvidence] = []
        document_rows: Iterable[Any] = (
            hits[:candidate_depth] if "documents" in record else hits
        )
        for index, raw in enumerate(document_rows, 1):
            if not isinstance(raw, dict):
                continue
            doc_id = str(
                raw.get("doc_id", raw.get("document_id", raw.get("id", "")))
            ).strip()
            nested = raw.get("evidence", [])
            nested_text = (
                "\n\n".join(
                    str(part.get("text", "")).strip()
                    for part in nested
                    if isinstance(part, dict) and str(part.get("text", "")).strip()
                )
                if isinstance(nested, list)
                else ""
            )
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
                    evidence_id=str(
                        raw.get("chunk_id", raw.get("id", f"{doc_id}:{index}"))
                    ),
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
    return str(format_document_evidence(candidate))


def _selected_negative(record: dict[str, Any], ablation: str) -> bool:
    if ablation == "baseline":
        return False
    kind = str(record.get("negative_type", "")).strip()
    return kind in NEGATIVE_TYPES[ablation]


def _balanced_training_subset(
    records: Sequence[dict[str, Any]], *, limit: int, fold: int
) -> list[dict[str, Any]]:
    """Deterministically cap pairs without biasing low-sorted query IDs."""

    if limit <= 0 or len(records) <= limit:
        return list(records)
    type_priority = {
        "same_law": 0,
        "semantic_confuser": 1,
        "hard_false_positive": 2,
        "lexical_confuser": 3,
        "semi_hard": 4,
        "easy_random": 5,
        "easy": 5,
    }

    def stable_hash(value: str) -> str:
        return hashlib.sha256(f"{fold}:{value}".encode()).hexdigest()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["query_id"]), []).append(record)
    queues: dict[str, list[dict[str, Any]]] = {}
    for query_id, values in grouped.items():
        queues[query_id] = sorted(
            values,
            key=lambda row: (
                type_priority.get(str(row.get("negative_type", "")), 99),
                stable_hash(
                    "|".join(
                        (
                            query_id,
                            str(row.get("positive_doc", "")),
                            str(row.get("negative_doc", "")),
                            str(row.get("_copy_index", 0)),
                        )
                    )
                ),
            ),
        )
    query_ids = sorted(grouped, key=stable_hash)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for query_id in query_ids:
            queue = queues[query_id]
            if offset < len(queue):
                selected.append(queue[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


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
    selected = _balanced_training_subset(selected, limit=max_training_pairs, fold=fold)
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
    for index, query_id in enumerate(validation_ids, start=1):
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
        if index % 50 == 0 or index == len(validation_ids):
            print(
                f"fold {fold}: scored {index}/{len(validation_ids)} validation queries",
                flush=True,
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
        inference_batch_size: int,
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
        self._inference_batch_size = inference_batch_size
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
        self._scaler: Any | None = (
            torch.cuda.amp.GradScaler(enabled=True)
            if self._fp16
            else _DisabledGradScaler()
        )
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
        if self._optimizer is None or self._scheduler is None or self._scaler is None:
            raise RuntimeError("prepare_training must run before fit_epoch")
        torch = self._torch
        ordered = list(examples)
        random.shuffle(ordered)
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        total_batches = math.ceil(len(ordered) / self._batch_size)
        progress_interval = max(1, total_batches // 20)
        for start in range(0, len(ordered), self._batch_size):
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
            if (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                print(
                    "training progress: "
                    f"{batch_index}/{total_batches} batches "
                    f"({100.0 * batch_index / total_batches:.1f}%), "
                    f"latest_loss={losses[-1]:.6f}",
                    flush=True,
                )
        return sum(losses) / len(losses)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(documents), self._inference_batch_size):
                pairs = [
                    TrainingExample("", query, "", document, 0.0)
                    for document in documents[
                        start : start + self._inference_batch_size
                    ]
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

        # Adam states retain references to every trained parameter. Release them
        # and the trained model before loading the saved copy, otherwise the
        # smoke/reload step can briefly hold two 0.6B models plus optimizer state.
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        del self._model
        gc.collect()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        self._tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True
        ).to(self._device)

    def close(self) -> None:
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        gc.collect()
        if getattr(self, "_device", None) is None or self._device.type == "cuda":
            self._torch.cuda.empty_cache()


class _DisabledGradScaler:
    """CPU-safe no-op matching the GradScaler API used by the trainer."""

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
    """Resolve the execution device without touching CUDA during a dry run."""

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
        inference_batch_size=args.inference_batch_size,
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
    }


def _checkpoint_complete(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    weight_markers = (
        directory / "model.safetensors",
        directory / "pytorch_model.bin",
        directory / "mock_checkpoint.json",
    )
    return any(path.is_file() and path.stat().st_size > 0 for path in weight_markers)


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
    current_inputs = {
        "train": str(args.train),
        "train_sha256": _sha256(args.train),
        "strict_folds": str(args.folds),
        "strict_folds_sha256": _sha256(args.folds),
        "negatives_dir": str(args.negatives_dir),
        "candidates": str(args.candidates),
        "candidates_sha256": _sha256(args.candidates),
    }
    resumed_base_revision: str | None = None
    if args.resume and (output_dir / "training_manifest.json").is_file():
        previous_manifest = json.loads(
            (output_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        previous = previous_manifest.get("hyperparameters", {})
        current = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16,
            "seed": args.seed,
            "same_law_boost": args.same_law_boost,
            "max_training_pairs": args.max_training_pairs,
            "max_validation_queries": args.max_validation_queries,
            "selection_policy": args.selection_policy,
        }
        mismatches = [
            key for key, value in current.items() if previous.get(key) != value
        ]
        if mismatches:
            raise ValueError("resume hyperparameter mismatch: " + ", ".join(mismatches))
        previous_inputs = previous_manifest.get("inputs", {})
        input_mismatches = [
            key
            for key in ("train_sha256", "strict_folds_sha256", "candidates_sha256")
            if previous_inputs.get(key) != current_inputs[key]
        ]
        previous_unit = previous_manifest.get("document_unit", {})
        if previous_unit.get("candidate_depth") != args.candidate_depth:
            input_mismatches.append("candidate_depth")
        if previous_unit.get("evidence_limit") != args.evidence_limit:
            input_mismatches.append("evidence_limit")
        if str(previous_manifest.get("base_model", {}).get("path")) != str(
            args.base_model
        ):
            input_mismatches.append("base_model")
        if previous_manifest.get("ablation") != ablation:
            input_mismatches.append("ablation")
        if input_mismatches:
            raise ValueError(
                "resume input/config mismatch: " + ", ".join(input_mismatches)
            )
        previous_revision = previous_manifest.get("base_model", {}).get("revision")
        if isinstance(previous_revision, str) and previous_revision:
            resumed_base_revision = previous_revision
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
    # P12 must preserve canonical positive evidence.  Do this before creating
    # the backend so an invalid bundle can never consume GPU/model resources.
    unresolved_positive: list[str] = []
    for check_fold in sorted(set(fold_map.values())):
        check_path = args.negatives_dir / f"fold_{check_fold}.jsonl"
        for record in _iter_json_records(check_path):
            if (
                not str(record.get("positive_evidence", "")).strip()
                or not str(record.get("negative_evidence", "")).strip()
            ):
                unresolved_positive.append(
                    f"{record.get('query_id')}:{record.get('positive_doc')}"
                )
    if unresolved_positive:
        raise ValueError(
            "P12 contains unresolved positive evidence; refusing P13 training: "
            + ", ".join(unresolved_positive[:5])
        )
    fold_reports: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    base_revision = resumed_base_revision
    requested_folds = set(args.folds_to_run) if args.folds_to_run else set(range(5))
    for fold in sorted(set(fold_map.values()).intersection(requested_folds)):
        paths = _fold_paths(output_dir, fold)
        validation_ids = [
            query_id
            for query_id in all_ids
            if fold_map[query_id] == fold and query_id in candidates
        ]
        if args.max_validation_queries:
            validation_ids = validation_ids[: args.max_validation_queries]
        train_ids = [query_id for query_id in all_ids if fold_map[query_id] != fold]
        negative_path = args.negatives_dir / f"fold_{fold}.jsonl"
        negative_records = _load_json_records(negative_path)
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
        dataset_payload = {
            "schema_version": "task1-p13-fold-dataset-v1",
            "fold": fold,
            "ablation": ablation,
            "pair_count": dataset.pair_count,
            "example_count": len(dataset.examples),
            "training_query_count": len(dataset.query_ids),
            "training_query_ids": list(dataset.query_ids),
            "dataset_sha256": dataset.dataset_sha256,
            "negative_type_counts": dataset.negative_type_counts,
            "skipped_missing_evidence": dataset.skipped_missing_evidence,
            "negative_source": str(negative_path),
            "negative_source_sha256": _sha256(negative_path),
            "leakage_policy": (
                "all fold_F training records must exclude fold F validation IDs"
            ),
        }
        if (
            args.resume
            and paths["metrics"].is_file()
            and paths["baseline"].is_file()
            and paths["finetuned"].is_file()
            and (ablation == "baseline" or _checkpoint_complete(paths["checkpoint"]))
        ):
            if not paths["dataset"].is_file():
                raise ValueError(f"resume dataset manifest missing for fold {fold}")
            previous_dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
            if previous_dataset.get("dataset_sha256") != dataset.dataset_sha256:
                raise ValueError(
                    f"resume hash mismatch for fold {fold}: dataset changed"
                )
            previous_report = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            if previous_report.get("dataset_sha256") != dataset.dataset_sha256:
                raise ValueError(
                    f"resume hash mismatch for fold {fold}: metrics changed"
                )
            if previous_report.get("status") != "COMPLETE" or previous_report.get(
                "validation_query_count"
            ) != len(validation_ids):
                raise ValueError(f"resume metrics are incomplete for fold {fold}")
            baseline_rows = _load_json_records(paths["baseline"])
            tuned_rows = _load_json_records(paths["finetuned"])
            if len(baseline_rows) != len(validation_ids) or len(tuned_rows) != len(
                validation_ids
            ):
                raise ValueError(f"resume artifact row count mismatch for fold {fold}")
            fold_reports.append(previous_report)
            oof_rows.extend(
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
                for before, after in zip(baseline_rows, tuned_rows)
            )
            continue
        _write_json(paths["dataset"], dataset_payload)
        if not validation_ids:
            raise ValueError(f"no covered validation candidates for fold {fold}")
        if args.dry_run:
            fold_reports.append(
                {
                    "fold": fold,
                    "status": "DRY_RUN",
                    "validation_query_count": len(validation_ids),
                    "training_pair_count": dataset.pair_count,
                    "dataset_sha256": dataset.dataset_sha256,
                }
            )
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
                if paths["checkpoint"].exists():
                    shutil.rmtree(paths["checkpoint"])
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
    manifest = {
        "schema_version": "task1-p13-finetune-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRY_RUN" if args.dry_run else "OOF_PENDING",
        "diagnostic_only": args.diagnostic_only,
        "promotable": False,
        "ablation": ablation,
        "base_model": {"path": args.base_model, "revision": base_revision},
        "inputs": current_inputs,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
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
    worst_fold = cast(dict[str, float], comparison["worst_fold_recall"])
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
                worst_fold["baseline"],
                worst_fold["fine_tuned"],
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
    complete_oof = (
        len(completed_folds) == 5
        and len(oof_rows) == len(all_ids)
        and args.max_validation_queries == 0
    )
    promotable = wins and not partial and complete_oof and not args.diagnostic_only
    decision = {
        "schema_version": "task1-p13-decision-v1",
        "status": "PROMOTE_CANDIDATE" if promotable else "REJECT_CHECKPOINT",
        "promotable": promotable,
        "complete_oof": complete_oof,
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
                if complete_oof and not wins
                else (
                    "Incomplete, bounded, or diagnostic OOF cannot promote "
                    "a checkpoint."
                )
            )
        ),
        "oof": comparison,
    }
    _write_json(output_dir / "final_decision.json", decision)
    manifest["status"] = "COMPLETE" if complete_oof else "OOF_INCOMPLETE"
    manifest["promotable"] = promotable
    _write_json(output_dir / "training_manifest.json", manifest)
    return manifest


def run(
    args: argparse.Namespace,
    *,
    backend_factory: Any = _backend_factory,
) -> list[Path]:
    """Run one ablation, or the bounded A-E ablation suite on Kaggle."""

    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than zero")
    args.device = _resolve_device(args.device, dry_run=args.dry_run)
    if args.diagnostic_only:
        _print_device_diagnostics(args.device, diagnostic_only=True)
    ablations: tuple[str, ...]
    if args.ablation == "all":
        ablations = ABLATATIONS
    else:
        ablations = (args.ablation,)
    train = load_train(args.train)
    fold_map = load_fold_map(args.folds)
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
