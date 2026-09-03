"""Lazy, injectable clients for cross-encoder relevance scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from numbers import Real
from threading import Lock
from typing import Any, Protocol, runtime_checkable


class RerankerDependencyError(ImportError):
    """Raised when an optional reranker dependency is unavailable."""


class RerankerModelLoadError(RuntimeError):
    """Raised when the configured cross-encoder cannot be loaded."""


class RerankerScoringError(RuntimeError):
    """Raised when a loaded cross-encoder cannot score candidate pairs."""


@runtime_checkable
class RerankerClient(Protocol):
    """Structural interface accepted by retrieval rerankers.

    Implementations receive one query and an ordered document collection.  The
    returned score at position ``i`` must correspond to ``documents[i]``.
    """

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """Return one finite relevance score for every document."""


class CrossEncoderClient:
    """Score query-document pairs with a lazily loaded CrossEncoder.

    ``model_name_or_path`` may be a local directory or a Hugging Face model ID.
    By default, ``local_files_only=True`` prevents network downloads; a Hub ID
    must already exist in the local cache unless remote download is explicitly
    enabled.  Importing this module and constructing the client never imports
    ``sentence_transformers`` or loads weights.  Loading happens on the first
    non-empty call to :meth:`score`, which keeps lightweight services and unit
    tests independent of the optional ML dependency.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        local_files_only: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """Initialize a local cross-encoder reranking model."""
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be None or a non-empty string")
        _validate_positive_integer(batch_size, "batch_size")
        _validate_positive_integer(max_length, "max_length")
        if not isinstance(local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        if not isinstance(use_fp16, bool):
            raise ValueError("use_fp16 must be a boolean")
        if (
            use_fp16
            and device is not None
            and not device.strip().casefold().startswith("cuda")
        ):
            raise ValueError("use_fp16 requires a CUDA device or automatic selection")

        self.model_name_or_path = model_name_or_path.strip()
        self.device = device.strip() if device is not None else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.use_fp16 = use_fp16
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Report whether model weights have already been loaded."""

        return self._model is not None

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """Return relevance scores in exactly the same order as ``documents``."""

        _validate_non_empty_text(query, "query")
        validated_documents = _validate_documents(documents)
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        _validate_positive_integer(effective_batch_size, "batch_size")

        if not validated_documents:
            return []

        model = self._get_or_load_model()
        pairs = [(query, document) for document in validated_documents]
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=effective_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerScoringError(
                "Cross-encoder scoring failed for "
                f"{len(validated_documents)} candidate(s)"
            ) from exc

        return _coerce_scores(raw_scores, expected_count=len(validated_documents))

    def _get_or_load_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                self._initialize_model()
        model = self._model
        if model is None:
            raise RerankerModelLoadError(
                "Cross-encoder initialization completed without a model instance"
            )
        return model

    def _initialize_model(self) -> None:
        """Load once while allowing another thread to win the initialization."""

        if self._model is None:
            self._model = self._load_model()

    def _load_model(self) -> Any:
        if "qwen3-vl-reranker" in self.model_name_or_path.casefold():
            return _load_qwen3_vl_reranker(
                self.model_name_or_path,
                device=self.device,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
                use_fp16=self.use_fp16,
            )
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise RerankerDependencyError(
                "Cross-encoder reranking requires the optional "
                "'sentence-transformers' package. Install it with "
                "`pip install sentence-transformers`."
            ) from exc

        try:
            cross_encoder_class = sentence_transformers.CrossEncoder
        except AttributeError as exc:
            raise RerankerDependencyError(
                "The installed 'sentence-transformers' package does not expose "
                "CrossEncoder. Upgrade it to a compatible version."
            ) from exc

        automodel_args: dict[str, Any] = {}
        if self.use_fp16:
            try:
                torch = import_module("torch")
                automodel_args["torch_dtype"] = torch.float16
            except (AttributeError, ImportError) as exc:
                raise RerankerDependencyError(
                    "FP16 reranking requires a compatible PyTorch installation"
                ) from exc

        try:
            return cross_encoder_class(
                self.model_name_or_path,
                device=self.device,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
                automodel_args=automodel_args,
            )
        except Exception as exc:
            raise RerankerModelLoadError(
                "Could not load cross-encoder from "
                f"'{self.model_name_or_path}'. Verify the local path or Hugging "
                "Face model ID, model files, cache/network access, and device."
            ) from exc


class _Qwen3VLReranker:
    """Small adapter exposing the same ``predict`` API as CrossEncoder.

    Qwen3-VL reranking is generative: the model scores the final yes/no
    decision token instead of returning a sequence-classification logit.
    Keeping this adapter private lets the rest of the retrieval contract stay
    unchanged.
    """

    def __init__(
        self, model: Any, processor: Any, device: str | None, max_length: int
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.max_length = max_length

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> list[float]:
        del show_progress_bar, convert_to_numpy
        import torch

        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            prompts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": (
                        "Given a query and a document, determine whether the "
                        "document is relevant to answering the query.\n"
                        f"Query: {query}\nDocument: {document}\n"
                        "Answer only yes or no."
                    )}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for query, document in batch
            ]
            inputs = self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
            logits = output.logits[:, -1, :]
            yes_ids = self.processor.tokenizer.encode(" yes", add_special_tokens=False)
            no_ids = self.processor.tokenizer.encode(" no", add_special_tokens=False)
            if not yes_ids or not no_ids:
                raise RerankerScoringError("Qwen3-VL tokenizer has no yes/no token")
            scores.extend((logits[:, yes_ids[-1]] - logits[:, no_ids[-1]]).float().tolist())
        return scores


def _load_qwen3_vl_reranker(
    model_name_or_path: str,
    *,
    device: str | None,
    max_length: int,
    local_files_only: bool,
    use_fp16: bool,
) -> _Qwen3VLReranker:
    """Load Qwen3-VL with Transformers, without importing it at module import."""

    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RerankerDependencyError(
            "Qwen3-VL reranking requires torch and transformers."
        ) from exc
    try:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if use_fp16 else "auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(target_device)
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        processor.tokenizer.model_max_length = max_length
        model.eval()
        return _Qwen3VLReranker(model, processor, target_device, max_length)
    except Exception as exc:
        raise RerankerModelLoadError(
            f"Could not load Qwen3-VL reranker from '{model_name_or_path}'."
        ) from exc


def _validate_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_documents(documents: Sequence[str]) -> tuple[str, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence of strings")

    validated = tuple(documents)
    for index, document in enumerate(validated):
        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"documents[{index}] must be a non-empty string")
    return validated


def _coerce_scores(raw_scores: Any, *, expected_count: int) -> list[float]:
    """Convert common tensor/array outputs into validated scalar scores."""

    values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if expected_count == 1 and _is_real_number(values):
        values = [values]
    elif not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RerankerScoringError("Cross-encoder returned a non-sequence score output")

    if len(values) != expected_count:
        raise RerankerScoringError(
            "Cross-encoder returned "
            f"{len(values)} score(s) for {expected_count} candidate(s)"
        )

    scores: list[float] = []
    for index, value in enumerate(values):
        scalar = value.tolist() if hasattr(value, "tolist") else value
        if (
            isinstance(scalar, Sequence)
            and not isinstance(scalar, (str, bytes))
            and len(scalar) == 1
        ):
            scalar = scalar[0]
        if not _is_real_number(scalar):
            raise RerankerScoringError(
                f"Cross-encoder score at index {index} is not a scalar real number"
            )
        score = float(scalar)
        if not math.isfinite(score):
            raise RerankerScoringError(
                f"Cross-encoder score at index {index} must be finite"
            )
        scores.append(score)
    return scores


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
