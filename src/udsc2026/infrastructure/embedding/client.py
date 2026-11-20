"""Sentence-Transformer client used to create Vietnamese retrieval vectors."""

import math
from pathlib import Path
from typing import Any, Protocol, cast

from udsc2026.retrieval.tokenization import tokenize_vi

SentenceTransformer: Any
try:  # Keep config/tests importable without installing optional ML packages.
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover - exercised on lightweight environments
    SentenceTransformer = None
else:
    SentenceTransformer = _SentenceTransformer


class _SupportsToList(Protocol):
    """Structural result type returned when encode converts output to NumPy."""

    def tolist(self) -> Any:
        """Convert the array-like result into nested Python lists."""


class EmbeddingClient:
    """Encode raw text as normalized float vectors for indexing and query search."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
        query_max_length: int | None = None,
        document_max_length: int | None = None,
        normalize_embeddings: bool = True,
        output_dimension: int | None = None,
        preprocessor: str = "pyvi",
        window_long_texts: bool = False,
        window_overlap_tokens: int = 32,
    ) -> None:
        """Initialize a model from a configured path or an explicitly allowed Hugging Face ID."""
        if not model_path:
            raise ValueError("model_path must not be empty")
        query_length = max_length if query_max_length is None else query_max_length
        document_length = (
            max_length if document_max_length is None else document_max_length
        )
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be greater than zero")
        if query_length <= 0 or document_length <= 0:
            raise ValueError("query/document max lengths must be greater than zero")
        if output_dimension is not None and output_dimension <= 0:
            raise ValueError("output_dimension must be greater than zero")
        if preprocessor not in {"identity", "pyvi"}:
            raise ValueError("preprocessor must be 'identity' or 'pyvi'")
        if not isinstance(window_long_texts, bool):
            raise ValueError("window_long_texts must be a boolean")
        if window_overlap_tokens < 0 or window_overlap_tokens >= document_length:
            raise ValueError(
                "window_overlap_tokens must be non-negative and less than "
                "document_max_length"
            )
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"configured embedding model path not found: {model_path}"
            )
        if SentenceTransformer is None:
            raise ImportError(
                "Embedding requires sentence-transformers. Install the "
                "retrieval optional dependencies first."
            )
        self.batch_size = batch_size
        self.query_max_length = query_length
        self.document_max_length = document_length
        self.normalize_embeddings = normalize_embeddings
        self.output_dimension = output_dimension
        self.preprocessor = preprocessor
        self.window_long_texts = window_long_texts
        self.window_overlap_tokens = window_overlap_tokens
        self.windowed_document_count = 0
        self.encoded_window_count = 0
        self.model = SentenceTransformer(
            model_path,
            device=device,
        )
        self.model.max_seq_length = document_length

    def _prepare_text(self, text: str) -> str:
        """Apply the configured model input normalization."""
        if self.preprocessor == "identity":
            return text
        return " ".join(tokenize_vi(text))

    def _finalize_vector(self, vector: Any) -> list[float]:
        raw_values = vector.tolist() if hasattr(vector, "tolist") else vector
        values = [float(value) for value in raw_values]
        if self.output_dimension is None:
            return values
        if self.output_dimension > len(values):
            raise ValueError(
                "output_dimension exceeds model embedding size "
                f"({self.output_dimension} > {len(values)})"
            )
        truncated = values[: self.output_dimension]
        if not self.normalize_embeddings:
            return truncated
        norm = math.sqrt(sum(value * value for value in truncated))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("embedding model returned a zero or non-finite vector")
        return [value / norm for value in truncated]

    def _token_windows(self, prepared_text: str, max_length: int) -> list[str]:
        """Split an over-length input with the model tokenizer when requested."""
        if not self.window_long_texts:
            return [prepared_text]
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return [prepared_text]
        try:
            token_ids = tokenizer.encode(prepared_text, add_special_tokens=False)
        except (AttributeError, TypeError, ValueError):
            return [prepared_text]
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            return [prepared_text]
        try:
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        except (AttributeError, TypeError, ValueError):
            special_tokens = 2
        capacity = max_length - max(special_tokens, 0)
        if capacity <= 0:
            raise ValueError("max_length is too small for the tokenizer special tokens")
        if len(token_ids) <= capacity:
            return [prepared_text]
        overlap = min(self.window_overlap_tokens, capacity - 1)
        step = capacity - overlap
        windows: list[str] = []
        for start in range(0, len(token_ids), step):
            window_ids = token_ids[start : start + capacity]
            if not window_ids:
                break
            decoded = tokenizer.decode(
                window_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, str) or not decoded.strip():
                raise ValueError("embedding tokenizer produced an empty text window")
            windows.append(decoded.strip())
            if start + capacity >= len(token_ids):
                break
        return windows

    @staticmethod
    def _mean_vector(vectors: list[list[float]], *, normalize: bool) -> list[float]:
        if not vectors:
            raise ValueError("cannot aggregate an empty embedding window list")
        dimension = len(vectors[0])
        if dimension == 0 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding windows returned inconsistent dimensions")
        mean = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(dimension)
        ]
        if not normalize:
            return mean
        norm = math.sqrt(sum(value * value for value in mean))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("embedding windows produced a zero or non-finite vector")
        return [value / norm for value in mean]

    def embed_query(self, query: str) -> list[float]:
        """Encode one non-empty query into a Python list of floats."""
        return self.embed_queries([query])[0]

    def embed_queries(
        self, queries: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Encode query texts with the configured query token limit."""
        if not queries:
            raise ValueError("queries must not be empty")
        if any(query is None or not query.strip() for query in queries):
            raise ValueError("queries must contain only non-empty strings")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model.max_seq_length = self.query_max_length
        vectors = cast(
            _SupportsToList,
            self.model.encode(
                [self._prepare_text(query) for query in queries],
                batch_size=effective_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        rows = cast(list[list[float]], vectors.tolist())
        if len(rows) != len(queries):
            raise ValueError(
                f"model returned {len(rows)} vectors for {len(queries)} queries"
            )
        return [self._finalize_vector(vector) for vector in rows]

    def embed_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Encode non-empty documents in batches into lists of floats."""
        if not texts:
            raise ValueError("texts must not be empty")
        if any(text is None or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model.max_seq_length = self.document_max_length
        expanded_texts: list[str] = []
        owners: list[int] = []
        window_counts: list[int] = []
        for owner, text in enumerate(texts):
            windows = self._token_windows(
                self._prepare_text(text), self.document_max_length
            )
            expanded_texts.extend(windows)
            owners.extend([owner] * len(windows))
            window_counts.append(len(windows))
        self.windowed_document_count += sum(count > 1 for count in window_counts)
        self.encoded_window_count += len(expanded_texts)
        vectors = cast(
            _SupportsToList,
            self.model.encode(
                expanded_texts,
                batch_size=effective_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        rows = cast(list[list[float]], vectors.tolist())
        if len(rows) != len(owners):
            raise ValueError(
                f"model returned {len(rows)} vectors for {len(owners)} text windows"
            )
        grouped: list[list[list[float]]] = [[] for _ in texts]
        for owner, vector in zip(owners, rows):
            grouped[owner].append(vector)
        aggregated = [
            self._mean_vector(group, normalize=self.normalize_embeddings)
            if len(group) > 1
            else group[0]
            for group in grouped
        ]
        return [self._finalize_vector(vector) for vector in aggregated]

    def embed_documents_resilient(
        self, items: list[tuple[str, str]], batch_size: int | None = None
    ) -> tuple[list[tuple[str, list[float]]], list[dict[str, str]]]:
        """Encode batches while recording failed chunk IDs instead of aborting.

        The strict :meth:`embed_documents` API remains unchanged. This method is
        intended for large indexing/benchmark jobs where one bad document must
        not discard all successfully encoded batches.
        """
        if not items:
            raise ValueError("items must not be empty")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        encoded: list[tuple[str, list[float]]] = []
        errors: list[dict[str, str]] = []
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            try:
                vectors = self.embed_documents(
                    [text for _, text in batch], batch_size=effective_batch_size
                )
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"model returned {len(vectors)} vectors for {len(batch)} texts"
                    )
                encoded.extend(
                    (chunk_id, vector) for (chunk_id, _), vector in zip(batch, vectors)
                )
            except Exception as exc:  # noqa: BLE001 - batch boundary must be resilient
                for chunk_id, _ in batch:
                    errors.append({"chunk_id": chunk_id, "error": str(exc)})
        return encoded, errors
