"""Local HCMUTE embedding v2 client used to create retrieval embeddings."""

from pathlib import Path
from typing import Any, Protocol, cast

from sentence_transformers import SentenceTransformer

from udsc2026.retrieval.sparse.tokenizer import tokenize_vi


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
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the local HCMUTE Embedding v2 embedding model."""
        if not model_path:
            raise ValueError("model_path must not be empty")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be greater than zero")
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"Local embedding model path not found: {model_path}"
            )
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(
            model_path, device=device, local_files_only=True
        )
        self.model.max_seq_length = max_length

    def _prepare_text(self, text: str) -> str:
        """Segment Vietnamese text before embedding so legal terms and word boundaries are preserved better."""
        tokens = tokenize_vi(text)
        return " ".join(tokens)

    def embed_query(self, query: str) -> list[float]:
        """Encode one non-empty query into a Python list of floats."""
        if query is None or not query.strip():
            raise ValueError("query must not be empty")
        vector = cast(
            _SupportsToList,
            self.model.encode(
                self._prepare_text(query),
                batch_size=1,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        return cast(list[float], vector.tolist())

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
        prepared_texts = [self._prepare_text(text) for text in texts]
        vectors = cast(
            _SupportsToList,
            self.model.encode(
                prepared_texts,
                batch_size=effective_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        return cast(list[list[float]], vectors.tolist())

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
