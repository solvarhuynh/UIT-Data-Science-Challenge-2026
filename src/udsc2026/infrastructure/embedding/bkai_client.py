"""Local BKAI bi-encoder client used to create retrieval embeddings."""

from pathlib import Path

from sentence_transformers import SentenceTransformer


class EmbeddingClient:
    """Encode raw text as normalized float vectors for indexing and query search."""

    def __init__(
        self, model_path: str, device: str = "cpu", batch_size: int = 32,
        max_length: int = 256, normalize_embeddings: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must not be empty")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be greater than zero")
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Local embedding model path not found: {model_path}")
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = max_length

    def embed_query(self, query: str) -> list[float]:
        """Encode one non-empty query into a Python list of floats."""
        if query is None or not query.strip():
            raise ValueError("query must not be empty")
        vector = self.model.encode(query, batch_size=1,
                                   normalize_embeddings=self.normalize_embeddings,
                                   convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode non-empty documents in batches into lists of floats."""
        if not texts:
            raise ValueError("texts must not be empty")
        if any(text is None or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        vectors = self.model.encode(texts, batch_size=effective_batch_size,
                                    normalize_embeddings=self.normalize_embeddings,
                                    convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
