"""Manual smoke check for the local BKAI embedding client."""

from udsc2026.infrastructure.config import load_config
from udsc2026.infrastructure.embedding import EmbeddingClient


def main() -> None:
    config = load_config()
    embedding = config["embedding"]
    client = EmbeddingClient(
        model_path=embedding["model_path"],
        device=embedding.get("device", "cpu"),
        batch_size=embedding.get("batch_size", 32),
        max_length=embedding.get("max_length", 256),
        normalize_embeddings=embedding.get("normalize_embeddings", True),
    )
    query_vector = client.embed_query("Điều 10 Bộ luật Lao động quy định gì?")
    document_vectors = client.embed_documents(["a", "b"], batch_size=2)
    print(f"query dimension: {len(query_vector)}")
    document_count = len(document_vectors)
    vector_dimension = len(document_vectors[0])
    print(f"document vectors: {document_count}, dimension: {vector_dimension}")


if __name__ == "__main__":
    main()
