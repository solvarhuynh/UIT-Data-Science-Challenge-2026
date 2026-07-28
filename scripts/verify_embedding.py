"""Manual smoke check for the local BKAI embedding client."""

from udsc2026.infrastructure.embedding import EmbeddingClient
from udsc2026.infrastructure.embedding.config import load_embedding_config


def main() -> None:
    config = load_embedding_config()
    client = EmbeddingClient(model_path=config["embedder_model_path"])
    query_vector = client.embed_query("Điều 10 Bộ luật Lao động quy định gì?")
    document_vectors = client.embed_documents(["a", "b"], batch_size=2)
    print(f"query dimension: {len(query_vector)}")
    print(f"document vectors: {len(document_vectors)}, dimension: {len(document_vectors[0])}")


if __name__ == "__main__":
    main()
