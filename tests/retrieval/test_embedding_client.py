from unittest.mock import patch

import numpy as np
import pytest

from udsc2026.infrastructure.embedding.client import EmbeddingClient
from udsc2026.retrieval.sparse.tokenizer import tokenize_vi


@pytest.fixture
def client():
    with patch("udsc2026.infrastructure.embedding.client.SentenceTransformer") as model:
        model.return_value.encode.side_effect = lambda texts, **kwargs: np.asarray(
            [0.1, 0.2, 0.3]
            if isinstance(texts, str)
            else [[0.1, 0.2, 0.3] for _ in texts]
        )
        with patch(
            "udsc2026.infrastructure.embedding.client.Path.is_dir",
            return_value=True,
        ):
            yield EmbeddingClient("local-model")


def test_embed_query_returns_float_list(client):
    vector = client.embed_query("Điều 10 quy định gì?")
    assert isinstance(vector, list)
    assert all(isinstance(value, float) for value in vector)


def test_embed_documents_preserves_count(client):
    vectors = client.embed_documents(["a", "b", "c"], batch_size=2)
    assert len(vectors) == 3
    assert all(isinstance(vector, list) for vector in vectors)


def test_embed_query_applies_vietnamese_segmentation_before_encoding(client):
    seen = {}

    def encode(texts, **kwargs):
        seen["texts"] = texts
        return np.asarray(
            [0.1, 0.2, 0.3]
            if isinstance(texts, str)
            else [[0.1, 0.2, 0.3] for _ in texts]
        )

    client.model.encode.side_effect = encode

    client.embed_query("Điều 10 quy định gì?")

    assert seen["texts"] == " ".join(tokenize_vi("Điều 10 quy định gì?"))


@pytest.mark.parametrize(
    "method, value", [("embed_query", ""), ("embed_documents", [])]
)
def test_empty_input_raises(client, method, value):
    with pytest.raises(ValueError):
        getattr(client, method)(value)


def test_resilient_embedding_skips_failed_batch_and_reports_chunk_ids(client):
    calls = {"count": 0}

    def encode_batch(texts, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("max_length exceeded")
        return np.asarray([[0.1, 0.2, 0.3] for _ in texts])

    client.model.encode.side_effect = encode_batch
    encoded, errors = client.embed_documents_resilient(
        [("c1", "a"), ("c2", "b"), ("c3", "c")], batch_size=2
    )

    assert [chunk_id for chunk_id, _ in encoded] == ["c1", "c2"]
    assert errors == [{"chunk_id": "c3", "error": "max_length exceeded"}]


def test_long_documents_are_windowed_and_aggregated(client):
    client.window_long_texts = True
    client.max_length = 6
    client.window_overlap_tokens = 1
    client.model.tokenizer.encode.return_value = list(range(10))
    client.model.tokenizer.num_special_tokens_to_add.return_value = 2
    client.model.tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(
        str(token_id) for token_id in ids
    )
    client.model.encode.side_effect = lambda texts, **kwargs: np.asarray(
        [[1.0, 0.0, 0.0] for _ in texts]
    )

    vectors = client.embed_documents(["điều luật rất dài"])

    assert vectors == [[1.0, 0.0, 0.0]]
    assert client.windowed_document_count == 1
    assert client.encoded_window_count == 3
