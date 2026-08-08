"""Tests for compact parent-child metadata persisted through vector payloads."""

from udsc2026.contracts import LegalChunk
from udsc2026.infrastructure.vector_db.base import chunk_payload, payload_to_hit


def test_vector_payload_preserves_parent_link_without_repeating_parent_text():
    chunk = LegalChunk(
        chunk_id="law_article_1_clause_1",
        parent_id="law_article_1",
        doc_id="law",
        text="Nội dung khoản 1.",
        parent_text="Điều 1. Nội dung khoản 1.",
        law_name="Luật mẫu",
        article="Điều 1",
        clause="Khoản 1",
    )

    payload = chunk_payload(chunk)
    hit = payload_to_hit(payload, 0.9)

    assert payload["parent_id"] == "law_article_1"
    assert "parent_text" not in payload
    assert hit.metadata["parent_id"] == "law_article_1"
    assert "parent_text" not in hit.metadata
