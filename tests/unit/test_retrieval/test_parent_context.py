"""Tests for lazy, bounded parent-context hydration."""

import json

import pytest

from udsc2026.contracts import LegalParent, RetrievalHit
from udsc2026.retrieval.parent_context import (
    JsonlParentStore,
    ParentContextExpander,
    ParentContextSettings,
    ParentStoreError,
)


def _write_parent(tmp_path, parent: LegalParent) -> None:
    path = tmp_path / "parents" / "{0}.jsonl".format(parent.doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(parent.model_dump(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _hit(chunk_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        parent_id="doc_article_1",
        doc_id="doc",
        text=text,
        law_name="Luật mẫu",
        article="Điều 1",
        clause="Khoản 1",
        score=0.9,
    )


def test_expander_deduplicates_children_and_uses_short_full_parent(tmp_path):
    parent = LegalParent(
        parent_id="doc_article_1",
        doc_id="doc",
        text="Điều 1. Mở đầu. Nội dung khoản một. Nội dung khoản hai.",
        law_name="Luật mẫu",
        article="Điều 1",
    )
    _write_parent(tmp_path, parent)
    hits = [
        _hit("child_1", "Nội dung khoản một."),
        _hit("child_2", "Nội dung khoản hai."),
    ]
    original = [hit.model_copy(deep=True) for hit in hits]
    expander = ParentContextExpander(JsonlParentStore(tmp_path / "parents"))

    expanded = expander.expand(hits)

    assert len(expanded) == 1
    assert expanded[0].text == parent.text
    assert expanded[0].clause is None
    assert expanded[0].metadata["parent_context_anchor_chunk_ids"] == [
        "child_1",
        "child_2",
    ]
    assert hits == original


def test_expander_uses_anchor_window_for_oversized_parent(tmp_path):
    tokens = ["token{0}".format(index) for index in range(300)]
    anchor = " ".join(tokens[145:150])
    parent = LegalParent(
        parent_id="doc_article_1",
        doc_id="doc",
        text=" ".join(tokens),
        law_name="Luật mẫu",
        article="Điều 1",
    )
    _write_parent(tmp_path, parent)
    expander = ParentContextExpander(
        JsonlParentStore(tmp_path / "parents"),
        max_parent_tokens=40,
        max_total_tokens=40,
    )

    expanded = expander.expand([_hit("child_anchor", anchor)])

    assert len(expanded[0].text.split()) <= 40
    assert anchor in expanded[0].text
    assert expanded[0].metadata["parent_context_truncated"] is True


def test_expander_falls_back_to_children_when_parent_is_missing(tmp_path):
    hit = _hit("child", "Nội dung child an toàn.")
    expander = ParentContextExpander(JsonlParentStore(tmp_path / "parents"))

    expanded = expander.expand([hit])

    assert expanded[0].text == hit.text
    assert expanded[0].metadata["parent_context_fallback"] is True


def test_parent_store_rejects_path_traversal(tmp_path):
    store = JsonlParentStore(tmp_path / "parents")

    with pytest.raises(ParentStoreError, match="unsafe doc_id"):
        store.get("../secret", "parent")


def test_short_parent_preserves_legal_line_breaks(tmp_path):
    parent = LegalParent(
        parent_id="doc_article_1",
        doc_id="doc",
        text="Điều 1. Tiêu đề\n1. Khoản thứ nhất\na) Điểm thứ nhất",
        article="Điều 1",
    )
    _write_parent(tmp_path, parent)

    expanded = ParentContextExpander(JsonlParentStore(tmp_path / "parents")).expand(
        [_hit("child", "1. Khoản thứ nhất")]
    )

    assert expanded[0].text == parent.text


def test_conflicting_first_class_and_metadata_parent_ids_fall_back(tmp_path):
    hit = _hit("child", "Nội dung child phải được giữ lại.").model_copy(
        update={"metadata": {"parent_id": "different_parent"}}
    )
    parent = LegalParent(
        parent_id="doc_article_1",
        doc_id="doc",
        text="Nội dung child phải được giữ lại. Phần parent bổ sung.",
        article="Điều 1",
    )
    _write_parent(tmp_path, parent)

    expanded = ParentContextExpander(JsonlParentStore(tmp_path / "parents")).expand(
        [hit]
    )

    assert expanded[0].text == hit.text
    assert expanded[0].metadata["parent_context_fallback"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": "false"},
        {"max_cached_documents": 1.5},
        {"max_parent_tokens": 20, "max_total_tokens": 10},
        {"unexpected": True},
    ],
)
def test_parent_context_settings_reject_ambiguous_or_invalid_values(payload):
    with pytest.raises(ValueError):
        ParentContextSettings.model_validate(payload)
