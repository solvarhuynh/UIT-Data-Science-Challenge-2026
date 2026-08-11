"""CPU tests for leakage-safe canonical Task1 positive evidence."""

from __future__ import annotations

import json
from pathlib import Path

from udsc2026.evaluation.task1_canonical_evidence import CanonicalEvidenceResolver


def test_gold_is_resolved_when_dense_candidates_do_not_contain_it(tmp_path: Path) -> None:
    root = tmp_path / "processed_v3"
    (root / "documents").mkdir(parents=True)
    (root / "parents").mkdir()
    (root / "documents" / "gold.json").write_text(
        json.dumps({"doc_id": "gold", "law_name": "Luật thử"}), encoding="utf-8"
    )
    (root / "parents" / "gold.jsonl").write_text(
        json.dumps({"parent_id": "gold_article_1", "doc_id": "gold", "text": "Điều 1 quy định thử."}) + "\n",
        encoding="utf-8",
    )
    resolved = CanonicalEvidenceResolver(root).resolve("gold")
    assert resolved is not None
    assert resolved.source == "canonical_corpus"
    assert resolved.evidence_ids == ("gold_article_1",)
    assert "quy định thử" in resolved.text
