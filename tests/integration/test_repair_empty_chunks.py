"""Coverage for repairing documents omitted by the old structure parser."""

import argparse
import json
from pathlib import Path

from scripts.data_prep.repair_empty_chunks import run

from udsc2026.ingestion.cleaners.models import CleanDocument


def _arguments(root: Path, *, apply: bool) -> argparse.Namespace:
    return argparse.Namespace(
        documents_dir=root / "documents",
        chunks_dir=root / "chunks",
        parents_dir=root / "parents",
        report=root / "metadata" / "repair.json",
        chunk_size=4,
        chunk_overlap=1,
        apply=apply,
    )


def test_repair_is_dry_run_by_default_and_preserves_existing_chunks(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, apply=False)
    arguments.documents_dir.mkdir()
    arguments.chunks_dir.mkdir()
    recoverable = CleanDocument(
        doc_id="missing",
        source_path="source.json",
        cleaned_text="má»™t hai ba bá»‘n nÄƒm sĂ¡u báº£y",
        file_format="json",
    )
    existing = recoverable.model_copy(update={"doc_id": "existing"})
    for document in (recoverable, existing):
        (arguments.documents_dir / f"{document.doc_id}.json").write_text(
            document.model_dump_json(), encoding="utf-8"
        )
    empty_path = arguments.chunks_dir / "missing.jsonl"
    empty_path.write_text("", encoding="utf-8")
    existing_path = arguments.chunks_dir / "existing.jsonl"
    existing_content = '{"chunk_id":"keep"}\n'
    existing_path.write_text(existing_content, encoding="utf-8")

    audited = run(arguments)

    assert audited["recoverable_document_count"] == 1
    assert audited["recovered_chunk_count"] == 2
    assert empty_path.read_text(encoding="utf-8") == ""
    assert existing_path.read_text(encoding="utf-8") == existing_content

    arguments.apply = True
    repaired = run(arguments)
    records = [
        json.loads(line) for line in empty_path.read_text(encoding="utf-8").splitlines()
    ]
    assert repaired["recoverable_document_count"] == 1
    assert [record["chunk_id"] for record in records] == [
        "missing_document_part_1",
        "missing_document_part_2",
    ]
    assert all(record["metadata"]["fallback_chunking"] for record in records)
    assert arguments.report.is_file()
