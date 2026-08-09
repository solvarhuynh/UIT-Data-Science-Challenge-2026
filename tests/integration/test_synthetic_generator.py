"""Tests for citation-grounded synthetic Legal RAG benchmark generation."""

import json

import pytest

from udsc2026.contracts import LegalChunk
from udsc2026.evaluation.loaders import load_benchmark
from udsc2026.evaluation.synthetic_generator import (
    QUESTION_TYPES,
    generate_synthetic_benchmark,
    iter_legal_chunks,
    load_legal_chunks,
    regenerate_synthetic_benchmark,
    validate_synthetic_benchmark_sources,
    write_synthetic_benchmark,
)


def _chunk(chunk_id, text, article="Điều 10", clause="Khoản 1"):
    return LegalChunk(
        chunk_id=chunk_id,
        doc_id="law-2019",
        text=text,
        law_name="Bộ luật Lao động 2019",
        article=article,
        clause=clause,
        source="data/raw/btc/law-2019.txt",
    )


def _coverage_chunks():
    return [
        _chunk(
            "law_article_10_clause_1", "Khái niệm hợp đồng lao động là sự thỏa thuận."
        ),
        _chunk(
            "law_article_10_clause_2",
            "Người lao động có quyền yêu cầu bảo vệ quyền lợi.",
            clause="Khoản 2",
        ),
        _chunk(
            "law_article_10_clause_3",
            "Trường hợp áp dụng khi hợp đồng còn hiệu lực.",
            clause="Khoản 3",
        ),
        _chunk(
            "law_article_11_clause_1",
            "Mức phạt tiền đối với hành vi vi phạm là 5 triệu đồng.",
            article="Điều 11",
        ),
        _chunk(
            "law_article_12_clause_1",
            "Hồ sơ và trình tự thực hiện thủ tục được quy định như sau.",
            article="Điều 12",
        ),
    ]


def test_generator_builds_100_to_200_grounded_records_for_all_question_types():
    source_chunks = _coverage_chunks()

    records = generate_synthetic_benchmark(source_chunks, target_count=105, seed=7)

    assert len(records) == 105
    assert records[0].question_id == "syn_0001"
    assert records[-1].question_id == "syn_0105"
    assert {record.question_type for record in records} == set(QUESTION_TYPES)
    source_ids = {chunk.chunk_id for chunk in source_chunks}
    for record in records:
        assert record.question
        assert record.answer
        assert record.answer.startswith("Căn cứ ")
        assert record.law_name == "Bộ luật Lao động 2019"
        assert record.article
        assert record.gold_chunk_ids
        assert set(record.gold_chunk_ids).issubset(source_ids)
        assert len(record.gold_chunk_ids) == len(record.gold_citations)
        assert len(record.gold_chunk_ids) == len(record.metadata["gold_chunk_sha256"])
        assert record.metadata["source_chunk_count"] == len(source_chunks)
    assert any(record.difficulty == "hard" for record in records)


def test_generator_requires_grounded_coverage_by_default():
    chunks = [_chunk("only-right", "Người lao động có quyền được bảo vệ.")]

    with pytest.raises(ValueError, match="Corpus lacks grounded chunks"):
        generate_synthetic_benchmark(chunks)


def test_generator_can_use_available_types_for_exploratory_corpora():
    chunks = [_chunk("only-right", "Người lao động có quyền được bảo vệ.")]

    records = generate_synthetic_benchmark(
        chunks, target_count=100, require_all_question_types=False
    )

    assert len(records) == 100
    assert {record.question_type for record in records} == {"rights_obligations"}


def test_jsonl_round_trip_preserves_benchmark_contract(tmp_path):
    records = generate_synthetic_benchmark(_coverage_chunks(), target_count=100)
    output_path = write_synthetic_benchmark(records, tmp_path / "synthetic.jsonl")
    lines = output_path.read_text(encoding="utf-8").splitlines()
    decoded = [json.loads(line) for line in lines]

    assert len(decoded) == 100
    assert all("gold_chunk_ids" in record for record in decoded)
    assert all("gold_citations" in record for record in decoded)


def test_chunk_loader_reads_jsonl_records_written_by_chunking(tmp_path):
    chunk = _coverage_chunks()[0]
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    loaded = load_legal_chunks(path)

    assert loaded == [chunk]


def test_streaming_regeneration_binds_gold_ids_to_content_hashes(tmp_path):
    chunks_dir = tmp_path / "processed_v3" / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks_path = chunks_dir / "law-2019.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
            for chunk in _coverage_chunks()
        ),
        encoding="utf-8",
    )
    output = tmp_path / "processed_v3" / "benchmarks" / "synthetic_qa.jsonl"

    artifact = regenerate_synthetic_benchmark(
        chunks_dir, output, target_count=100, seed=7
    )

    assert artifact.record_count == 100
    assert artifact.source_chunk_count == len(_coverage_chunks())
    assert len(artifact.source_chunk_corpus_sha256) == 64
    assert len(artifact.benchmark_sha256) == 64
    assert output.is_file()
    assert len(load_benchmark(output)) == 100
    records = generate_synthetic_benchmark(
        iter_legal_chunks(chunks_dir), target_count=100, seed=7
    )
    assert validate_synthetic_benchmark_sources(records, chunks_dir) == (
        artifact.source_chunk_count,
        artifact.source_chunk_corpus_sha256,
    )


def test_source_validation_rejects_a_gold_id_whose_text_changed(tmp_path):
    chunks = _coverage_chunks()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    records = generate_synthetic_benchmark(
        iter_legal_chunks(chunks_path), target_count=100, seed=7
    )
    gold_id = records[0].gold_chunk_ids[0]
    changed = [
        chunk.model_copy(update={"text": "Ná»™i dung Ä‘Ă£ bá»‹ thay Ä‘á»•i."})
        if chunk.chunk_id == gold_id
        else chunk
        for chunk in chunks
    ]
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
            for chunk in changed
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gold chunk identity changed"):
        validate_synthetic_benchmark_sources(records, chunks_path)


def test_source_validation_rejects_changed_gold_citation_provenance(tmp_path):
    chunks = _coverage_chunks()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    records = generate_synthetic_benchmark(
        iter_legal_chunks(chunks_path), target_count=100, seed=7
    )
    gold_id = records[0].gold_chunk_ids[0]
    changed = [
        chunk.model_copy(update={"article": "\u0110i\u1ec1u 999"})
        if chunk.chunk_id == gold_id
        else chunk
        for chunk in chunks
    ]
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
            for chunk in changed
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gold chunk identity changed"):
        validate_synthetic_benchmark_sources(records, chunks_path)
