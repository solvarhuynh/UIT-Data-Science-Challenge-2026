"""Write a TV2-only baseline submission for Task 1 (LegalIR).

This script intentionally does not import or call any reranker. It searches with
HybridRetriever when dense and sparse indexes are available, and falls back to
BM25 when the dense side is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        default="data/raw/btc/LegalIR/public-official.json",
        help="LegalIR question JSON; data/task1/warmup.json is also supported.",
    )
    parser.add_argument("--output-json", default="artifacts/task1/baseline_submission.json")
    parser.add_argument("--output-zip", default="artifacts/task1/baseline_submission.zip")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--mode", choices=("hybrid", "sparse", "dense"), default="hybrid")
    return parser.parse_args()


def load_questions(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.items()
        return [(str(question_id), str(record["question"])) for question_id, record in records]
    if isinstance(payload, list):
        return [(str(record["id"]), str(record["question"])) for record in payload]
    raise ValueError("questions JSON must be an object or array")


def build_retriever(mode: str) -> Any:
    from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

    if mode == "sparse":
        from udsc2026.config import load_project_config

        config = load_project_config()
        path = config.get("sparse", {}).get("bm25_index_path")
        retriever = BM25Retriever(path)
        retriever.load()
        return retriever

    if mode == "dense":
        from udsc2026.config import load_project_config
        from udsc2026.infrastructure.embedding.client import EmbeddingClient
        from udsc2026.infrastructure.embedding.config import load_embedding_config
        from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
        from udsc2026.retrieval.dense.dense_retriever import DenseRetriever

        config = load_project_config()
        embedding_config = load_embedding_config()
        client = EmbeddingClient(
            model_path=embedding_config["embedder_model_path"],
            device=embedding_config.get("device", "cpu"),
            batch_size=embedding_config.get("batch_size", 32),
            max_length=embedding_config.get("max_length", 256),
            normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        )
        return DenseRetriever(client, get_vector_db_adapter(config))

    from udsc2026.config import load_project_config
    from udsc2026.infrastructure.embedding.client import EmbeddingClient
    from udsc2026.infrastructure.embedding.config import load_embedding_config
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
    from udsc2026.retrieval.dense.dense_retriever import DenseRetriever
    from udsc2026.retrieval.hybrid.config import load_hybrid_settings
    from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever

    config = load_project_config()
    embedding_config = load_embedding_config()
    settings = load_hybrid_settings()
    client = EmbeddingClient(
        model_path=embedding_config["embedder_model_path"],
        device=embedding_config.get("device", "cpu"),
        batch_size=embedding_config.get("batch_size", 32),
        max_length=embedding_config.get("max_length", 256),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )
    dense = DenseRetriever(client, get_vector_db_adapter(config))
    sparse = BM25Retriever(settings.bm25_index_path)
    sparse.load()
    return HybridRetriever(
        dense,
        sparse,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        candidate_k=settings.candidate_k,
        min_score=settings.min_score,
    )


def hit_document_id(hit: Any) -> str:
    metadata = getattr(hit, "metadata", {}) or {}
    parent_id = metadata.get("parent_id")
    if isinstance(parent_id, str) and parent_id.strip():
        return parent_id
    doc_id = getattr(hit, "doc_id", None)
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"retrieval hit {getattr(hit, 'chunk_id', '<unknown>')} has no doc_id")
    return doc_id


def write_artifacts(records: list[dict[str, Any]], json_path: Path, zip_path: Path) -> None:
    encoded = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(encoded, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", encoded.encode("utf-8"))


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.final_k <= 0 or args.final_k > 5:
        raise ValueError("top-k must be positive and final-k must be between 1 and 5")
    questions = load_questions(REPO_ROOT / args.questions)
    try:
        retriever = build_retriever(args.mode)
        selected_mode = args.mode
    except Exception as exc:
        if args.mode != "hybrid":
            raise
        print(f"Hybrid unavailable ({exc}); falling back to BM25.", file=sys.stderr)
        retriever = build_retriever("sparse")
        selected_mode = "sparse"

    predictions: list[dict[str, Any]] = []
    for index, (question_id, question) in enumerate(questions, 1):
        hits = retriever.search(question, args.top_k)
        documents: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            document_id = hit_document_id(hit)
            if document_id not in seen:
                seen.add(document_id)
                documents.append(document_id)
            if len(documents) == args.final_k:
                break
        predictions.append({"id": question_id, "documents": documents})
        if index % 100 == 0 or index == len(questions):
            print(f"Processed {index}/{len(questions)} questions")

    write_artifacts(
        predictions,
        REPO_ROOT / args.output_json,
        REPO_ROOT / args.output_zip,
    )
    print(f"Mode: {selected_mode}")
    print(f"Questions: {len(predictions)}")
    print(f"JSON: {REPO_ROOT / args.output_json}")
    print(f"ZIP: {REPO_ROOT / args.output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


