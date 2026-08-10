"""Build a reusable dense-vector cache for the processed v3 parent corpus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "experiments/tv3"))

from mock_hybrid_retriever import MockHybridRetriever

LOGGER = logging.getLogger("build_vector_cache")

DEFAULT_EMBEDDING_MODEL = "huyydangg/DEk21_hcmute_embedding_v2"
DEFAULT_PARENTS_DIR = BASE_DIR / "data/processed_v3/parents"
DEFAULT_CACHE_DIR = BASE_DIR / "artifacts/tv3/cache"
LOCAL_EMBEDDING_DIR = BASE_DIR / "models/dek21-v2"


def _resolve_local_model(
    requested: str,
    default_id: str = DEFAULT_EMBEDDING_MODEL,
    local_dir: Path = LOCAL_EMBEDDING_DIR,
) -> str:
    """Prefer the repository model only when the default id was requested."""

    if requested == default_id and local_dir.is_dir():
        return str(local_dir)
    return requested


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when a requested CUDA device is unavailable."""

    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return requested


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the processed-v3 parent embeddings cache."
    )
    parser.add_argument(
        "--parents_dir",
        "--parents-dir",
        dest="parents_dir",
        type=Path,
        default=DEFAULT_PARENTS_DIR,
    )
    parser.add_argument(
        "--embedding_model",
        "--embedding-model",
        dest="embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        dest="cache_dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_docs",
        "--max-docs",
        dest="max_docs",
        type=int,
        default=0,
        help="Maximum parent records to index; 0 loads the complete corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_docs < 0:
        parser.error("--max_docs must be non-negative")

    parents_dir = args.parents_dir.resolve()
    if not parents_dir.is_dir():
        raise FileNotFoundError(
            f"processed-v3 parents directory does not exist: {parents_dir}"
        )
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    embedding_model = _resolve_local_model(args.embedding_model)
    device = _resolve_device(args.device)
    LOGGER.info(
        "Building vector cache | parents=%s | cache=%s | model=%s | device=%s",
        parents_dir,
        cache_dir,
        embedding_model,
        device,
    )
    with MockHybridRetriever(
        parents_dir=parents_dir,
        embedding_model_path=embedding_model,
        device=device,
        cache_dir=cache_dir,
        max_docs=args.max_docs,
        enable_dense=True,
    ) as retriever:
        if retriever._doc_embeddings is None:
            raise RuntimeError("dense vector cache was not initialized")
        LOGGER.info(
            "Dense vector cache shape: %s",
            retriever._doc_embeddings.shape,
        )
    LOGGER.info("Vector cache is ready: %s", cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
