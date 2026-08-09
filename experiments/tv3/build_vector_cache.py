"""Utility script to build and export vector embeddings cache (.npy) to Kaggle Output directory."""

import sys
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "experiments/tv3"))

from mock_hybrid_retriever import MockHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("build_vector_cache")

def main():
    parser = argparse.ArgumentParser(description="Export parent embeddings cache .npy")
    parser.add_argument("--parents_dir", type=str, default="/kaggle/input/datasets/phamthequan/uit-ds-task2-test/processed/processed/chunks")
    parser.add_argument("--embedding_model", type=str, default="huyydangg/DEk21_hcmute_embedding_v2")
    parser.add_argument("--cache_dir", type=str, default="/kaggle/working")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Đang khởi chạy Build Vector Embeddings Cache sang '%s'...", cache_dir)
    retriever = MockHybridRetriever(
        parents_dir=Path(args.parents_dir),
        embedding_model_path=args.embedding_model,
        device=args.device,
        cache_dir=cache_dir,
    )
    logger.info("🎉 Đã xuất thành công file Vector Embeddings Cache tại: %s!", cache_dir)

if __name__ == "__main__":
    main()
