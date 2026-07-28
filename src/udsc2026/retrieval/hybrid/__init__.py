"""Hybrid dense and sparse retrieval with weighted score fusion."""

from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever, search
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores, normalize_scores

__all__ = ["HybridRetriever", "search", "fuse_scores", "normalize_scores"]
