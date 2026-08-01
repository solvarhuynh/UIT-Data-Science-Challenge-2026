"""Hybrid dense and sparse retrieval with weighted score fusion."""

from udsc2026.retrieval.hybrid.config import HybridSettings, load_hybrid_settings
from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever, search
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores, normalize_scores

__all__ = [
    "HybridRetriever",
    "HybridSettings",
    "fuse_scores",
    "load_hybrid_settings",
    "normalize_scores",
    "search",
]
