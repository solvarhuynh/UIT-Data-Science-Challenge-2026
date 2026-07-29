
"""Evaluation datasets and benchmark helpers for the legal RAG system."""

from udsc2026.evaluation.synthetic_generator import (
    SyntheticQA,
    generate_synthetic_benchmark,
    load_legal_chunks,
    write_synthetic_benchmark,
)

__all__ = [
    "SyntheticQA",
    "generate_synthetic_benchmark",
    "load_legal_chunks",
    "write_synthetic_benchmark",
]
