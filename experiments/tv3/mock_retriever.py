"""Mock BM25 Retriever for TV3 local testing.

Loads all parent chunks from data/processed/parents/*.jsonl, builds a BM25 index
in-memory, and provides a simple search(question, top_k) method that returns
a list of RetrievalHit objects. This mimics TV2's retrieval output so that TV3
can test its QAEngine with REAL legal passages instead of golden answers.

NOT FOR PRODUCTION USE. This is a temporary local-only tool for TV3 benchmark.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class MockBM25Retriever:
    """Simple BM25 retriever over parent chunks for TV3 local testing."""

    def __init__(self, parents_dir: str | Path, max_docs: int = 0) -> None:
        """Load all parent JSONL files and build BM25 index.

        Args:
            parents_dir: Path to data/processed/parents/
            max_docs: Maximum number of parent docs to load (0 = all).
                      Use a small number (e.g. 5000) for faster startup.
        """
        self._docs: list[dict] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        self._load_corpus(Path(parents_dir), max_docs)
        self._build_index()

    def _load_corpus(self, parents_dir: Path, max_docs: int) -> None:
        """Read all *.jsonl files from parents directory."""
        files = sorted(parents_dir.glob("*.jsonl"))
        logger.info("Found %d parent JSONL files in '%s'.", len(files), parents_dir)

        count = 0
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        if len(text) < 20:
                            continue
                        self._docs.append(doc)
                        # Tokenize bằng split() đơn giản (khớp với BTC scoring)
                        self._corpus_tokens.append(text.lower().split())
                        count += 1
                        if max_docs > 0 and count >= max_docs:
                            logger.info("Reached max_docs=%d, stopping load.", max_docs)
                            return
                    except json.JSONDecodeError:
                        continue

        logger.info("Loaded %d parent documents into corpus.", len(self._docs))

    def _build_index(self) -> None:
        """Build BM25 index from loaded corpus."""
        if not self._corpus_tokens:
            logger.warning("Empty corpus, BM25 index not built.")
            return
        logger.info("Building BM25 index over %d documents...", len(self._corpus_tokens))
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built successfully.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant parent chunks given a question.

        Args:
            query: The user's question string.
            top_k: Number of top results to return.

        Returns:
            List of parent doc dicts with added 'bm25_score' field,
            sorted by relevance (highest first).
        """
        if self._bm25 is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            doc = dict(self._docs[idx])
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

        return results
