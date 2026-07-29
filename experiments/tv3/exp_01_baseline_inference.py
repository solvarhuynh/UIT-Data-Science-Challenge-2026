"""Experiment script: smoke-test Qwen3 1.7B baseline loading and inference.

Run from the repo root:
    python experiments/tv3/exp_01_baseline_inference.py

This script is intentionally NOT imported into src/ – it lives in
experiments/ as a one-off manual test.  Move stable logic to
src/udsc2026/ only after the interface is settled.
"""

import logging
import sys
from pathlib import Path

# Allow importing src/udsc2026 from the experiments directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.qa.citation_parser import CitationParser
from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.qa.qa_engine import QAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_CONTEXTS = [
    RetrievalHit(
        chunk_id="doc001_article_10_clause_1",
        doc_id="doc001",
        text=(
            "Người lao động có các quyền sau đây: Làm việc; tự do lựa chọn "
            "việc làm, nghề nghiệp, học nghề, nâng cao trình độ nghề nghiệp "
            "và không bị phân biệt đối xử."
        ),
        score=0.91,
        law_name="Bộ luật Lao động 2019",
        article="Điều 5",
        clause="Khoản 1",
        source="data/raw/bllđ2019.txt",
    ),
]

QUESTION = "Người lao động có những quyền cơ bản nào theo Bộ luật Lao động 2019?"


def main() -> None:
    """Load the baseline Qwen3 1.7B model and run a single QA inference."""
    model_path = "./models/qwen3-legal"

    if not Path(model_path).exists():
        logger.error(
            "Model not found at '%s'. Run 'python download_models.py' first.",
            model_path,
        )
        sys.exit(1)

    logger.info("Initialising LLMClient (backend=transformers)…")
    config = LLMConfig(
        model_path=model_path,
        backend="transformers",
        device="cuda",
        dtype="float16",  # Use float16 on GPU for stability
        max_new_tokens=512,
        temperature=0.1,
    )

    import asyncio

    async def run() -> None:
        llm = LLMClient(config)
        engine = QAEngine(
            llm_client=llm,
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
        )

        logger.info("Question: %s", QUESTION)
        response = await engine.generate_answer(
            question=QUESTION,
            contexts=SAMPLE_CONTEXTS,
        )

        print("\n" + "=" * 60)
        print("ANSWER:")
        print(response.answer)
        print("\nCITATIONS:")
        for c in response.citations:
            status = "✓ verified" if c.is_verified else "✗ UNVERIFIED"
            print(f"  [{status}] {c.law_name}, {c.article}, {c.clause}")
        print(f"\nCONFIDENCE: {response.confidence}")
        print(f"WARNINGS:   {response.warnings}")
        print("=" * 60)

    asyncio.run(run())


if __name__ == "__main__":
    main()
