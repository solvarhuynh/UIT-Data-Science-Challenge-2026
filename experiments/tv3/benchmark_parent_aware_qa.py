"""Evaluate Qwen LegalQA with child-to-parent expansion and parent-aware reranking.

This experiment consumes an existing TV5 ``predictions_after.jsonl`` candidate
pool.  It intentionally performs a second rerank *after* resolving child hits
to parent passages:

    top-K child hits -> anchor-centred parents -> BGE parent rerank
    -> bounded Top-N parent contexts -> Qwen reader

Use a labelled train candidate artifact for metric-based A/B testing.  The
same script also accepts public questions, but those runs only write
diagnostics because public answers are null.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "Scoring-Program-Task-LegalQA"))

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.infrastructure.reranker import CrossEncoderClient
from udsc2026.qa.qa_engine import QAEngine
from udsc2026.retrieval.parent_context import JsonlParentStore, ParentContextExpander
from udsc2026.retrieval.reranking import CrossEncoderReranker

LOGGER = logging.getLogger("benchmark_parent_aware_qa")
_PROMPT_ECHO_PREFIXES = (
    "dữ liệu pháp lý",
    "các trích đoạn pháp lý",
    "context:",
    "câu hỏi:",
    "trả lời dựa trên context",
    "chỉ sử dụng thông tin có trong context",
    "[1] (law_name=",
)
_REFUSAL_MARKERS = (
    "không có đủ căn cứ",
    "không đủ căn cứ",
    "không có thông tin",
    "không thể trả lời",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark child-to-parent expansion plus parent-aware BGE reranking."
    )
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--parents-dir", type=Path, required=True)
    parser.add_argument(
        "--reranker-model",
        default="models/qwen3-vl-reranker-2b",
        help="Local path or Hugging Face model id for the second-stage parent reranker.",
    )
    parser.add_argument(
        "--allow-remote-reranker",
        action="store_true",
        help="Allow Hugging Face download when --reranker-model is not available locally.",
    )
    parser.add_argument(
        "--model-path",
        default="thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--candidate-k", type=_positive_int, default=50)
    parser.add_argument("--parent-candidate-tokens", type=_positive_int, default=768)
    parser.add_argument("--parent-top-n", type=_positive_int, default=5)
    parser.add_argument(
        "--disable-parent-rerank",
        action="store_true",
        help="A/B baseline: retain parent candidates in their child-rerank order.",
    )
    parser.add_argument("--context-char-budget", type=_positive_int, default=8000)
    parser.add_argument("--reranker-batch-size", type=_positive_int, default=8)
    parser.add_argument("--reranker-max-length", type=_positive_int, default=1024)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    parser.add_argument("--prompt-version", default="legal_qa_v5")
    parser.add_argument("--rag-template", default="default_rag_v5")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate a reproducible random subset; 0 evaluates every matching candidate.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip shuffled question ids; combine with --limit for a frozen holdout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "experiments/tv3/parent_aware_benchmark_report.json",
    )
    return parser.parse_args()


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("questions file must be an object keyed by question id")

    questions: dict[str, dict[str, Any]] = {}
    for raw_qid, raw_item in payload.items():
        if not isinstance(raw_item, dict):
            continue
        question = raw_item.get("question")
        if isinstance(question, str) and question.strip():
            questions[str(raw_qid)] = raw_item
    if not questions:
        raise ValueError("questions file contains no usable question")
    return questions


def load_candidate_pools(path: Path) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"candidate line {line_number} must be an object")
            qid = row.get("question_id")
            hits = row.get("hits")
            if not isinstance(qid, str) or not isinstance(hits, list):
                raise ValueError(
                    f"candidate line {line_number} needs question_id and hits fields"
                )
            pools[qid] = [hit for hit in hits if isinstance(hit, dict)]
    if not pools:
        raise ValueError("candidate file contains no usable candidate pool")
    return pools


def _score_from_hit(hit: dict[str, Any]) -> float | None:
    for key in ("final_score", "rerank_score", "score", "dense_score"):
        value = hit.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def child_hit_from_json(hit: dict[str, Any], index: int) -> RetrievalHit | None:
    text = hit.get("text")
    doc_id = hit.get("doc_id")
    if not isinstance(text, str) or not text.strip() or not isinstance(doc_id, str) or not doc_id.strip():
        return None

    metadata = hit.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    parent_id = hit.get("parent_id")
    if not isinstance(parent_id, str) or not parent_id.strip():
        metadata_parent_id = metadata.get("parent_id")
        parent_id = metadata_parent_id if isinstance(metadata_parent_id, str) else None
    if parent_id:
        metadata["parent_id"] = parent_id

    rank = hit.get("rank")
    return RetrievalHit(
        chunk_id=str(hit.get("chunk_id") or f"candidate_{doc_id}_{index}"),
        parent_id=parent_id,
        doc_id=doc_id,
        text=text.strip(),
        score=_score_from_hit(hit),
        source=hit.get("source") if isinstance(hit.get("source"), str) else None,
        law_name=hit.get("law_name") if isinstance(hit.get("law_name"), str) else None,
        article=hit.get("article") if isinstance(hit.get("article"), str) else None,
        clause=hit.get("clause") if isinstance(hit.get("clause"), str) else None,
        metadata=metadata,
        rank=rank if isinstance(rank, int) and rank > 0 else None,
    )


def unique_child_hits(raw_hits: list[dict[str, Any]], candidate_k: int) -> list[RetrievalHit]:
    children: list[RetrievalHit] = []
    seen_chunk_ids: set[str] = set()
    for index, raw_hit in enumerate(raw_hits[:candidate_k], start=1):
        child = child_hit_from_json(raw_hit, index)
        if child is None or child.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(child.chunk_id)
        children.append(child)
    return children


def fit_context_budget(
    parents: list[RetrievalHit], total_char_budget: int
) -> list[RetrievalHit]:
    """Keep ranked parent windows inside one explicit character budget."""

    if not parents:
        return []
    base_weights = (0.35, 0.25, 0.18, 0.13, 0.09)
    weights = base_weights[: len(parents)]
    if len(weights) < len(parents):
        weights = weights + (0.0,) * (len(parents) - len(weights))
    weight_total = sum(weights)
    output: list[RetrievalHit] = []
    for parent, weight in zip(parents, weights):
        limit = max(1, int(total_char_budget * weight / weight_total))
        clipped = parent.text[:limit].strip()
        text = clipped.rsplit(" ", 1)[0].strip() if " " in clipped else clipped
        if text:
            output.append(parent.model_copy(update={"text": text}))
    return output


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().casefold()
    return len(normalized) < 20 or any(marker in normalized for marker in _REFUSAL_MARKERS)


def is_prompt_echo(answer: str) -> bool:
    return answer.strip().casefold().startswith(_PROMPT_ECHO_PREFIXES)


def evaluate_scores(records: list[dict[str, Any]]) -> dict[str, float] | None:
    scored = [
        record
        for record in records
        if isinstance(record.get("reference"), str) and record["reference"].strip()
    ]
    if not scored:
        return None
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_values: list[float] = []
    meteor_values: list[float] = []
    for record in scored:
        reference = str(record["reference"])
        answer = str(record["answer"])
        rouge_values.append(rouge.score(reference, answer)["rougeL"].fmeasure)
        meteor_values.append(meteor_score([reference.split()], answer.split()))
    return {
        "rougeL": sum(rouge_values) / len(rouge_values),
        "meteor": sum(meteor_values) / len(meteor_values),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions_file)
    candidate_pools = load_candidate_pools(args.candidate_file)
    qids = [qid for qid in candidate_pools if qid in questions]
    if not qids:
        raise ValueError("candidate file and questions file share no question ids")
    random.Random(args.seed).shuffle(qids)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    qids = qids[args.offset :]
    if args.limit > 0:
        qids = qids[: args.limit]

    if not args.parents_dir.is_dir():
        raise FileNotFoundError(f"parents directory does not exist: {args.parents_dir}")
    device = args.device if torch.cuda.is_available() else "cpu"
    parent_store = JsonlParentStore(args.parents_dir)
    # Candidate expansion is intentionally generous: it preserves evidence for
    # the parent reranker.  The reader receives a separate 8k-char budget below.
    candidate_expander = ParentContextExpander(
        parent_store,
        max_parent_tokens=args.parent_candidate_tokens,
        max_total_tokens=args.candidate_k * args.parent_candidate_tokens,
    )
    reranker = CrossEncoderReranker(
        CrossEncoderClient(
            args.reranker_model,
            device=device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            local_files_only=not args.allow_remote_reranker,
            use_fp16=device.startswith("cuda"),
        )
    )
    llm = LLMClient(
        LLMConfig(
            model_path=args.model_path,
            backend="transformers",
            device=device,
            dtype="float16",
            quantization=args.quantization if device.startswith("cuda") else "none",
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=300.0,
        )
    )
    # Contexts passed below are already parent-resolved and budgeted.
    qa_engine = QAEngine(llm_client=llm, use_chat_template=True)

    LOGGER.info(
        "Evaluating %d questions | child_k=%d | parent_top_n=%d | char_budget=%d",
        len(qids),
        args.candidate_k,
        args.parent_top_n,
        args.context_char_budget,
    )
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    for index, qid in enumerate(qids, start=1):
        question_item = questions[qid]
        question = str(question_item["question"])
        children = unique_child_hits(candidate_pools[qid], args.candidate_k)
        parents = candidate_expander.expand(children)
        # Parent lookup can legitimately fail for a malformed/missing parent
        # file. Keep those child hits observable instead of aborting a whole run.
        selected = (
            reranker.rerank(question, parents, top_n=min(args.parent_top_n, len(parents)))
            if parents and not args.disable_parent_rerank
            else parents[: args.parent_top_n]
            if parents
            else children[: args.parent_top_n]
        )
        contexts = fit_context_budget(selected, args.context_char_budget)
        response = await qa_engine.generate_answer(
            question=question,
            contexts=contexts,
            prompt_version=args.prompt_version,
            rag_template=args.rag_template,
            trace_id=f"parent_aware_{qid}",
        )
        answer = response.answer
        record = {
            "id": qid,
            "question": question,
            "reference": question_item.get("answer"),
            "answer": answer,
            "is_refusal": is_refusal(answer),
            "is_prompt_echo": is_prompt_echo(answer),
            "warnings": response.warnings,
            "selected_parents": [
                {
                    "doc_id": hit.doc_id,
                    "parent_id": hit.parent_id,
                    "law_name": hit.law_name,
                    "article": hit.article,
                    "rerank_score": hit.rerank_score,
                    "anchor_chunk_ids": hit.metadata.get(
                        "parent_context_anchor_chunk_ids", []
                    ),
                    "context_chars": len(hit.text),
                }
                for hit in contexts
            ],
        }
        records.append(record)
        LOGGER.info(
            "[%d/%d] qid=%s parents=%d chars=%d refusal=%s echo=%s",
            index,
            len(qids),
            qid,
            len(contexts),
            sum(len(hit.text) for hit in contexts),
            record["is_refusal"],
            record["is_prompt_echo"],
        )
        if torch.cuda.is_available() and index % 10 == 0:
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    metrics = evaluate_scores(records)
    return {
        "pipeline": "child_to_parent_then_parent_rerank",
        "num_questions": len(records),
        "refusal_count": sum(record["is_refusal"] for record in records),
        "prompt_echo_count": sum(record["is_prompt_echo"] for record in records),
        "elapsed_seconds": elapsed,
        "avg_seconds_per_question": elapsed / max(1, len(records)),
        "metrics": metrics,
        "settings": {
            "candidate_k": args.candidate_k,
            "parent_candidate_tokens": args.parent_candidate_tokens,
            "parent_top_n": args.parent_top_n,
            "context_char_budget": args.context_char_budget,
            "parent_rerank_enabled": not args.disable_parent_rerank,
            "reranker_model": args.reranker_model,
            "prompt_version": args.prompt_version,
            "rag_template": args.rag_template,
        },
        "records": records,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    report = asyncio.run(run(args))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = report["metrics"]
    print(f"Report: {args.output_json}")
    print(
        "Questions={num_questions} | refusal={refusal_count} | prompt_echo={prompt_echo_count}"
        .format(**report)
    )
    if metrics is not None:
        print("METEOR={meteor:.4f} | ROUGE-L={rougeL:.4f}".format(**metrics))


if __name__ == "__main__":
    main()
