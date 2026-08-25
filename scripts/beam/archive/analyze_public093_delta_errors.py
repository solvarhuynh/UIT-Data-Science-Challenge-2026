"""Phân tích 35 query V3-delta đổi outcome mà không dùng gold làm feature."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from udsc2026.evaluation.legal_ir_lexical import parse_legal_citations  # noqa: E402


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _load_contexts(path: Path) -> tuple[list[str], list[str]]:
    documents: dict[str, str] = {}
    for item in sorted(path.glob("*.json"), key=lambda value: value.name):
        row = _json(item)
        documents[str(row["id"])] = str(row["passage"])
    ids = sorted(documents, key=lambda value: (len(value), value))
    return ids, [documents[item] for item in ids]


def _knn_details(
    questions: dict[str, str], labeled: Sequence[Any]
) -> dict[str, dict[str, dict[str, float | int]]]:
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True, norm="l2", dtype=np.float32
    )
    matrix = vectorizer.fit_transform([item.question for item in labeled])
    ids = list(questions)
    similarities = (vectorizer.transform([questions[item] for item in ids]) @ matrix.T).tocsr()
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, qid in enumerate(ids):
        row = similarities.getrow(index)
        neighbors = sorted(
            ((int(i), float(score)) for i, score in zip(row.indices, row.data) if score > 0),
            key=lambda item: (-item[1], labeled[item[0]].question_id),
        )[:20]
        scores: dict[str, tuple[float, int]] = {}
        for neighbor_rank, (label_index, score) in enumerate(neighbors, 1):
            for document_id in labeled[label_index].documents:
                prior = scores.get(document_id, (-math.inf, 10**9))
                if score > prior[0] or (score == prior[0] and neighbor_rank < prior[1]):
                    scores[document_id] = (score, neighbor_rank)
        ranked = sorted(scores, key=lambda doc: (-scores[doc][0], scores[doc][1], doc))
        output[qid] = {
            doc: {"rank": rank, "score": scores[doc][0], "neighbor_rank": scores[doc][1]}
            for rank, doc in enumerate(ranked, 1)
        }
    return output


def _bm25_details(
    questions: dict[str, str], document_ids: Sequence[str], passages: Sequence[str]
) -> dict[str, dict[str, dict[str, float | int]]]:
    word_re = re.compile(r"(?u)\b\w+\b")
    terms: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for question in questions.values():
        tokens = word_re.findall(question.casefold())
        for width in (1, 2, 3):
            for start in range(len(tokens) - width + 1):
                term = " ".join(tokens[start : start + width])
                lengths[term] = width
    vocabulary = {term: index for index, term in enumerate(sorted(lengths))}
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 3), lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    qids = list(questions)
    query = vectorizer.transform([questions[qid] for qid in qids]).tocsr()
    counts = vectorizer.transform(passages).tocsr()
    doc_lengths = np.asarray([max(1, len(word_re.findall(text))) for text in passages], dtype=np.float32)
    average = float(doc_lengths.mean())
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log1p((len(document_ids) - df + 0.5) / (df + 0.5)).astype(np.float32)
    weights = np.asarray([lengths[term] for term in sorted(lengths)], dtype=np.float32)
    rows = np.repeat(np.arange(len(document_ids), dtype=np.int64), np.diff(counts.indptr))
    norm = 0.7 * (1.0 - 0.3 + 0.3 * doc_lengths / average)
    counts.data = ((counts.data * 1.7) / (counts.data + norm[rows]) * idf[counts.indices] * weights[counts.indices]).astype(np.float32)
    query.data.fill(1.0)
    scores = (query @ counts.T).tocsr()
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, qid in enumerate(qids):
        row = scores.getrow(index)
        ranked = sorted(
            ((document_ids[int(i)], float(score)) for i, score in zip(row.indices, row.data) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )[:100]
        output[qid] = {doc: {"rank": rank, "score": score} for rank, (doc, score) in enumerate(ranked, 1)}
    return output


def _aggregate_doc_features(hits: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["doc_id"])].append(hit)
    docs: dict[str, dict[str, Any]] = {}
    bge_maxes: list[float] = []
    dense_maxes: list[float] = []
    for doc, values in grouped.items():
        bge = sorted((float(item["bge_score"]) for item in values), reverse=True)
        dense = sorted((float(item["dense_score"]) for item in values), reverse=True)
        docs[doc] = {
            "support_count": len(values),
            "bge_max": bge[0],
            "bge_top2_mean": mean(bge[:2]),
            "bge_top3_mean": mean(bge[:3]),
            "bge_lse3": max(bge[:3]) + math.log(sum(math.exp(value - max(bge[:3])) for value in bge[:3])),
            "dense_max": dense[0],
            "best_dense_rank": min(int(item["dense_rank"]) for item in values),
        }
        bge_maxes.append(bge[0])
        dense_maxes.append(dense[0])
    bge_maxes.sort(reverse=True)
    dense_maxes.sort(reverse=True)
    exp = np.exp(np.asarray(bge_maxes) - bge_maxes[0])
    probs = exp / exp.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))
    query_features = {
        "unique_docs_raw_top200": len(docs),
        "bge_top_margin": bge_maxes[0] - bge_maxes[1] if len(bge_maxes) > 1 else 0.0,
        "dense_top_margin": dense_maxes[0] - dense_maxes[1] if len(dense_maxes) > 1 else 0.0,
        "bge_doc_entropy": entropy,
    }
    return docs, query_features


def _outcome(gold: set[str], baseline: Sequence[str], candidate: Sequence[str]) -> str:
    delta = len(gold & set(candidate)) / len(gold) - len(gold & set(baseline)) / len(gold)
    return "better" if delta > 0 else "worse" if delta < 0 else "same"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--warmup", type=Path, default=ROOT / "data/task1/warmup.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json")
    parser.add_argument("--fold0-cache", type=Path, required=True)
    parser.add_argument("--fold1to4-cache", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, default=ROOT / "data/raw/btc/LegalIR/selected-contexts")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    builder = _module("_public093_builder_errors", ROOT / "scripts/submission/build_legal_ir_ensemble.py")
    train = _json(args.train)
    baseline = {str(row["query_id"]): row for row in _jsonl(args.baseline)}
    candidate = {str(row["query_id"]): row for row in _jsonl(args.candidate)}
    compact = {str(row["query_id"]): row for row in _jsonl(args.fold0_cache) + _jsonl(args.fold1to4_cache)}
    selected: list[dict[str, Any]] = []
    for qid, base in baseline.items():
        gold = set(str(value) for value in base["gold_documents"])
        label = _outcome(gold, base["top5"], candidate[qid]["top5"])
        if label != "same":
            selected.append({"query_id": qid, "fold": int(base["fold"]), "outcome": label, "gold_documents": sorted(gold), "baseline_top5": base["top5"], "candidate_top5": candidate[qid]["top5"]})
    if len(selected) != 35:
        raise ValueError(f"Cần đúng 35 delta outcome, nhận {len(selected)}")

    questions = {row["query_id"]: str(train[row["query_id"]]["question"]) for row in selected}
    fold_payload = _json(args.folds)
    fold_ids = {int(item["fold"]): {str(value) for value in item["validation_ids"]} for item in fold_payload["folds"]}
    knn: dict[str, dict[str, dict[str, float | int]]] = {}
    for fold in range(5):
        subset = {qid: question for qid, question in questions.items() if qid in fold_ids[fold]}
        if not subset:
            continue
        excluded_texts = {builder.normalize_question(str(train[qid]["question"])) for qid in fold_ids[fold]}
        labeled = builder._load_labeled_questions([args.train, args.warmup], fold_ids[fold], excluded_texts)
        knn.update(_knn_details(subset, labeled))
    document_ids, passages = _load_contexts(args.contexts_dir)
    bm25 = _bm25_details(questions, document_ids, passages)

    output: list[dict[str, Any]] = []
    for item in selected:
        qid = item["query_id"]
        doc_features, query_features = _aggregate_doc_features(compact[qid]["hits"])
        dense_ranked = sorted(doc_features, key=lambda doc: (doc_features[doc]["best_dense_rank"], doc))
        bge_ranked = sorted(doc_features, key=lambda doc: (-doc_features[doc]["bge_max"], doc_features[doc]["best_dense_rank"], doc))
        added = [doc for doc in item["candidate_top5"] if doc not in item["baseline_top5"]]
        removed = [doc for doc in item["baseline_top5"] if doc not in item["candidate_top5"]]
        all_docs = list(dict.fromkeys(item["baseline_top5"] + item["candidate_top5"]))
        details: dict[str, Any] = {}
        for doc in all_docs:
            raw = dict(doc_features.get(doc, {}))
            raw["dense_doc_rank"] = dense_ranked.index(doc) + 1 if doc in doc_features else None
            raw["bge_doc_rank"] = bge_ranked.index(doc) + 1 if doc in doc_features else None
            raw["bm25"] = bm25[qid].get(doc)
            raw["knn_word"] = knn[qid].get(doc)
            raw["in_baseline"] = doc in item["baseline_top5"]
            raw["in_candidate"] = doc in item["candidate_top5"]
            raw["source_agreement_count"] = sum((doc in item["baseline_top5"], doc in item["candidate_top5"], doc in dense_ranked[:5], doc in bge_ranked[:5], doc in bm25[qid], doc in knn[qid]))
            details[doc] = raw
        citation = parse_legal_citations(questions[qid])
        added_details = [details[doc] for doc in added if doc in details]
        feature = {
            **query_features,
            "set_overlap": len(set(item["baseline_top5"]) & set(item["candidate_top5"])),
            "documents_changed": len(added),
            "query_token_count": len(re.findall(r"(?u)\b\w+\b", questions[qid])),
            "citation_cue": citation.explicit,
            "candidate_added_mean_bge_max": mean([float(row["bge_max"]) for row in added_details]) if added_details else None,
            "candidate_added_mean_dense_max": mean([float(row["dense_max"]) for row in added_details]) if added_details else None,
            "candidate_added_mean_support_count": mean([float(row["support_count"]) for row in added_details]) if added_details else None,
            "candidate_added_mean_bm25_score": mean([float(row["bm25"]["score"]) for row in added_details if row["bm25"] is not None]) if any(row["bm25"] is not None for row in added_details) else None,
            "candidate_added_mean_knn_score": mean([float(row["knn_word"]["score"]) for row in added_details if row["knn_word"] is not None]) if any(row["knn_word"] is not None for row in added_details) else None,
            "candidate_added_mean_source_agreement": mean([float(row["source_agreement_count"]) for row in added_details]) if added_details else None,
        }
        gold = set(item["gold_documents"])
        item.update({
            "question": questions[qid],
            "baseline_recall": len(gold & set(item["baseline_top5"])) / len(gold),
            "candidate_recall": len(gold & set(item["candidate_top5"])) / len(gold),
            "removed_documents": removed,
            "added_documents": added,
            "features": feature,
            "document_features": details,
        })
        output.append(item)
    output.sort(key=lambda row: (row["outcome"], row["query_id"]))
    _write_jsonl(args.output_dir / "error_analysis.jsonl", output)

    numerical = [key for key, value in output[0]["features"].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    summary: list[dict[str, Any]] = []
    better = [row for row in output if row["outcome"] == "better"]
    worse = [row for row in output if row["outcome"] == "worse"]
    for key in numerical:
        left = [float(row["features"][key]) for row in better if row["features"][key] is not None]
        right = [float(row["features"][key]) for row in worse if row["features"][key] is not None]
        if not left or not right:
            continue
        pooled = math.sqrt((pstdev(left) ** 2 + pstdev(right) ** 2) / 2)
        summary.append({"feature": key, "better_mean": mean(left), "worse_mean": mean(right), "difference": mean(left) - mean(right), "standardized_difference": (mean(left) - mean(right)) / pooled if pooled > 1e-12 else None})
    summary.sort(key=lambda row: abs(row["standardized_difference"] or 0.0), reverse=True)
    lines = ["# Phân tích 35 query V3-delta đổi outcome", "", "- 15 better, 20 worse.", "- Gold chỉ dùng để gắn nhãn outcome; mọi feature trong JSONL có thể tính khi inference.", "", "## Feature phân biệt mô tả tốt nhất", "", "| Feature | TB better | TB worse | Chênh lệch | Hiệu ứng chuẩn hóa |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {row['feature']} | {row['better_mean']:.4f} | {row['worse_mean']:.4f} | {row['difference']:.4f} | {row['standardized_difference']:.4f} |" for row in summary[:10]]
    lines += ["", "Các chênh lệch trên chỉ mang tính mô tả với mẫu nhỏ 35 query; không được xem là threshold promotion nếu chưa qua nested OOF.", "", "## Query", "", "| Outcome | Số query |", "|---|---:|", "| better | 15 |", "| worse | 20 |"]
    (args.output_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"query_count": len(output), "top_discriminative_features": summary[:10]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
