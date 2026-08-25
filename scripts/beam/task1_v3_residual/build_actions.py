"""Build V3 residual swap actions and train-only fractional utility labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import jsonl, recall, sort_query_ids, write_jsonl
except ImportError:
    from common import jsonl, recall, sort_query_ids, write_jsonl


BASE_FEATURES = (
    "neural_max", "neural_second", "neural_mean", "neural_min", "neural_std", "neural_max_minus_mean", "neural_top2_mean",
    "bm25_max", "bm25_mean", "union_rank", "reciprocal_union_rank", "source_support", "min_source_rank", "has_bge_support",
    "bm25_rank", "adaptive_k500_rank", "knn_char_rank", "knn_word_rank", "bm25_missing", "adaptive_k500_missing", "knn_char_missing", "knn_word_missing",
    "question_token_length", "question_char_length", "baseline_rank", "is_baseline_top5",
)
DIFF_FEATURES = ("neural_max", "neural_mean", "neural_second", "union_rank", "source_support", "bm25_max")


def numeric(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def action_label(gold: set[str], baseline_top5: list[str], incoming: str, drop_rank: int) -> tuple[float, str]:
    before = recall(gold, baseline_top5)
    swapped = list(baseline_top5)
    swapped[drop_rank - 1] = str(incoming)
    gain = recall(gold, swapped) - before
    return gain, "BENEFIT" if gain > 0 else "HARM" if gain < 0 else "NEUTRAL"


def build_action(query_id: str, fold: int, baseline_top5: list[str], incoming: dict[str, Any], dropped: dict[str, Any], drop_rank: int, gold: set[str] | None) -> dict[str, Any]:
    features: dict[str, float] = {}
    for name in BASE_FEATURES:
        features[f"incoming_{name}"] = numeric(incoming.get(name))
        features[f"dropped_{name}"] = numeric(dropped.get(name))
    for name in DIFF_FEATURES:
        features[f"diff_{name}"] = numeric(incoming.get(name)) - numeric(dropped.get(name))
    features["dropped_baseline_rank"] = float(drop_rank)
    features["incoming_is_baseline_top5"] = 0.0
    output = {"query_id": query_id, "fold": fold, "incoming_doc_id": str(incoming["doc_id"]), "dropped_doc_id": str(dropped["doc_id"]), "drop_rank": drop_rank, "baseline_top5": baseline_top5, "features": features}
    if gold is not None:
        gain, label = action_label(gold, baseline_top5, str(incoming["doc_id"]), drop_rank)
        output.update({"baseline_recall": recall(gold, baseline_top5), "gain": gain, "label": label})
    return output


def build_public_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build unlabeled public one-swap actions with the exact V3A schema."""
    actions: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row["query_id"])
        if "fold" in row or set(row) & {"answer", "gold", "label", "gain"}:
            raise ValueError(f"public features contain train-only fields: {query_id}")
        baseline = [str(value) for value in row["baseline_top5"]]
        if len(baseline) != 5 or len(set(baseline)) != 5:
            raise ValueError(f"invalid public baseline top5: {query_id}")
        docs = [dict(doc) for doc in row["docs"]]
        by_doc = {str(doc["doc_id"]): doc for doc in docs}
        if not set(baseline).issubset(by_doc):
            raise ValueError(f"public baseline docs absent from frozen features: {query_id}")
        for incoming in sorted((doc for doc in docs if str(doc["doc_id"]) not in set(baseline)), key=lambda doc: (int(doc["union_rank"]), str(doc["doc_id"]))):
            for drop_rank in (5, 4):
                action = build_action(query_id, -1, baseline, incoming, by_doc[baseline[drop_rank - 1]], drop_rank, None)
                action.pop("fold", None)
                if len(action["features"]) != 58:
                    raise ValueError(f"public action schema is not 58 features: {query_id}")
                actions.append(action)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label-folds", default="1,2,3,4")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    label_folds = {int(value) for value in args.label_folds.split(",") if value.strip()}
    if 0 in label_folds:
        raise ValueError("fold0 action labels are prohibited")
    if args.preflight:
        gold = {"A", "B"}
        benefit = action_label(gold, ["A", "X", "Y", "Z", "W"], "B", 5)
        neutral = action_label(gold, ["A", "X", "Y", "Z", "W"], "N", 5)
        harm = action_label(gold, ["A", "X", "Y", "Z", "B"], "N", 5)
        if (benefit[1], neutral[1], harm[1]) != ("BENEFIT", "NEUTRAL", "HARM") or benefit[0] != 0.5 or harm[0] != -0.5:
            raise AssertionError("fractional action label toy failed")
        print(json.dumps({"status": "PREFLIGHT_PASS", "action_label_toy": True, "fold0_labels_prohibited": True, "gpu_launched": False}, indent=2))
        return
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(args.features):
        by_query[str(row["query_id"])].append(row)
    if len(by_query) != 7000:
        raise ValueError("V3 actions require frozen features for exactly 7,000 queries")
    actions: list[dict[str, Any]] = []
    counts: dict[int, Counter] = defaultdict(Counter)
    for query_id in sort_query_ids(by_query):
        docs = by_query[query_id]
        fold = int(docs[0]["fold"])
        if any(int(doc["fold"]) != fold for doc in docs):
            raise ValueError(f"inconsistent feature fold: {query_id}")
        baseline = [str(doc_id) for doc_id in docs[0]["baseline_top5"]]
        if len(baseline) != 5 or len(set(baseline)) != 5:
            raise ValueError(f"invalid baseline top5: {query_id}")
        by_doc = {str(doc["doc_id"]): doc for doc in docs}
        if not set(baseline).issubset(by_doc):
            raise ValueError(f"baseline docs absent from shortlist features: {query_id}")
        gold = {str(doc_id) for doc_id in questions[query_id].get("answer", [])} if fold in label_folds else None
        for incoming in sorted((doc for doc in docs if str(doc["doc_id"]) not in set(baseline)), key=lambda doc: (int(doc["union_rank"]), str(doc["doc_id"]))):
            for drop_rank in (5, 4):
                action = build_action(query_id, fold, baseline, incoming, by_doc[baseline[drop_rank - 1]], drop_rank, gold)
                actions.append(action)
                if gold is not None:
                    counts[fold][str(action["label"])] += 1
    write_jsonl(args.output, actions)
    report = {"status": "ACTIONS_COMPLETE", "query_count": len(by_query), "action_count": len(actions), "label_folds": sorted(label_folds), "fold0_labeled_action_count": sum(1 for row in actions if int(row["fold"]) == 0 and "label" in row), "class_counts_by_train_fold": {str(fold): dict(counter) for fold, counter in counts.items()}, "features": list(BASE_FEATURES), "differences": list(DIFF_FEATURES)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["fold0_labeled_action_count"]:
        raise RuntimeError("fold0 gold leaked into V3 action labels")


if __name__ == "__main__":
    main()
