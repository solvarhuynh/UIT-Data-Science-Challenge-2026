"""Dựng strict-OOF CPU-only cho producer public 0.93 đã xác nhận.

Script chỉ dùng cache điểm dense/BGE có sẵn; không tải model, không train và
không tạo submission public. Mọi nguồn có giám sát đều loại fold giữ lại và
các câu hỏi trùng sau chuẩn hóa.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from udsc2026.evaluation.legal_ir_chunk_aggregation import top_document_ids  # noqa: E402


def _load_builder() -> Any:
    path = ROOT / "scripts/submission/build_legal_ir_ensemble.py"
    spec = importlib.util.spec_from_file_location("_public093_ensemble", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} không phải object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _metrics(rows: Sequence[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    recalls: list[float] = []
    precisions: list[float] = []
    full = 0
    multi: list[float] = []
    for row in rows:
        gold = set(row["gold_documents"])
        predicted = set(row[prediction_key])
        recall = len(gold & predicted) / len(gold)
        precision = len(gold & predicted) / len(predicted)
        recalls.append(recall)
        precisions.append(precision)
        full += gold <= predicted
        if len(gold) > 1:
            multi.append(recall)
    return {
        "macro_recall": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": sum(multi) / len(multi) if multi else None,
        "full_gold_query_count": full,
        "query_count": len(rows),
    }


def _deduplicated_docs(hits: Sequence[dict[str, Any]], *, source: str) -> list[str]:
    indexed = list(enumerate(hits))
    if source == "dense":
        indexed.sort(key=lambda pair: (int(pair[1]["dense_rank"]), pair[0]))
    elif source == "bge":
        indexed.sort(
            key=lambda pair: (
                -float(pair[1]["bge_score"]),
                int(pair[1]["dense_rank"]),
                str(pair[1]["doc_id"]),
                str(pair[1]["chunk_id"]),
            )
        )
    else:
        raise ValueError(source)
    output: list[str] = []
    seen: set[str] = set()
    for _, hit in indexed:
        document_id = str(hit["doc_id"])
        if document_id not in seen:
            seen.add(document_id)
            output.append(document_id)
    return output


def _fold_metrics(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        str(fold): _metrics([row for row in rows if row["fold"] == fold], key)
        for fold in range(5)
    }


def _recall_delta(row: dict[str, Any], left: str, right: str) -> float:
    gold = set(row["gold_documents"])
    return (len(gold & set(row[right])) - len(gold & set(row[left]))) / len(gold)


def _rule_functions() -> list[tuple[str, Callable[[int], bool]]]:
    rules: list[tuple[str, Callable[[int], bool]]] = [
        ("always_baseline", lambda overlap: False),
        ("always_candidate", lambda overlap: True),
    ]
    for threshold in range(6):
        rules.append((f"switch_if_overlap_ge_{threshold}", lambda x, t=threshold: x >= t))
        rules.append((f"switch_if_overlap_le_{threshold}", lambda x, t=threshold: x <= t))
    return rules


def _policy_oof(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = _rule_functions()
    output: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for holdout in range(5):
        train = [row for row in rows if row["fold"] != holdout]
        test = [row for row in rows if row["fold"] == holdout]
        candidates: list[tuple[tuple[float, float, int, int], str, Callable[[int], bool]]] = []
        for order, (name, rule) in enumerate(rules):
            recall_sum = 0.0
            precision_sum = 0.0
            changes = 0
            for row in train:
                overlap = len(set(row["baseline_top5"]) & set(row["v3_top5"]))
                key = "v3_top5" if rule(overlap) else "baseline_top5"
                gold = set(row["gold_documents"])
                predicted = set(row[key])
                recall_sum += len(gold & predicted) / len(gold)
                precision_sum += len(gold & predicted) / len(predicted)
                changes += key == "v3_top5"
            score = (
                recall_sum / len(train),
                precision_sum / len(train),
                -changes,
                -order,
            )
            candidates.append((score, name, rule))
        best_score, best_name, best_rule = max(candidates, key=lambda item: item[0])
        selections.append(
            {
                "fold": holdout,
                "selected_rule": best_name,
                "training_macro_recall": best_score[0],
                "training_macro_precision": best_score[1],
                "training_switch_count": -best_score[2],
            }
        )

        bucket_training: dict[int, list[float]] = {value: [] for value in range(6)}
        for row in train:
            overlap = len(set(row["baseline_top5"]) & set(row["v3_top5"]))
            bucket_training[overlap].append(
                _recall_delta(row, "baseline_top5", "v3_top5")
            )
        for row in test:
            overlap = len(set(row["baseline_top5"]) & set(row["v3_top5"]))
            use_candidate = best_rule(overlap)
            deltas = bucket_training[overlap]
            confidence = (
                sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0
            )
            output.append(
                {
                    "query_id": row["query_id"],
                    "fold": holdout,
                    "gold_documents": row["gold_documents"],
                    "top5": row["v3_top5"] if use_candidate else row["baseline_top5"],
                    "source": "v3_locked" if use_candidate else "baseline_093",
                    "top5_overlap": overlap,
                    "training_bucket_candidate_win_probability": confidence,
                }
            )

    by_id = {row["query_id"]: row for row in output}
    output = [by_id[row["query_id"]] for row in rows]
    calibration: dict[str, dict[str, Any]] = {}
    ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001)]
    for low, high in ranges:
        selected = [
            row
            for row in output
            if low <= row["training_bucket_candidate_win_probability"] < high
        ]
        label = f"[{low:.2f},{min(high, 1.0):.2f}{']' if high > 1 else ')'}"
        calibration[label] = {
            "query_count": len(selected),
            "mean_confidence": (
                sum(row["training_bucket_candidate_win_probability"] for row in selected)
                / len(selected)
                if selected
                else None
            ),
        }
    return output, {"fold_selections": selections, "calibration_buckets": calibration}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--warmup", type=Path, default=ROOT / "data/task1/warmup.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json")
    parser.add_argument("--contexts-dir", type=Path, default=ROOT / "data/raw/btc/LegalIR/selected-contexts")
    parser.add_argument("--fold0-cache", type=Path, required=True)
    parser.add_argument("--fold1to4-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    builder = _load_builder()
    train_payload = _json(args.train)
    warmup_payload = _json(args.warmup)
    folds_payload = _json(args.folds)
    if not isinstance(train_payload, dict) or len(train_payload) != 7000:
        raise ValueError("Tập train phải có đúng 7.000 câu")
    question_text = {str(qid): str(record["question"]).strip() for qid, record in train_payload.items()}
    gold = {str(qid): [str(doc) for doc in record["answer"]] for qid, record in train_payload.items()}
    qid_to_fold: dict[str, int] = {}
    fold_ids: dict[int, list[str]] = {}
    for item in folds_payload["folds"]:
        fold = int(item["fold"])
        ids = [str(value) for value in item["validation_ids"]]
        fold_ids[fold] = ids
        for qid in ids:
            qid_to_fold[qid] = fold
    if set(qid_to_fold) != set(question_text):
        raise ValueError("Fold coverage không khớp đúng 7.000 ID train")

    cache_rows = _jsonl(args.fold0_cache) + _jsonl(args.fold1to4_cache)
    cache_by_id = {str(row["query_id"]): row for row in cache_rows}
    if len(cache_rows) != 7000 or set(cache_by_id) != set(question_text):
        raise ValueError("Cache dense/BGE không phủ đúng 7.000 ID")

    print("[1/4] Nạp 8.532 context và dựng BM25 dùng chung", flush=True)
    document_ids, passages = builder._load_contexts(args.contexts_dir)
    bm25 = builder._build_bm25_rankings(question_text, document_ids, passages)

    knn: dict[str, list[str]] = {}
    exact: dict[str, list[str]] = {}
    leakage_audit: list[dict[str, Any]] = []
    for fold in range(5):
        heldout = fold_ids[fold]
        heldout_questions = {qid: question_text[qid] for qid in heldout}
        excluded_questions = {
            builder.normalize_question(question_text[qid]) for qid in heldout
        }
        labeled = builder._load_labeled_questions(
            [args.train, args.warmup], set(heldout), excluded_questions
        )
        print(f"[2/4] Fold {fold}: dựng KNN từ {len(labeled)} câu an toàn", flush=True)
        knn.update(builder._build_knn_rankings(heldout_questions, labeled))
        exact_map = builder._exact_label_map(labeled)
        fold_exact = 0
        for qid in heldout:
            known = exact_map.get(builder.normalize_question(question_text[qid]), [])
            if known:
                exact[qid] = known
                fold_exact += 1
        leakage_audit.append(
            {
                "fold": fold,
                "heldout_count": len(heldout),
                "labeled_pool_count_after_exclusion": len(labeled),
                "heldout_normalized_exact_matches_after_exclusion": fold_exact,
            }
        )

    print("[3/4] Ghép producer 0.93 và nguồn V3 khóa trước", flush=True)
    rows: list[dict[str, Any]] = []
    for qid in train_payload:
        cache = cache_by_id[qid]
        hits = cache["hits"]
        dense = _deduplicated_docs(hits, source="dense")
        bge = _deduplicated_docs(hits, source="bge")
        baseline = builder._weighted_rrf(
            [dense, bge, knn[qid], bm25[qid]], [0.20, 0.30, 0.20, 0.30], 2
        )[:5]
        known = exact.get(qid, [])
        if known:
            baseline = (known + [doc for doc in baseline if doc not in known])[:5]
        dense_v3 = top_document_ids(
            hits, method="rank_cap5_k5", rank_source="dense", limit=200
        )
        bge_v3 = top_document_ids(
            hits, method="rank_cap10_k5", rank_source="bge", limit=200
        )
        v3 = builder._weighted_rrf([dense_v3, bge_v3], [0.30, 0.70], 0)[:5]
        bge_only = bge_v3[:5]
        rows.append(
            {
                "query_id": qid,
                "fold": qid_to_fold[qid],
                "gold_documents": gold[qid],
                "baseline_top5": baseline,
                "v3_top5": v3,
                "bge_chunk_only_top5": bge_only,
            }
        )

    baseline_metrics = _metrics(rows, "baseline_top5")
    v3_metrics = _metrics(rows, "v3_top5")
    bge_metrics = _metrics(rows, "bge_chunk_only_top5")
    outcomes = {"candidate_better": 0, "same": 0, "candidate_worse": 0}
    overlap_buckets: dict[str, dict[str, Any]] = {}
    for overlap in range(6):
        selected = [
            row
            for row in rows
            if len(set(row["baseline_top5"]) & set(row["v3_top5"])) == overlap
        ]
        deltas = [_recall_delta(row, "baseline_top5", "v3_top5") for row in selected]
        overlap_buckets[str(overlap)] = {
            "query_count": len(selected),
            "candidate_better": sum(delta > 0 for delta in deltas),
            "same": sum(delta == 0 for delta in deltas),
            "candidate_worse": sum(delta < 0 for delta in deltas),
            "mean_recall_delta": sum(deltas) / len(deltas) if deltas else None,
        }
    for row in rows:
        delta = _recall_delta(row, "baseline_top5", "v3_top5")
        outcomes[
            "candidate_better" if delta > 0 else "candidate_worse" if delta < 0 else "same"
        ] += 1

    policy_rows, policy_diagnostics = _policy_oof(rows)
    policy_metrics = _metrics(policy_rows, "top5")
    policy_changes = sum(row["source"] == "v3_locked" for row in policy_rows)
    policy_improvements = 0
    policy_harms = 0
    row_by_id = {row["query_id"]: row for row in rows}
    for item in policy_rows:
        source_row = row_by_id[item["query_id"]]
        gold_set = set(item["gold_documents"])
        delta = (
            len(gold_set & set(item["top5"]))
            - len(gold_set & set(source_row["baseline_top5"]))
        ) / len(gold_set)
        policy_improvements += delta > 0
        policy_harms += delta < 0

    output_dir = args.output_dir
    baseline_output = [
        {"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold_documents"], "top5": row["baseline_top5"]}
        for row in rows
    ]
    v3_output = [
        {"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold_documents"], "top5": row["v3_top5"]}
        for row in rows
    ]
    bge_output = [
        {"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold_documents"], "top5": row["bge_chunk_only_top5"]}
        for row in rows
    ]
    _write_jsonl(output_dir / "predictions.jsonl", baseline_output)
    _write_jsonl(output_dir / "v3_locked_predictions.jsonl", v3_output)
    _write_jsonl(output_dir / "bge_chunk_only_predictions.jsonl", bge_output)
    _write_jsonl(output_dir / "policy_predictions.jsonl", policy_rows)

    report = {
        "schema_version": "public-093-strict-oof-v1",
        "status": "COMPLETE",
        "query_count": len(rows),
        "producer": {
            "dense_weight": 0.20,
            "bge_weight": 0.30,
            "knn_weight": 0.20,
            "bm25_weight": 0.30,
            "rrf_k": 2,
            "component_top_k": 200,
            "labeled_sources": [str(args.train), str(args.warmup)],
            "exact_label_overlay": True,
        },
        "leakage_audit": leakage_audit,
        "baseline_093_oof": {
            "pooled": baseline_metrics,
            "per_fold": _fold_metrics(rows, "baseline_top5"),
        },
        "v3_locked_oof": {
            "config": {
                "dense_aggregation": "rank_cap5_k5",
                "bge_aggregation": "rank_cap10_k5",
                "weights": {"dense": 0.30, "bge": 0.70},
                "rrf_k": 0,
            },
            "pooled": v3_metrics,
            "per_fold": _fold_metrics(rows, "v3_top5"),
            "recall_delta_vs_baseline": v3_metrics["macro_recall"] - baseline_metrics["macro_recall"],
            "outcomes": outcomes,
            "overlap_buckets": overlap_buckets,
        },
        "bge_chunk_only_control": {
            "config": "rank_cap10_k5",
            "pooled": bge_metrics,
            "per_fold": _fold_metrics(rows, "bge_chunk_only_top5"),
        },
        "change_policy_oof": {
            "pooled": policy_metrics,
            "recall_delta_vs_baseline": policy_metrics["macro_recall"] - baseline_metrics["macro_recall"],
            "changed_query_count": policy_changes,
            "improved_query_count": policy_improvements,
            "harmed_query_count": policy_harms,
            **policy_diagnostics,
        },
    }
    _write_json(output_dir / "report.json", report)
    manifest = {
        "schema_version": "public-093-strict-oof-manifest-v1",
        "script": str(Path(__file__).resolve()),
        "inputs": {
            str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (args.train, args.warmup, args.folds, args.fold0_cache, args.fold1to4_cache)
        },
        "outputs": {
            name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in (
                "predictions.jsonl",
                "v3_locked_predictions.jsonl",
                "bge_chunk_only_predictions.jsonl",
                "policy_predictions.jsonl",
                "report.json",
            )
        },
        "constraints": {"neural_training": False, "public_labels_used": False, "submission_created": False},
    }
    _write_json(output_dir / "manifest.json", manifest)
    print("[4/4] Hoàn tất strict OOF", flush=True)
    print(json.dumps({"baseline": baseline_metrics, "v3": v3_metrics, "policy": policy_metrics}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
