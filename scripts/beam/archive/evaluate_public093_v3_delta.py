"""Đánh giá V3 như nguồn delta bên cạnh baseline 0.93 bằng nested OOF."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _score(rows: Sequence[dict[str, Any]], key: str) -> tuple[float, float]:
    recall = 0.0
    precision = 0.0
    for row in rows:
        gold = set(row["gold_documents"])
        predicted = set(row[key])
        recall += len(gold & predicted) / len(gold)
        precision += len(gold & predicted) / len(predicted)
    return recall / len(rows), precision / len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strict = _module(
        "_public093_strict", ROOT / "scripts/beam/archive/build_public093_strict_oof.py"
    )
    builder = _module(
        "_public093_builder", ROOT / "scripts/submission/build_legal_ir_ensemble.py"
    )
    baseline_rows = _jsonl(args.baseline)
    v3_by_id = {str(row["query_id"]): row for row in _jsonl(args.v3)}
    if len(baseline_rows) != 7000 or len(v3_by_id) != 7000:
        raise ValueError("Hai nguồn phải phủ đúng 7.000 câu")
    rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        qid = str(baseline["query_id"])
        v3 = v3_by_id[qid]
        if int(v3["fold"]) != int(baseline["fold"]):
            raise ValueError(f"Fold không khớp tại {qid}")
        rows.append(
            {
                "query_id": qid,
                "fold": int(baseline["fold"]),
                "gold_documents": [str(value) for value in baseline["gold_documents"]],
                "baseline_top5": [str(value) for value in baseline["top5"]],
                "v3_source_top5": [str(value) for value in v3["top5"]],
            }
        )

    configs = [
        {"baseline_weight": weight / 100.0, "v3_weight": 1.0 - weight / 100.0, "rrf_k": rrf_k}
        for weight in range(50, 100, 5)
        for rrf_k in (0, 2, 5, 10, 60)
    ]
    predictions: dict[tuple[float, int], list[list[str]]] = {}
    for config in configs:
        key = (config["baseline_weight"], config["rrf_k"])
        predictions[key] = [
            builder._weighted_rrf(
                [row["baseline_top5"], row["v3_source_top5"]],
                [config["baseline_weight"], config["v3_weight"]],
                config["rrf_k"],
            )[:5]
            for row in rows
        ]

    selected: list[dict[str, Any]] = []
    delta_tops: list[list[str] | None] = [None] * len(rows)
    for holdout in range(5):
        train_indices = [index for index, row in enumerate(rows) if row["fold"] != holdout]
        test_indices = [index for index, row in enumerate(rows) if row["fold"] == holdout]
        best: tuple[tuple[float, float, int, float, int], dict[str, Any]] | None = None
        for order, config in enumerate(configs):
            key = (config["baseline_weight"], config["rrf_k"])
            recall = 0.0
            precision = 0.0
            changes = 0
            for index in train_indices:
                gold = set(rows[index]["gold_documents"])
                predicted = set(predictions[key][index])
                recall += len(gold & predicted) / len(gold)
                precision += len(gold & predicted) / len(predicted)
                changes += predictions[key][index] != rows[index]["baseline_top5"]
            score = (
                recall / len(train_indices),
                precision / len(train_indices),
                -changes,
                config["baseline_weight"],
                -order,
            )
            if best is None or score > best[0]:
                best = (score, config)
        assert best is not None
        score, config = best
        key = (config["baseline_weight"], config["rrf_k"])
        for index in test_indices:
            delta_tops[index] = predictions[key][index]
        selected.append(
            {
                "fold": holdout,
                "selected_config": config,
                "training_macro_recall": score[0],
                "training_macro_precision": score[1],
                "training_changed_query_count": -score[2],
            }
        )
    if any(top is None for top in delta_tops):
        raise RuntimeError("Nested OOF bị thiếu prediction")
    for row, top in zip(rows, delta_tops):
        row["v3_top5"] = top

    baseline_metrics = strict._metrics(rows, "baseline_top5")
    delta_metrics = strict._metrics(rows, "v3_top5")
    outcomes = {"candidate_better": 0, "same": 0, "candidate_worse": 0}
    for row in rows:
        delta = strict._recall_delta(row, "baseline_top5", "v3_top5")
        outcomes[
            "candidate_better" if delta > 0 else "candidate_worse" if delta < 0 else "same"
        ] += 1
    policy_rows, policy_diagnostics = strict._policy_oof(rows)
    policy_metrics = strict._metrics(policy_rows, "top5")
    changed = sum(row["v3_top5"] != row["baseline_top5"] for row in rows)
    policy_changed = sum(row["source"] == "v3_locked" for row in policy_rows)

    output_dir = args.output_dir
    delta_output = [
        {
            "query_id": row["query_id"],
            "fold": row["fold"],
            "gold_documents": row["gold_documents"],
            "top5": row["v3_top5"],
        }
        for row in rows
    ]
    strict._write_jsonl(output_dir / "v3_delta_full_stack_predictions.jsonl", delta_output)
    strict._write_jsonl(output_dir / "v3_delta_policy_predictions.jsonl", policy_rows)
    report = {
        "schema_version": "public-093-v3-delta-nested-oof-v1",
        "status": "COMPLETE",
        "note": "Baseline được giữ làm nguồn neo; V3 là nguồn delta. Cấu hình được chọn trên bốn fold rồi áp dụng lên fold giữ lại.",
        "config_grid": configs,
        "fold_selections": selected,
        "baseline": baseline_metrics,
        "v3_delta_full_stack": {
            "pooled": delta_metrics,
            "per_fold": strict._fold_metrics(rows, "v3_top5"),
            "recall_delta_vs_baseline": delta_metrics["macro_recall"] - baseline_metrics["macro_recall"],
            "changed_query_count": changed,
            "outcomes": outcomes,
        },
        "change_policy_oof": {
            "pooled": policy_metrics,
            "recall_delta_vs_baseline": policy_metrics["macro_recall"] - baseline_metrics["macro_recall"],
            "changed_query_count": policy_changed,
            **policy_diagnostics,
        },
        "constraints": {"neural_training": False, "public_labels_used": False, "submission_created": False},
    }
    strict._write_json(output_dir / "v3_delta_report.json", report)
    print(json.dumps({"baseline": baseline_metrics, "delta": delta_metrics, "policy": policy_metrics, "outcomes": outcomes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
