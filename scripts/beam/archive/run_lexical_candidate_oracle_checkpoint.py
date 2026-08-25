"""Resume-safe strict-OOF lexical candidate-oracle ablation.

Each (source, fold) ranking is atomically checkpointed, so a timeout preserves
completed work.  Rankings use no gold at inference; gold is read only for the
final candidate-oracle report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from udsc2026.evaluation.legal_ir import normalize_legal_ir_matching_question  # noqa: E402
from udsc2026.evaluation.legal_ir_lexical import LegalContext, build_bm25f_rankings, build_citation_rankings, build_knn_rankings  # noqa: E402


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def contexts(path: Path) -> list[LegalContext]:
    out = []
    for item in sorted(path.glob("*.json")):
        row = json.loads(item.read_text(encoding="utf-8-sig"))
        out.append(LegalContext(str(row["id"]), str(row["passage"]), str(row.get("title", row.get("name", "")) or "")))
    return out


def coverage(gold: set[str], candidates: Sequence[str]) -> float:
    return len(gold & set(candidates)) / len(gold)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    p.add_argument("--folds", type=Path, default=ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json")
    p.add_argument("--contexts-dir", type=Path, default=ROOT / "data/raw/btc/LegalIR/selected-contexts")
    p.add_argument("--baseline", type=Path, default=ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl")
    p.add_argument("--adaptive", type=Path, default=ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_decisions.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/task1/recovery_096/candidate_union_ablations/checkpointed_lexical_v1")
    p.add_argument("--sources", nargs="+", choices=("bm25", "bm25f", "knn_word", "knn_char", "citation"), default=("bm25", "bm25f", "knn_word", "knn_char", "citation"))
    args = p.parse_args(argv)
    baseline = {str(x["query_id"]): x for x in rows(args.baseline)}
    adaptive = {str(x["query_id"]): x for x in rows(args.adaptive)}
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    fold_data = json.loads(args.folds.read_text(encoding="utf-8-sig"))["folds"]
    folds = {str(q): int(f["fold"]) for f in fold_data for q in f["validation_ids"]}
    if set(baseline) != set(adaptive) or set(baseline) != set(folds):
        raise ValueError("baseline/adaptive/folds coverage mismatch")
    corpus = contexts(args.contexts_dir)
    fingerprint = {"train": digest(args.train), "folds": digest(args.folds), "baseline": digest(args.baseline), "adaptive": digest(args.adaptive)}
    checkpoint_dir = args.output_dir / "checkpoints"
    timings: dict[str, float] = {}
    for source in args.sources:
        for fold in range(5):
            out = checkpoint_dir / f"{source}_fold{fold}.json"
            if out.is_file():
                continue
            start = time.monotonic()
            valid = sorted(q for q, f in folds.items() if f == fold)
            questions = {q: str(train[q]["question"]) for q in valid}
            if source in ("knn_word", "knn_char"):
                excluded = {normalize_legal_ir_matching_question(questions[q]) for q in valid}
                pool = [(q, str(train[q]["question"]), train[q]["answer"]) for q in baseline if q not in set(valid) and normalize_legal_ir_matching_question(str(train[q]["question"])) not in excluded]
                ranked = build_knn_rankings(questions, pool, analyzer="word", ngram_range=(1, 2)) if source == "knn_word" else build_knn_rankings(questions, pool, analyzer="char_wb", ngram_range=(3, 5))
            elif source == "citation":
                ranked = build_citation_rankings(questions, corpus)
            else:
                ranked = build_bm25f_rankings(questions, corpus, title_weight=0.0 if source == "bm25" else 1.0, top_k=200)
            atomic(out, {"source": source, "fold": fold, "input_sha256": fingerprint, "rankings": ranked, "seconds": time.monotonic() - start})
    report: dict[str, Any] = {"schema_version": "checkpointed-lexical-candidate-oracle-v1", "input_sha256": fingerprint, "sources": {}}
    for source in args.sources:
        ranking = {}
        seconds = 0.0
        for fold in range(5):
            value = json.loads((checkpoint_dir / f"{source}_fold{fold}.json").read_text(encoding="utf-8"))
            if value["input_sha256"] != fingerprint:
                raise ValueError(f"stale checkpoint: {source} fold {fold}")
            ranking.update(value["rankings"]); seconds += float(value["seconds"])
        source_oracle = [coverage(set(baseline[q]["gold_documents"]), ranking[q]) for q in baseline]
        k200_union = [coverage(set(baseline[q]["gold_documents"]), adaptive[q]["k200_documents"] + ranking[q]) for q in baseline]
        adaptive_union = [coverage(set(baseline[q]["gold_documents"]), adaptive[q]["k200_documents"] + adaptive[q]["k500_added_documents"] + ranking[q]) for q in baseline]
        k200_miss = [q for q in baseline if adaptive[q]["miss_k200"]]
        multi = [q for q in baseline if len(baseline[q]["gold_documents"]) > 1]
        base_k200 = sum(coverage(set(baseline[q]["gold_documents"]), adaptive[q]["k200_documents"]) for q in baseline) / len(baseline)
        union_value = sum(k200_union) / len(k200_union)
        report["sources"][source] = {"coverage": len(ranking), "oracle_source": sum(source_oracle) / len(source_oracle), "oracle_k200_union": union_value, "oracle_adaptive_union": sum(adaptive_union) / len(adaptive_union), "oracle_gain_vs_k200": union_value - base_k200, "k200_miss_rescued": sum(coverage(set(baseline[q]["gold_documents"]), ranking[q]) > 0 for q in k200_miss), "multi_gold_miss_rescued": sum(coverage(set(baseline[q]["gold_documents"]), ranking[q]) > 0 and coverage(set(baseline[q]["gold_documents"]), adaptive[q]["k200_documents"]) < 1 for q in multi), "runtime_seconds": seconds, "gain_per_second": (union_value - base_k200) / seconds if seconds else None}
    ordered = sorted(report["sources"], key=lambda s: report["sources"][s]["gain_per_second"], reverse=True)
    report["ranking_by_oracle_gain_per_cost"] = ordered
    atomic(args.output_dir / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
