"""Nested OOF adaptive K500 candidate rescue, không rerank K500 hàng loạt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
ADAPTIVE_FEATURE_NAMES = [
    "dense_top_score", "dense_top_margin_raw", "raw_unique_docs200",
    "raw_mean_chunks_per_doc200", "raw_max_chunks_per_doc200",
    "bge_top_margin", "dense_top_margin", "bge_entropy",
    "compact_unique_docs", "compact_mean_chunks_per_doc",
    "compact_max_chunks_per_doc", "query_token_count", "citation_cue",
    "baseline_candidate_set_overlap",
]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): sanitize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(value) for value in item]
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            item = float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, float):
            if math.isinf(item):
                return "+inf" if item > 0 else "-inf"
            if math.isnan(item):
                raise ValueError("NaN is not allowed in JSON reports")
        return item
    path.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_features(hits: Sequence[dict[str, Any]]) -> dict[str, float]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        by_doc[str(hit["doc_id"])].append(hit)
    bge = sorted((max(float(hit["bge_score"]) for hit in values) for values in by_doc.values()), reverse=True)
    dense = sorted((max(float(hit["dense_score"]) for hit in values) for values in by_doc.values()), reverse=True)
    probability = np.exp(np.asarray(bge) - bge[0])
    probability /= probability.sum()
    return {
        "bge_top_margin": bge[0] - bge[1] if len(bge) > 1 else 0.0,
        "dense_top_margin": dense[0] - dense[1] if len(dense) > 1 else 0.0,
        "bge_entropy": float(-(probability * np.log(probability + 1e-12)).sum() / math.log(len(probability))),
        "compact_unique_docs": float(len(by_doc)),
        "compact_mean_chunks_per_doc": float(len(hits) / len(by_doc)),
        "compact_max_chunks_per_doc": float(max(len(values) for values in by_doc.values())),
    }


def _load_k500(path: Path) -> dict[str, dict[str, Any]]:
    """Streaming parse, only retain document IDs/rank statistics, never text."""
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            hits = row.get("hits", [])
            docs200: list[str] = []
            docs500: list[str] = []
            seen200: set[str] = set()
            seen500: set[str] = set()
            first_scores: list[float] = []
            chunk_count: dict[str, int] = defaultdict(int)
            alignment200: list[dict[str, str]] = []
            for hit_index, hit in enumerate(hits, 1):
                doc = str(hit["doc_id"])
                score = float(hit.get("dense_score", hit.get("score", 0.0)))
                if hit_index <= 200:
                    if "chunk_id" not in hit:
                        raise ValueError(f"raw K500 hit lacks chunk_id: {qid}:{hit_index}")
                    first_scores.append(score)
                    chunk_count[doc] += 1
                    alignment200.append({"doc_id": doc, "chunk_id": str(hit["chunk_id"])})
                    if doc not in seen200:
                        seen200.add(doc)
                        docs200.append(doc)
                if doc not in seen500:
                    seen500.add(doc)
                    docs500.append(doc)
            result[qid] = {
                "docs200": docs200,
                "docs500": docs500,
                "dense_top_score": first_scores[0] if first_scores else 0.0,
                "dense_top_margin_raw": first_scores[0] - first_scores[1] if len(first_scores) > 1 else 0.0,
                "raw_unique_docs200": float(len(docs200)),
                "raw_mean_chunks_per_doc200": float(min(200, len(hits)) / len(docs200)) if docs200 else 0.0,
                "raw_max_chunks_per_doc200": float(max(chunk_count.values())) if chunk_count else 0.0,
                "alignment200": alignment200,
            }
            if index % 1000 == 0:
                print(f"[K500] parsed={index}", flush=True)
    return result


def _load_public_bge_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load a label-free public K200 compact cache.

    Historical adaptive features require BGE scores over the compact K200
    source.  Fresh dense K500 contains no BGE score, so it is never silently
    substituted.  The recovered public reranker cache uses ``rerank_score``;
    the historical compact cache uses ``bge_score``.  Both are accepted only
    at this explicit adapter boundary.
    """
    if path.suffix == ".jsonl":
        rows = _jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("predictions") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"unsupported public BGE cache schema: {path}")
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = str(row.get("query_id", row.get("question_id", "")))
        if not qid or qid in result:
            raise ValueError(f"invalid or duplicate public BGE query: {qid}")
        hits = row.get("hits")
        if not isinstance(hits, list) or len(hits) != 200:
            raise ValueError(f"public BGE cache must contain exactly 200 hits: {qid}")
        normalized = []
        for hit in hits:
            value = dict(hit)
            if "bge_score" not in value:
                if "rerank_score" not in value:
                    raise ValueError(f"public BGE hit lacks bge_score/rerank_score: {qid}")
                value["bge_score"] = value["rerank_score"]
            if "dense_score" not in value and "score" in value:
                value["dense_score"] = value["score"]
            if "dense_score" not in value:
                raise ValueError(f"public BGE hit lacks dense_score: {qid}")
            normalized.append(value)
        result[qid] = normalized
    if len(result) != 1000:
        raise ValueError(f"public BGE cache must contain 1000 queries, got {len(result)}")
    return result


def _verify_public_bge_alignment(
    raw: dict[str, dict[str, Any]],
    bge: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prove the recovered public BGE pool is the fresh dense K200 prefix."""
    if set(raw) != set(bge) or len(raw) != 1000:
        raise ValueError("public BGE/raw K500 query IDs do not match")
    for qid, raw_item in raw.items():
        raw_prefix = raw_item.get("alignment200")
        hits = bge[qid]
        if not isinstance(raw_prefix, list) or len(raw_prefix) != 200 or len(hits) < 200:
            raise ValueError(f"public BGE/raw K500 alignment is not exactly K200: {qid}")
        raw_pairs = {(str(item["doc_id"]), str(item["chunk_id"])) for item in raw_prefix}
        bge_pairs = {(str(item.get("doc_id", "")), str(item.get("chunk_id", ""))) for item in hits[:200]}
        if raw_pairs != bge_pairs or len(raw_pairs) != 200:
            raise ValueError(f"public BGE/raw K500 candidate/chunk alignment mismatch: {qid}")
        if any(int(item.get("dense_rank", item.get("rank", 0))) > 200 for item in hits[:200]):
            raise ValueError(f"public BGE cache contains a hit outside dense K200: {qid}")
    return {
        "status": "PASS",
        "query_count": len(raw),
        "prefix_hits_per_query": 200,
        "comparison": "exact_doc_id_and_chunk_id_set",
    }


def _top5_documents(hits: Sequence[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.get("doc_id", ""))
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
            if len(result) == 5:
                break
    if len(result) != 5:
        raise ValueError("public BGE cache cannot produce five unique candidate documents")
    return result


def _verify_public_bge_manifest(
    path: Path,
    *,
    cache_path: Path | None = None,
    raw_k500_path: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"public BGE provenance manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # The historical cache is retained as forensic evidence only.  Public
    # application requires the explicitly compatible fresh-prefix manifest.
    if manifest.get("schema_version") != "public-bge-k200-compatible-v1":
        raise ValueError("public BGE manifest is not the compatible fresh-prefix schema")
    if manifest.get("status") != "PUBLIC_BGE_K200_COMPATIBLE_PASS":
        raise ValueError("public BGE compatible manifest is not PASS")
    compatibility = manifest.get("compatibility", {})
    if compatibility.get("status") != "FRESH_RAW_K500_PREFIX_EXACT":
        raise ValueError("public BGE manifest lacks exact fresh raw K500-prefix provenance")
    if int(manifest.get("candidate_k", -1)) != 200 or int(manifest.get("top_n", -1)) != 200 or int(manifest.get("query_count", -1)) != 1000:
        raise ValueError("public BGE manifest is not a 1000-query K200 producer")
    inputs = manifest.get("inputs", {})
    model = manifest.get("model", {})
    if not inputs.get("raw_k500_sha256") or model.get("max_length") != 512 or model.get("single_logit") is not True:
        raise ValueError("public BGE manifest lacks scorer contract")
    if cache_path is not None and manifest.get("output_sha256") != _sha256(cache_path):
        raise ValueError("public BGE cache hash does not match its manifest")
    if raw_k500_path is not None and inputs.get("raw_k500_sha256") != _sha256(raw_k500_path):
        raise ValueError("public BGE cache was not produced from this fresh raw K500")
    return {"status": "PASS", "manifest": str(path), "candidate_k": 200, "query_count": 1000, "candidate_source": inputs.get("raw_k500"), "compatibility": compatibility}


def _training_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    baseline = {str(row["query_id"]): row for row in _jsonl(args.baseline)}
    candidate = {str(row["query_id"]): row for row in _jsonl(args.candidate)}
    compact = {str(row["query_id"]): row for row in _jsonl(args.fold0_cache) + _jsonl(args.fold1to4_cache)}
    print("[1/3] Streaming raw K500 (giữ metadata candidate, không giữ text)", flush=True)
    raw = _load_k500(args.raw_k500)
    if set(raw) != set(baseline) or set(compact) != set(baseline) or set(candidate) != set(baseline):
        raise ValueError("Coverage K500/K200 compact/candidate/baseline không khớp")
    rows: list[dict[str, Any]] = []
    for qid, item in baseline.items():
        question = str(train[qid]["question"])
        features = {**raw[qid], **_compact_features(compact[qid]["hits"])}
        features["query_token_count"] = float(len(re.findall(r"(?u)\b\w+\b", question)))
        features["citation_cue"] = float(bool(re.search(r"\b(điều|khoản|điểm|nghị định|thông tư|quyết định)\b", question, flags=re.IGNORECASE)))
        features["baseline_candidate_set_overlap"] = float(len(set(item["top5"]) & set(candidate[qid]["top5"])))
        gold = set(str(value) for value in item["gold_documents"])
        rows.append({"query_id": qid, "fold": int(item["fold"]), "gold_documents": sorted(gold), "features": features, "feature_names": ADAPTIVE_FEATURE_NAMES, "miss_k200": int(not gold <= set(raw[qid]["docs200"])), "k200_coverage": _coverage(gold, raw[qid]["docs200"]), "k500_coverage": _coverage(gold, raw[qid]["docs500"]), "baseline_top5": item["top5"]})
    return rows, raw


def _select_final_threshold(rows: list[dict[str, Any]], oof_path: Path) -> tuple[float, float, dict[str, Any]]:
    """Select one budget/threshold from train-only outer-OOF probabilities."""
    oof = {str(row["query_id"]): row for row in _jsonl(oof_path)}
    if set(oof) != {str(row["query_id"]) for row in rows}:
        raise ValueError("adaptive OOF artifact does not cover the training rows")
    probabilities = np.asarray([float(oof[str(row["query_id"])]["miss_probability"]) for row in rows], dtype=float)
    options = []
    for budget in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        count = int(math.ceil(len(rows) * budget))
        threshold = float("inf") if count == 0 else float(np.partition(probabilities, -count)[-count])
        expanded = probabilities >= threshold if count else np.zeros(len(rows), dtype=bool)
        coverage = mean([row["k500_coverage"] if use else row["k200_coverage"] for row, use in zip(rows, expanded)])
        fraction = float(expanded.mean())
        utility = coverage - 0.01 * fraction
        options.append({"utility": utility, "coverage": coverage, "budget": budget, "threshold": threshold, "expanded_query_count": int(expanded.sum()), "expanded_fraction": fraction})
    selected = max(options, key=lambda item: (item["utility"], item["coverage"], -item["expanded_query_count"]))
    return float(selected["budget"]), float(selected["threshold"]), {"options": options, "selected": selected, "source": "outer_OOF_probabilities_train_only"}


def _final_fit(args: argparse.Namespace) -> int:
    import joblib

    rows, _ = _training_rows(args)
    oof_path = args.oof_decisions
    if not oof_path.is_file():
        raise FileNotFoundError(f"train-only adaptive OOF decisions missing: {oof_path}")
    selected_budget, threshold, threshold_report = _select_final_threshold(rows, oof_path)
    x_train = np.asarray([_features(row) for row in rows], dtype=np.float64)
    y_train = np.asarray([row["miss_k200"] for row in rows], dtype=np.int64)
    model = CalibratedClassifierCV(Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2026))]), method="sigmoid", cv=3)
    model.fit(x_train, y_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "adaptive_k500_final_fit.joblib"
    joblib.dump({"model": model, "feature_names": ADAPTIVE_FEATURE_NAMES, "classifier": {"scaler": "StandardScaler", "estimator": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000, "random_state": 2026}, "selected_budget": selected_budget, "threshold": threshold}, model_path)
    manifest = {"schema_version": "adaptive-k500-final-fit-v1", "feature_names": ADAPTIVE_FEATURE_NAMES, "classifier": {"scaler": "StandardScaler", "estimator": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000, "random_state": 2026, "calibration": {"method": "sigmoid", "cv": 3}}, "selected_budget": selected_budget, "oof_reference_threshold": threshold, "gating_rule": "rank probabilities and expand exactly ceil(query_count * selected_budget); OOF threshold is diagnostic only", "training_query_count": len(rows), "training_input_sha256": {"train": _sha256(args.train), "baseline": _sha256(args.baseline), "candidate": _sha256(args.candidate), "fold0_cache": _sha256(args.fold0_cache), "fold1to4_cache": _sha256(args.fold1to4_cache), "raw_k500": _sha256(args.raw_k500), "oof_decisions": _sha256(oof_path)}, "source_code_sha256": _sha256(Path(__file__)), "threshold_selection": threshold_report, "model_sha256": _sha256(model_path), "no_public_labels_used": True, "public_application_not_run": True}
    _write_json(args.output_dir / "adaptive_k500_final_fit_manifest.json", manifest)
    print(json.dumps({"status": "ADAPTIVE_K500_FINAL_FIT_PASS", "model": str(model_path), "manifest": str(args.output_dir / "adaptive_k500_final_fit_manifest.json"), "selected_budget": selected_budget, "threshold": threshold, "training_query_count": len(rows), "no_public_labels_used": True}, indent=2))
    return 0


def _public_apply(args: argparse.Namespace) -> int:
    import joblib

    if args.public_output.exists() or args.public_report.exists():
        raise RuntimeError("refusing to overwrite existing public adaptive output/report")
    questions = json.loads(args.public_questions.read_text(encoding="utf-8-sig"))
    public_ids = {str(qid) for qid in questions}
    raw_path = args.public_raw_k500
    raw = _load_k500(raw_path)
    bge = _load_public_bge_cache(args.public_bge_k200)
    if set(raw) != public_ids or set(bge) != public_ids:
        raise ValueError("public question/raw K500/BGE cache IDs differ")
    alignment_report = _verify_public_bge_alignment(raw, bge)
    bge_manifest_report = _verify_public_bge_manifest(
        args.public_bge_manifest,
        cache_path=args.public_bge_k200,
        raw_k500_path=raw_path,
    )
    with zipfile.ZipFile(args.public_baseline) as archive:
        baseline_payload = json.loads(archive.read("submission.json"))
    baseline = {str(qid): [str(doc) for doc in row["answer"]] for qid, row in baseline_payload.items()}
    if set(baseline) != public_ids or any(len(docs) != 5 or len(set(docs)) != 5 for docs in baseline.values()):
        raise ValueError("public baseline must contain five unique docs for every public ID")
    bundle = joblib.load(args.model)
    if bundle.get("feature_names") != ADAPTIVE_FEATURE_NAMES:
        raise ValueError("adaptive final-fit feature schema mismatch")
    model = bundle["model"]
    selected_budget = float(bundle["selected_budget"])
    outputs = []
    for qid in sorted(public_ids, key=lambda value: int(value) if value.isdigit() else value):
        question = str(questions[qid].get("question", ""))
        features = {**raw[qid], **_compact_features(bge[qid])}
        features["query_token_count"] = float(len(re.findall(r"(?u)\b\w+\b", question)))
        features["citation_cue"] = float(bool(re.search(r"\b(điều|khoản|điểm|nghị định|thông tư|quyết định)\b", question, flags=re.IGNORECASE)))
        features["baseline_candidate_set_overlap"] = float(len(set(baseline[qid]) & set(_top5_documents(bge[qid]))))
        probability = float(model.predict_proba(np.asarray([_features({"features": features, "feature_names": ADAPTIVE_FEATURE_NAMES})], dtype=np.float64))[:, 1][0])
        outputs.append({"query_id": qid, "miss_probability": probability})
    expand_count = int(math.ceil(len(outputs) * selected_budget))
    ranked_ids = [row["query_id"] for row in sorted(outputs, key=lambda row: (-float(row["miss_probability"]), str(row["query_id"])))[:expand_count]]
    expanded_ids = set(ranked_ids)
    final_outputs = []
    for row in outputs:
        qid = row["query_id"]
        docs200 = raw[qid]["docs200"]
        docs500 = raw[qid]["docs500"]
        expand = qid in expanded_ids
        added = [doc for doc in docs500 if doc not in set(docs200)] if expand else []
        final_outputs.append({"query_id": qid, "miss_probability": row["miss_probability"], "expand_to_k500": bool(expand), "k200_documents": docs200, "k500_added_documents": added, "ranking": docs200 + added})
    _write_jsonl(args.public_output, final_outputs)
    expanded = [row for row in final_outputs if row["expand_to_k500"]]
    report = {"schema_version": "adaptive-k500-public-v1", "status": "PUBLIC_ADAPTIVE_K500_PASS", "query_count": len(final_outputs), "expanded_query_count": len(expanded), "expanded_fraction": len(expanded) / len(final_outputs), "selected_budget": selected_budget, "gating_rule": "rank public calibrated probabilities and expand exactly ceil(query_count * selected_budget)", "model_sha256": _sha256(args.model), "public_raw_k500_sha256": _sha256(raw_path), "public_bge_k200_sha256": _sha256(args.public_bge_k200), "public_bge_manifest_sha256": _sha256(args.public_bge_manifest), "public_baseline_sha256": _sha256(args.public_baseline), "public_bge_alignment": alignment_report, "public_bge_manifest": bge_manifest_report, "adaptive_feature_parity": True, "adaptive_gating_rule_parity": True, "no_public_labels_used": True}
    _write_json(args.public_report, report)
    print(json.dumps(report, indent=2))
    return 0


def _coverage(gold: set[str], docs: Sequence[str]) -> float:
    return len(gold & set(docs)) / len(gold)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _features(row: dict[str, Any]) -> list[float]:
    return [float(row["features"][name]) for name in row["feature_names"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl")
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/task1/recovery_096/baseline_093_oof/bge_chunk_only_predictions.jsonl")
    parser.add_argument("--fold0-cache", type=Path, default=ROOT / "artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl")
    parser.add_argument("--fold1to4-cache", type=Path, default=ROOT / "artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl")
    parser.add_argument("--raw-k500", type=Path, default=ROOT / "artifacts/task1/raw_k500.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1")
    parser.add_argument("--final-fit", action="store_true", help="fit and serialize the production gate using train-only OOF threshold selection")
    parser.add_argument("--oof-decisions", type=Path, default=ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_decisions.jsonl")
    parser.add_argument("--public", action="store_true", help="apply a serialized final-fit gate to public raw K500")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_final_fit.joblib")
    parser.add_argument("--public-questions", type=Path, default=ROOT / "data/raw/btc/LegalIR/public-official.json")
    parser.add_argument("--public-bge-k200", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_bge_k200_compatible.jsonl")
    parser.add_argument("--public-bge-manifest", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_bge_k200_compatible_manifest.json")
    parser.add_argument("--public-raw-k500", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl")
    parser.add_argument("--public-baseline", type=Path, default=ROOT / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip")
    parser.add_argument("--public-output", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_adaptive_k500.jsonl")
    parser.add_argument("--public-report", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_adaptive_k500_report.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.final_fit and args.public:
        raise ValueError("choose exactly one of --final-fit or --public")
    if args.final_fit:
        return _final_fit(args)
    if args.public:
        return _public_apply(args)
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    baseline = {str(row["query_id"]): row for row in _jsonl(args.baseline)}
    candidate = {str(row["query_id"]): row for row in _jsonl(args.candidate)}
    compact = {str(row["query_id"]): row for row in _jsonl(args.fold0_cache) + _jsonl(args.fold1to4_cache)}
    print("[1/3] Streaming raw K500 (giữ metadata candidate, không giữ text)", flush=True)
    raw = _load_k500(args.raw_k500)
    if set(raw) != set(baseline) or set(compact) != set(baseline):
        raise ValueError("Coverage K500/K200 compact/baseline không khớp")
    feature_names = ["dense_top_score", "dense_top_margin_raw", "raw_unique_docs200", "raw_mean_chunks_per_doc200", "raw_max_chunks_per_doc200", "bge_top_margin", "dense_top_margin", "bge_entropy", "compact_unique_docs", "compact_mean_chunks_per_doc", "compact_max_chunks_per_doc", "query_token_count", "citation_cue", "baseline_candidate_set_overlap"]
    rows: list[dict[str, Any]] = []
    for qid, item in baseline.items():
        question = str(train[qid]["question"])
        features = {**raw[qid], **_compact_features(compact[qid]["hits"])}
        features["query_token_count"] = float(len(re.findall(r"(?u)\b\w+\b", question)))
        features["citation_cue"] = float(bool(re.search(r"\b(điều|khoản|điểm|nghị định|thông tư|quyết định)\b", question, flags=re.IGNORECASE)))
        features["baseline_candidate_set_overlap"] = float(len(set(item["top5"]) & set(candidate[qid]["top5"])))
        gold = set(str(value) for value in item["gold_documents"])
        rows.append({"query_id": qid, "fold": int(item["fold"]), "gold_documents": sorted(gold), "features": features, "feature_names": feature_names, "miss_k200": int(not gold <= set(raw[qid]["docs200"])), "k200_coverage": _coverage(gold, raw[qid]["docs200"]), "k500_coverage": _coverage(gold, raw[qid]["docs500"]), "baseline_top5": item["top5"]})
    outputs: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    budgets = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    for holdout in range(5):
        outer_train = [row for row in rows if row["fold"] != holdout]
        outer_test = [row for row in rows if row["fold"] == holdout]
        x_train = np.asarray([_features(row) for row in outer_train], dtype=np.float64)
        y_train = np.asarray([row["miss_k200"] for row in outer_train], dtype=np.int64)
        model = CalibratedClassifierCV(Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2026))]), method="sigmoid", cv=3)
        model.fit(x_train, y_train)
        p_train = model.predict_proba(x_train)[:, 1]
        options: list[tuple[tuple[float, float, float], float, float]] = []
        for budget in budgets:
            count = int(math.ceil(len(outer_train) * budget))
            threshold = float("inf") if count == 0 else float(np.partition(p_train, -count)[-count])
            expanded = [value >= threshold for value in p_train] if count else [False] * len(outer_train)
            coverage = mean([row["k500_coverage"] if use else row["k200_coverage"] for row, use in zip(outer_train, expanded)])
            utility = coverage - 0.01 * (sum(expanded) / len(expanded))
            options.append(((utility, coverage, -sum(expanded)), budget, threshold))
        _, selected_budget, threshold = max(options, key=lambda item: item[0])
        p_test = model.predict_proba(np.asarray([_features(row) for row in outer_test], dtype=np.float64))[:, 1]
        expanded = [value >= threshold for value in p_test] if selected_budget else [False] * len(outer_test)
        true_miss = [row["miss_k200"] for row in outer_test]
        tp = sum(pred and label for pred, label in zip(expanded, true_miss))
        fp = sum(pred and not label for pred, label in zip(expanded, true_miss))
        fn = sum(not pred and label for pred, label in zip(expanded, true_miss))
        fold_reports.append({"fold": holdout, "selected_budget": selected_budget, "threshold": threshold, "expanded_query_count": sum(expanded), "miss_predictor_recall": tp / (tp + fn) if tp + fn else 0.0, "miss_predictor_precision": tp / (tp + fp) if tp + fp else 0.0, "oracle_k200": mean([row["k200_coverage"] for row in outer_test]), "oracle_adaptive": mean([row["k500_coverage"] if use else row["k200_coverage"] for row, use in zip(outer_test, expanded)]), "oracle_k500_blanket": mean([row["k500_coverage"] for row in outer_test])})
        for row, probability, use in zip(outer_test, p_test, expanded):
            raw_item = raw[row["query_id"]]
            docs200 = raw_item["docs200"]
            docs500 = raw_item["docs500"]
            outputs.append({"query_id": row["query_id"], "fold": holdout, "miss_k200": row["miss_k200"], "miss_probability": float(probability), "expand_to_k500": bool(use), "k200_unique_docs": len(docs200), "k500_unique_docs": len(docs500), "candidate_oracle_k200": row["k200_coverage"], "candidate_oracle_selected": row["k500_coverage"] if use else row["k200_coverage"], "baseline_top5": row["baseline_top5"], "k200_documents": docs200, "k500_added_documents": [doc for doc in docs500 if doc not in set(docs200)] if use else []})
    by_id = {row["query_id"]: row for row in outputs}
    outputs = [by_id[row["query_id"]] for row in rows]
    expanded = [row for row in outputs if row["expand_to_k500"]]
    adaptive = mean([row["candidate_oracle_selected"] for row in outputs])
    k200 = mean([row["k200_coverage"] for row in rows])
    k500 = mean([row["k500_coverage"] for row in rows])
    tp = sum(row["expand_to_k500"] and row["miss_k200"] for row in outputs)
    fp = sum(row["expand_to_k500"] and not row["miss_k200"] for row in outputs)
    fn = sum(not row["expand_to_k500"] and row["miss_k200"] for row in outputs)
    report = {"schema_version": "adaptive-k500-rescue-v1", "status": "CANDIDATE_ORACLE_ONLY", "note": "Không rerank K500: chỉ cache BGE K200 hiện có. Kết quả top5 giữ baseline và không phải bằng chứng promotion rerank.", "query_count": len(rows), "miss_k200_query_count": sum(row["miss_k200"] for row in rows), "expanded_query_count": len(expanded), "expanded_fraction": len(expanded) / len(rows), "miss_predictor": {"recall": tp / (tp + fn) if tp + fn else 0.0, "precision": tp / (tp + fp) if tp + fp else 0.0, "true_positive": tp, "false_positive": fp, "false_negative": fn}, "candidate_oracle": {"k200": k200, "adaptive": adaptive, "adaptive_delta_vs_k200": adaptive - k200, "blanket_k500": k500, "adaptive_fraction_of_blanket_gain": (adaptive - k200) / (k500 - k200) if k500 > k200 else None}, "top5_after_rerank": {"status": "NOT_RUN", "baseline_recall_unchanged": mean([len(set(row["gold_documents"]) & set(row["baseline_top5"])) / len(row["gold_documents"]) for row in rows])}, "per_fold": fold_reports, "cost": {"bge_new_chunks_scored": 0, "blanket_k500_bge_scoring": False, "raw_k500_streamed": True}, "constraints": {"neural_training": False, "public_submission_created": False}}
    _write_jsonl(args.output_dir / "adaptive_k500_decisions.jsonl", outputs)
    _write_json(args.output_dir / "report.json", report)
    print(f"Đã ghi {args.output_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
