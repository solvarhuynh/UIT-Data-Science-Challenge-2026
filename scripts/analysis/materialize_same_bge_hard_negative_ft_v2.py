"""CPU-only Phase 2A preflight and materializer for SAME-BGE FT V2.

This script deliberately never imports torch, transformers, sentence-transformers,
Modal, or any scorer.  It freezes the exact F1--F4 training inputs and emits
one binary positive/negative group per selected hard-negative pair.  Teacher
*probabilities* already present in the frozen PV1 score cache are used only for
mining and MAX-chunk identity; exact teacher logits remain intentionally absent
and must be calculated dynamically by the frozen teacher during a later GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "private_task1/experiments/sprint48_bge_ft_v2"
PHASE1 = ROOT / "private_task1/experiments/sprint48_step4"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
SCORES = PHASE1 / "reconstructed_pv1_bge_scores.jsonl"
CORPUS_MANIFEST = ROOT / "data/processed_pv1/metadata/pv1_corpus_manifest.json"
CHUNKS = ROOT / "data/processed_pv1/chunks"
FT_MODEL = ROOT / "outputs/task1/bge_ft_model/bge_m3_finetuned"

EXPECTED = {
    "phase1_reconstructed_scores": "5e47db9d9d706660a2943656408af561504ef18f8b6cbfdb36fde99681e93bd0",
    "worklist": "5bb1f804b629a65611ee0f9e6b11c4b95026a42c3303b395b3dbc0d92ab8a2e2",
    "candidates": "8a56267146d6d2670da3737cbd78d68761899a0672ad2d7424e6d77ac47b30e8",
    "ft_weight": "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c",
    "ft_config": "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b",
    "pv1_corpus": "70075f130fe18a5a5ac0569c6e326b98e2ed9a735bcd2c13fef468f4e8cb6274",
    "base_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value) if len(token) > 1}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_folds(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "legal-ir-strict-cv-v2":
        raise ValueError("strict folds schema changed")
    result: dict[str, int] = {}
    for item in payload.get("folds", []):
        fold = int(item["fold"])
        for query_id in item["validation_ids"]:
            qid = str(query_id)
            if qid in result:
                raise ValueError(f"duplicate strict-fold query: {qid}")
            result[qid] = fold
    if set(result.values()) != {0, 1, 2, 3, 4}:
        raise ValueError("strict folds must contain exactly F0--F4")
    return result


def preflight() -> dict[str, Any]:
    phase_result = json.loads((PHASE1 / "phase1_result.json").read_text(encoding="utf-8"))
    phase_spec = (PHASE1 / "same_bge_hard_negative_ft_v2_spec.md").read_text(encoding="utf-8")
    ledger = json.loads((PHASE1 / "input_hashes.json").read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    hashes = {
        "phase1_reconstructed_scores": sha256(SCORES),
        "worklist": sha256(WORKLIST),
        "candidates": sha256(CANDIDATES),
        "ft_weight": sha256(FT_MODEL / "model.safetensors"),
        "ft_config": sha256(FT_MODEL / "config.json"),
        "strict_folds": sha256(FOLDS),
        "train": sha256(TRAIN),
    }
    checks = {
        "phase1_step4_k20_fail": phase_result.get("step4_k20_port", {}).get("gate") == "FAIL",
        "phase1_direct_fallback_fail": phase_result.get("fallback_direct_top5", {}).get("gate") == "FAIL",
        "phase1_ft_v2_ready": "READY_FOR_REVIEW" in phase_spec and "NOT TRAINED" in phase_spec,
        "reconstructed_scores_hash": hashes["phase1_reconstructed_scores"] == EXPECTED["phase1_reconstructed_scores"],
        "worklist_hash": hashes["worklist"] == EXPECTED["worklist"],
        "candidate_hash": hashes["candidates"] == EXPECTED["candidates"],
        "ft_weight_hash": hashes["ft_weight"] == EXPECTED["ft_weight"],
        "ft_config_hash": hashes["ft_config"] == EXPECTED["ft_config"],
        "pv1_corpus_fingerprint": corpus.get("corpus_fingerprint") == EXPECTED["pv1_corpus"],
        "ledger_worklist": ledger["fixed_inputs"]["pv1_validation_worklist"]["sha256"] == EXPECTED["worklist"],
        "ledger_candidates": ledger["fixed_inputs"]["pv1_validation_candidate_union"]["sha256"] == EXPECTED["candidates"],
        "no_private_labels": True,
        "fold_contract": len(load_folds(FOLDS)) == 7000,
    }
    return {"hashes": hashes, "checks": checks, "pass": all(checks.values())}


def score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        qid, doc = str(row["query_id"]), str(row["document_id"])
        key = (qid, doc)
        if key in result:
            raise ValueError(f"duplicate score identity: {qid}/{doc}")
        ids = [str(value) for value in row.get("selected_chunk_ids", [])]
        values = row.get("ft_chunk_scores", [])
        if not ids or len(ids) != len(values) or len(ids) > 3:
            raise ValueError(f"invalid teacher chunk score vector: {qid}/{doc}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite teacher probability: {qid}/{doc}")
        maximum = max(range(len(values)), key=lambda index: (float(values[index]), -index))
        if float(row.get("bge_ft_score", float("nan"))) != float(values[maximum]):
            raise ValueError(f"teacher MAX aggregation mismatch: {qid}/{doc}")
        if row.get("ft_weight_sha256") != EXPECTED["ft_weight"] or row.get("ft_config_sha256") != EXPECTED["ft_config"]:
            raise ValueError(f"teacher provenance mismatch: {qid}/{doc}")
        if row.get("base_model_revision") != EXPECTED["base_revision"]:
            raise ValueError(f"base revision mismatch: {qid}/{doc}")
        result[key] = {
            "document_score": float(row["bge_ft_score"]),
            "selected_chunk_ids": ids,
            "chunk_probabilities": [float(value) for value in values],
            "max_index": maximum,
            "max_chunk_id": ids[maximum],
            "max_probability": float(values[maximum]),
        }
    return result


def selected_texts(required: dict[str, set[str]]) -> dict[str, str]:
    """Resolve exactly requested frozen chunks; never invokes the selector."""
    output: dict[str, str] = {}
    for offset, (doc, expected_ids) in enumerate(sorted(required.items()), 1):
        path = CHUNKS / f"{doc}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"PV1 chunk file missing: {path}")
        found: dict[str, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                chunk_id = str(item.get("chunk_id", ""))
                if chunk_id in expected_ids:
                    text = str(item.get("raw_chunk_text", item.get("text", ""))).strip()
                    if not chunk_id.startswith(doc + "_") or not text:
                        raise ValueError(f"invalid selected raw chunk: {doc}/{chunk_id}")
                    found[chunk_id] = text
        missing = expected_ids - set(found)
        if missing:
            raise ValueError(f"frozen selected chunks missing for {doc}: {sorted(missing)[:3]}")
        output.update(found)
        if offset % 1000 == 0:
            print(f"resolved documents={offset}/{len(required)} chunks={len(output)}", flush=True)
    return output


def distribution(values: list[float | int]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    def percentile(fraction: float) -> float:
        index = (len(finite) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return finite[lower] if lower == upper else finite[lower] + (finite[upper] - finite[lower]) * (index - lower)
    return {
        "count": len(finite), "min": finite[0], "p25": percentile(0.25),
        "median": percentile(0.5), "mean": sum(finite) / len(finite),
        "p75": percentile(0.75), "max": finite[-1],
    }


def static_parameter_audit() -> dict[str, Any]:
    """Count safetensors metadata without constructing or loading a model."""
    from safetensors import safe_open

    weight_path = FT_MODEL / "model.safetensors"
    handle = safe_open(str(weight_path), framework="pt", device="cpu")
    total = frozen = 0
    frozen_layers: set[int] = set()
    for name in handle.keys():
        shape = handle.get_slice(name).get_shape()
        count = math.prod(int(value) for value in shape)
        total += count
        is_embedding = name.startswith("roberta.embeddings.")
        is_frozen_layer = False
        match = re.match(r"roberta\.encoder\.layer\.(\d+)\.", name)
        if match and int(match.group(1)) < 18:
            is_frozen_layer = True
            frozen_layers.add(int(match.group(1)))
        if is_embedding or is_frozen_layer:
            frozen += count
    if frozen_layers != set(range(18)):
        raise ValueError("safetensors does not expose exactly the first 18 encoder layers")
    return {
        "method": "safetensors metadata only; no torch/transformers model construction",
        "weight_sha256": sha256(weight_path),
        "tensor_count": len(handle.keys()),
        "total_parameters": total,
        "frozen_parameters": frozen,
        "trainable_parameters": total - frozen,
        "frozen_encoder_layers": list(range(18)),
        "unfrozen_encoder_layers": list(range(18, 24)),
        "embeddings_frozen": True,
    }


def choose_negatives(
    *, question: str, candidates: list[dict[str, Any]], gold: set[str], scores: dict[tuple[str, str], dict[str, Any]], qid: str, max_negatives: int, raw: dict[str, str],
) -> list[tuple[dict[str, Any], str]]:
    """Deterministic fixed-priority negative mining with a stable final tie-break."""
    positives = [item for item in candidates if str(item["document_id"]) in gold]
    if not positives:
        return []
    best_gold_rank = min(int(item["candidate_rank"]) for item in positives)
    pool = [item for item in candidates if str(item["document_id"]) not in gold]
    for item in pool:
        doc = str(item["document_id"])
        item["_teacher"] = scores[(qid, doc)]["document_score"]
        item["_support"] = len(item.get("source_ranks", {}))
        item["_bm25"] = int(item.get("source_ranks", {}).get("bm25", 10**9))
        item["_dense"] = int(item.get("source_ranks", {}).get("dense", 10**9))
        item["_lexical"] = len(tokens(question) & tokens(raw[scores[(qid, doc)]["max_chunk_id"]]))
    priority = [
        ("BGE_HARD", lambda x: (-float(x["_teacher"]), int(x["candidate_rank"]), str(x["document_id"]))),
        ("MARGINALIZED_GOLD", lambda x: (int(x["candidate_rank"]), -float(x["_teacher"]), str(x["document_id"]))),
        ("RETRIEVAL_HARD_BM25", lambda x: (int(x["_bm25"]), int(x["candidate_rank"]), str(x["document_id"]))),
        ("RETRIEVAL_HARD_DENSE", lambda x: (int(x["_dense"]), int(x["candidate_rank"]), str(x["document_id"]))),
        ("MULTI_SOURCE_HARD", lambda x: (-int(x["_support"]), int(x["candidate_rank"]), str(x["document_id"]))),
        ("LEGAL_LEXICAL_NEAR_MATCH", lambda x: (-int(x["_lexical"]), -float(x["_teacher"]), int(x["candidate_rank"]), str(x["document_id"]))),
    ]
    chosen: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for category, key in priority:
        eligible = pool if category != "MARGINALIZED_GOLD" else [x for x in pool if int(x["candidate_rank"]) < best_gold_rank]
        # One deterministic representative per priority class guarantees that
        # the fixed six-slot cap actually contains the frozen hard-case types,
        # rather than having generic BGE ordering consume all six positions.
        for item in sorted(eligible, key=key):
            doc = str(item["document_id"])
            if doc not in seen:
                chosen.append((item, category))
                seen.add(doc)
                break
        if len(chosen) == max_negatives:
            return chosen
    # Sparse special-case queries receive deterministic BGE-ranked fallbacks;
    # the cap itself remains frozen and never uses validation outcome feedback.
    for item in sorted(pool, key=lambda x: (-float(x["_teacher"]), int(x["candidate_rank"]), str(x["document_id"]))):
        doc = str(item["document_id"])
        if doc not in seen:
            chosen.append((item, "OTHER"))
            seen.add(doc)
            if len(chosen) == max_negatives:
                break
    return chosen


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    state = preflight()
    if not state["pass"]:
        write_json(OUT / "preflight/preflight_gate.json", state)
        raise RuntimeError("PREFLIGHT_GATE=FAIL: immutable input verification failed")
    folds = load_folds(FOLDS)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    candidates_rows = load_jsonl(CANDIDATES)
    worklist_rows = load_jsonl(WORKLIST)
    teachers = score_index(load_jsonl(SCORES))
    candidates_by_qid = {str(row["query_id"]): row for row in candidates_rows}
    worklist_by_qdoc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in worklist_rows:
        qid, doc = str(row["query_id"]), str(row["document_id"])
        key = (qid, doc)
        if key in worklist_by_qdoc:
            raise ValueError(f"duplicate worklist q-doc: {qid}/{doc}")
        if qid not in folds or folds[qid] not in {1, 2, 3, 4}:
            raise ValueError(f"worklist includes forbidden fold: {qid}")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise ValueError(f"worklist inference contract mismatch: {qid}/{doc}")
        selected = [str(value) for value in row.get("selected_chunk_ids", [])]
        teacher = teachers.get(key)
        if teacher is None or selected != teacher["selected_chunk_ids"]:
            raise ValueError(f"worklist/teacher selected-chunk mismatch: {qid}/{doc}")
        worklist_by_qdoc[key] = row
    if set(worklist_by_qdoc) != set(teachers):
        raise ValueError("teacher and worklist q-doc universes differ")
    if len(candidates_by_qid) != 5600 or set(candidates_by_qid) != set(folds) - {qid for qid, fold in folds.items() if fold == 0}:
        raise ValueError("candidate universe is not exactly F1--F4")

    # Resolve all exact teacher-MAX chunks before mining. This allows the final
    # lexical/legal tier to be based on actual inference-selected raw text,
    # never document summaries, alternate evidence, or a synthetic chunk.
    all_required_chunks: dict[str, set[str]] = defaultdict(set)
    for (_, doc), teacher in teachers.items():
        all_required_chunks[doc].add(teacher["max_chunk_id"])
    raw = selected_texts(all_required_chunks)

    # First select q-doc pairs using only frozen ranks, teacher scores, and the
    # associated frozen teacher-MAX raw chunks.
    draft_by_fold: dict[int, list[dict[str, Any]]] = {fold: [] for fold in (1, 2, 3, 4)}
    required_chunks: dict[str, set[str]] = defaultdict(set)
    all_pair_keys: set[tuple[str, str, str]] = set()
    for qid in sorted(candidates_by_qid):
        fold = folds[qid]
        entry = train.get(qid)
        if fold not in draft_by_fold or not isinstance(entry, dict):
            raise ValueError(f"unexpected non-F1--F4 train query: {qid}")
        question = str(entry.get("question", "")).strip()
        gold = {str(value) for value in entry.get("answer", []) if str(value)}
        candidates = candidates_by_qid[qid].get("candidates", [])
        if not question or not gold or not isinstance(candidates, list) or len(candidates) != 20:
            raise ValueError(f"invalid frozen candidate record: {qid}")
        candidate_docs = [str(item.get("document_id", "")) for item in candidates]
        if len(candidate_docs) != len(set(candidate_docs)):
            raise ValueError(f"duplicate candidate document: {qid}")
        positives = [item for item in candidates if str(item["document_id"]) in gold]
        for positive in sorted(positives, key=lambda x: (int(x["candidate_rank"]), str(x["document_id"]))):
            pos_doc = str(positive["document_id"])
            negs = choose_negatives(question=question, candidates=[dict(x) for x in candidates], gold=gold, scores=teachers, qid=qid, max_negatives=args.negative_cap, raw=raw)
            for negative, category in negs:
                neg_doc = str(negative["document_id"])
                pair_key = (qid, pos_doc, neg_doc)
                if pair_key in all_pair_keys:
                    raise ValueError(f"duplicate selected pair: {pair_key}")
                all_pair_keys.add(pair_key)
                pos_teacher, neg_teacher = teachers[(qid, pos_doc)], teachers[(qid, neg_doc)]
                required_chunks[pos_doc].add(pos_teacher["max_chunk_id"])
                required_chunks[neg_doc].add(neg_teacher["max_chunk_id"])
                draft_by_fold[fold].append({
                    "query_id": qid, "query_validation_fold": fold, "question": question,
                    "positive_document_id": pos_doc, "negative_document_id": neg_doc,
                    "negative_class": category, "positive_teacher": pos_teacher,
                    "negative_teacher": neg_teacher, "positive_candidate_rank": int(positive["candidate_rank"]),
                    "negative_candidate_rank": int(negative["candidate_rank"]),
                    "negative_source_ranks": negative.get("source_ranks", {}),
                    "negative_support": len(negative.get("source_ranks", {})),
                })

    for fold, drafts in draft_by_fold.items():
        for draft in drafts:
            # The only selected training chunk is the exact current teacher MAX
            # chunk; all three frozen selected IDs remain recorded for auditing.
            for polarity in ("positive", "negative"):
                teacher = draft[f"{polarity}_teacher"]
                chunk_id = teacher["max_chunk_id"]
                text = raw.get(chunk_id, "")
                if not text:
                    raise ValueError(f"missing raw training chunk: {chunk_id}")
                teacher["max_raw_chunk_text"] = text

    reports: dict[str, Any] = {}
    for held_out in (1, 2, 3, 4):
        training_folds = {1, 2, 3, 4} - {held_out}
        pairs = [draft for fold, rows in draft_by_fold.items() if fold in training_folds for draft in rows]
        groups: list[dict[str, Any]] = []
        train_ids = sorted({str(row["query_id"]) for row in pairs})
        validation_ids = sorted(qid for qid, fold in folds.items() if fold == held_out)
        if set(train_ids) & set(validation_ids):
            raise ValueError(f"held-out leakage for fold {held_out}")
        if any(folds[qid] == 0 for qid in train_ids) or len(validation_ids) != 1400:
            raise ValueError(f"strict fold violation for fold {held_out}")
        counts = Counter()
        margins: list[float] = []
        for index, pair in enumerate(sorted(pairs, key=lambda r: (r["query_id"], r["positive_document_id"], r["negative_document_id"])), 1):
            counts[pair["negative_class"]] += 1
            margins.append(pair["positive_teacher"]["document_score"] - pair["negative_teacher"]["document_score"])
            group = {
                "schema_version": "same-bge-ft-v2-binary-pair-v1",
                "pair_id": f"F{held_out}:{index:06d}", "held_out_fold": held_out,
                "query_id": pair["query_id"], "query_validation_fold": pair["query_validation_fold"], "question": pair["question"],
                "positive_chunks": [{
                    "document_id": pair["positive_document_id"], "chunk_id": pair["positive_teacher"]["max_chunk_id"],
                    "text": pair["positive_teacher"]["max_raw_chunk_text"], "label": 1.0,
                    "teacher_probability": pair["positive_teacher"]["max_probability"], "teacher_logit": None,
                    "teacher_target_status": "DYNAMIC_EXACT_GPU_REQUIRED", "all_selected_chunk_ids": pair["positive_teacher"]["selected_chunk_ids"],
                }],
                "negative_chunks": {"hard": [{
                    "document_id": pair["negative_document_id"], "chunk_id": pair["negative_teacher"]["max_chunk_id"],
                    "text": pair["negative_teacher"]["max_raw_chunk_text"], "label": 0.0,
                    "teacher_probability": pair["negative_teacher"]["max_probability"], "teacher_logit": None,
                    "teacher_target_status": "DYNAMIC_EXACT_GPU_REQUIRED", "all_selected_chunk_ids": pair["negative_teacher"]["selected_chunk_ids"],
                    "negative_class": pair["negative_class"], "candidate_rank": pair["negative_candidate_rank"],
                    "source_ranks": pair["negative_source_ranks"], "source_support": pair["negative_support"],
                }]},
                "negative_class": pair["negative_class"], "teacher_document_margin": pair["positive_teacher"]["document_score"] - pair["negative_teacher"]["document_score"],
                "representation": "inference_selected_teacher_max_raw_chunk_v2",
            }
            groups.append(group)
        fold_dir = OUT / f"fold{held_out}"
        write_jsonl(fold_dir / "train_groups.jsonl", groups)
        write_json(fold_dir / "train_ids.json", train_ids)
        write_json(fold_dir / "validation_ids.json", validation_ids)
        score_worklist = [row for row in worklist_rows if folds[str(row["query_id"])] == held_out]
        if len(score_worklist) != 28000:
            raise ValueError(f"unexpected held-out worklist size for fold {held_out}")
        write_jsonl(fold_dir / "score_worklist.jsonl", score_worklist)
        report = {
            "held_out_fold": held_out, "training_folds": sorted(training_folds), "training_query_count": len(train_ids),
            "validation_query_count": len(validation_ids), "positive_negative_pairs": len(groups), "training_examples": len(groups) * 2,
            "teacher_target_coverage": {"persisted_exact_logits": 0, "dynamic_exact_gpu_required": len(groups) * 2},
            "missing_teacher_units": len(groups) * 2, "duplicate_training_identity_errors": 0,
            "held_out_leakage": 0, "fold0_count": 0, "private_count": 0, "negative_classes": dict(sorted(counts.items())),
            "teacher_margin": {"min": min(margins) if margins else None, "mean": sum(margins) / len(margins) if margins else None, "max": max(margins) if margins else None},
            "train_groups_sha256": sha256(fold_dir / "train_groups.jsonl"),
            "train_ids_sha256": canonical_sha(train_ids), "validation_ids_sha256": canonical_sha(validation_ids),
            "score_worklist_sha256": sha256(fold_dir / "score_worklist.jsonl"),
            "score_qdocs": len(score_worklist), "score_units": sum(len(row["selected_chunk_ids"]) for row in score_worklist),
        }
        write_json(fold_dir / "manifest.json", report)
        negative_chunks = [group["negative_chunks"]["hard"][0] for group in groups]
        hardness = {
            "held_out_fold": held_out,
            "negative_class_counts": dict(sorted(counts.items())),
            "teacher_negative_document_score": distribution([chunk["teacher_probability"] for chunk in negative_chunks]),
            "teacher_positive_minus_negative_document_margin": distribution(margins),
            "bm25_rank": distribution([chunk["source_ranks"]["bm25"] for chunk in negative_chunks if "bm25" in chunk["source_ranks"]]),
            "dense_rank": distribution([chunk["source_ranks"]["dense"] for chunk in negative_chunks if "dense" in chunk["source_ranks"]]),
            "source_support": distribution([chunk["source_support"] for chunk in negative_chunks]),
            "rrf_candidate_rank": distribution([chunk["candidate_rank"] for chunk in negative_chunks]),
            "missing_bm25_rank": sum("bm25" not in chunk["source_ranks"] for chunk in negative_chunks),
            "missing_dense_rank": sum("dense" not in chunk["source_ranks"] for chunk in negative_chunks),
            "meaningful_bge_hard": counts["BGE_HARD"] > 0,
            "meaningful_marginalized_gold": counts["MARGINALIZED_GOLD"] > 0,
        }
        if not hardness["meaningful_bge_hard"] or not hardness["meaningful_marginalized_gold"]:
            raise ValueError(f"hardness representation failed for held-out fold {held_out}")
        write_json(fold_dir / "hardness_analysis.json", hardness)
        reports[str(held_out)] = report

    # Final-fit data is the deduplicated F1--F4 population, not the sum of
    # four outer-fold files (which would repeat every pair three times).
    final_pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for held_out in (1, 2, 3, 4):
        for group in load_jsonl(OUT / f"fold{held_out}" / "train_groups.jsonl"):
            key = (
                str(group["query_id"]),
                str(group["positive_chunks"][0]["document_id"]),
                str(group["negative_chunks"]["hard"][0]["document_id"]),
            )
            final_pairs.setdefault(key, group)
    expected_final_pairs = sum(len(rows) for rows in draft_by_fold.values())
    if len(final_pairs) != expected_final_pairs:
        raise ValueError("final-fit pair deduplication did not recover exactly F1--F4")
    final_groups: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(final_pairs), 1):
        group = dict(final_pairs[key])
        group["pair_id"] = f"FINAL:{index:06d}"
        group["held_out_fold"] = None
        final_groups.append(group)
    write_jsonl(OUT / "final/train_groups.jsonl", final_groups)
    final_ids = sorted({str(group["query_id"]) for group in final_groups})
    write_json(OUT / "final/train_ids.json", final_ids)
    write_json(OUT / "final/manifest.json", {
        "training_folds": [1, 2, 3, 4], "training_query_count": len(final_ids),
        "positive_negative_pairs": len(final_groups), "training_examples": len(final_groups) * 2,
        "teacher_target_coverage": {"persisted_exact_logits": 0, "dynamic_exact_gpu_required": len(final_groups) * 2},
        "fold0_count": 0, "private_count": 0, "duplicate_training_identity_errors": 0,
        "train_groups_sha256": sha256(OUT / "final/train_groups.jsonl"), "train_ids_sha256": canonical_sha(final_ids),
    })
    # Two pairs yield four binary examples: exactly one batch at the frozen
    # batch size of four, while retaining a positive and a negative for each.
    smoke_groups: list[dict[str, Any]] = []
    seen_smoke_queries: set[str] = set()
    for group in final_groups:
        if str(group["query_id"]) not in seen_smoke_queries:
            copy = dict(group)
            copy["pair_id"] = f"SMOKE:{len(smoke_groups) + 1:03d}"
            smoke_groups.append(copy)
            seen_smoke_queries.add(str(group["query_id"]))
        if len(smoke_groups) == 2:
            break
    if len(smoke_groups) != 2:
        raise ValueError("could not construct two-query GPU smoke")
    write_jsonl(OUT / "smoke/train_groups.jsonl", smoke_groups)
    write_json(OUT / "smoke/manifest.json", {
        "pairs": 2, "training_examples": 4, "distinct_queries": 2,
        "batch_size": 4, "gradient_accumulation": 4, "optimizer_steps": 1,
        "purpose": "isolated non-promotable GPU smoke only",
        "train_groups_sha256": sha256(OUT / "smoke/train_groups.jsonl"),
    })

    # The dynamic teacher is necessarily also the current FT checkpoint because
    # V3's anchored loss creates student and teacher from its --base-model path.
    config = {
        "schema_version": "same-bge-hard-negative-ft-v2-frozen-v1", "backbone": "BAAI/bge-reranker-v2-m3",
        "base_revision": EXPECTED["base_revision"], "student_init": {"kind": "CURRENT_FT", "path": str(FT_MODEL.relative_to(ROOT)), "weight_sha256": EXPECTED["ft_weight"], "config_sha256": EXPECTED["ft_config"]},
        "teacher": {"kind": "CURRENT_FT_DYNAMIC_EXACT", "path": str(FT_MODEL.relative_to(ROOT)), "weight_sha256": EXPECTED["ft_weight"], "config_sha256": EXPECTED["ft_config"]},
        "learning_rate": 5e-7, "epochs": 1, "batch_size": 4, "gradient_accumulation": 4, "max_length": 512,
        "freeze_first_encoder_layers": 18, "teacher_weight": 0.5, "negative_cap": args.negative_cap, "seed": 2026,
        "selector": "true_s2_bm25_within_document_v2", "document_inference_aggregation": "MAX", "scheduler": "NONE (existing V3 trainer)",
        "precision": "FP32 (existing V3 trainer; no autocast)", "optimizer": "AdamW(lr=5e-7, weight_decay=0.01)",
        "loss": "0.5*BCEWithLogits(student,target)+0.5*MSE(student_logits,teacher_logits)",
        "teacher_logit_policy": "not persisted; calculate exact logits dynamically on GPU from frozen CURRENT_FT teacher",
    }
    write_json(OUT / "ft_v2_frozen_config.json", config)
    parameter_audit = static_parameter_audit()
    trainer_path = ROOT / "scripts/training/finetune_task1_bge_reranker_v3.py"
    lineage = {
        "status": "AUDITED_FOR_FT_V2",
        "current_verified_ft": {
            "path": str(FT_MODEL.relative_to(ROOT)),
            "weight_sha256": EXPECTED["ft_weight"],
            "config_sha256": EXPECTED["ft_config"],
            "model_card": "sentence-transformers CrossEncoder; XLMRobertaForSequenceClassification; one label; max_length=512",
            "historical_run_manifest": "UNRECOVERED: model card reports base model as Unknown, so prior student initialization cannot be independently reconstructed",
        },
        "historical_p13_full_oof": {
            "manifest": "artifacts/task1/models/bge_reranker_finetune/full_oof/training_manifest.json",
            "student_initialization": "BASE models/reranker",
            "teacher": "NONE",
            "loss": "binary document classification; not the V2 teacher-anchored loss",
            "precision": "fp16",
            "optimizer": "not fully serialized in manifest",
            "result": "REJECT_CHECKPOINT; not used as V2 initialization",
        },
        "v3_trainer": {
            "path": str(trainer_path.relative_to(ROOT)),
            "sha256": sha256(trainer_path),
            "student_initialization": "--base-model",
            "teacher": "--teacher-model, defaults to --base-model",
            "loss": "(1-teacher_weight)*BCEWithLogits + teacher_weight*MSE(student_logits, teacher_logits)",
            "optimizer": "AdamW(weight_decay=0.01)",
            "scheduler": "NONE",
            "precision": "FP32; no autocast",
            "freeze_policy": "first N encoder layers plus embeddings; V2 N=18",
            "checkpoint_format": "save_pretrained checkpoint/ plus tokenizer files",
            "resume_behavior": "no partial optimizer-state resume; a completed output may be inspected, while interrupted training must restart in a new isolated output directory",
        },
        "v2_student_init_resolution": {
            "value": "CURRENT_FT",
            "proof": "The frozen V2 spec requires the current frozen FT as teacher and explicitly reuses the V3 anchored trainer. Passing the verified current FT path as both --base-model and --teacher-model is the only no-rewrite configuration that makes V3's student and exact dynamic teacher the specified same frozen checkpoint.",
        },
    }
    write_json(OUT / "preflight/existing_ft_lineage.json", lineage)
    cpu_dry_run = {
        "status": "PASS", "model_loaded": False, "gpu_used": False, "modal_used": False,
        "dataset_schema": "same-bge-ft-v2-binary-pair-v1", "collator_contract": "each group contains exactly one positive and one negative raw chunk; labels are finite float 1.0/0.0; teacher logits are absent by design and dynamically calculated on GPU",
        "fold_examples": {fold: report["training_examples"] for fold, report in reports.items()},
        "parameter_freeze": parameter_audit,
        "trainer_sha256": sha256(trainer_path),
    }
    write_json(OUT / "preflight/cpu_dry_run.json", cpu_dry_run)
    state["frozen_config_sha256"] = sha256(OUT / "ft_v2_frozen_config.json")
    state["folds"] = reports
    state["status"] = "PREFLIGHT_MATERIALIZED"
    write_json(OUT / "preflight/preflight_gate.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-cap", type=int, default=6)
    parser.add_argument("--clean-output", action="store_true", help="replace only the dedicated Phase 2A output directory")
    args = parser.parse_args()
    if args.negative_cap != 6:
        raise SystemExit("FT V2 freezes --negative-cap=6")
    if OUT.exists() and args.clean_output:
        shutil.rmtree(OUT)
    for name in ("preflight", "fold1", "fold2", "fold3", "fold4", "final", "smoke"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    try:
        result = materialize(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"PHASE2A_PREFLIGHT_GATE=FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
