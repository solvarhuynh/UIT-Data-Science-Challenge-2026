"""Prepare the isolated V3 workspace from immutable V2 inputs (CPU only)."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "private_task1/experiments/sprint48_bge_ft_v2"
V3 = ROOT / "private_task1/experiments/sprint48_bge_ft_v3_conflict_anchor"
EXPECTED_WEIGHT = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
EXPECTED_CONFIG = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
CLASSES = ("BGE_HARD", "MARGINALIZED_GOLD", "RETRIEVAL_HARD_BM25", "RETRIEVAL_HARD_DENSE", "MULTI_SOURCE_HARD", "LEGAL_LEXICAL_NEAR_MATCH", "OTHER")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    v2_audit = json.loads((V2 / "fold1/forensic_audit.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads((V2 / "fold1/checkpoint_tensor_diff_manifest.json").read_text(encoding="utf-8"))
    if v2_audit.get("ft_v2_status") != "CLOSED":
        raise SystemExit("V2 is not CLOSED")
    if v2_manifest.get("current_ft_weight_sha256") != EXPECTED_WEIGHT:
        raise SystemExit("CURRENT_FT weight identity mismatch")
    if v2_manifest.get("current_ft_config_sha256") != EXPECTED_CONFIG:
        raise SystemExit("CURRENT_FT config identity mismatch")

    V3.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, dict[str, str]] = {}
    fold_analysis: dict[str, Any] = {}
    total_positive_consistent = total_positive_conflict = 0
    total_negative_consistent = total_negative_conflict = 0
    for fold in range(1, 5):
        source = V2 / f"fold{fold}"
        target = V3 / f"fold{fold}"
        target.mkdir(parents=True, exist_ok=True)
        names = ("train_groups.jsonl", "score_worklist.jsonl", "train_ids.json", "validation_ids.json")
        source_hashes[str(fold)] = {}
        for name in names:
            source_path = source / name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_hashes[str(fold)][name] = sha256(source_path)
            shutil.copy2(source_path, target / name)
        v2_fold_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if source_hashes[str(fold)]["train_groups.jsonl"] != v2_fold_manifest["train_groups_sha256"]:
            raise SystemExit(f"V2 train group hash mismatch F{fold}")
        if source_hashes[str(fold)]["score_worklist.jsonl"] != v2_fold_manifest["score_worklist_sha256"]:
            raise SystemExit(f"V2 worklist hash mismatch F{fold}")

        positive_consistent = positive_conflict = 0
        negative_consistent = negative_conflict = 0
        class_counts: Counter[str] = Counter()
        class_consistent: Counter[str] = Counter()
        class_conflict: Counter[str] = Counter()
        group_count = 0
        example_count = 0
        for row in iter_jsonl(source / "train_groups.jsonl"):
            group_count += 1
            positives = row.get("positive_chunks", [])
            negatives = row.get("negative_chunks", {}).get("hard", [])
            example_count += len(positives) + len(negatives)
            for item in positives:
                probability = float(item["teacher_probability"])
                if not math.isfinite(probability):
                    raise ValueError(f"nonfinite positive teacher probability F{fold}")
                if probability >= 0.5:
                    positive_consistent += 1
                else:
                    positive_conflict += 1
            for item in negatives:
                probability = float(item["teacher_probability"])
                if not math.isfinite(probability):
                    raise ValueError(f"nonfinite negative teacher probability F{fold}")
                cls = str(item.get("negative_class", "OTHER"))
                class_counts[cls] += 1
                if probability < 0.5:
                    negative_consistent += 1
                    class_consistent[cls] += 1
                else:
                    negative_conflict += 1
                    class_conflict[cls] += 1
        total_positive_consistent += positive_consistent
        total_positive_conflict += positive_conflict
        total_negative_consistent += negative_consistent
        total_negative_conflict += negative_conflict
        fold_analysis[f"F{fold}"] = {
            "train_groups": group_count,
            "training_examples": example_count,
            "positive_teacher_consistent": positive_consistent,
            "positive_teacher_conflict": positive_conflict,
            "negative_teacher_consistent": negative_consistent,
            "negative_teacher_conflict": negative_conflict,
            "negative_class_counts": dict(sorted(class_counts.items())),
            "negative_class_consistent": dict(sorted(class_consistent.items())),
            "negative_class_conflict": dict(sorted(class_conflict.items())),
        }

    config = {
        "schema_version": "same-bge-hard-negative-ft-v3-conflict-anchor-frozen-v1",
        "hypothesis": "CONFLICT_AWARE_TEACHER_ANCHORING",
        "backbone": "BAAI/bge-reranker-v2-m3",
        "base_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "student_init": {"kind": "CURRENT_FT", "path": "outputs\\task1\\bge_ft_model\\bge_m3_finetuned", "weight_sha256": EXPECTED_WEIGHT, "config_sha256": EXPECTED_CONFIG},
        "teacher": {"kind": "CURRENT_FT_DYNAMIC_EXACT", "path": "outputs\\task1\\bge_ft_model\\bge_m3_finetuned", "weight_sha256": EXPECTED_WEIGHT, "config_sha256": EXPECTED_CONFIG},
        "learning_rate": 5e-7,
        "epochs": 1,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "max_length": 512,
        "freeze_first_encoder_layers": 18,
        "teacher_weight": 0.5,
        "teacher_anchor_mode": "conflict-aware",
        "teacher_consistency_threshold": 0.5,
        "negative_cap": 6,
        "seed": 2026,
        "selector": "true_s2_bm25_within_document_v2",
        "document_inference_aggregation": "MAX",
        "ranking_policy": {"name": "RETRIEVAL_RRF_NO_LABEL", "dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3, "rrf_k": 2},
        "loss": "consistent: 0.5*BCE + 0.5*MSE; conflict: BCE-only",
        "scientific_change_from_v2": "teacher anchoring behavior only",
        "v2_source_workspace": str(V2.relative_to(ROOT)),
        "v2_source_hashes": source_hashes,
    }
    config_path = V3 / "v3_frozen_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conflict = {
        "schema_version": "same-bge-ft-v3-conflict-motivation-v1",
        "rule": {"positive_consistent": "p_teacher >= 0.5", "negative_consistent": "p_teacher < 0.5", "conflict_loss": "BCE-only"},
        "folds": fold_analysis,
        "pooled": {
            "positive_teacher_consistent": total_positive_consistent,
            "positive_teacher_conflict": total_positive_conflict,
            "negative_teacher_consistent": total_negative_consistent,
            "negative_teacher_conflict": total_negative_conflict,
            "teacher_conflict_positives": total_positive_conflict,
            "teacher_conflict_negatives": total_negative_conflict,
            "negative_class_counts": dict(sorted(sum((Counter(value["negative_class_counts"]) for value in fold_analysis.values()), Counter()).items())),
        },
        "labels_used_for_rule": False,
        "heldout_metrics_used_for_rule": False,
    }
    (V3 / "conflict_analysis.json").write_text(json.dumps(conflict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "status": "PASS",
        "v2_status": "CLOSED",
        "v2_artifacts_untouched": True,
        "train_group_parity": "PASS",
        "worklist_parity": "PASS",
        "v3_config_sha256": sha256(config_path),
        "source_hashes": source_hashes,
        "conflict_analysis": str((V3 / "conflict_analysis.json").relative_to(ROOT)),
        "v2_forensic_report_sha256": sha256(V2 / "fold1/forensic_audit.json"),
        "v2_checkpoint_manifest_sha256": sha256(V2 / "fold1/checkpoint_tensor_diff_manifest.json"),
    }
    (V3 / "v3_preflight.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"workspace": str(V3), "train_group_parity": "PASS", "worklist_parity": "PASS", "v3_config_sha256": freeze["v3_config_sha256"], "teacher_conflict_positives": total_positive_conflict, "teacher_conflict_negatives": total_negative_conflict, "folds": fold_analysis}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
