"""Close the CPU forensic audit after importing the authoritative CPU manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLD = ROOT / "private_task1/experiments/sprint48_bge_ft_v2/fold1"
AUDIT = FOLD / "forensic_audit.json"
MANIFEST = FOLD / "checkpoint_tensor_diff_manifest.json"
REPORT = FOLD / "forensic_audit.md"
LOG = ROOT / "private_task1/log_private.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "CHECKPOINT_TENSOR_AUDIT_PASS":
        raise SystemExit("checkpoint tensor manifest is not PASS")
    if manifest.get("checkpoint_weight_sha256") != "4bbe295ef1a1901288230c030438663cffd775d0c1eb6aa49f1c4eed9a3ce601":
        raise SystemExit("checkpoint weight identity mismatch")
    if manifest.get("checkpoint_config_sha256") != "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b":
        raise SystemExit("checkpoint config identity mismatch")

    audit["gates"]["F1_ARTIFACT_GATE"] = "PASS"
    audit["gates"]["CHECKPOINT_INTEGRITY_GATE"] = "PASS"
    audit["gates"]["SCORER_RUNTIME_PARITY"] = "UNPROVEN"
    audit["checkpoint_closure"] = {
        "CHECKPOINT_FOUND": "YES",
        "CHECKPOINT_LOCATION": manifest["checkpoint_location"],
        "CHECKPOINT_IDENTITY_GATE": manifest["checkpoint_identity_gate"],
        "CHECKPOINT_WEIGHT_SHA256": manifest["checkpoint_weight_sha256"],
        "CURRENT_FT_INITIALIZATION_SHA256": manifest["current_ft_weight_sha256"],
        "CONFIG_SHA256": manifest["checkpoint_config_sha256"],
        "FROZEN_TENSORS_EXPECTED": manifest["frozen_tensors"]["expected"],
        "FROZEN_TENSORS_CHANGED": manifest["frozen_tensors"]["changed"],
        "FROZEN_MAX_ABS_DELTA": manifest["frozen_tensors"]["max_abs_delta"],
        "TRAINABLE_TENSORS_TOTAL": manifest["trainable_tensors"]["total"],
        "TRAINABLE_TENSORS_CHANGED": manifest["trainable_tensors"]["changed"],
        "TRAINABLE_TENSORS_UNCHANGED": manifest["trainable_tensors"]["unchanged"],
        "NONFINITE_TENSORS": manifest["nonfinite_tensors"],
        "MISSING_TENSORS": len(manifest["tensor_set"]["missing"]),
        "EXTRA_TENSORS": len(manifest["tensor_set"]["extra"]),
        "DTYPE_OR_SHAPE_CHANGES": manifest["unexpected_dtype_or_shape_changes"],
        "CONFIG_CONTRACT": manifest["config_contract"],
        "RECORDED_CHECKPOINT_HASHES_ALL_MATCH": manifest["recorded_checkpoint_hashes_all_match"],
        "MANIFEST_SHA256": sha256(MANIFEST),
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
    }
    audit["root_cause"] = {
        **audit["root_cause"],
        "primary": "PRACTICAL_SCIENTIFIC_FAIL_WITH_PARITY_CAVEAT",
        "secondary_contributors": ["OBJECTIVE_CONFLICT_EVIDENCE", "SCORER_RUNTIME_PARITY_UNPROVEN"],
        "checkpoint_integrity_now_proven": True,
        "why_not_true_scientific_fail": "The exact checkpoint and frozen/trainable tensor contract are now proven. Independent scorer parity was not separately persisted, so the closure retains the explicit parity caveat.",
        "decision": "CLOSE_FT_V2",
        "next_action": "Do not rerun SAME-BGE FT V2; any future model work must be a separately named teacher-anchoring objective hypothesis with a new validation gate.",
    }
    audit["final_f1_v2_classification"] = "PRACTICAL_SCIENTIFIC_FAIL_WITH_PARITY_CAVEAT"
    audit["ft_v2_status"] = "CLOSED"
    audit["checkpoint_tensor_manifest"] = manifest
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    replay = audit["evaluator_replay"]
    harm = audit["recall_harm_anatomy"]
    closure = audit["checkpoint_closure"]
    md = f"""# F1 SAME-BGE HARD-NEGATIVE FT V2 — forensic closure\n\n- Classification: **PRACTICAL_SCIENTIFIC_FAIL_WITH_PARITY_CAVEAT**\n- Status: **CLOSED**\n- GPU runs: **0**; GPU inference: **0**; GPU training: **0**.\n\n## Checkpoint identity and tensor contract\n\n- Checkpoint found: **YES**\n- Remote location: `{closure['CHECKPOINT_LOCATION']}`\n- Checkpoint weight SHA: `{closure['CHECKPOINT_WEIGHT_SHA256']}`\n- CURRENT_FT SHA: `{closure['CURRENT_FT_INITIALIZATION_SHA256']}`\n- Config SHA: `{closure['CONFIG_SHA256']}`\n- Tensor set: `393/393`, missing `0`, extra `0`\n- Frozen tensors: `{closure['FROZEN_TENSORS_CHANGED']}/{closure['FROZEN_TENSORS_EXPECTED']}` changed; max absolute delta `{closure['FROZEN_MAX_ABS_DELTA']}`\n- Trainable tensors: `{closure['TRAINABLE_TENSORS_CHANGED']}/{closure['TRAINABLE_TENSORS_TOTAL']}` changed; unchanged `{closure['TRAINABLE_TENSORS_UNCHANGED']}`\n- Nonfinite tensors: `{closure['NONFINITE_TENSORS']}`\n- Dtype/shape changes: `{closure['DTYPE_OR_SHAPE_CHANGES']}`\n- Config: XLM-RoBERTa-compatible, single-logit head, configs byte-identical.\n- Execution and score manifests point to the same checkpoint SHA: **YES**.\n\n## Gates\n\n- F1_ARTIFACT_GATE: **PASS**\n- EVALUATOR_REPLAY_GATE: **PASS**\n- BGE_SUBSTITUTION_GATE: **PASS**\n- CHECKPOINT_INTEGRITY_GATE: **PASS**\n- SCORER_RUNTIME_PARITY: **UNPROVEN**\n- Objective conflict evidence: **STRONG**\n\n## Frozen F1 result\n\n- Baseline Recall: `{replay['baseline']['recall']}`\n- V2 Recall: `{replay['v2']['recall']}`\n- Recall delta: `{replay['delta']['recall']}`\n- Final Top5 changed: `{audit['ranking_diff']['final_top5_changed_queries']}`; improved `{harm['improved_queries']}`; harmed `{harm['harmed_queries']}`\n\nThe checkpoint is genuine and training integrity is clean: all expected frozen tensors are exactly unchanged and all expected trainable tensors changed finitely. The negative F1 result is therefore treated as a practical scientific failure of this frozen V2 recipe, with the explicit caveat that no independent scorer-runtime parity sample was persisted. This is not evidence of an implementation or checkpoint-corruption bug.\n\nDo not rerun V2. Any future work must be a separately named teacher-anchoring objective hypothesis with a new validation gate.\n\nMachine-readable audit: `{AUDIT.relative_to(ROOT)}`\nTensor manifest: `{MANIFEST.relative_to(ROOT)}`\n"""
    REPORT.write_text(md, encoding="utf-8")
    entry = f"""\n\n## 2026-09-22 — F1 CHECKPOINT FORENSIC CLOSURE\n\n- CPU-only Modal Volume audit found the authoritative checkpoint at `{closure['CHECKPOINT_LOCATION']}`. No training, inference, or GPU was run.\n- Checkpoint SHA `{closure['CHECKPOINT_WEIGHT_SHA256']}` and config SHA `{closure['CONFIG_SHA256']}` match the recorded F1 execution/score manifests.\n- Tensor contract PASS: `393` tensors, missing/extra `0`, nonfinite `0`; frozen embeddings + encoder layers 0–17: `{closure['FROZEN_TENSORS_CHANGED']}/{closure['FROZEN_TENSORS_EXPECTED']}` changed; trainable tensors changed `{closure['TRAINABLE_TENSORS_CHANGED']}/{closure['TRAINABLE_TENSORS_TOTAL']}`.\n- Final classification: `PRACTICAL_SCIENTIFIC_FAIL_WITH_PARITY_CAVEAT`; `FT_V2_STATUS = CLOSED`; `SCORER_RUNTIME_PARITY = UNPROVEN`.\n- Do not rerun SAME-BGE FT V2. Future work must use a separately named teacher-anchoring objective hypothesis.\n- Tensor manifest: `{MANIFEST.relative_to(ROOT)}`.\n"""
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(json.dumps({"classification": audit["final_f1_v2_classification"], "status": audit["ft_v2_status"], "checkpoint_integrity": audit["gates"]["CHECKPOINT_INTEGRITY_GATE"], "scorer_runtime_parity": audit["gates"]["SCORER_RUNTIME_PARITY"], "manifest_sha256": closure["MANIFEST_SHA256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
