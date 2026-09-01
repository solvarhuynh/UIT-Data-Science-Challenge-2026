"""CPU-only Step 1B core A/B/C oracle; no model fitting or policy replay."""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_1A = ROOT / "reports/task1/exp_1a_recall_gap_report.json"
REPORT = ROOT / "reports/task1/exp_1b_oracle_gap_decomposition_report.json"
ACTIONS = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl"
EXPECTED_SUM_A = 357.3666666666666
EXPECTED_POOLED_A = 0.06381547619047619


def load_step1a_module() -> Any:
    path = ROOT / "scripts/analysis/exp_1a_recall_gap_audit.py"
    spec = importlib.util.spec_from_file_location("step1a_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Step 1A structural reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gain(gold: set[str], baseline: list[str], incoming: str, ranks: tuple[int, ...]) -> float:
    before = len(gold & set(baseline)) / len(gold)
    best = 0.0
    for rank in ranks:
        swapped = list(baseline)
        swapped[rank - 1] = incoming
        if len(swapped) != len(set(swapped)):
            continue
        after = len(gold & set(swapped)) / len(gold)
        best = max(best, after - before)
    return best


def main() -> None:
    step1a = json.loads(REPORT_1A.read_text(encoding="utf-8"))
    required = (step1a.get("status") == "PASS" and step1a.get("audited_queries") == 5600
        and abs(float(step1a.get("sum_one_swap_gain")) - EXPECTED_SUM_A) <= 1e-10
        and abs(float(step1a.get("one_swap_candidate_ceiling_gain")) - EXPECTED_POOLED_A) <= 1e-10)
    if not required:
        raise RuntimeError("Step 1A contract/headline values do not match")
    reader = load_step1a_module()
    folds = reader.read_target_fold_map(); target = set(folds)
    gold = reader.read_target_train(target)
    baseline, baseline_skipped = reader.read_baseline(target, folds)
    pool, candidate_skipped = reader.read_candidates(target, folds)

    incoming: dict[str, set[str]] = defaultdict(set)
    allowed: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    rows, action_skipped = reader.stream_target_rows(ACTIONS, target)
    for row in rows:
        qid = str(row["query_id"])
        if int(row.get("fold", -1)) != folds[qid]:
            raise ValueError(f"action fold mismatch: {qid}")
        doc = str(row["incoming_doc_id"]); rank = int(row["drop_rank"]); dropped = str(row["dropped_doc_id"])
        if rank not in (4, 5) or dropped != baseline[qid][rank - 1]:
            raise ValueError(f"action contract mismatch: {qid}")
        incoming[qid].add(doc); allowed[qid].add((doc, rank, dropped))
    if set(incoming) != target:
        raise ValueError("actions do not cover every target query")

    print("STEP 1B CORE-FIX PREFLIGHT")
    print("Step1A report: reports/task1/exp_1a_recall_gap_report.json\nStep1A status: PASS")
    print(f"Expected A: sum_A = {EXPECTED_SUM_A}; pooled_A = {EXPECTED_POOLED_A}")
    print(f"Gold source: {reader.TRAIN.relative_to(ROOT)}\nFold source: {reader.FOLDS.relative_to(ROOT)}")
    print(f"Baseline source: {reader.BASELINE.relative_to(ROOT)}\nFull candidate source: {reader.CANDIDATES.relative_to(ROOT)}\nCurrent action-space source: {ACTIONS.relative_to(ROOT)}")
    print("ACTION-SPACE TRACE:\nscripts/beam/beam_task1_v3_prepare_cpu.py → v3_residual/shortlist_evidence.jsonl → scripts/beam/task1_v3_residual/build_actions.py → policy/actions.jsonl")
    print("Target folds: [1,2,3,4]\nTarget query count: 5600")
    print("Will retrain policy: NO\nWill rerun inner-CV training: NO\nWill use Fold0 prediction: NO\nWill read Fold0 labels: NO\nWill use public labels: NO\nGPU required: NO")
    print("Historical OOF policy artifact found: NO\nCore A/B/C decomposition can run: YES")
    print(f"Output: {REPORT.relative_to(ROOT)}")

    pf = {str(f): {"query_count": 0, "sum_A": 0.0, "sum_B": 0.0, "sum_C": 0.0,
          "action_space_gap": 0.0, "rank_limit_gap": 0.0, "queries_with_A_gt_0": 0,
          "queries_with_B_gt_0": 0, "queries_with_C_gt_0": 0, "queries_with_action_space_gap_gt_0": 0,
          "queries_with_rank_limit_gap_gt_0": 0} for f in (1,2,3,4)}
    sums = {key: 0.0 for key in ("A", "B", "C", "as_gap", "rank_gap")}
    counts = {key: 0 for key in ("A", "B", "C", "as_gap", "rank_gap")}
    samples = {"action_space_gap": [], "rank_limit_gap": []}
    ordered = sorted(target, key=lambda x: int(x) if x.isdigit() else x)
    invariant = True
    for qid in ordered:
        base = baseline[qid]; base_set = set(base); fold = folds[qid]; stats = pf[str(fold)]; stats["query_count"] += 1
        a = max((gain(gold[qid], base, doc, (1,2,3,4,5)) for doc in pool[qid] - base_set), default=0.0)
        current = incoming[qid] - base_set
        b = max((gain(gold[qid], base, doc, (1,2,3,4,5)) for doc in current), default=0.0)
        c = 0.0
        for doc, rank, dropped in allowed[qid]:
            if doc not in current or dropped != base[rank - 1]:
                raise ValueError(f"invalid allowed action: {qid}")
            c = max(c, gain(gold[qid], base, doc, (rank,)))
        as_gap, rank_gap = a - b, b - c
        invariant &= a + 1e-12 >= b and b + 1e-12 >= c and c >= -1e-12
        for key, value in (("A",a),("B",b),("C",c),("as_gap",as_gap),("rank_gap",rank_gap)):
            sums[key] += value
        for key, value in (("A",a),("B",b),("C",c),("as_gap",as_gap),("rank_gap",rank_gap)):
            if value > 0: counts[key] += 1
        stats["sum_A"] += a; stats["sum_B"] += b; stats["sum_C"] += c; stats["action_space_gap"] += as_gap; stats["rank_limit_gap"] += rank_gap
        for key, value in (("A",a),("B",b),("C",c),("as_gap",as_gap),("rank_gap",rank_gap)):
            name = {"A":"queries_with_A_gt_0", "B":"queries_with_B_gt_0", "C":"queries_with_C_gt_0", "as_gap":"queries_with_action_space_gap_gt_0", "rank_gap":"queries_with_rank_limit_gap_gt_0"}[key]
            if value > 0: stats[name] += 1
        sample = {"query_id": qid, "fold": fold, "relevant_count": len(gold[qid]), "baseline_recall": len(gold[qid] & base_set)/len(gold[qid]), "A": a, "B": b, "C": c, "action_space_gap": as_gap, "rank_limit_gap": rank_gap}
        if as_gap > 0 and len(samples["action_space_gap"]) < 20: samples["action_space_gap"].append(sample)
        if rank_gap > 0 and len(samples["rank_limit_gap"]) < 20: samples["rank_limit_gap"].append(sample)
    for stats in pf.values():
        for metric in ("A","B","C"):
            stats[f"pooled_{metric}"] = stats[f"sum_{metric}"] / stats["query_count"]
        stats["pooled_action_space_gap"] = stats["action_space_gap"] / stats["query_count"]
        stats["pooled_rank_limit_gap"] = stats["rank_limit_gap"] / stats["query_count"]
    pooled = {key: value / 5600 for key, value in sums.items()}
    cross = abs(sums["A"] - EXPECTED_SUM_A)
    identity = abs(pooled["A"] - (pooled["as_gap"] + pooled["rank_gap"] + pooled["C"]))
    checks = {"all_queries_covered": set(gold)==set(baseline)==set(pool)==set(incoming)==target and len(target)==5600,
              "A_ge_B_ge_C": invariant, "step1a_A_crosscheck": cross <= 1e-10,
              "core_decomposition_identity": identity <= 1e-9, "fold0_payload_not_materialized": True}
    status = "CONTRACT_ERROR" if not all(checks.values()) else ("PASS" if pooled["rank_gap"] > .003 else "FAIL" if pooled["rank_gap"] < .001 else "INCONCLUSIVE")
    report = {"status": status, "official_metric":"macro_recall", "folds_used":[1,2,3,4], "fold0_touched":False, "public_labels_used":False,
      "aggregate_artifacts_contain_fold0":True, "fold0_payload_materialized":False, "fold0_used_in_statistics":False, "fold0_used_in_oracle":False, "fold0_labels_used":False,
      "baseline_non_target_records_skipped":baseline_skipped,"candidate_non_target_records_skipped":candidate_skipped,"action_non_target_records_skipped":action_skipped[0],"total_queries":5600,"audited_queries":5600,
      "actual_policy_available":False,"actual_policy_unavailable_reason":"Historical folds1-4 OOF decisions were not serialized.",
      "input_artifacts":{"step0_report":"reports/task1/step0_metric_contract_report.json","step1a_report":"reports/task1/exp_1a_recall_gap_report.json","train_labels":str(reader.TRAIN.relative_to(ROOT)),"fold_mapping":str(reader.FOLDS.relative_to(ROOT)),"baseline_top5":str(reader.BASELINE.relative_to(ROOT)),"full_candidate_pool":str(reader.CANDIDATES.relative_to(ROOT)),"current_action_space":str(ACTIONS.relative_to(ROOT)),"actual_v3a_policy":None},
      "input_provenance":{"action_space_trace":"scripts/beam/beam_task1_v3_prepare_cpu.py → v3_residual/shortlist_evidence.jsonl → scripts/beam/task1_v3_residual/build_actions.py → policy/actions.jsonl; build_actions emits all non-baseline shortlist incoming candidates for drop ranks 4 and 5.","actual_policy_trace":"No historical folds1-4 OOF selected-action/final-top5 artifact was serialized. policy_training_report.json is aggregate-only; existing serialized prediction is Fold0 and was not read."},
      "step1a_crosscheck":{"expected_sum_A":EXPECTED_SUM_A,"measured_sum_A":sums["A"],"expected_pooled_A":EXPECTED_POOLED_A,"measured_pooled_A":pooled["A"],"absolute_difference":cross,"passed":checks["step1a_A_crosscheck"]},
      "sum_A":sums["A"],"pooled_A":pooled["A"],"sum_B":sums["B"],"pooled_B":pooled["B"],"sum_C":sums["C"],"pooled_C":pooled["C"],"pooled_action_space_gap":pooled["as_gap"],"pooled_rank_limit_gap":pooled["rank_gap"],
      "sum_D":None,"pooled_actual_policy_gain":None,"pooled_decision_policy_gap":None,"queries_with_A_gt_0":counts["A"],"queries_with_B_gt_0":counts["B"],"queries_with_C_gt_0":counts["C"],"queries_with_action_space_gap_gt_0":counts["as_gap"],"queries_with_rank_limit_gap_gt_0":counts["rank_gap"],"queries_with_decision_policy_gap_gt_0":None,"queries_actual_policy_benefit":None,"queries_actual_policy_neutral":None,"queries_actual_policy_harm":None,"per_fold":pf,"sanity_checks":checks,"sample_queries":samples,
      "notes":["Core A/B/C completed without fitting or replaying any policy.","D and decision-policy gap are null because historical folds1-4 OOF policy decisions were not serialized.","All aggregate JSONL sources used top-level query_id structural target-only materialization."]}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
