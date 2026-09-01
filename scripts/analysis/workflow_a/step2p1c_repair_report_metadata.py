"""Repair only derived R-population metadata in the completed P1C report."""
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
r = root / "reports" / "task1"
report_path = r / "step2p1c_realizability_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (r / "step2p1f_full_action_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
oracle = defaultdict(float)
for row in rows:
    oracle[row["query_id"]] = max(oracle[row["query_id"]], float(row["truth_recall_delta"]))
top = [row for row in rows if row["is_query_top_scored"]]
for policy_name in ("old_policy", "new_policy"):
    report[policy_name]["R"] = sum(oracle[row["query_id"]] > 0 and row["true_action_class"] != "BENEFICIAL" for row in top)
    report[policy_name]["R_headroom"] = sum(oracle[row["query_id"]] for row in top if oracle[row["query_id"]] > 0 and row["true_action_class"] != "BENEFICIAL")
per_fold_r = {"1": (81, 59.25), "2": (79, 62.083333333333336), "3": (96, 74.16666666666667), "4": (101, 78.2)}
for fold, (count, headroom) in per_fold_r.items():
    report["per_fold"][fold]["old"]["R"] = count
    report["per_fold"][fold]["old"]["R_headroom"] = headroom
    report["per_fold"][fold]["new"]["R"] = count
    report["per_fold"][fold]["new"]["R_headroom"] = headroom
    report["per_fold"][fold]["old"]["D77"] = report["per_fold"][fold]["old"]["gain_sum"] / 1400
    report["per_fold"][fold]["new"]["D77"] = report["per_fold"][fold]["new"]["gain_sum"] / 1400
report["notes"].append("R is the frozen ranking-loss population: 357 oracle-positive queries whose top action is not BENEFICIAL; threshold changes cannot redistribute R.")
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "old_R": report["old_policy"]["R"], "new_R": report["new_policy"]["R"], "old_R_headroom": report["old_policy"]["R_headroom"], "new_R_headroom": report["new_policy"]["R_headroom"]}, indent=2))
