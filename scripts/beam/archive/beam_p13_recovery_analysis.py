from beam import Sandbox, Image, PythonVersion, Volume
import time

ANALYSIS_CODE = r"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUNTIME = Path("/workspace/p13/runtime")
ROOT = RUNTIME / "artifacts/task1/models/bge_reranker_finetune/full_oof"
OOF = ROOT / "oof_predictions.jsonl"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
NEG = RUNTIME / "artifacts/task1/training/negatives"
REPORT_DIR = ROOT / "recovery_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_TYPES = {
    "semi_hard",
    "lexical_confuser",
    "hard_false_positive",
    "semantic_confuser",
}

def load_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {e}") from e

def macro_metrics(rows, key):
    recalls, precisions = [], []
    full = 0
    multi = []
    for r in rows:
        gold = set(map(str, r["gold_documents"]))
        pred = set(map(str, r[key][:5]))
        matched = gold & pred
        rec = len(matched) / len(gold) if gold else 0.0
        pre = len(matched) / len(pred) if pred else 0.0
        recalls.append(rec)
        precisions.append(pre)
        if len(gold) > 1:
            multi.append(rec)
        if matched == gold:
            full += 1
    return {
        "macro_recall": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": (sum(multi) / len(multi)) if multi else None,
        "full_gold_query_rate": full / len(rows),
        "query_count": len(rows),
    }

def qtile(values, p):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    x = (len(xs)-1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi-x) + xs[hi] * (x-lo)

def summarize_num(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p10": qtile(values, .10),
        "median": qtile(values, .50),
        "p90": qtile(values, .90),
        "p99": qtile(values, .99),
        "max": max(values),
        "mean": sum(values)/len(values),
    }

def evidence_len(v):
    if isinstance(v, str):
        return len(v.strip())
    if isinstance(v, list):
        total = 0
        for x in v:
            if isinstance(x, str):
                total += len(x.strip())
            elif isinstance(x, dict):
                total += len(str(x.get("text", "")).strip())
        return total
    if isinstance(v, dict):
        return len(str(v.get("text", "")).strip())
    return 0

print("=== P13 RECOVERY ANALYSIS ===", flush=True)

success = ROOT / "BEAM_P13_FULL_5FOLD_SUCCESS.txt"
if not success.is_file():
    raise RuntimeError("Full 5-fold success marker is missing")

rows = list(load_jsonl(OOF))
if len(rows) != 7000:
    raise RuntimeError(f"Expected 7000 OOF rows, got {len(rows)}")
ids = [str(r["query_id"]) for r in rows]
if len(set(ids)) != 7000:
    raise RuntimeError("OOF query IDs are not unique")

train = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
gold_map = {
    str(qid): set(map(str, item.get("answer", [])))
    for qid, item in train.items()
    if isinstance(item, dict)
}

baseline = macro_metrics(rows, "baseline_top5")
tuned = macro_metrics(rows, "fine_tuned_top5")

# ------------------------------------------------------------
# Candidate ceiling / rank distribution
# ------------------------------------------------------------
ks = [1, 5, 10, 20, 50, 100, 200]
candidate_recall_sums = {k: 0.0 for k in ks}
candidate_full = {k: 0 for k in ks}
gold_rank_buckets = Counter()
gold_rank_values = []
missing_gold_count = 0
total_gold = 0
candidate_duplicate_queries = 0

baseline_harmed = 0
baseline_rescued = 0
same = 0
harm_amount = 0.0
rescue_amount = 0.0
harm_samples = []
rescue_samples = []

group_rows = {
    "single_gold": [],
    "multi_gold": [],
}
fold_rows = defaultdict(list)

for r in rows:
    gold = list(map(str, r["gold_documents"]))
    gold_set = set(gold)
    candidates = list(map(str, r.get("candidate_documents", [])))
    if len(candidates) != len(set(candidates)):
        candidate_duplicate_queries += 1

    for k in ks:
        matched = gold_set & set(candidates[:k])
        candidate_recall_sums[k] += len(matched) / len(gold_set)
        if matched == gold_set:
            candidate_full[k] += 1

    pos = {d: i+1 for i, d in enumerate(candidates)}
    for g in gold:
        total_gold += 1
        rank = pos.get(g)
        if rank is None:
            missing_gold_count += 1
            gold_rank_buckets["missing"] += 1
        else:
            gold_rank_values.append(rank)
            if rank <= 5:
                gold_rank_buckets["1-5"] += 1
            elif rank <= 10:
                gold_rank_buckets["6-10"] += 1
            elif rank <= 20:
                gold_rank_buckets["11-20"] += 1
            elif rank <= 50:
                gold_rank_buckets["21-50"] += 1
            elif rank <= 100:
                gold_rank_buckets["51-100"] += 1
            else:
                gold_rank_buckets["101-200"] += 1

    bpred = set(map(str, r["baseline_top5"][:5]))
    tpred = set(map(str, r["fine_tuned_top5"][:5]))
    br = len(gold_set & bpred) / len(gold_set)
    tr = len(gold_set & tpred) / len(gold_set)
    delta = tr - br
    if delta < 0:
        baseline_harmed += 1
        harm_amount += -delta
        if len(harm_samples) < 100:
            harm_samples.append({
                "query_id": r["query_id"],
                "fold": r["fold"],
                "gold": gold,
                "baseline_top5": r["baseline_top5"],
                "fine_tuned_top5": r["fine_tuned_top5"],
                "candidate_gold_ranks": {g: pos.get(g) for g in gold},
                "recall_delta": delta,
            })
    elif delta > 0:
        baseline_rescued += 1
        rescue_amount += delta
        if len(rescue_samples) < 100:
            rescue_samples.append({
                "query_id": r["query_id"],
                "fold": r["fold"],
                "gold": gold,
                "baseline_top5": r["baseline_top5"],
                "fine_tuned_top5": r["fine_tuned_top5"],
                "candidate_gold_ranks": {g: pos.get(g) for g in gold},
                "recall_delta": delta,
            })
    else:
        same += 1

    group_rows["multi_gold" if len(gold) > 1 else "single_gold"].append(r)
    fold_rows[int(r["fold"])].append(r)

candidate = {
    f"recall_at_{k}": candidate_recall_sums[k] / len(rows)
    for k in ks
}
candidate.update({
    f"full_gold_query_rate_at_{k}": candidate_full[k] / len(rows)
    for k in ks
})

# ------------------------------------------------------------
# Training-record audit: this is where we look for a smoking gun.
# ------------------------------------------------------------
neg_type_counts = Counter()
selected_type_counts = Counter()
fold_record_counts = Counter()
selected_pairs_by_query = Counter()
all_pairs_by_query = Counter()
positive_not_gold = 0
negative_is_gold = 0
positive_eq_negative = 0
unknown_query = 0
selected_records = 0
total_records = 0
positive_evidence_lens = []
negative_evidence_lens = []
provenance_key_counts = Counter()
provenance_law_name_count = 0
false_negative_samples = []
bad_positive_samples = []

for fold in range(5):
    path = NEG / f"fold_{fold}.jsonl"
    for rec in load_jsonl(path):
        total_records += 1
        fold_record_counts[fold] += 1
        qid = str(rec.get("query_id", "")).strip()
        pos = str(rec.get("positive_doc", "")).strip()
        neg = str(rec.get("negative_doc", "")).strip()
        kind = str(rec.get("negative_type", "")).strip()
        neg_type_counts[kind] += 1
        all_pairs_by_query[qid] += 1

        prov = rec.get("provenance")
        if isinstance(prov, dict):
            for k in prov:
                provenance_key_counts[str(k)] += 1
            if str(prov.get("law_name", "")).strip():
                provenance_law_name_count += 1

        positive_evidence_lens.append(evidence_len(rec.get("positive_evidence")))
        negative_evidence_lens.append(evidence_len(rec.get("negative_evidence")))

        gold = gold_map.get(qid)
        if gold is None:
            unknown_query += 1
        else:
            if pos not in gold:
                positive_not_gold += 1
                if len(bad_positive_samples) < 50:
                    bad_positive_samples.append({
                        "fold": fold, "query_id": qid,
                        "positive_doc": pos, "gold": sorted(gold), "negative_type": kind,
                    })
            if neg in gold:
                negative_is_gold += 1
                if len(false_negative_samples) < 50:
                    false_negative_samples.append({
                        "fold": fold, "query_id": qid,
                        "negative_doc": neg, "gold": sorted(gold), "negative_type": kind,
                    })
        if pos and neg and pos == neg:
            positive_eq_negative += 1

        if kind in SELECTED_TYPES:
            selected_records += 1
            selected_type_counts[kind] += 1
            selected_pairs_by_query[qid] += 1

selected_counts = list(selected_pairs_by_query.values())
all_counts = list(all_pairs_by_query.values())

training_audit = {
    "total_records": total_records,
    "selected_records_for_semi_hard_plus_hard": selected_records,
    "negative_type_counts": dict(neg_type_counts),
    "selected_type_counts": dict(selected_type_counts),
    "fold_record_counts": dict(fold_record_counts),
    "unknown_query_records": unknown_query,
    "positive_not_gold_records": positive_not_gold,
    "negative_is_gold_records_FALSE_NEGATIVES": negative_is_gold,
    "positive_equals_negative_records": positive_eq_negative,
    "false_negative_rate_selected_denominator_approx": (
        negative_is_gold / total_records if total_records else None
    ),
    "selected_pairs_per_query": summarize_num(selected_counts),
    "all_pairs_per_query": summarize_num(all_counts),
    "positive_evidence_char_len": summarize_num(positive_evidence_lens),
    "negative_evidence_char_len": summarize_num(negative_evidence_lens),
    "provenance_keys": dict(provenance_key_counts),
    "provenance_law_name_records": provenance_law_name_count,
    "formatting_asymmetry_warning": (
        "Trainer code formats positive_document WITHOUT provenance law_name, "
        "but formats negative_document WITH provenance law_name. This creates a "
        "potential label-correlated formatting artifact and train/inference mismatch."
    ),
}

# ------------------------------------------------------------
# Actionable diagnosis
# ------------------------------------------------------------
ceiling200 = candidate["recall_at_200"]
ceiling20 = candidate["recall_at_20"]
gap_to_ceiling = ceiling200 - baseline["macro_recall"]

flags = []
if positive_not_gold:
    flags.append("CRITICAL_POSITIVE_LABEL_ERRORS")
if negative_is_gold:
    flags.append("CRITICAL_FALSE_NEGATIVES")
flags.append("POSITIVE_NEGATIVE_FORMATTING_ASYMMETRY_IN_TRAINER")
if selected_counts and max(selected_counts) > 3 * max(1, statistics.median(selected_counts)):
    flags.append("QUERY_WEIGHT_SKEW_FROM_REPEATED_PAIRS")
if gap_to_ceiling > 0.05:
    flags.append("LARGE_RERANKING_HEADROOM")
if ceiling200 < 0.95:
    flags.append("RETRIEVAL_CEILING_BELOW_0_95_OFFLINE")

next_move = []
if positive_not_gold or negative_is_gold:
    next_move.append(
        "Do NOT train again until mislabeled positive/negative records are fixed."
    )
next_move.append(
    "Fix training representation symmetry before any new GPU run: positive and "
    "negative documents must be formatted by the same code path with equivalent metadata."
)
next_move.append(
    "Run a cheap single-fold rescue sweep (small max_training_pairs, 1 epoch) "
    "before any new 5-fold training."
)
if gap_to_ceiling > 0.05:
    next_move.append(
        "Prioritize reranking/post-processing experiments: candidate@200 contains "
        "substantial oracle headroom over pretrained BGE."
    )
if ceiling200 < 0.95:
    next_move.append(
        "Also improve retrieval/candidate generation because reranking alone cannot "
        "cross the measured candidate ceiling."
    )

report = {
    "status": "PASS",
    "oof_rows": len(rows),
    "baseline": baseline,
    "fine_tuned": tuned,
    "fine_tune_damage": {
        "recall_delta": tuned["macro_recall"] - baseline["macro_recall"],
        "precision_delta": tuned["macro_precision"] - baseline["macro_precision"],
        "queries_harmed": baseline_harmed,
        "queries_rescued": baseline_rescued,
        "queries_unchanged": same,
        "mean_harm_on_harmed_queries": (
            harm_amount / baseline_harmed if baseline_harmed else 0.0
        ),
        "mean_rescue_on_rescued_queries": (
            rescue_amount / baseline_rescued if baseline_rescued else 0.0
        ),
    },
    "candidate_ceiling": candidate,
    "candidate_gap_to_baseline_at_200": gap_to_ceiling,
    "gold_candidate_rank_buckets": dict(gold_rank_buckets),
    "gold_candidate_rank_summary_present_only": summarize_num(gold_rank_values),
    "total_gold_labels": total_gold,
    "missing_gold_from_candidate200": missing_gold_count,
    "candidate_duplicate_queries": candidate_duplicate_queries,
    "by_group": {
        name: {
            "baseline": macro_metrics(group, "baseline_top5"),
            "fine_tuned": macro_metrics(group, "fine_tuned_top5"),
        }
        for name, group in group_rows.items()
    },
    "by_fold": {
        str(fold): {
            "baseline": macro_metrics(group, "baseline_top5"),
            "fine_tuned": macro_metrics(group, "fine_tuned_top5"),
        }
        for fold, group in sorted(fold_rows.items())
    },
    "training_audit": training_audit,
    "diagnostic_flags": flags,
    "recommended_next_move": next_move,
}

(REPORT_DIR / "recovery_analysis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

with (REPORT_DIR / "fine_tune_harmed_samples.jsonl").open("w", encoding="utf-8") as f:
    for x in harm_samples:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with (REPORT_DIR / "fine_tune_rescued_samples.jsonl").open("w", encoding="utf-8") as f:
    for x in rescue_samples:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with (REPORT_DIR / "false_negative_samples.jsonl").open("w", encoding="utf-8") as f:
    for x in false_negative_samples:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with (REPORT_DIR / "bad_positive_samples.jsonl").open("w", encoding="utf-8") as f:
    for x in bad_positive_samples:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

md = []
md.append("# P13 Recovery Analysis")
md.append("")
md.append(f"- OOF rows: **{len(rows)}**")
md.append(f"- Baseline Recall@5: **{baseline['macro_recall']:.6f}**")
md.append(f"- Fine-tuned Recall@5: **{tuned['macro_recall']:.6f}**")
md.append(f"- Delta: **{tuned['macro_recall']-baseline['macro_recall']:+.6f}**")
md.append(f"- Candidate Recall@200 ceiling: **{ceiling200:.6f}**")
md.append(f"- Candidate Recall@20: **{ceiling20:.6f}**")
md.append(f"- Candidate@200 gap over baseline: **{gap_to_ceiling:+.6f}**")
md.append("")
md.append("## Fine-tune damage")
md.append(f"- Queries harmed: **{baseline_harmed}**")
md.append(f"- Queries rescued: **{baseline_rescued}**")
md.append(f"- Queries unchanged: **{same}**")
md.append("")
md.append("## Training audit")
md.append(f"- Total negative records: **{total_records}**")
md.append(f"- Selected semi-hard+hard records: **{selected_records}**")
md.append(f"- Positive doc NOT gold: **{positive_not_gold}**")
md.append(f"- Negative doc IS gold (false negative): **{negative_is_gold}**")
md.append(f"- Positive == negative: **{positive_eq_negative}**")
md.append(f"- Provenance law_name present: **{provenance_law_name_count}**")
md.append("")
md.append("### Important code-level warning")
md.append(
    "- The current trainer formats positive documents without `provenance.law_name`, "
    "while negative documents receive `provenance.law_name`. This can create a "
    "label-correlated formatting shortcut and a severe train/inference mismatch."
)
md.append("")
md.append("## Diagnostic flags")
for x in flags:
    md.append(f"- **{x}**")
md.append("")
md.append("## Recommended next move")
for x in next_move:
    md.append(f"- {x}")

(REPORT_DIR / "recovery_analysis.md").write_text(
    "\n".join(md) + "\n", encoding="utf-8"
)

print("")
print("============================================================")
print("RECOVERY SUMMARY")
print("============================================================")
print(f"BASELINE_RECALL={baseline['macro_recall']:.6f}")
print(f"TUNED_RECALL={tuned['macro_recall']:.6f}")
print(f"TUNED_DELTA={tuned['macro_recall']-baseline['macro_recall']:+.6f}")
print(f"CANDIDATE_RECALL_AT_20={ceiling20:.6f}")
print(f"CANDIDATE_RECALL_AT_200={ceiling200:.6f}")
print(f"CANDIDATE_GAP_OVER_BASELINE={gap_to_ceiling:+.6f}")
print(f"QUERIES_HARMED={baseline_harmed}")
print(f"QUERIES_RESCUED={baseline_rescued}")
print(f"POSITIVE_NOT_GOLD={positive_not_gold}")
print(f"NEGATIVE_IS_GOLD_FALSE_NEGATIVE={negative_is_gold}")
print(f"SELECTED_PAIR_QUERY_MEDIAN={qtile(selected_counts,.50) if selected_counts else None}")
print(f"SELECTED_PAIR_QUERY_P99={qtile(selected_counts,.99) if selected_counts else None}")
print("FLAGS=" + ",".join(flags))
print(f"REPORT_JSON={REPORT_DIR / 'recovery_analysis.json'}")
print(f"REPORT_MD={REPORT_DIR / 'recovery_analysis.md'}")
print("============================================================")
"""

sb = None
try:
    sb = Sandbox(
        name="udsc-p13-recovery-analysis",
        cpu=4,
        memory="16Gi",
        image=Image(python_version=PythonVersion.Python311),
        volumes=[
            Volume(name="udsc-p13", mount_path="/workspace/p13"),
        ],
    ).create()

    print("Sandbox created; waiting for readiness...", flush=True)
    time.sleep(5)

    p = sb.process.exec("python", "-c", ANALYSIS_CODE)
    p.wait()

    for line in p.logs:
        print(line, end="")

finally:
    if sb is not None:
        try:
            sb.terminate()
            print("\nSandbox terminated.", flush=True)
        except Exception as e:
            print("\nWARNING: sandbox termination:", repr(e), flush=True)
