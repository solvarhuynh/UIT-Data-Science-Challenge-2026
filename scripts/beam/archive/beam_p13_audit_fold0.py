from __future__ import annotations

from beam import function, Image, Volume

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
NEGATIVES = RUNTIME / "artifacts/task1/training/negatives/fold_0.jsonl"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
FOLDS = RUNTIME / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
OUT_DIR = RUNTIME / "artifacts/task1/diagnostics/p13_fold0_negative_audit"

image = Image(python_version="python3.11")


def _evidence_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
            elif isinstance(item, dict):
                s = str(item.get("text", "")).strip()
            else:
                s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, dict):
        s = str(value.get("text", "")).strip()
        return [s] if s else []
    return []


def _hash_texts(texts: list[str]) -> str | None:
    if not texts:
        return None
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _load_gold() -> dict[str, set[str]]:
    raw = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    out: dict[str, set[str]] = {}
    for qid, row in raw.items():
        if not isinstance(row, dict):
            continue
        answers = row.get("answer")
        if isinstance(answers, list):
            out[str(qid)] = {str(x).strip() for x in answers if str(x).strip()}
    return out


def _load_fold_map() -> dict[str, int]:
    raw = json.loads(FOLDS.read_text(encoding="utf-8-sig"))
    out: dict[str, int] = {}
    for item in raw.get("folds", []):
        if not isinstance(item, dict):
            continue
        fold = int(item.get("fold"))
        for qid in item.get("validation_ids", []):
            out[str(qid)] = fold
    return out


@function(
    name="udsc-p13-audit-fold0-negatives",
    cpu=4,
    memory="16Gi",
    image=image,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def audit_fold0():
    print("=== P13 FOLD0 NEGATIVE AUDIT ===", flush=True)
    for path in (NEGATIVES, TRAIN, FOLDS):
        print(f"[CHECK] {path} exists={path.is_file()}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _load_gold()
    fold_map = _load_fold_map()

    total = 0
    invalid_json = 0
    type_counts = Counter()
    query_counts = Counter()
    positive_doc_not_gold = 0
    negative_doc_is_gold = 0
    positive_equals_negative = 0
    missing_positive_evidence = 0
    missing_negative_evidence = 0
    identical_evidence = 0
    wrong_record_fold = 0
    leakage_query_in_validation_fold = 0
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, str, str]] = set()
    pos_chars = 0
    neg_chars = 0
    samples: list[dict[str, Any]] = []

    def sample(reason: str, row: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        if len(samples) >= 500:
            return
        pe = _evidence_texts(row.get("positive_evidence"))
        ne = _evidence_texts(row.get("negative_evidence"))
        item = {
            "reason": reason,
            "query_id": str(row.get("query_id", "")),
            "fold": row.get("fold"),
            "negative_type": row.get("negative_type"),
            "positive_doc": str(row.get("positive_doc", "")),
            "negative_doc": str(row.get("negative_doc", "")),
            "positive_evidence_count": len(pe),
            "negative_evidence_count": len(ne),
            "positive_evidence_chars": sum(map(len, pe)),
            "negative_evidence_chars": sum(map(len, ne)),
            "positive_evidence_preview": pe[0][:240] if pe else "",
            "negative_evidence_preview": ne[0][:240] if ne else "",
            "provenance": row.get("provenance"),
        }
        if extra:
            item.update(extra)
        samples.append(item)

    with NEGATIVES.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                if invalid_json <= 10:
                    print(f"[BAD JSON] line={line_no}", flush=True)
                continue
            if not isinstance(row, dict):
                continue

            total += 1
            if total % 100000 == 0:
                print(f"[PROGRESS] records={total:,}", flush=True)

            qid = str(row.get("query_id", "")).strip()
            pdoc = str(row.get("positive_doc", "")).strip()
            ndoc = str(row.get("negative_doc", "")).strip()
            ntype = str(row.get("negative_type", "")).strip() or "<missing>"
            type_counts[ntype] += 1
            query_counts[qid] += 1

            pe = _evidence_texts(row.get("positive_evidence"))
            ne = _evidence_texts(row.get("negative_evidence"))
            pos_chars += sum(map(len, pe))
            neg_chars += sum(map(len, ne))

            g = gold.get(qid, set())
            if pdoc not in g:
                positive_doc_not_gold += 1
                sample("positive_doc_not_gold", row, {"gold_documents": sorted(g)})
            if ndoc in g:
                negative_doc_is_gold += 1
                sample("negative_doc_is_gold", row, {"gold_documents": sorted(g)})
            if pdoc and pdoc == ndoc:
                positive_equals_negative += 1
                sample("positive_equals_negative", row)
            if not pe:
                missing_positive_evidence += 1
                sample("missing_positive_evidence", row)
            if not ne:
                missing_negative_evidence += 1
                sample("missing_negative_evidence", row)
            if pe and ne and _hash_texts(pe) == _hash_texts(ne):
                identical_evidence += 1
                sample("identical_positive_negative_evidence", row)

            pair = (qid, pdoc, ndoc)
            if pair in seen_pairs:
                duplicate_pairs += 1
            else:
                seen_pairs.add(pair)

            if row.get("fold") != 0:
                wrong_record_fold += 1
                sample("wrong_record_fold", row)
            if fold_map.get(qid) == 0:
                leakage_query_in_validation_fold += 1
                sample("validation_query_leakage", row)

    counts = list(query_counts.values())
    summary = {
        "schema_version": "p13-fold0-negative-audit-v1",
        "negatives_file": str(NEGATIVES),
        "total_records": total,
        "invalid_json_lines": invalid_json,
        "unique_queries": len(query_counts),
        "negative_type_counts": dict(type_counts.most_common()),
        "label_integrity": {
            "positive_doc_not_gold": positive_doc_not_gold,
            "negative_doc_is_gold": negative_doc_is_gold,
            "positive_equals_negative": positive_equals_negative,
        },
        "evidence_integrity": {
            "missing_positive_evidence": missing_positive_evidence,
            "missing_negative_evidence": missing_negative_evidence,
            "identical_positive_negative_evidence": identical_evidence,
            "avg_positive_evidence_chars": pos_chars / total if total else 0.0,
            "avg_negative_evidence_chars": neg_chars / total if total else 0.0,
        },
        "fold_integrity": {
            "wrong_record_fold": wrong_record_fold,
            "validation_query_leakage": leakage_query_in_validation_fold,
        },
        "duplicates": {"duplicate_query_positive_negative_pairs": duplicate_pairs},
        "per_query_negative_count": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
        },
        "basic_integrity_gate": (
            invalid_json == 0
            and positive_doc_not_gold == 0
            and negative_doc_is_gold == 0
            and positive_equals_negative == 0
            and wrong_record_fold == 0
            and leakage_query_in_validation_fold == 0
        ),
        "suspicious_sample_count": len(samples),
    }

    report = OUT_DIR / "p13_fold0_negative_audit.json"
    suspicious = OUT_DIR / "p13_fold0_suspicious_sample.jsonl"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with suspicious.open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== AUDIT SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={report}", flush=True)
    print(f"SAMPLES={suspicious}", flush=True)
    return {
        "status": "PASS" if summary["basic_integrity_gate"] else "FAIL_INTEGRITY",
        "report": str(report),
        "samples": str(suspicious),
        "summary": summary,
    }


if __name__ == "__main__":
    print("Enqueuing P13 fold0 negative audit on Beam CPU...", flush=True)
    result = audit_fold0.remote()
    print("REMOTE RESULT:", result, flush=True)
