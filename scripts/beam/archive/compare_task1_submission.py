#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, list[str]]:
    def row_value(row: dict[str, Any]) -> tuple[str, list[str]] | None:
        qid = row.get("id", row.get("query_id", row.get("question_id")))
        docs = row.get("documents", row.get("answer", row.get("top5")))
        if docs is None and isinstance(row.get("hits"), list):
            docs = [
                hit.get("doc_id", hit.get("document_id"))
                for hit in row["hits"]
                if isinstance(hit, dict)
            ]
        if qid is None or not isinstance(docs, list):
            return None
        distinct = list(dict.fromkeys(str(value) for value in docs if value is not None))
        return str(qid), distinct[:5]

    def normalize(payload: Any) -> dict[str, list[str]]:
        if isinstance(payload, dict):
            if isinstance(payload.get("predictions"), list):
                return normalize(payload["predictions"])
            direct = row_value(payload)
            if direct is not None:
                return {direct[0]: direct[1]}
            out: dict[str, list[str]] = {}
            for qid, value in payload.items():
                if isinstance(value, dict):
                    docs = value.get(
                        "answer", value.get("documents", value.get("top5"))
                    )
                else:
                    docs = value
                if isinstance(docs, list):
                    out[str(qid)] = [str(x) for x in docs[:5]]
            if out:
                return out

        if isinstance(payload, list):
            out = {}
            for row in payload:
                if not isinstance(row, dict):
                    continue
                value = row_value(row)
                if value is not None:
                    out[value[0]] = value[1]
            if out:
                return out
        raise ValueError("Unsupported prediction schema")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            if "submission.json" not in zf.namelist():
                raise ValueError(f"{path}: ZIP has no submission.json")
            return normalize(json.loads(zf.read("submission.json")))

    if path.suffix.lower() == ".jsonl":
        out: dict[str, list[str]] = {}
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    value = row_value(row)
                    if value is not None:
                        out[value[0]] = value[1]
        if not out:
            raise ValueError(f"{path}: không đọc được ranking document")
        return out

    return normalize(json.loads(path.read_text(encoding="utf-8-sig")))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare Task1 public submissions/prediction artifacts against a known reference."
    )
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path, action="append")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    ref = _load_object(args.reference)
    ref_ids = set(ref)
    results = []

    for cand_path in args.candidate:
        cand = _load_object(cand_path)
        cand_ids = set(cand)
        common = sorted(ref_ids & cand_ids)

        exact_order = 0
        exact_set = 0
        overlap_sum = 0
        jaccard_sum = 0.0
        position_hits = [0] * 5
        changed = []

        for qid in common:
            a = ref[qid][:5]
            b = cand[qid][:5]
            sa, sb = set(a), set(b)
            overlap = len(sa & sb)
            union = len(sa | sb)
            overlap_sum += overlap
            jaccard_sum += overlap / union if union else 1.0
            if a == b:
                exact_order += 1
            if sa == sb:
                exact_set += 1
            for i in range(min(5, len(a), len(b))):
                if a[i] == b[i]:
                    position_hits[i] += 1
            if a != b:
                changed.append(
                    {
                        "query_id": qid,
                        "reference": a,
                        "candidate": b,
                        "set_overlap": overlap,
                    }
                )

        n = len(common)
        result = {
            "candidate": str(cand_path),
            "reference_query_count": len(ref),
            "candidate_query_count": len(cand),
            "common_query_count": n,
            "missing_from_candidate": len(ref_ids - cand_ids),
            "extra_in_candidate": len(cand_ids - ref_ids),
            "exact_order_count": exact_order,
            "exact_order_rate": exact_order / n if n else 0.0,
            "exact_set_count": exact_set,
            "exact_set_rate": exact_set / n if n else 0.0,
            "mean_set_overlap_at5": overlap_sum / n if n else 0.0,
            "mean_jaccard_at5": jaccard_sum / n if n else 0.0,
            "position_match_rates": [
                x / n if n else 0.0 for x in position_hits
            ],
            "changed_query_count": len(changed),
            "changed_sample": changed[:30],
        }
        results.append(result)

    payload = {
        "reference": str(args.reference),
        "reference_query_count": len(ref),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
