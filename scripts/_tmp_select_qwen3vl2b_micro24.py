"""One-shot CPU selector for the frozen 24-qdoc micro-replay diagnostic."""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOOD = ROOT / "artifacts/task1/workflow_b/tv2/b2a/acc_a_400q/predictions.jsonl"
FULL_PREDICTIONS = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
ACC_B = ROOT / "artifacts/task1/workflow_b/tv2/b2a/acc_b_500k/chunk_scores.sqlite3"
ACC_C = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/acc_c_600k_fix_sqlite/chunk_scores_692k_before_sqlite_fix.sqlite3"
FINAL = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/chunk_scores_FULL.sqlite3"
OUTPUT = ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/qwen3vl2b_micro_same_item_24qdoc_selection.jsonl"


def read_keys(path: Path) -> set[tuple[str, str, str]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return {(str(q), str(d), str(c)) for q, d, c in connection.execute("SELECT query_id, doc_id, chunk_id FROM chunk_scores")}
    finally:
        connection.close()


def read_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row["query_id"])
            for score in row["document_scores"]:
                key = (query, str(score["doc_id"]))
                if key in scores:
                    raise RuntimeError(f"duplicate q-doc score: {key}")
                scores[key] = float(score["score"])
    return scores


def key_order(row: dict[str, object]) -> tuple[int, int]:
    return int(str(row["query_id"])), int(str(row["doc_id"]))


def select_stage(stage: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    top = sorted(rows, key=lambda row: (-float(row["abs_delta"]), *key_order(row)))[:4]
    selected = list(top)
    remaining = sorted((row for row in rows if row not in top), key=key_order)
    indices = [math.floor(fraction * (len(remaining) - 1)) for fraction in (0.2, 0.4, 0.6, 0.8)]
    used: set[int] = set()
    for index in indices:
        while index in used:
            index += 1
        used.add(index)
        row = dict(remaining[index])
        row["selection_reason"] = "DETERMINISTIC_SPREAD"
        selected.append(row)
    for row in selected[:4]:
        row["selection_reason"] = "TOP_DELTA"
    if len(selected) != 8 or any(row["stage"] != stage for row in selected):
        raise RuntimeError(f"selection invariant failed for stage {stage}")
    return selected


def main() -> None:
    good, full = read_scores(GOOD), read_scores(FULL_PREDICTIONS)
    if len(good) != 30_800 or set(good) - set(full):
        raise RuntimeError("400Q score-universe mismatch")

    stage_a = read_keys(ACC_B)
    stage_ab = read_keys(ACC_C)
    stage_final = read_keys(FINAL)
    stage_b, stage_c = stage_ab - stage_a, stage_final - stage_ab
    if (len(stage_a), len(stage_b), len(stage_c)) != (559_170, 134_152, 599_876):
        raise RuntimeError("checkpoint stage cardinality mismatch")

    chunks_by_doc: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in stage_final:
        if key[:2] in good:
            chunks_by_doc[key[:2]].append(key)
    rows_by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key, good_score in good.items():
        memberships = {
            "A" if chunk in stage_a else "B" if chunk in stage_b else "C"
            for chunk in chunks_by_doc[key]
        }
        if len(memberships) != 1:
            raise RuntimeError(f"mixed/missing stage membership: {key}")
        stage = memberships.pop()
        rows_by_stage[stage].append({
            "stage": stage,
            "query_id": key[0],
            "doc_id": key[1],
            "good_document_score": good_score,
            "bad_full_document_score": full[key],
            "abs_delta": abs(good_score - full[key]),
        })
    if {stage: len(rows_by_stage[stage]) for stage in "ABC"} != {"A": 13321, "B": 3157, "C": 14322}:
        raise RuntimeError("400Q pure-stage cardinality mismatch")

    selected = [row for stage in "ABC" for row in select_stage(stage, rows_by_stage[stage])]
    counts, reasons = Counter(row["stage"] for row in selected), Counter(row["selection_reason"] for row in selected)
    if len(selected) != 24 or len({(row["query_id"], row["doc_id"]) for row in selected}) != 24 or counts != Counter({"A": 8, "B": 8, "C": 8}) or reasons != Counter({"TOP_DELTA": 12, "DETERMINISTIC_SPREAD": 12}):
        raise RuntimeError("final selection invariant failed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("MICRO24_SELECTION_COMPLETE")
    print("A=8\nB=8\nC=8\nTOTAL=24")
    print(f"OUTPUT={OUTPUT}")
    for row in selected:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
