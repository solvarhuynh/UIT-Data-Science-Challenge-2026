from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SELECTION = (
    ROOT
    / "reports/task1/workflow_b/tv2/b2a/manifests/"
    "qwen3vl2b_micro_same_item_24qdoc_selection.jsonl"
)

WORKLIST = (
    ROOT
    / "reports/task1/workflow_b/tv2/b2a/manifests/"
    "b2a_qwen_true_s2_top3_worklist.jsonl"
)

OUTPUT = (
    ROOT
    / "reports/task1/workflow_b/tv2/b2a/manifests/"
    "qwen3vl2b_micro_same_item_24qdoc_manifest.jsonl"
)

EXPECTED_WORKLIST_SHA = (
    "a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {lineno}: {exc}"
                ) from exc
    return rows


# ============================================================
# 1. Verify frozen 24-qdoc selection
# ============================================================

selection = read_jsonl(SELECTION)

assert len(selection) == 24, f"Expected 24 selection rows, got {len(selection)}"

keys = [
    (str(r["query_id"]), str(r["doc_id"]))
    for r in selection
]

assert len(set(keys)) == 24, "Duplicate (query_id, doc_id) in selection"

stage_counts = Counter(str(r["stage"]) for r in selection)
reason_counts = Counter(str(r["selection_reason"]) for r in selection)

assert stage_counts == {"A": 8, "B": 8, "C": 8}, stage_counts
assert reason_counts == {
    "TOP_DELTA": 12,
    "DETERMINISTIC_SPREAD": 12,
}, reason_counts


# ============================================================
# 2. Verify canonical worklist SHA
# ============================================================

actual_sha = sha256_file(WORKLIST)

if actual_sha != EXPECTED_WORKLIST_SHA:
    raise RuntimeError(
        "WORKLIST SHA MISMATCH\n"
        f"expected={EXPECTED_WORKLIST_SHA}\n"
        f"actual={actual_sha}"
    )

print(f"WORKLIST_SHA_OK={actual_sha}")


# ============================================================
# 3. Stream canonical worklist and collect ONLY frozen 24 q-docs
# ============================================================

wanted = set(keys)

resolved: dict[tuple[str, str], dict] = {
    key: {
        "canonical_k77_rank": None,
        "chunks": [],
    }
    for key in wanted
}

with WORKLIST.open("r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue

        obj = json.loads(line)

        key = (str(obj["query_id"]), str(obj["doc_id"]))

        if key not in wanted:
            continue

        k77_rank = int(obj["canonical_k77_rank"])

        current_rank = resolved[key]["canonical_k77_rank"]

        if current_rank is None:
            resolved[key]["canonical_k77_rank"] = k77_rank
        elif current_rank != k77_rank:
            raise RuntimeError(
                f"Inconsistent canonical_k77_rank for {key}: "
                f"{current_rank} vs {k77_rank}"
            )

        resolved[key]["chunks"].append(
            {
                "chunk_id": str(obj["chunk_id"]),
                "bm25_within_doc_rank": int(obj["bm25_within_doc_rank"]),
                "source_chunk_artifact": str(obj["source_chunk_artifact"]),
            }
        )


# ============================================================
# 4. Validate exact worklist resolution
# ============================================================

for key, info in resolved.items():
    chunks = info["chunks"]

    if not chunks:
        raise RuntimeError(f"Q-doc not found in canonical worklist: {key}")

    if not 1 <= len(chunks) <= 3:
        raise RuntimeError(
            f"Expected 1-3 selected chunks for {key}, got {len(chunks)}"
        )

    ranks = [x["bm25_within_doc_rank"] for x in chunks]

    if len(set(ranks)) != len(ranks):
        raise RuntimeError(
            f"Duplicate bm25_within_doc_rank for {key}: {ranks}"
        )

    chunks.sort(
        key=lambda x: (
            x["bm25_within_doc_rank"],
            x["chunk_id"],
        )
    )


# ============================================================
# 5. Verify canonical chunk files and exact chunk IDs
# ============================================================

doc_cache: dict[str, dict[str, dict]] = {}

def load_doc_chunks(doc_id: str) -> dict[str, dict]:
    if doc_id in doc_cache:
        return doc_cache[doc_id]

    path = ROOT / "data/processed_v3/chunks" / f"{doc_id}.jsonl"

    if not path.exists():
        raise RuntimeError(f"Missing canonical chunk file: {path}")

    by_id = {}

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            cid = str(obj["chunk_id"])

            if cid in by_id:
                raise RuntimeError(
                    f"Duplicate chunk_id={cid} in {path}"
                )

            by_id[cid] = obj

    doc_cache[doc_id] = by_id
    return by_id


total_chunks = 0
chunk_count_distribution = Counter()

for query_id, doc_id in keys:
    info = resolved[(query_id, doc_id)]
    doc_chunks = load_doc_chunks(doc_id)

    for selected in info["chunks"]:
        cid = selected["chunk_id"]

        if cid not in doc_chunks:
            raise RuntimeError(
                f"Canonical selected chunk missing: "
                f"query_id={query_id} doc_id={doc_id} chunk_id={cid}"
            )

        chunk_obj = doc_chunks[cid]

        if str(chunk_obj["doc_id"]) != doc_id:
            raise RuntimeError(
                f"doc_id mismatch for chunk {cid}: "
                f"expected={doc_id} actual={chunk_obj['doc_id']}"
            )

        if not isinstance(chunk_obj.get("text"), str):
            raise RuntimeError(
                f"Chunk text is not readable string: {cid}"
            )

    n = len(info["chunks"])
    total_chunks += n
    chunk_count_distribution[n] += 1


assert total_chunks <= 72, f"Chunk guard exceeded: {total_chunks} > 72"

assert (
    chunk_count_distribution[1]
    + chunk_count_distribution[2]
    + chunk_count_distribution[3]
    == 24
)

assert (
    chunk_count_distribution[1]
    + 2 * chunk_count_distribution[2]
    + 3 * chunk_count_distribution[3]
    == total_chunks
)


# ============================================================
# 6. Write final frozen replay manifest
# ============================================================

final_rows = []

for original in selection:
    query_id = str(original["query_id"])
    doc_id = str(original["doc_id"])

    info = resolved[(query_id, doc_id)]

    row = dict(original)

    row["canonical_k77_rank"] = info["canonical_k77_rank"]
    row["selected_chunk_ids"] = [
        x["chunk_id"] for x in info["chunks"]
    ]
    row["chunk_count"] = len(info["chunks"])

    final_rows.append(row)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
    for row in final_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


# ============================================================
# 7. Read-back validation
# ============================================================

check = read_jsonl(OUTPUT)

assert len(check) == 24
assert len(
    {
        (str(r["query_id"]), str(r["doc_id"]))
        for r in check
    }
) == 24

assert Counter(r["stage"] for r in check) == {
    "A": 8,
    "B": 8,
    "C": 8,
}

assert Counter(r["selection_reason"] for r in check) == {
    "TOP_DELTA": 12,
    "DETERMINISTIC_SPREAD": 12,
}

for before, after in zip(selection, check):
    for field in (
        "stage",
        "query_id",
        "doc_id",
        "selection_reason",
        "good_document_score",
        "bad_full_document_score",
        "abs_delta",
    ):
        if before[field] != after[field]:
            raise RuntimeError(
                f"Frozen selection field changed: "
                f"{field}: {before[field]} != {after[field]}"
            )


# ============================================================
# 8. Final output
# ============================================================

print()
print("B2A QWEN3-VL-2B MICRO24 CHUNK RESOLUTION")
print("Status: PASS_CHUNK_MANIFEST")
print(f"Frozen selection: {SELECTION.relative_to(ROOT)}")
print(f"Final replay manifest: {OUTPUT.relative_to(ROOT)}")
print("Worklist SHA verified: YES")
print("Selection rows: 24")
print("Unique q-docs: 24")
print("Selected A: 8")
print("Selected B: 8")
print("Selected C: 8")
print("Canonical worklist matches: 24/24")
print(f"Selected total chunks: {total_chunks}")
print(f"1-chunk q-docs: {chunk_count_distribution[1]}")
print(f"2-chunk q-docs: {chunk_count_distribution[2]}")
print(f"3-chunk q-docs: {chunk_count_distribution[3]}")
print(f"Canonical chunk files resolved: {total_chunks}/{total_chunks}")
print("Missing chunks: 0")
print("Frozen q-doc selection changed: NO")
print("Fold0 used: NO")
print("Public labels used: NO")
print("GPU launched: NO")
print("Modal used: NO")
print("Model inference: NO")
print()
print("RESOLVED MANIFEST:")
print(
    "stage | query_id | doc_id | reason | "
    "k77_rank | chunk_count | selected_chunk_ids"
)

for row in check:
    print(
        f"{row['stage']} | "
        f"{row['query_id']} | "
        f"{row['doc_id']} | "
        f"{row['selection_reason']} | "
        f"{row['canonical_k77_rank']} | "
        f"{row['chunk_count']} | "
        f"{row['selected_chunk_ids']}"
    )

print()
print("Safe next action: BUILD_MICRO_REPLAY_FROM_FROZEN_MANIFEST")