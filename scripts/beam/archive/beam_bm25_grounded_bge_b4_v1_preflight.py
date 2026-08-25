"""CPU-only B4 preflight; it never loads torch/reranker and never scores GPU."""
from __future__ import annotations

from beam import Image, Volume, function

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/p13")
RUNTIME = ROOT / "runtime"
WORKLIST = RUNTIME / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
K500 = RUNTIME / "artifacts/task1/recovery_096/selective_bge_k500_v1/new_chunk_scores.jsonl"
TRAIN = RUNTIME / "data/raw/btc/LegalIR/train.json"
PAYLOADS = RUNTIME / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
K200_A = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl"
K200_B = RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
ARCHIVES = (ROOT / "udsc_p13_reranker.zip", ROOT / "udsc_reranker.zip")
EXPECTED_SHA = "34d3ba1f0f9e202222c19a921f10725faace86af8869809d3c947f66b035ba32"

image = Image(python_version="python3.11")


def canonical_sha(rows: list[dict]) -> str:
    return hashlib.sha256("".join(f"{r['query_id']}\t{r['doc_id']}\n" for r in sorted(rows, key=lambda r: (str(r['query_id']), str(r['doc_id'])))).encode()).hexdigest()


@function(name="udsc-bm25-grounded-bge-b4-v1-preflight", cpu=2, memory="4Gi", image=image, volumes=[Volume(name="udsc-p13", mount_path=str(ROOT))], timeout=600, headless=True)
def preflight() -> dict:
    paths = {"worklist": WORKLIST, "train": TRAIN, "payloads": PAYLOADS, "k200_fold0": K200_A, "k200_fold1to4": K200_B, "k500_scores": K500}
    missing = [name for name, path in paths.items() if not path.exists()]
    archive_ok = next((path for path in ARCHIVES if path.is_file()), None)
    if archive_ok is None: missing.append("reranker_archive")
    result = {"status": "PREFLIGHT_FAIL" if missing else "PREFLIGHT_PASS", "missing": missing, "paths": {name: {"exists": path.exists(), "is_dir": path.is_dir(), "bytes": path.stat().st_size if path.is_file() else None} for name, path in paths.items()}, "reranker_archive": str(archive_ok) if archive_ok else None}
    if not missing:
        payload = json.loads(WORKLIST.read_text(encoding="utf-8")); rows = payload.get("rows", [])
        result["worklist"] = {"doc_occurrences": len(rows), "query_count": len({str(r['query_id']) for r in rows}), "canonical_sha256": canonical_sha(rows), "sha_matches": canonical_sha(rows) == EXPECTED_SHA, "expected_chunks_at3": 161763}
        if len(rows) != 53924 or canonical_sha(rows) != EXPECTED_SHA:
            result["status"] = "PREFLIGHT_FAIL"
            result["error"] = "B4 worklist count/SHA mismatch"
    return result


if __name__ == "__main__":
    print(json.dumps(preflight.remote(), ensure_ascii=False, indent=2))
