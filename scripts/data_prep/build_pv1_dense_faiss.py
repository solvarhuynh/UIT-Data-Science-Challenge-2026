"""Build the PV1 dense embedding set by reusing the immutable V3 FAISS vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PV1 = ROOT / "data/processed_pv1"
V3 = ROOT / "data/processed_v3"
OLD_COLLECTION = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768"
OUT = PV1 / "dense_faiss"
META = PV1 / "metadata"
MODEL = ROOT / "models/dek21-v2"
TOTAL = 1_272_971
REPAIR_SET = set(json.loads((META / "pv1_corpus_manifest.json").read_text(encoding="utf-8"))["changed_document_ids"])


def chunks(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_chunk_ids(path: Path) -> list[str]:
    # Payloads are a large JSON object.  Extracting only the unescaped chunk_id
    # fields avoids loading 1.8 GB of metadata into RAM.
    pattern = re.compile(rb'"chunk_id":"([^"]+)"')
    ids: list[str] = []
    tail = b""
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            data = tail + block
            ids.extend(item.decode("utf-8") for item in pattern.findall(data))
            tail = data[-256:]
    # The overlap can see the boundary item twice.
    result: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    started = time.perf_counter()
    manifest = json.loads((META / "pv1_corpus_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("PV1_CORPUS_GATE") != "PASS" or manifest.get("total_chunks") != TOTAL:
        raise RuntimeError("PV1 corpus gate/size is not the frozen expected state")

    v3: dict[str, str] = {}
    pv1: dict[str, tuple[str, str]] = {}
    for item in chunks(V3 / "chunks"):
        v3[item["chunk_id"]] = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
    for item in chunks(PV1 / "chunks"):
        cid = item["chunk_id"]
        pv1[cid] = (item["doc_id"], hashlib.sha256(item["text"].encode("utf-8")).hexdigest())
    if len(pv1) != TOTAL or len(v3) != 1_270_356:
        raise RuntimeError(f"unexpected corpus sizes: PV1={len(pv1)} V3={len(v3)}")
    missing = set(pv1) - set(v3)
    removed = set(v3) - set(pv1)
    changed = {cid for cid in set(pv1) & set(v3) if pv1[cid][1] != v3[cid]}
    changed_ids = missing | changed
    if any(pv1[cid][0] not in REPAIR_SET for cid in changed_ids):
        raise RuntimeError("delta contains a changed chunk outside the repair set")
    if changed:
        raise RuntimeError(f"unexpected text changes in inherited chunks: {len(changed)}")

    old_index_path = OLD_COLLECTION / "index.faiss"
    old_payload_path = OLD_COLLECTION / "payloads.json"
    old_manifest = json.loads((OLD_COLLECTION / "manifest.json").read_text(encoding="utf-8"))
    payload_ids = payload_chunk_ids(old_payload_path)
    if len(payload_ids) != len(set(payload_ids)) or set(payload_ids) != set(v3):
        raise RuntimeError("V3 payload mapping does not exactly cover V3 chunks")
    if old_manifest.get("chunk_count") != len(v3) or old_manifest.get("embedding_dimension") != 768:
        raise RuntimeError("V3 FAISS manifest is inconsistent")

    from udsc2026.infrastructure.embedding.client import EmbeddingClient
    import faiss

    old_index = faiss.read_index(str(old_index_path))
    if type(old_index).__name__ != "IndexFlatIP" or old_index.ntotal != len(v3) or old_index.d != 768:
        raise RuntimeError("V3 FAISS index contract is not the expected IndexFlatIP/768")
    old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
    if not np.isfinite(old_vectors).all():
        raise RuntimeError("V3 vectors contain non-finite values")
    old_rows = {cid: row for row, cid in enumerate(payload_ids)}
    if set(old_rows) != set(v3) or len(old_rows) != old_index.ntotal:
        raise RuntimeError("ambiguous or incomplete V3 row mapping")

    worklist = [item for item in chunks(PV1 / "chunks") if item["chunk_id"] in changed_ids]
    if len(worklist) != len(changed_ids) or any(not item["text"].strip() for item in worklist):
        raise RuntimeError("invalid changed embedding worklist")
    embedder = EmbeddingClient(
        model_path=str(MODEL), device="cpu", batch_size=args.batch_size,
        max_length=256, normalize_embeddings=True, output_dimension=768,
        window_long_texts=True, window_overlap_tokens=32,
    )
    new_vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(worklist), args.batch_size):
        batch = worklist[start : start + args.batch_size]
        vectors = embedder.embed_documents([item["text"] for item in batch], batch_size=args.batch_size)
        for item, vector in zip(batch, vectors):
            array = np.asarray(vector, dtype=np.float32)
            if array.shape != (768,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid new vector for {item['chunk_id']}")
            new_vectors[item["chunk_id"]] = array
        print(f"EMBED_PROGRESS completed={len(new_vectors)}/{len(worklist)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    changed_path = OUT / "changed_vectors.npz"
    np.savez_compressed(changed_path, chunk_ids=np.asarray(list(new_vectors)), vectors=np.asarray(list(new_vectors.values()), dtype=np.float32))
    full_path = OUT / "pv1_embeddings.float32.npy"
    full = np.lib.format.open_memmap(full_path, mode="w+", dtype="float32", shape=(TOTAL, 768))
    index = faiss.IndexFlatIP(768)
    mapping_path = OUT / "chunk_vector_mapping.jsonl"
    with mapping_path.open("w", encoding="utf-8", newline="\n") as mapping:
        batch_vectors: list[np.ndarray] = []
        batch_rows: list[dict[str, Any]] = []
        row = 0
        for item in chunks(PV1 / "chunks"):
            cid = item["chunk_id"]
            vector = new_vectors[cid] if cid in new_vectors else old_vectors[old_rows[cid]]
            full[row] = vector
            batch_vectors.append(vector)
            batch_rows.append({"row": row, "chunk_id": cid, "document_id": item["doc_id"]})
            if len(batch_vectors) == args.batch_size * 16:
                index.add(np.asarray(batch_vectors, dtype=np.float32))
                mapping.write("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in batch_rows) + "\n")
                batch_vectors.clear(); batch_rows.clear()
            row += 1
        if batch_vectors:
            index.add(np.asarray(batch_vectors, dtype=np.float32))
            mapping.write("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in batch_rows) + "\n")
    full.flush()
    del full
    if index.ntotal != TOTAL:
        raise RuntimeError(f"PV1 FAISS ntotal={index.ntotal}, expected {TOTAL}")
    index_path = OUT / "index.faiss"
    faiss.write_index(index, str(index_path))
    report = {
        "pv1_corpus_fingerprint": manifest["corpus_fingerprint"],
        "embedding_model": "huyydangg/DEk21_hcmute_embedding_v2",
        "embedding_revision": "LOCAL_SNAPSHOT_TREE_99a2963b2f51fa7a570a3e7f550d7993b9de90a8",
        "embedding_revision_status": "LOCAL_REVISION_UNPROVEN",
        "pooling": "mean",
        "normalization": True,
        "max_length": 256,
        "dimension": 768,
        "dtype": "float32",
        "total_chunks": TOTAL,
        "reused_vectors": len(set(v3) & set(pv1)),
        "new_vectors": len(new_vectors),
        "removed_v3_chunks": len(removed),
        "embedding_artifact_path": str(full_path),
        "embedding_artifact_sha256": sha256_file(full_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "faiss_path": str(index_path),
        "faiss_sha256": sha256_file(index_path),
        "faiss_index_type": "IndexFlatIP",
        "faiss_metric": "inner_product; cosine contract via normalized vectors",
        "faiss_ntotal": index.ntotal,
        "delta_reuse_gate": "PASS",
        "dense_faiss_gate": "PASS",
        "v3_artifacts_mutated": "NO",
        "pv1_corpus_mutated": "NO",
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = META / "pv1_dense_faiss_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = META / "pv1_dense_faiss_report.md"
    report_path.write_text("# PV1 Dense + FAISS\n\n" + "\n".join(f"- **{key}:** `{value}`" for key, value in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
