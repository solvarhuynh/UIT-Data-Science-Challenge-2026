"""Mock Hybrid Retriever (BM25 + Dense Vector Search) for TV3 testing.

Implements full Hybrid Search combining:
1. BM25 (Sparse keyword search)
2. BKAI Bi-Encoder (Dense vector embeddings via sentence-transformers)
3. Reciprocal Rank Fusion (RRF) to merge and rank results.

Saves embeddings to numpy cache file (.npy) so reloading is instant!
Supports CPU and CUDA (e.g. for Kaggle 2x T4 or local GPU).

NOT FOR PRODUCTION USE. This is a mock retriever for TV3 benchmark experiments.
"""

import gc
import hashlib
import json
import logging
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from pyvi import ViTokenizer
    _PYVI_AVAILABLE = True
except ImportError:
    _PYVI_AVAILABLE = False

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NPY_MAGIC = b"\x93NUMPY"


def _contains_jsonl(directory: Path) -> bool:
    """Return whether *directory* contains at least one JSONL corpus file."""

    return directory.is_dir() and next(directory.glob("*.jsonl"), None) is not None


def _resolve_parent_corpus(requested: Path) -> Path:
    """Resolve a parent corpus without silently falling back to child chunks."""

    candidates = [requested, _PROJECT_ROOT / "data/processed_v3/parents"]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(sorted(kaggle_input.glob("**/processed_v3/parents")))
        candidates.extend(sorted(kaggle_input.glob("**/parents")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _contains_jsonl(resolved):
            if resolved != requested.expanduser().resolve():
                logger.warning(
                    "Parent corpus '%s' is unavailable; using '%s'.",
                    requested,
                    resolved,
                )
            return resolved

    raise FileNotFoundError(
        "No parent JSONL corpus found. Pass --parents_dir pointing to "
        "data/processed_v3/parents (or its copied equivalent)."
    )


def _corpus_cache_fingerprint(parents_dir: Path, max_docs: int) -> str:
    """Build a stable cache key from the audited corpus and document limit."""

    digest = hashlib.sha256()
    manifest_path = parents_dir.parent / "metadata" / "processing_manifest.json"
    manifest_hash = ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("processed_corpus_tree_hash")
        if isinstance(value, str):
            manifest_hash = value.strip()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        pass

    if manifest_hash:
        digest.update(manifest_hash.encode("ascii"))
    else:
        for path in sorted(parents_dir.glob("*.jsonl")):
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))

    digest.update(f"\0max_docs={max_docs}".encode("ascii"))
    return digest.hexdigest()[:16]


def _finite_embedding_sample(array: np.ndarray) -> bool:
    """Check representative rows without materializing a full mmap in RAM."""

    if array.shape[0] == 0:
        return False
    indices = sorted({0, array.shape[0] // 2, array.shape[0] - 1})
    return bool(np.isfinite(np.asarray(array[indices])).all())


def _load_cached_embeddings(
    cache_file: Path,
    expected_rows: int,
) -> np.ndarray | np.memmap | None:
    """Load a validated standard NPY file or an exact legacy raw float32 mmap."""

    if expected_rows <= 0 or not cache_file.is_file():
        return None

    try:
        array = np.load(cache_file, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        try:
            with cache_file.open("rb") as stream:
                magic = stream.read(len(_NPY_MAGIC))
            size = cache_file.stat().st_size
        except OSError:
            return None

        # A damaged standard NPY must be rebuilt, never reinterpreted with a
        # shifted header. Only headerless legacy float32 memmaps are accepted.
        if magic == _NPY_MAGIC:
            return None
        bytes_per_column = expected_rows * np.dtype(np.float32).itemsize
        if size <= 0 or size % bytes_per_column != 0:
            return None
        dimension = size // bytes_per_column
        if dimension <= 0 or dimension > 65_536:
            return None
        try:
            legacy: np.memmap = np.memmap(
                cache_file,
                dtype=np.float32,
                mode="r",
                shape=(expected_rows, dimension),
            )
        except (OSError, ValueError):
            return None
        return legacy if _finite_embedding_sample(legacy) else None

    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 2
        or array.shape[0] != expected_rows
        or array.dtype != np.dtype(np.float32)
    ):
        return None
    return array if _finite_embedding_sample(array) else None


_STOPWORDS = {
    "và", "của", "thì", "là", "được", "cho", "trong", "về", "theo", "các", "những", "có",
    "này", "đã", "khi", "tại", "để", "ra", "bởi", "với", "hoặc", "như", "nào", "gì", "đó",
    "từ", "trên", "dưới", "sau", "trước", "phải", "người", "cơ", "quan", "tổ", "chức"
}


def _fast_corpus_tokenize(text: str) -> list[str]:
    """Lightweight tokenization filtering stopwords to save 75% RAM memory."""
    if not text:
        return []
    words = [w for w in text.lower().split() if len(w) > 1 and w not in _STOPWORDS]
    tokens = list(words)
    # Thêm 2-gram từ đôi kề nhau để khớp cụm từ ghép mà không làm tốn RAM
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    return tokens


def _tokenize_query(query: str) -> list[str]:
    """Tokenize query using PyVi + 1-grams + Law Code Identifiers (e.g. 08/ck-tncn)."""
    if not query:
        return []
    q_clean = query.lower()
    words = [w for w in q_clean.split() if w not in _STOPWORDS]
    tokens = set(words)

    # Trích xuất các mã văn bản/biểu mẫu đặc biệt (ví dụ: 08/ck-tncn, 80/2021/tt-btc)
    code_matches = re.findall(r"\b\d+[a-z0-9/\-_]*[a-z0-9]\b", q_clean)
    for code in code_matches:
        tokens.add(code)
        tokens.add(code.replace("/", "_").replace("-", "_"))

    if _PYVI_AVAILABLE:
        try:
            vi_tokens = ViTokenizer.tokenize(q_clean).split()
            tokens.update([w for w in vi_tokens if w not in _STOPWORDS])
        except Exception:
            pass
    return list(tokens)


def _tokenize_text(text: str) -> list[str]:
    """Tokenize Vietnamese text using PyVi if available."""
    if not text:
        return []
    text_clean = text.lower()
    if _PYVI_AVAILABLE:
        try:
            return str(ViTokenizer.tokenize(text_clean)).split()
        except Exception:
            return text_clean.split()
    return text_clean.split()


import sqlite3


class DiskBM25:
    """SQLite-backed Disk BM25 Index. Optimized with batching for 15s build time & <150MB disk size!"""

    def __init__(self, db_path: Path, parents_dir: Path, max_docs: int = 0, k1: float = 1.5, b: float = 0.75):
        self.db_path = Path(db_path)
        self.parents_dir = _resolve_parent_corpus(Path(parents_dir))
        self.k1 = k1
        self.b = b
        self.max_docs = max_docs
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode = OFF;")
            self.conn.execute("PRAGMA synchronous = OFF;")
            self.conn.execute("PRAGMA temp_store = MEMORY;")
            self.conn.execute("PRAGMA page_size = 65536;")
            self._init_db()
        except sqlite3.DatabaseError:
            logger.warning("⚠️ File DB '%s' bị malformed. Đang dọn dẹp sạch sẽ và tạo lại DB mới...", self.db_path)
            try:
                self.conn.close()
            except Exception:
                pass
            for ext in ["", "-journal", "-wal", "-shm"]:
                f = Path(str(self.db_path) + ext)
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode = OFF;")
            self.conn.execute("PRAGMA synchronous = OFF;")
            self.conn.execute("PRAGMA temp_store = MEMORY;")
            self.conn.execute("PRAGMA page_size = 65536;")
            self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, val REAL);")
        cursor.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, json_str TEXT, doc_len INTEGER);")
        cursor.execute("CREATE TABLE IF NOT EXISTS postings (term TEXT, doc_id INTEGER, tf INTEGER);")
        self.conn.commit()

        cursor.execute("SELECT val FROM meta WHERE key = 'N'")
        row = cursor.fetchone()
        if row is None or int(row[0]) == 0:
            self._build_disk_index()
        else:
            self.N = int(row[0])
            cursor.execute("SELECT val FROM meta WHERE key = 'avgdl'")
            self.avgdl = cursor.fetchone()[0]

    def _build_disk_index(self) -> None:
        logger.info("💾 Đang xây dựng Disk BM25 Index siêu tốc (executemany) từ '%s'...", self.parents_dir)
        files = sorted(self.parents_dir.glob("*.jsonl"))
        cursor = self.conn.cursor()
        cursor.execute("BEGIN TRANSACTION;")

        total_docs = 0
        total_len = 0
        
        docs_batch = []
        postings_batch = []
        
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        if len(text) < 20:
                            continue
                        tokens = _fast_corpus_tokenize(text)
                        doc_len = len(tokens)
                        if doc_len == 0:
                            continue

                        doc_id = total_docs
                        docs_batch.append((doc_id, line, doc_len))

                        counts = Counter(tokens)
                        for term, count in counts.items():
                            postings_batch.append((term, doc_id, count))

                        total_docs += 1
                        total_len += doc_len

                        if len(docs_batch) >= 10000:
                            cursor.executemany("INSERT INTO documents (id, json_str, doc_len) VALUES (?, ?, ?);", docs_batch)
                            cursor.executemany("INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?);", postings_batch)
                            docs_batch.clear()
                            postings_batch.clear()

                        if self.max_docs > 0 and total_docs >= self.max_docs:
                            break
                    except Exception:
                        continue
            if self.max_docs > 0 and total_docs >= self.max_docs:
                break

        if docs_batch:
            cursor.executemany("INSERT INTO documents (id, json_str, doc_len) VALUES (?, ?, ?);", docs_batch)
            cursor.executemany("INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?);", postings_batch)
            docs_batch.clear()
            postings_batch.clear()

        logger.info("⚡ Đang tạo Index postings trên SQLite...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);")

        self.N = total_docs
        self.avgdl = (total_len / total_docs) if total_docs > 0 else 1.0

        cursor.execute("INSERT OR REPLACE INTO meta (key, val) VALUES ('N', ?);", (self.N,))
        cursor.execute("INSERT OR REPLACE INTO meta (key, val) VALUES ('avgdl', ?);", (self.avgdl,))
        self.conn.commit()
        logger.info("🎉 Hoàn tất xây dựng SQLite Disk BM25 Index siêu tốc cho %d văn bản!", self.N)

    def get_scores_and_docs(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        if not query_tokens or self.N == 0:
            return []

        cursor = self.conn.cursor()
        doc_scores: defaultdict[int, float] = defaultdict(float)

        for q in query_tokens:
            cursor.execute("SELECT postings.doc_id, postings.tf, documents.doc_len FROM postings JOIN documents ON postings.doc_id = documents.id WHERE postings.term = ?;", (q,))
            postings = cursor.fetchall()
            n_q = len(postings)
            if n_q == 0:
                continue
            idf_val = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            if idf_val <= 0:
                continue

            for doc_id, tf, doc_len in postings:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                numer = tf * (self.k1 + 1.0)
                doc_scores[doc_id] += idf_val * (numer / denom)

        sorted_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_hits

    def get_doc(self, doc_id: int) -> Optional[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT json_str FROM documents WHERE id = ?;", (doc_id,))
        row = cursor.fetchone()
        if row:
            payload = json.loads(row[0])
            return payload if isinstance(payload, dict) else None
        return None

    def close(self) -> None:
        """Release the SQLite handle so temporary/removable caches can close."""

        self.conn.close()


class MockHybridRetriever:
    """Hybrid Retriever combining Disk BM25 (SQLite) + BKAI Dense Vector Search + RRF."""

    def __init__(
        self,
        parents_dir: str | Path,
        embedding_model_path: str | Path,
        device: str = "cpu",
        cache_dir: Optional[str | Path] = None,
        max_docs: int = 0,
        rrf_k: int = 60,
        alpha: float = 0.5,
        enable_dense: bool = True,
        enable_query_decomposition: bool = False,
    ) -> None:
        if max_docs < 0:
            raise ValueError("max_docs must be non-negative")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.parents_dir = _resolve_parent_corpus(Path(parents_dir))
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.enable_dense = enable_dense
        self.enable_query_decomposition = enable_query_decomposition

        # Keep generated caches outside the immutable processed_v3 corpus.
        candidate_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else _PROJECT_ROOT / "artifacts/tv3/cache"
        )
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            test_file = candidate_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            self.cache_dir = candidate_dir.resolve()
        except OSError:
            self.cache_dir = Path(tempfile.gettempdir()) / "udsc2026_tv3_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rrf_k = rrf_k
        self.alpha = alpha
        self.max_docs = max_docs

        self._corpus_cache_key = _corpus_cache_fingerprint(
            self.parents_dir,
            self.max_docs,
        )
        db_path = self.cache_dir / f"bm25_{self._corpus_cache_key}.sqlite"
        self._disk_bm25 = DiskBM25(db_path=db_path, parents_dir=self.parents_dir, max_docs=self.max_docs)
        self._doc_embeddings: Optional[np.ndarray] = None
        self._dense_model: Optional[SentenceTransformer] = None

        self._load_corpus()
        if self.enable_dense:
            self._build_dense_index()

    def _load_corpus(self) -> None:
        """Lazily set corpus size N from SQLite disk index to keep RAM under 200MB."""
        self.N = self._disk_bm25.N
        logger.info("Loaded %d documents from SQLite database into retriever.", self.N)

    def close(self) -> None:
        """Release model, mmap, and SQLite resources owned by this retriever."""

        self.unload_dense_model()
        self._doc_embeddings = None
        self._disk_bm25.close()

    def __enter__(self) -> "MockHybridRetriever":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _build_dense_index(self) -> None:
        """Load dense embedding model and compute cached embeddings with streaming to keep RAM under 300MB."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed. Dense vector search disabled.")
            return

        model_cache_key = hashlib.sha256(
            (
                f"{self._corpus_cache_key}\0{self.embedding_model_path}"
                f"\0rows={self.N}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_file = self.cache_dir / f"parent_embeddings_{model_cache_key}.npy"

        if cache_file.exists():
            logger.info("⚡ Đã tìm thấy cache embeddings: '%s'. Đang nạp memory-mapped cực nhanh...", cache_file)
            cached = _load_cached_embeddings(cache_file, self.N)
            if cached is not None:
                self._doc_embeddings = cached
                logger.info(
                    "⚡ Nạp xong cache embeddings shape: %s.",
                    self._doc_embeddings.shape,
                )
                return
            logger.warning("Cache embeddings không hợp lệ; sẽ build lại: '%s'", cache_file)

        model_name_or_path = str(self.embedding_model_path)
        logger.info("Loading embedding model from '%s' (device=%s)...", model_name_or_path, self.device)
        self._dense_model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
        )

        logger.info("Encoding %d documents in streaming batches of 5,000 to prevent RAM OOM...", self.N)
        start = time.monotonic()
        
        sample_vec = self._dense_model.encode(["test"], normalize_embeddings=True)
        emb_dim = sample_vec.shape[1]

        raw_cache_file = cache_file.with_suffix(".raw.tmp")
        npy_cache_file = cache_file.with_suffix(".npy.tmp")
        mmap_arr: np.memmap = np.memmap(
            raw_cache_file,
            dtype="float32",
            mode="w+",
            shape=(self.N, emb_dim),
        )

        batch_sz = 256 if (_TORCH_AVAILABLE and "cuda" in str(self.device) and torch.cuda.is_available()) else 64
        chunk_size = 5000
        cursor = self._disk_bm25.conn.cursor()

        for offset in range(0, self.N, chunk_size):
            cursor.execute(
                "SELECT json_str FROM documents ORDER BY id LIMIT ? OFFSET ?;",
                (chunk_size, offset),
            )
            rows = cursor.fetchall()
            chunk_texts = []
            for (json_str,) in rows:
                try:
                    doc = json.loads(json_str)
                    chunk_texts.append(doc.get("text", ""))
                except Exception:
                    chunk_texts.append("")
            
            if _TORCH_AVAILABLE and "cuda" in str(self.device) and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    vecs = self._dense_model.encode(
                        chunk_texts,
                        batch_size=batch_sz,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
            else:
                vecs = self._dense_model.encode(
                    chunk_texts,
                    batch_size=batch_sz,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

            mmap_arr[offset : offset + len(vecs)] = np.array(vecs, dtype=np.float32)
            del chunk_texts, vecs, rows
            gc.collect()
            if (offset + chunk_size) % 50000 < chunk_size or (offset + chunk_size) >= self.N:
                logger.info("Progress: %d / %d documents encoded into vector embeddings...", min(offset + chunk_size, self.N), self.N)

        mmap_arr.flush()
        logger.info("💾 Đang xuất file vector cache chuẩn .npy...")
        with npy_cache_file.open("wb") as stream:
            np.save(stream, mmap_arr, allow_pickle=False)
        del mmap_arr
        npy_cache_file.replace(cache_file)
        raw_cache_file.unlink(missing_ok=True)

        self._doc_embeddings = _load_cached_embeddings(cache_file, self.N)
        if self._doc_embeddings is None:
            raise RuntimeError(f"Embedding cache validation failed: {cache_file}")

        logger.info("🎉 Encoding finished in %.1fs. Saved memory-mapped cache to '%s'!", time.monotonic() - start, cache_file)
        
        # GIẢI PHÓNG GPU VRAM NGAY LẬP TỨC CHO QWEN LLM
        self.unload_dense_model()

    def unload_dense_model(self) -> None:
        """Unload embedding model from GPU memory to free VRAM for LLM."""
        if self._dense_model is not None:
            import torch
            del self._dense_model
            self._dense_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("🧹 Đã giải phóng hoàn toàn VRAM của model Embedding cho LLM!")

    def _decompose_queries(self, query: str) -> list[str]:
        """Create bounded experimental sub-queries while preserving intent words."""

        original = query.strip()
        queries = [original]

        def append_unique(candidate: str) -> None:
            normalized = " ".join(candidate.split())
            if (
                len(normalized) >= 10
                and normalized.casefold() not in {item.casefold() for item in queries}
                and len(queries) < 4
            ):
                queries.append(normalized)

        # Clean query
        clean_q = re.sub(r"\bnăm\s+(?:19|20)\d{2}\b", "", original, flags=re.IGNORECASE)
        clean_q = re.sub(
            r"\b(mới nhất|hiện hành|theo quy định mới)\b",
            "",
            clean_q,
            flags=re.IGNORECASE,
        )
        append_unique(clean_q)

        # Preserve the delimiter in the right-hand query; removing phrases such
        # as "bị phạt" or "thủ tục" destroys the legal intent being retrieved.
        split_pattern = re.compile(
            r"\b(?:thì có|có được|bị phạt|bao nhiêu|như thế nào|thủ tục|thẩm quyền)\b",
            flags=re.IGNORECASE,
        )
        match = split_pattern.search(original)
        if match is not None:
            append_unique(original[: match.start()])
            append_unique(original[match.start() :])
        return queries

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid") -> list[dict]:
        """Search parent records with optional, bounded multi-query retrieval."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be one of: bm25, dense, hybrid")

        N = self.N
        if N == 0:
            return []

        sub_queries = (
            self._decompose_queries(query)
            if self.enable_query_decomposition
            else [query.strip()]
        )

        candidate_ids = set()
        bm25_scores: defaultdict[int, float] = defaultdict(float)
        dense_scores_by_doc: defaultdict[int, float] = defaultdict(float)
        k = self.rrf_k

        # 1. BM25 Search cho từng Sub-Query
        if mode in {"bm25", "hybrid"}:
            for query_index, sq in enumerate(sub_queries):
                query_weight = 1.0 if query_index == 0 else 0.5
                tokens = _tokenize_query(sq)
                hits = self._disk_bm25.get_scores_and_docs(tokens, top_k=200)
                for rank, (doc_id, _score) in enumerate(hits, start=1):
                    bm25_scores[doc_id] += query_weight / (k + rank)
                    candidate_ids.add(doc_id)

        # 2. Dense Vector Search cho từng Sub-Query
        if mode in {"dense", "hybrid"} and self._doc_embeddings is not None:
            if self._dense_model is None:
                self._dense_model = SentenceTransformer(
                    str(self.embedding_model_path),
                    device="cpu",
                )

            for query_index, sq in enumerate(sub_queries):
                query_weight = 1.0 if query_index == 0 else 0.5
                q_vec = self._dense_model.encode(sq, normalize_embeddings=True)
                dense_scores = np.dot(self._doc_embeddings, q_vec)
                dense_limit = min(200, N)
                if dense_limit == N:
                    top_indices = np.arange(N)
                else:
                    top_indices = np.argpartition(
                        -dense_scores,
                        dense_limit - 1,
                    )[:dense_limit]
                top_sorted = sorted(top_indices, key=lambda i: dense_scores[i], reverse=True)
                for rank, idx in enumerate(top_sorted, start=1):
                    doc_id = int(idx)
                    dense_scores_by_doc[doc_id] += query_weight / (k + rank)
                    candidate_ids.add(doc_id)

        # 3. Combine with RRF over ONLY candidate_ids
        final_scores = []
        alpha = self.alpha

        for doc_id in candidate_ids:
            if mode == "bm25":
                rrf = bm25_scores[doc_id]
            elif mode == "dense":
                rrf = dense_scores_by_doc[doc_id]
            else:  # hybrid
                rrf = (
                    (1.0 - alpha) * bm25_scores[doc_id]
                    + alpha * dense_scores_by_doc[doc_id]
                )
            final_scores.append((rrf, doc_id))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        top_hits = final_scores[:top_k]

        results = []
        for rrf_score, doc_id in top_hits:
            doc = self._disk_bm25.get_doc(doc_id)
            if doc:
                doc["retrieval_score"] = float(rrf_score)
                results.append(doc)
        return results
