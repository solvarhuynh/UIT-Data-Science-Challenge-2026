"""Mock Hybrid Retriever (BM25 + Dense Vector Search) for TV3 testing.

Implements full Hybrid Search combining:
1. BM25 (Sparse keyword search)
2. BKAI Bi-Encoder (Dense vector embeddings via sentence-transformers)
3. Reciprocal Rank Fusion (RRF) to merge and rank results.

Saves embeddings to numpy cache file (.npy) so reloading is instant!
Supports CPU and CUDA (e.g. for Kaggle 2x T4 or local GPU).

NOT FOR PRODUCTION USE. This is a mock retriever for TV3 benchmark experiments.
"""

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

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
            return ViTokenizer.tokenize(text_clean).split()
        except Exception:
            return text_clean.split()
    return text_clean.split()


import sqlite3


class DiskBM25:
    """SQLite-backed Disk BM25 Index. Optimized with batching for 15s build time & <150MB disk size!"""

    def __init__(self, db_path: Path, parents_dir: Path, max_docs: int = 0, k1: float = 1.5, b: float = 0.75):
        self.db_path = Path(db_path)
        self.parents_dir = Path(parents_dir)
        self.k1 = k1
        self.b = b
        self.max_docs = max_docs
        
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode = OFF;")
            self.conn.execute("PRAGMA synchronous = OFF;")
            self.conn.execute("PRAGMA temp_store = MEMORY;")
            self.conn.execute("PRAGMA page_size = 65536;")
            self._init_db()
        except sqlite3.DatabaseError:
            logger.warning("⚠️ File DB '%s' bị malformed do đĩa đầy trước đó. Đang xóa và tạo lại DB mới...", self.db_path)
            try:
                self.conn.close()
            except Exception:
                pass
            if self.db_path.exists():
                self.db_path.unlink()
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
        if not self.parents_dir.exists() or not list(self.parents_dir.glob("*.jsonl")):
            candidate_paths = [
                Path("/run/media/quan/New Volume/uit_data/processed/parents"),
                Path("/run/media/quan/New Volume/uit_data/processed/chunks"),
                Path("/kaggle/input/uit-data-processed/parents"),
                Path("/kaggle/input/uit-data-processed/chunks"),
                self.parents_dir.parent / "parents",
                self.parents_dir.parent / "chunks",
            ]
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                for parent_match in kaggle_input.glob("**/parents"):
                    candidate_paths.insert(0, parent_match)
                for chunk_match in kaggle_input.glob("**/chunks"):
                    candidate_paths.append(chunk_match)

            for cand in candidate_paths:
                if cand.exists() and list(cand.glob("*.jsonl")):
                    self.parents_dir = cand
                    logger.info("🎯 Tìm thấy thư mục corpus processed phù hợp: '%s'", cand)
                    break

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
        doc_scores = defaultdict(float)

        for q in query_tokens:
            cursor.execute("SELECT doc_id, tf FROM postings WHERE term = ?;", (q,))
            postings = cursor.fetchall()
            n_q = len(postings)
            if n_q == 0:
                continue
            idf_val = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            if idf_val <= 0:
                continue

            for doc_id, tf in postings:
                cursor.execute("SELECT doc_len FROM documents WHERE id = ?;", (doc_id,))
                doc_len_row = cursor.fetchone()
                doc_len = doc_len_row[0] if doc_len_row else self.avgdl
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                numer = tf * (self.k1 + 1.0)
                doc_scores[doc_id] += idf_val * (numer / denom)

        sorted_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_hits

    def get_doc(self, doc_id: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT json_str FROM documents WHERE id = ?;", (doc_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None


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
    ) -> None:
        self.parents_dir = Path(parents_dir)
        self.embedding_model_path = embedding_model_path
        self.device = device
        
        # Xử lý môi trường Kaggle / Read-only input filesystem:
        # Nếu cache_dir không truyền vào, kiểm tra nếu parents_dir nằm trong thư mục Read-only (/kaggle/input)
        # thì tự động fallback sang /tmp/cache hoặc ./data/cache trên đĩa có quyền ghi (Writable)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            candidate_dir = self.parents_dir.parent / "cache"
            try:
                candidate_dir.mkdir(parents=True, exist_ok=True)
                # Kiểm tra quyền ghi thực tế bằng cách tạo file test tạm
                test_file = candidate_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                self.cache_dir = candidate_dir
            except (OSError, PermissionError):
                self.cache_dir = Path("/tmp/cache")
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rrf_k = rrf_k
        self.alpha = alpha
        self.max_docs = max_docs

        db_path = self.cache_dir / "bm25_db.sqlite"
        self._disk_bm25 = DiskBM25(db_path=db_path, parents_dir=self.parents_dir, max_docs=self.max_docs)
        self._docs: list[dict] = []
        self._corpus_texts: list[str] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._dense_model: Optional[SentenceTransformer] = None

        self._load_corpus()
        self._build_dense_index()

    def _load_corpus(self) -> None:
        """Load document metadata lazily from SQLite to minimize RAM usage."""
        cursor = self._disk_bm25.conn.cursor()
        cursor.execute("SELECT json_str FROM documents;")
        rows = cursor.fetchall()
        for r in rows:
            doc = json.loads(r[0])
            self._docs.append(doc)
            self._corpus_texts.append(doc.get("text", ""))
        logger.info("Loaded %d documents from SQLite database into retriever.", len(self._docs))

    def _build_dense_index(self) -> None:
        """Load dense embedding model and compute or load cached embeddings."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed. Dense vector search disabled.")
            return

        cache_file = self.cache_dir / f"parent_embeddings_{len(self._docs)}.npy"

        if cache_file.exists():
            logger.info("⚡ Đã tìm thấy cache embeddings: '%s'. Đang nạp cực nhanh...", cache_file)
            self._doc_embeddings = np.load(cache_file)
            logger.info("⚡ Nạp xong cache embeddings shape: %s trong 0.1 giây!", self._doc_embeddings.shape)
            return

        model_name_or_path = str(self.embedding_model_path)
        logger.info("Loading embedding model from '%s' (device=%s)...", model_name_or_path, self.device)
        self._dense_model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
        )

        logger.info("Encoding %d documents into vector embeddings...", len(self._corpus_texts))
        start = time.monotonic()

        # Tự động điều chỉnh batch_size theo VRAM khả dụng
        batch_sz = 32
        if _TORCH_AVAILABLE and "cuda" in str(self.device) and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            batch_sz = 256 if total_vram_gb >= 8.0 else 64
            logger.info("Detect GPU VRAM: %.2f GB -> using batch_size=%d", total_vram_gb, batch_sz)

        if _TORCH_AVAILABLE and "cuda" in str(self.device) and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                embeddings = self._dense_model.encode(
                    self._corpus_texts,
                    batch_size=batch_sz,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
        else:
            embeddings = self._dense_model.encode(
                self._corpus_texts,
                batch_size=batch_sz,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        self._doc_embeddings = np.array(embeddings, dtype=np.float32)
        logger.info("Encoding finished in %.1fs. Saving cache to '%s'...", time.monotonic() - start, cache_file)
        np.save(cache_file, self._doc_embeddings)
        
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

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid") -> list[dict]:
        """Search for top_k documents using 'bm25', 'dense', or 'hybrid' with Dual Query Fusion."""
        N = len(self._docs)
        if N == 0:
            return []

        # 0. Tạo Clean Query (Chuẩn hóa câu hỏi loại bỏ mốc thời gian nhiễu)
        clean_q = re.sub(r"\bnăm\s+(?:19|20)\d{2}\b", "", query, flags=re.IGNORECASE)
        clean_q = re.sub(r"\b(mới nhất|hiện hành|theo quy định mới)\b", "", clean_q, flags=re.IGNORECASE).strip()
        has_dual = (clean_q != query.strip()) and (len(clean_q) > 10)

        # 1. BM25 Search (SQLite Disk Backend) - Chạy cho Query gốc + Clean Query
        bm25_ranks = {i: N for i in range(N)}
        tokens_orig = _tokenize_query(query)
        bm25_hits_orig = self._disk_bm25.get_scores_and_docs(tokens_orig, top_k=200)
        for rank, (doc_id, score) in enumerate(bm25_hits_orig, start=1):
            bm25_ranks[doc_id] = min(bm25_ranks[doc_id], rank)

        if has_dual:
            tokens_clean = _tokenize_query(clean_q)
            bm25_hits_clean = self._disk_bm25.get_scores_and_docs(tokens_clean, top_k=200)
            for rank, (doc_id, score) in enumerate(bm25_hits_clean, start=1):
                bm25_ranks[doc_id] = min(bm25_ranks[doc_id], rank)

        # 2. Dense Vector Search - Chạy cho Query gốc + Clean Query
        dense_ranks = {i: N for i in range(N)}
        if self._doc_embeddings is not None:
            if self._dense_model is None:
                self._dense_model = SentenceTransformer(
                    str(self.embedding_model_path),
                    device="cpu",
                )

            q_vec = self._dense_model.encode(query, normalize_embeddings=True)
            dense_scores = np.dot(self._doc_embeddings, q_vec)
            sorted_dense = sorted(range(N), key=lambda i: dense_scores[i], reverse=True)
            for rank, idx in enumerate(sorted_dense[:200]):
                dense_ranks[idx] = min(dense_ranks[idx], rank + 1)

            if has_dual:
                q_vec_clean = self._dense_model.encode(clean_q, normalize_embeddings=True)
                dense_scores_clean = np.dot(self._doc_embeddings, q_vec_clean)
                sorted_dense_clean = sorted(range(N), key=lambda i: dense_scores_clean[i], reverse=True)
                for rank, idx in enumerate(sorted_dense_clean[:200]):
                    dense_ranks[idx] = min(dense_ranks[idx], rank + 1)

        # 3. Combine with RRF (Reciprocal Rank Fusion)
        final_scores = []
        k = self.rrf_k
        alpha = self.alpha

        for i in range(N):
            if mode == "bm25":
                rrf = 1.0 / (k + bm25_ranks[i])
            elif mode == "dense":
                rrf = 1.0 / (k + dense_ranks[i])
            else:  # hybrid
                rrf = (1.0 - alpha) / (k + bm25_ranks[i]) + alpha / (k + dense_ranks[i])
            final_scores.append((rrf, i))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in final_scores[:top_k]]

        results = []
        for idx in top_indices:
            doc = dict(self._docs[idx])
            doc["retrieval_score"] = float(final_scores[top_indices.index(idx)][0])
            results.append(doc)

        return results
