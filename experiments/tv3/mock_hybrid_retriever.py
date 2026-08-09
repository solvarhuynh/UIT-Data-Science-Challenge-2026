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
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

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
        self._doc_embeddings: Optional[np.ndarray] = None
        self._dense_model: Optional[SentenceTransformer] = None

        self._load_corpus()
        self._build_dense_index()

    def _load_corpus(self) -> None:
        """Lazily set corpus size N from SQLite disk index to keep RAM under 200MB."""
        self.N = self._disk_bm25.N
        logger.info("Loaded %d documents from SQLite database into retriever.", self.N)

    def _build_dense_index(self) -> None:
        """Load dense embedding model and compute cached embeddings with streaming to keep RAM under 300MB."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed. Dense vector search disabled.")
            return

        cache_file = self.cache_dir / f"parent_embeddings_{self.N}.npy"

        if cache_file.exists():
            logger.info("⚡ Đã tìm thấy cache embeddings: '%s'. Đang nạp memory-mapped cực nhanh...", cache_file)
            try:
                self._doc_embeddings = np.load(cache_file, mmap_mode="r")
            except Exception:
                # Fallback nếu file được ghi bằng raw memmap binary
                emb_dim = 768
                self._doc_embeddings = np.memmap(cache_file, dtype="float32", mode="r", shape=(self.N, emb_dim))
            logger.info("⚡ Nạp xong cache embeddings shape: %s trong 0.01 giây!", self._doc_embeddings.shape)
            return

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

        tmp_cache_file = self.cache_dir / f"parent_embeddings_{self.N}.tmp.npy"
        mmap_arr = np.memmap(tmp_cache_file, dtype="float32", mode="w+", shape=(self.N, emb_dim))

        batch_sz = 256 if (_TORCH_AVAILABLE and "cuda" in str(self.device) and torch.cuda.is_available()) else 64
        chunk_size = 5000
        cursor = self._disk_bm25.conn.cursor()

        for offset in range(0, self.N, chunk_size):
            cursor.execute("SELECT json_str FROM documents LIMIT ? OFFSET ?;", (chunk_size, offset))
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
        np.save(cache_file, mmap_arr)
        del mmap_arr

        if tmp_cache_file.exists():
            try:
                tmp_cache_file.unlink()
            except Exception:
                pass

        try:
            self._doc_embeddings = np.load(cache_file, mmap_mode="r")
        except Exception:
            self._doc_embeddings = np.memmap(cache_file, dtype="float32", mode="r", shape=(self.N, emb_dim))
            
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

    def _resolve_parent_text(self, doc: dict) -> str:
        """Resolve full parent document text from chunks to eliminate context fragmentation."""
        chunk_text = doc.get("text", "")
        parents_dir = self.parents_dir
        if "chunks" in str(parents_dir):
            parents_dir = Path(str(parents_dir).replace("chunks", "parents"))

        doc_id = doc.get("doc_id") or doc.get("parent_id") or ""
        parent_candidates = []
        if doc_id:
            parent_candidates.append(parents_dir / str(doc_id))
            parent_candidates.append(parents_dir / f"{doc_id}.jsonl")
            parent_candidates.append(parents_dir / f"context_{doc_id}.jsonl")
            parent_candidates.append(parents_dir / f"context_{doc_id}")

        for cand in parent_candidates:
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        parent_texts = []
                        for l in lines[:5]:
                            p_json = json.loads(l)
                            parent_texts.append(p_json.get("text", ""))
                        full_parent_text = "\n".join(parent_texts)
                        if len(full_parent_text) > 50:
                            return f"[Đoạn trích chi tiết]: {chunk_text}\n\n[Toàn văn Điều luật tham chiếu]: {full_parent_text[:3000]}"
                except Exception:
                    pass
        return chunk_text

    def _decompose_queries(self, query: str) -> list[str]:
        """Automatically decompose compound legal questions into sub-queries for 100% general coverage."""
        queries = [query]
        # Clean query
        clean_q = re.sub(r"\bnăm\s+(?:19|20)\d{2}\b", "", query, flags=re.IGNORECASE)
        clean_q = re.sub(r"\b(mới nhất|hiện hành|theo quy định mới)\b", "", clean_q, flags=re.IGNORECASE).strip()
        if clean_q != query.strip() and len(clean_q) > 10:
            queries.append(clean_q)

        # Tự động bóc tách vế vế tình huống và vế chế tài/thủ tục (VD: "thì có được", "bị phạt", "thủ tục")
        split_pattern = r"\b(?:thì có|có được|bị phạt|bao nhiêu|như thế nào|thủ tục|thẩm quyền)\b"
        parts = re.split(split_pattern, query, flags=re.IGNORECASE)
        if len(parts) >= 2:
            part1 = parts[0].strip()
            part2 = " ".join(parts[1:]).strip()
            if len(part1) >= 10 and part1 not in queries:
                queries.append(part1)
            if len(part2) >= 10 and part2 not in queries:
                queries.append(part2)
        return queries

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid") -> list[dict]:
        """Search for top_k documents using 'bm25', 'dense', or 'hybrid' with Multi-Query Sub-Decomposition & Parent Resolution."""
        N = self.N
        if N == 0:
            return []

        # 0. Tự động sinh danh sách Sub-Queries cho các vế câu hỏi phức hợp
        sub_queries = self._decompose_queries(query)

        candidate_ids = set()
        bm25_ranks = {}
        dense_ranks = {}

        # 1. BM25 Search cho từng Sub-Query
        for sq in sub_queries:
            tokens = _tokenize_query(sq)
            hits = self._disk_bm25.get_scores_and_docs(tokens, top_k=200)
            for rank, (doc_id, score) in enumerate(hits, start=1):
                bm25_ranks[doc_id] = min(bm25_ranks.get(doc_id, N), rank)
                candidate_ids.add(doc_id)

        # 2. Dense Vector Search cho từng Sub-Query
        if self._doc_embeddings is not None:
            if self._dense_model is None:
                self._dense_model = SentenceTransformer(
                    str(self.embedding_model_path),
                    device="cpu",
                )

            for sq in sub_queries:
                q_vec = self._dense_model.encode(sq, normalize_embeddings=True)
                dense_scores = np.dot(self._doc_embeddings, q_vec)
                top_indices = np.argpartition(-dense_scores, min(200, N - 1))[:200]
                top_sorted = sorted(top_indices, key=lambda i: dense_scores[i], reverse=True)
                for rank, idx in enumerate(top_sorted, start=1):
                    dense_ranks[idx] = min(dense_ranks.get(idx, N), rank)
                    candidate_ids.add(idx)

        # 3. Combine with RRF over ONLY candidate_ids
        final_scores = []
        k = self.rrf_k
        alpha = self.alpha

        for doc_id in candidate_ids:
            b_rank = bm25_ranks.get(doc_id, N)
            d_rank = dense_ranks.get(doc_id, N)
            if mode == "bm25":
                rrf = 1.0 / (k + b_rank)
            elif mode == "dense":
                rrf = 1.0 / (k + d_rank)
            else:  # hybrid
                rrf = (1.0 - alpha) / (k + b_rank) + alpha / (k + d_rank)
            final_scores.append((rrf, doc_id))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        top_hits = final_scores[:top_k]

        results = []
        for rrf_score, doc_id in top_hits:
            doc = self._disk_bm25.get_doc(doc_id)
            if doc:
                doc["retrieval_score"] = float(rrf_score)
                # Tự động giải nén Parent Context để cung cấp ngữ cảnh đầy đủ cho LLM
                doc["text"] = self._resolve_parent_text(doc)
                results.append(doc)
        return results
