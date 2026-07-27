# Docker Deployment cho LegalIR & LegalQA

## 1. Mục tiêu đóng gói

Hệ thống cần chạy được toàn bộ pipeline local gồm React frontend, FastAPI backend và model inference cho Qwen 3 1.7B cùng BKAI Bi-encoder. Docker setup phải phục vụ hai chế độ:

- Development: tách service để debug nhanh frontend, backend, VectorDB và model path.
- Submission/CodaLab: giảm số container, giảm dung lượng image và tránh copy model weights hoặc vector index lớn vào image.

Nguyên tắc mặc định là backend giữ vai trò orchestration, model inference chạy local trong cùng container backend ở chế độ nộp bài, còn frontend được build thành static asset để serve qua backend hoặc reverse proxy. Cách này giảm rủi ro khi CodaLab chỉ cho chạy một entrypoint hoặc giới hạn tài nguyên giữa nhiều container.

## 2. Kiến trúc service đề xuất

### 2.1 Development compose

Trong môi trường phát triển, nên tách service để mỗi thành phần restart độc lập:

```text
frontend  React dev server, hot reload, gọi backend qua HTTP
backend   FastAPI, orchestration, load Qwen/BKAI hoặc gọi model adapter local
vectordb  Qdrant nếu dùng VectorDB service; có thể bỏ nếu dùng FAISS local
```

Compose dev ưu tiên mount volume:

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    command: uvicorn udsc2026.api.app:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      UDSC2026_ENV: development
      MODEL_EMBEDDER_PATH: /app/models/bkai-bi-encoder
      MODEL_LLM_PATH: /app/models/qwen3-legal
      VECTOR_STORE_PATH: /app/data/vector_store
      QDRANT_URL: http://vectordb:6333
    volumes:
      - ./configs:/app/configs:ro
      - ./data:/app/data
      - ./models:/app/models:ro
      - ./prompts:/app/prompts:ro
    depends_on:
      - vectordb

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    command: npm run dev -- --host 0.0.0.0 --port 5173
    ports:
      - "5173:5173"
    environment:
      VITE_API_BASE_URL: http://localhost:8000
    depends_on:
      - backend

  vectordb:
    image: qdrant/qdrant:v1.10.1
    ports:
      - "6333:6333"
    volumes:
      - ./data/vector_store/qdrant:/qdrant/storage
```

Nếu chọn FAISS thay cho Qdrant, bỏ service `vectordb` và mount `./data/vector_store/faiss` vào backend. Điều này đơn giản hơn khi nộp bài vì chỉ còn một process chính.

### 2.2 Submission/CodaLab compose

Với CodaLab, nên ưu tiên một container chính để tránh overhead, lỗi networking giữa container và giới hạn RAM bị chia nhỏ. Frontend build static trước, backend serve API và static files; Qwen/BKAI được load trong cùng process hoặc cùng container qua adapter local.

```yaml
services:
  app:
    build:
      context: .
      dockerfile: docker/submission.Dockerfile
    command: python -m udsc2026.api.run_submission
    ports:
      - "8000:8000"
    environment:
      UDSC2026_ENV: submission
      MODEL_EMBEDDER_PATH: /app/models/bkai-bi-encoder
      MODEL_LLM_PATH: /app/models/qwen3-legal
      VECTOR_STORE_PATH: /app/data/vector_store
      INFERENCE_DEVICE: auto
      LLM_QUANTIZATION: int8
      MAX_CONTEXT_CHUNKS: "5"
    volumes:
      - ./models:/app/models:ro
      - ./data/vector_store:/app/data/vector_store:ro
      - ./configs:/app/configs:ro
```

Không nên tách Qwen 3 và BKAI thành microservices riêng trong submission trừ khi BTC cho phép multi-container rõ ràng và tài nguyên đủ lớn. Microservice giúp scale/debug tốt hơn, nhưng trong môi trường chấm điểm thường làm tăng memory duplication, cold start và rủi ro lỗi network nội bộ.

## 3. Multi-stage Dockerfile

### 3.1 Backend + frontend static cho submission

Mục tiêu là build frontend ở stage Node, build Python wheel ở stage builder, rồi copy artifact sang runtime slim. Runtime không chứa npm cache, test tool, source thừa hoặc model weights.

```dockerfile
# Stage 1: frontend build
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Stage 2: python build
FROM python:3.11-slim AS python-builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip build
COPY pyproject.toml ./
COPY src ./src
RUN python -m build --wheel

# Stage 3: runtime
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

COPY configs ./configs
COPY prompts ./prompts
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# models/ và data/vector_store/ được mount lúc chạy, không copy vào image.
EXPOSE 8000
CMD ["uvicorn", "udsc2026.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Nếu project chưa đóng gói dependency runtime trong `pyproject.toml`, có thể thay stage Python bằng `pip install -r requirements.txt`. Tuy nhiên với submission nên tách `requirements.txt` runtime khỏi `requirements_dev.txt` để không cài pytest, lint, locust, build tools vào image chạy thật.

### 3.2 Backend dev Dockerfile

Dockerfile dev có thể giữ editable install và mount source:

```dockerfile
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY pyproject.toml ./
COPY requirements_dev.txt ./
RUN pip install --no-cache-dir -r requirements_dev.txt
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "udsc2026.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. Quản lý model và dữ liệu lớn

Không copy các thư mục sau vào Docker image:

```text
models/
data/raw/
data/processed/
data/vector_store/
```

Thay vào đó:

- Mount `models/` read-only vào `/app/models`.
- Mount `data/vector_store/` read-only trong submission nếu index đã build sẵn.
- Với Qdrant, mount storage folder thay vì rebuild collection khi container start.
- Ghi rõ checksum/version model trong `configs/models.yaml` hoặc tài liệu submission để tái tạo.

Image nộp bài chỉ nên chứa code, config nhỏ, prompts và frontend static build.

## 5. Chiến lược RAM/VRAM

### 5.1 Nguyên tắc load model

Qwen 3 1.7B là thành phần chiếm bộ nhớ lớn nhất. BKAI Bi-encoder nhỏ hơn nhưng vẫn tạo áp lực RAM/VRAM khi load cùng lúc với VectorDB. Backend cần load theo thứ tự và lazy-load khi có thể:

1. Load config và vector index metadata.
2. Load embedder khi request retrieval đầu tiên đến hoặc khi warmup rõ ràng.
3. Load LLM sau embedder, ưu tiên quantization nếu thiếu VRAM.
4. Chỉ giữ một instance cho mỗi model trong process, dùng dependency/app state để tái sử dụng.

Không nên để mỗi worker Uvicorn/Gunicorn load một bản model riêng. Trong môi trường giới hạn, chạy `workers=1` và dùng async/concurrency nhẹ tốt hơn nhiều worker nhân đôi RAM.

### 5.2 CPU/RAM fallback

Khi không có GPU hoặc VRAM thấp:

- Dùng quantization `int8` hoặc `int4` cho Qwen nếu backend inference hỗ trợ.
- Giảm `max_new_tokens`, `max_context_chunks`, `top_k` và kích thước prompt.
- Dùng FAISS local hoặc BM25 in-process nếu Qdrant service vượt ngân sách RAM.
- Giữ embedding batch size nhỏ, ví dụ `8` hoặc `16` trên CPU.
- Tắt rerank hoặc giảm `top_n` nếu Cross-Encoder làm tăng latency/memory.

### 5.3 GPU/VRAM

Khi có GPU:

- Ưu tiên đặt Qwen trên GPU, embedder có thể chạy CPU nếu VRAM thiếu.
- Nếu embedder chạy GPU, unload hoặc chuyển về CPU sau khi build index offline; online query embedding thường nhẹ hơn generation.
- Dùng `torch_dtype=float16` hoặc `bfloat16` nếu GPU hỗ trợ.
- Tránh vLLM nếu VRAM không đủ cho KV cache; `transformers` baseline với quantization thường dễ kiểm soát hơn cho 1.7B trong môi trường nhỏ.

### 5.4 VectorDB

Qdrant phù hợp development và hệ thống có RAM ổn định. Với CodaLab, FAISS local hoặc Qdrant embedded/file-backed có thể đơn giản hơn:

```text
Qdrant service: dễ quan sát, API rõ, nhưng thêm container và RAM nền.
FAISS local: ít overhead, dễ đóng gói một container, nhưng cần code load index cẩn thận.
BM25 local: nên lưu index đã build sẵn để tránh cold start dài.
```

Nếu dùng Qdrant trong submission, collection nên được build trước và mount vào container. Startup không nên tự ingest toàn bộ `data/raw/`.

## 6. Khuyến nghị cấu hình nộp bài

Cấu hình mặc định cho CodaLab nên là:

```text
1 container app
1 FastAPI process
1 Uvicorn worker
Frontend static served by backend
Models mounted read-only
Vector index mounted read-only
Qwen quantized nếu VRAM/RAM không chắc chắn
top_k và max_context_chunks có giới hạn cấu hình
```

Các biến môi trường tối thiểu:

```text
MODEL_EMBEDDER_PATH=/app/models/bkai-bi-encoder
MODEL_LLM_PATH=/app/models/qwen3-legal
VECTOR_STORE_PATH=/app/data/vector_store
INFERENCE_DEVICE=auto
LLM_QUANTIZATION=int8
MAX_RETRIEVAL_TOP_K=20
MAX_CONTEXT_CHUNKS=5
MAX_NEW_TOKENS=512
```

## 7. Checklist trước khi build submission

- Docker image không chứa model weights hoặc vector store lớn.
- Container start không rebuild index từ `data/raw/`.
- `/health` không bắt buộc load Qwen; chỉ kiểm tra app/config cơ bản.
- Warmup endpoint hoặc startup task có timeout rõ ràng nếu cần load model trước.
- `RetrievalHit` dùng contract chung từ `src/udsc2026/contracts/retrieval.py`.
- Log không ghi toàn bộ context/model output quá dài làm phình disk.
- Chạy thử end-to-end với giới hạn giống BTC: RAM, VRAM, CPU, timeout và số request mẫu.
