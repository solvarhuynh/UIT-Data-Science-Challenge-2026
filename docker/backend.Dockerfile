# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11.9-slim-bookworm

FROM python:${PYTHON_VERSION} AS wheel-builder

ARG BACKEND_EXTRAS=llm,rerank,retrieval
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml ./
COPY requirements_runtime.txt ./
COPY src ./src

# Seed the wheelhouse with the official CPU-only PyTorch wheel. Supplying this
# wheel to the full resolver avoids pulling unused NVIDIA runtime packages into
# the default CPU image. GPU deployments should use a dedicated image/profile.
RUN python -m pip install --upgrade pip build \
    && if [ -n "${BACKEND_EXTRAS}" ]; then \
        python -m pip wheel \
            --no-deps \
            --index-url "${PYTORCH_CPU_INDEX_URL}" \
            --wheel-dir /wheels \
            "torch==2.2.2"; \
    fi

# Build the application and all remaining runtime dependencies as wheels. The
# production default includes TV5 reranking and TV2 retrieval; set
# BACKEND_EXTRAS="" only for a deliberately core-only diagnostic image.
RUN if [ -n "${BACKEND_EXTRAS}" ]; then \
        python -m pip wheel \
            --constraint requirements_runtime.txt \
            --find-links /wheels \
            --wheel-dir /wheels \
            ".[${BACKEND_EXTRAS}]"; \
    else \
        python -m pip wheel \
            --constraint requirements_runtime.txt \
            --wheel-dir /wheels \
            .; \
    fi


FROM python:${PYTHON_VERSION} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
ARG BACKEND_EXTRAS=llm,rerank,retrieval

ENV HOME=/app \
    HF_HOME=/app/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /app \
        --no-create-home \
        --shell /usr/sbin/nologin \
        app \
    && mkdir -p /app/cache/huggingface /app/data/processed /app/models /app/output \
    && chown -R app:app /app

RUN --mount=type=bind,from=wheel-builder,source=/wheels,target=/wheels \
    if [ -n "${BACKEND_EXTRAS}" ]; then \
        python -m pip install \
            --no-index \
            --find-links=/wheels \
            "udsc2026[${BACKEND_EXTRAS}]"; \
    else \
        python -m pip install --no-index --find-links=/wheels udsc2026; \
    fi

COPY --chown=app:app configs ./configs
COPY --chown=app:app prompts ./prompts
COPY --chown=app:app scripts/evaluate.py ./scripts/evaluate.py
COPY --chown=app:app scripts/smoke_test.py ./scripts/smoke_test.py
COPY --chown=app:app scripts/write_submission.py ./scripts/write_submission.py

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["python", "-m", "uvicorn", "udsc2026.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
