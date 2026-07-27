FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
COPY download_models.py ./download_models.py

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "udsc2026.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
