BACKEND_BIND_ADDRESS ?= 127.0.0.1
BACKEND_PORT ?= 8000
PIP_CONSTRAINT ?= requirements_runtime.txt

.PHONY: help install install-dev test lint format format-check typecheck docstyle security tv5-smoke docker-build docker-up docker-down docker-logs run-backend run-frontend ci-check ci-local clean

help:
	@echo "UDSC2026 - LegalIR & LegalQA"
	@echo "install        Install project dependencies"
	@echo "install-dev    Install development dependencies"
	@echo "test           Run pytest"
	@echo "lint           Run Ruff lint"
	@echo "format         Format Python code"
	@echo "format-check   Check Python formatting"
	@echo "typecheck      Run mypy"
	@echo "docstyle       Check Python docstrings"
	@echo "security       Run bandit"
	@echo "tv5-smoke      Run offline TV5 smoke checks"
	@echo "run-backend    Start FastAPI backend"
	@echo "run-frontend   Start React frontend"
	@echo "docker-up      Start backend, Qdrant, and Redis"

install:
	python -m pip install -c $(PIP_CONSTRAINT) -e ".[llm,rerank,retrieval]"

install-dev:
	python -m pip install -c $(PIP_CONSTRAINT) -r requirements_dev.txt
	python -m pip install -c $(PIP_CONSTRAINT) -e ".[llm,rerank,retrieval]"
	pre-commit install

test:
	python -m pytest -q --cov=src/udsc2026 --cov-report=term-missing --cov-fail-under=70

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	python -m mypy src --no-warn-unused-configs

docstyle:
	python -m pydocstyle src

security:
	bandit -r src

run-backend:
	python -m uvicorn udsc2026.api.app:app --reload --host $(BACKEND_BIND_ADDRESS) --port $(BACKEND_PORT)

run-frontend:
	cd "frontend/giao dien" && npm run dev

tv5-smoke:
	python scripts/ci_cd/smoke_test.py --mode host

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

ci-local:
	bash scripts/ci_cd/check-ci-local.sh

ci-check: format-check lint typecheck docstyle test security tv5-smoke
	@echo "CI check completed"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/ .pytest_cache/ .cache/pytest/ .coverage coverage.xml
