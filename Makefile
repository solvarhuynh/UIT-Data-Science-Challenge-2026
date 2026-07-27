.PHONY: help install install-dev test lint format security docker-build docker-up docker-down docker-logs run-backend run-frontend ci-check ci-local clean

help:
	@echo "UDSC2026 - LegalIR & LegalQA"
	@echo "install        Install project dependencies"
	@echo "install-dev    Install development dependencies"
	@echo "test           Run pytest"
	@echo "lint           Run flake8"
	@echo "format         Format Python code"
	@echo "security       Run bandit"
	@echo "run-backend    Start FastAPI backend"
	@echo "run-frontend   Start React frontend"
	@echo "docker-up      Start backend and frontend"

install:
	pip install -e .

install-dev:
	pip install -r requirements_dev.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src tests || true

format:
	black src tests || true

security:
	bandit -r src || true

run-backend:
	uvicorn udsc2026.api.app:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

ci-local:
	./scripts/check-ci-local.sh

ci-check: format lint test security
	@echo "CI check completed"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/ .pytest_cache/ .coverage coverage.xml
