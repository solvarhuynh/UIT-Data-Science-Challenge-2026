#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "UDSC2026 local CI check"

if [ ! -f "pyproject.toml" ]; then
    echo "Not in project root directory"
    exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
    project_python=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    project_python=".venv/Scripts/python.exe"
else
    project_python="python"
fi

"$project_python" --version

echo "1. Ruff format check"
"$project_python" -m ruff format --check .

echo "2. Ruff lint check"
"$project_python" -m ruff check .

echo "3. Type check"
"$project_python" -m mypy src --no-warn-unused-configs

echo "4. Docstring style"
"$project_python" -m pydocstyle src

echo "5a. Tests except VectorDB native shard"
# FAISS 1.8 can abort on macOS after the same process has loaded Torch/SciPy.
# Run every test, but isolate all FAISS adapter tests and append coverage across
# both processes.  Retrieval tests added outside tests/unit/test_vector_db must
# stay in the same native shard.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$project_python" -m pytest -q \
    --ignore=tests/unit/test_vector_db \
    --ignore=tests/retrieval/test_vector_db_adapters.py \
    --cov=src/udsc2026 \
    --cov-report=

echo "5b. VectorDB native shard and combined coverage"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$project_python" -m pytest -q \
    tests/unit/test_vector_db \
    tests/retrieval/test_vector_db_adapters.py \
    --cov=src/udsc2026 \
    --cov-append \
    --cov-report=term-missing \
    --cov-fail-under=70

echo "6. Security"
"$project_python" -m bandit -q -r src

echo "7. TV5 host smoke"
"$project_python" scripts/smoke_test.py --mode host

echo "Local CI check completed"
