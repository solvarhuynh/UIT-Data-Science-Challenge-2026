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

echo "5. Tests"
"$project_python" -m pytest -q --cov=src/udsc2026 --cov-report=term-missing --cov-fail-under=70

echo "6. Security"
"$project_python" -m bandit -q -r src

echo "7. TV5 host smoke"
"$project_python" scripts/smoke_test.py --mode host

echo "Local CI check completed"
