#!/bin/bash

set -e

cd "$(dirname "$0")/.."

echo "UDSC2026 local CI check"

if [ ! -f "pyproject.toml" ]; then
    echo "Not in project root directory"
    exit 1
fi

python --version

echo "1. Format check"
black --check src tests

echo "2. Lint check"
flake8 src tests --count --statistics

echo "3. Tests"
pytest tests/ -v --tb=short

echo "4. Security"
bandit -r src -f text

echo "Local CI check completed"
