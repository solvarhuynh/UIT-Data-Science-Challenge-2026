#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

test_type="${1:-working}"
if [[ -x ".venv/bin/python" ]]; then
    project_python=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    project_python=".venv/Scripts/python.exe"
else
    project_python="python"
fi

case "$test_type" in
    working)
        echo "[INFO] Running the complete test suite"
        "$project_python" -m pytest -q -rs
        ;;
    coverage)
        echo "[INFO] Running the complete suite with source coverage"
        "$project_python" -m pytest \
            --cov=src/udsc2026 \
            --cov-report=term \
            --cov-report=html:htmlcov \
            --cov-report=xml \
            --cov-fail-under=70 \
            -q
        ;;
    unit)
        echo "[INFO] Running unit tests"
        "$project_python" -m pytest tests/unit -q -rs
        ;;
    quick)
        echo "[INFO] Running fast contract/API sanity tests"
        "$project_python" -m pytest \
            tests/integration/test_basic.py \
            tests/integration/test_api_simple.py \
            tests/integration/test_utils.py \
            -x \
            --tb=line
        ;;
    *)
        echo "[ERROR] Unknown test type: $test_type" >&2
        echo "[INFO] Available types: working, coverage, unit, quick" >&2
        exit 2
        ;;
esac

echo "[INFO] Test execution completed"
