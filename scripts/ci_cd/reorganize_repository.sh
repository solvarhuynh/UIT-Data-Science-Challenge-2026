#!/usr/bin/env bash
set -euo pipefail

# Run from repository root: bash scripts/ci_cd/reorganize_repository.sh
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mkdir -p scripts/ci_cd scripts/data_prep scripts/evaluation scripts/submission \
  tests/integration docs/members docs/competition docs/project

move_if_present() {
  local source="$1" destination="$2"
  if [[ -f "$source" ]]; then
    mv -f "$source" "$destination/"
  fi
}

for file in check-all.ps1 check-tv5.ps1 check-ci-local.sh run_working_tests.sh \
  setup_worktrees.sh quality_gate.py smoke_test.py; do
  move_if_present "scripts/$file" scripts/ci_cd
done

for file in build_bm25.py index_chunks.py generate_mock_btc_data.py verify_embedding.py; do
  move_if_present "scripts/$file" scripts/data_prep
done

for file in evaluate.py evaluate_legal_ir.py evaluate_legal_qa.py \
  audit_legal_ir_warmup.py audit_legal_qa_warmup.py; do
  move_if_present "scripts/$file" scripts/evaluation
done

for pattern in 'make_*.py' 'write_*.py' 'validate_*.py'; do
  for file in scripts/$pattern; do
    [[ -f "$file" ]] && mv -f "$file" scripts/submission/
  done
done

for file in tests/test_*.py; do
  [[ -f "$file" ]] && mv -f "$file" tests/integration/
done

for file in docs/tv*_*.md; do
  [[ -f "$file" ]] && mv -f "$file" docs/members/
done
for file in docs/*.docx; do
  [[ -f "$file" ]] && mv -f "$file" docs/competition/
done
move_if_present docs/engineering_execution_plan.md docs/project

echo "Repository reorganization completed."
