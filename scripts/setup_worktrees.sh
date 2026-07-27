#!/usr/bin/env bash
set -euo pipefail

git worktree add ../dsc2026-tv1-api -b feature/tv1-api
git worktree add ../dsc2026-tv2-retrieval -b feature/tv2-retrieval
git worktree add ../dsc2026-tv3-qa -b feature/tv3-qa
git worktree add ../dsc2026-tv4-ingestion -b feature/tv4-ingestion
git worktree add ../dsc2026-tv5-evaluation -b feature/tv5-evaluation

