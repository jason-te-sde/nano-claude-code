#!/usr/bin/env bash
# scripts/preflight.sh — everything CI runs, locally, in CI's order.
set -euo pipefail
cd "$(dirname "$0")/.."

# This repository must be its own, not a subdirectory of a larger one. If that ever
# stops being true, every path and every push in this project is pointing somewhere
# unintended, and it is worth failing loudly before anything else runs.
if [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
  echo "preflight: $(pwd -P) is not the repository root ($(git rev-parse --show-toplevel))" >&2
  exit 1
fi

bash scripts/check-hygiene.sh
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -q --cov=nanoclaude --cov-report=term-missing
echo "preflight: ok"
