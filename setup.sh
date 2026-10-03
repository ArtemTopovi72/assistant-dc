#!/usr/bin/env bash
# One-command setup for Linux/macOS: ./setup.sh [--no-models] [--no-start] [--cpu] [--dev]
# Safe to run again: every step skips what is already done.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "== installing uv (Python package manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ ! -x venv/bin/python ]; then
  echo "== creating venv (Python 3.13)"
  uv venv -p 3.13 venv
fi
# uv venvs ship without pip; setup.py drives uv directly.
exec venv/bin/python scripts/setup.py "$@"
