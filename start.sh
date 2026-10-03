#!/usr/bin/env bash
# Start Assistant DC (LM Studio / ComfyUI / Docker are started by the launcher).
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { echo "Not installed yet: run ./setup.sh first" >&2; exit 1; }
exec venv/bin/python scripts/launch_all.py "$@"
