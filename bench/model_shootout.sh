#!/usr/bin/env bash
# House model A/B on our own agent benches: Russian chat, tool-calling loop, intent gate.
#   bash bench/model_shootout.sh <model_id> [<model_id> ...]   -> outputs/shootout/<model>/*
cd "$(dirname "$0")/.."
for M in "$@"; do
  D="outputs/shootout/$M"; mkdir -p "$D"
  echo "=== $M  $(date +%T)"
  export MODEL_NAME="$M" PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 F5_TEST_RUN=
  # The probes do not load a model themselves: load it the way the app does (REST,
  # ctx 40960, parallel 1 -- `lms load --parallel 1` left the slot count at 4).
  if ! venv/Scripts/python.exe bench/load_like_app.py "$M" > "$D/load.log" 2>&1; then
    echo "load failed"; cat "$D/load.log"; continue
  fi
  venv/Scripts/python.exe bench/model_ru_probe.py --model "$M" --out "$D/ru.json" > "$D/ru.log" 2>&1
  tail -3 "$D/ru.log"
  venv/Scripts/python.exe bench/tc_run.py --reps 1 --out "$D/tc.json" > "$D/tc.log" 2>&1
  grep -A14 "^=====" "$D/tc.log" | head -16
  venv/Scripts/python.exe bench/intent_live.py > "$D/intent.log" 2>&1
  tail -1 "$D/intent.log"
done
echo "=== done $(date +%T)"
