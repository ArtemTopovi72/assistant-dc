#!/bin/bash
# Runs each test file passed as args, with a 90s timeout, records result.
PY="venv/Scripts/python.exe"
OUT="$1"
shift
for f in "$@"; do
  start=$(date +%s)
  timeout 90 "$PY" "$f" > "/tmp/out_$(basename $f).log" 2>&1
  code=$?
  end=$(date +%s)
  dur=$((end-start))
  if [ $code -eq 0 ]; then
    status="PASS"
  elif [ $code -eq 124 ]; then
    status="TIMEOUT"
  else
    status="FAIL($code)"
  fi
  echo "$status $dur"s"  $f" >> "$OUT"
done
