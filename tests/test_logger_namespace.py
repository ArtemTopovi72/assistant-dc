"""A module that logs through logging.getLogger(__name__) lands outside the "assistant" tree, the only
one with a file handler: intent reads, GPU-lock waits and the LLM gate wrote nowhere, and a night of
«the reply promised a plan that never came» could not be traced (2026-10-18).

Run: venv/Scripts/python.exe tests/test_logger_namespace.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = ("agent", "bot", "core", "gui", "imaging", "knowledge", "media", "personalities", "research", "sandbox", "services", "voice")
bad = []
for d, dirs, files in (w for pk in PKGS for w in os.walk(os.path.join(ROOT, pk))):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(d, f)
            for n, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                if re.search(r"getLogger\(\s*__name__\s*\)", line):
                    bad.append(f"{os.path.relpath(p, ROOT)}:{n}")
print(("FAIL  " if bad else "PASS  ") + "every module logs under the assistant tree  " + ", ".join(bad[:8]))
sys.exit(1 if bad else 0)
