"""The shipped Madhouse cast: every character has a voice of its own.

A character with no voice falls back to the assistant's default reference
(Stepan_short.wav) -- «Сан» had an empty voice and spoke as Степан.
The reference .wav files are local (not in the repo): on a clean checkout
only the cast file itself is checked.
Run: venv/Scripts/python.exe tests/test_madhouse_cast_voices.py
"""
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cast = json.load(open(os.path.join(ROOT, "personalities", "madhouse_cast.json"), encoding="utf-8"))
cast = cast.get("characters", cast) if isinstance(cast, dict) else cast
voices = [c.get("voice") or "" for c in cast]
print("voices:", voices)
bad = 0


def check(name, cond, extra=""):
    global bad
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}"))
    bad += not cond


check("every cast member names a voice", all(voices), voices)
check("no two cast members share a voice", len(set(voices)) == len(voices), voices)
present = [v for v in voices if v and os.path.exists(os.path.join(ROOT, v))]
if present:     # the references are on this machine: each one must be
    check("every named voice file exists", len(present) == len(voices),
          sorted(set(voices) - set(present)))
else:
    print("SKIP  voice files are not on this machine (clean checkout)")
sys.exit(1 if bad else 0)
