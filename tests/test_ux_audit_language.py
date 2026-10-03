"""ux_audit's wrong-language rule: a real drift is flagged, a requested English reply is not."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import ux_audit as U
import intent   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py
intent.STUB = {"can you answer in English from now on?": {"language_mode": "en"}}.get

row = lambda k, ts, t: {"kind": k, "ts": ts, "data": {"text": t, "method": "sendMessage"}}
rules = lambda rows: [x[2] for x in U.audit_chat(rows, set())]
assert rules([row("in", "2026-09-29 00:00:00", "какая столица Перу?"),
              row("out", "2026-09-29 00:00:02", "The capital of Peru is Lima.")]) == ["wrong-language"]
assert rules([row("in", "2026-09-29 00:00:00", "привет"),
              row("in", "2026-09-29 00:00:05", "can you answer in English from now on?"),
              row("out", "2026-09-29 00:00:06", "Of course!"),
              row("in", "2026-09-29 00:00:10", "а еще одну?"),
              row("out", "2026-09-29 00:00:12", "Try Sapiens by Harari.")]) == []
print("PASS ux audit language rule")
