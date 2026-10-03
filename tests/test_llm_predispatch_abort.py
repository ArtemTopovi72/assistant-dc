"""Regression: a failure BEFORE the request is dispatched must not be retried.

Observed while proving the webp fix: a context missing `api_lock` produced three
identical "LM Studio request failed (attempt N/3)" lines with backoff sleeps
between them. throttle_external_calls runs before any bytes leave the process, so
anything it raises is our defect — asking the server again cannot help, and the
retries only turn one bug into three failures and a delay.

Genuine transport failures (the ones a retry CAN fix) must still get all
LLM_MAX_RETRIES attempts.

Run: venv/Scripts/python.exe tests/test_llm_predispatch_abort.py
"""
import os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.disable(logging.CRITICAL)

import llm
from config import LLM_MAX_RETRIES

ok = fail = 0

def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"PASS {label}")
    else:
        fail += 1
        print(f"FAIL {label} {extra}")


class Ctx:
    """Deliberately missing api_lock / api_min_interval — the observed bug."""
    model_name = "test-model"
    no_think = True
    def set_stage(self, *a, **k): pass


def run(monkey_throttle, monkey_stream):
    orig_t, orig_s = llm.throttle_external_calls, llm._stream_chat
    orig_delay = llm.LLM_RETRY_BASE_DELAY
    llm.throttle_external_calls = monkey_throttle
    llm._stream_chat = monkey_stream
    llm.LLM_RETRY_BASE_DELAY = 0.0          # keep the test fast
    try:
        return llm.send_to_lm_studio(Ctx(), [{"role": "user", "content": "hi"}])
    finally:
        llm.throttle_external_calls, llm._stream_chat = orig_t, orig_s
        llm.LLM_RETRY_BASE_DELAY = orig_delay


# ── 1. pre-dispatch failure: exactly ONE attempt, no stream call ─────────────
calls = {"throttle": 0, "stream": 0}

def bad_throttle(ctx):
    calls["throttle"] += 1
    raise AttributeError("'Ctx' object has no attribute 'api_lock'")

def never_stream(ctx, payload):
    calls["stream"] += 1
    return {"role": "assistant", "content": "should never happen"}

t0 = time.time()
out = run(bad_throttle, never_stream)
elapsed = time.time() - t0

check("pre-dispatch failure aborts after ONE attempt", calls["throttle"] == 1,
      f"throttled {calls['throttle']}x")
check("no request was dispatched", calls["stream"] == 0, f"{calls['stream']} streams")
check("the caller gets None (no fabricated reply)", out is None, repr(out))
check("it returns promptly, without the backoff ladder", elapsed < 2.0, f"{elapsed:.2f}s")

# ── 2. a TRANSPORT failure is still retried the full number of times ─────────
calls2 = {"throttle": 0, "stream": 0}

def good_throttle(ctx):
    calls2["throttle"] += 1

def flaky_stream(ctx, payload):
    calls2["stream"] += 1
    raise ConnectionError("connection reset by peer")

out2 = run(good_throttle, flaky_stream)
check(f"transport failure still retries {LLM_MAX_RETRIES}x",
      calls2["stream"] == LLM_MAX_RETRIES, f"{calls2['stream']} attempts")
check("exhausted retries return None", out2 is None, repr(out2))

# ── 3. a transport failure that RECOVERS returns the reply ───────────────────
calls3 = {"n": 0}

def recovering_stream(ctx, payload):
    calls3["n"] += 1
    if calls3["n"] == 1:
        raise ConnectionError("first attempt drops")
    return {"role": "assistant", "content": "recovered answer"}

out3 = run(good_throttle, recovering_stream)
check("a recovering call returns the reply",
      isinstance(out3, dict) and out3.get("content") == "recovered answer", repr(out3))
check("recovery took exactly two attempts", calls3["n"] == 2, calls3["n"])

# ── 4. the happy path is untouched ───────────────────────────────────────────
calls4 = {"n": 0}

def ok_stream(ctx, payload):
    calls4["n"] += 1
    return {"role": "assistant", "content": "fine"}

out4 = run(good_throttle, ok_stream)
check("success on the first attempt", calls4["n"] == 1 and out4.get("content") == "fine",
      repr(out4))

print(f"\n{ok}/{ok + fail} checks passed")
sys.exit(1 if fail else 0)
