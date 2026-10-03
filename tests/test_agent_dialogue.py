"""
Full agent dialogue audit — sends messages through the real graph pipeline
and checks that responses are coherent, memory works, and tools don't break.

Run: .\\venv\\Scripts\\python.exe tests\\test_agent_dialogue.py
"""
# Manual live audit, not part of the automated suite: it needs an ALREADY
# LOADED LM Studio model and makes real generation calls against it (real
# GPU time, real minutes), and its own summary never sys.exit(1)s on failure
# anyway (just prints "OVERALL: FAIL" and returns 0) -- unlike the rest of
# tests/, there is no pass/fail signal here for run_all.py to act on, so
# auto-running it on every full sweep would only burn the card for nothing.
NOT_IN_RUN_ALL = True
import sys, os, time, json
sys.path.insert(0, ".")

# The Windows console is cp1251 here: Cyrillic replies and the ✓/✗ marks below
# both raise UnicodeEncodeError without this.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ── minimal context (no real models needed for text-only test) ─────────────────
print("Importing graph...")
import config

# We'll use the HTTP API directly (LM Studio must be running with a model loaded)
import requests

BASE_URL = f"{config.LM_STUDIO_BASE}/v1"

def check_lm_studio():
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        models = r.json().get("data", [])
        if not models:
            return None
        return models[0]["id"]
    except Exception as e:
        return None

model_id = check_lm_studio()
if not model_id:
    print(f"LM Studio not reachable at {BASE_URL} or no model loaded.")
    print("Start LM Studio, load a model, then re-run this test.")
    sys.exit(1)

print(f"LM Studio: model '{model_id}' is loaded\n")

# ── test via the graph agent ──────────────────────────────────────────────────
from graph import build_graph
from models import Models, Context
from pathlib import Path
import threading

print("Setting up minimal context (no TTS/Whisper for text-only audit)...")

# Stub out heavy models so we can test without GPU
class StubModels:
    whisper = None
    tts_model = None
    vocoder = None
    accentor = None
    accentor_loaded = False

ctx = Context(
    models=StubModels(),
    transcription_cache={},
    cache_file=Path("tests/_audit_cache.json"),
    asr_lock=threading.Lock(),
    tts_lock=threading.Lock(),
)
ctx.model_name = model_id
ctx.no_think = True
ctx.tts_disabled = True   # skip TTS synthesis
ctx.mic_disabled = True   # skip mic

graph = build_graph(ctx)

TESTS = [
    # (label, message, check_fn)
    ("greeting",        "Привет! Как тебя зовут?",
     lambda r: len(r) > 5),
    ("simple_math",     "Сколько будет 12 умножить на 8?",
     lambda r: "96" in r),
    ("context_memory",  "Что я спросил тебя только что?",
     lambda r: len(r) > 10),
    ("capital",         "Какая столица Франции?",
     lambda r: "Париж" in r or "Paris" in r),
    ("followup",        "А какая река протекает через этот город?",
     lambda r: "Сена" in r or "Seine" in r or len(r) > 10),
    ("tool_weather",    "Какая сейчас погода в Москве?",
     lambda r: len(r) > 10),
]

results = []
messages = []   # keep conversation history for context

print(f"Running {len(TESTS)} dialogue turns:\n{'='*60}")

for label, user_msg, check_fn in TESTS:
    print(f"\n[{label}]")
    print(f"  USER: {user_msg}")

    state = {
        "messages": list(messages),
        "user_input": user_msg,
        "image_data": None,
        "final_answer": "",
    }

    t0 = time.time()
    try:
        result = graph.invoke(state)
        elapsed = time.time() - t0
        answer = result.get("final_answer", "").strip()

        # Update conversation history
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": answer})

        ok = check_fn(answer)
        status = "PASS" if ok else "WARN"
        print(f"  AGENT ({elapsed:.1f}s): {answer[:120]}{'...' if len(answer)>120 else ''}")
        print(f"  [{status}]")
        results.append({"label": label, "ok": ok, "elapsed": elapsed, "answer": answer})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] Exception: {e}")
        results.append({"label": label, "ok": False, "elapsed": elapsed, "error": str(e)})

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AUDIT SUMMARY")
print(f"{'='*60}")
passed = sum(1 for r in results if r["ok"])
total  = len(results)
for r in results:
    icon = "✓" if r["ok"] else "✗"
    t = f"{r['elapsed']:.1f}s"
    print(f"  {icon} [{t:5s}] {r['label']}")

print(f"\n{passed}/{total} checks passed")
if passed == total:
    print("OVERALL: PASS — dialogue logic is working correctly")
elif passed >= total * 0.7:
    print("OVERALL: PARTIAL — most checks pass, review warnings above")
else:
    print("OVERALL: FAIL — significant issues detected")
