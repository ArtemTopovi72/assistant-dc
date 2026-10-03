"""`lms load` exiting 0 is not the same as the model being served.

Live 2026-09-12 (journey 3): the chat model crashed on a vision call, the
revive ran `lms load`, which exited 0 in nine seconds, and the server then
answered "Model reloaded." and "No models loaded" for the next minute. The
revive cached "ok" and every turn inside its cooldown died. Now the loader
polls the model list and reports failure when the model never shows up.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import lmstudio as L

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

class R:
    returncode = 0; stdout = "Loaded"; stderr = ""

served = []
L._lms_unload_all = lambda: (True, "")
L.resolve_virtual_model = lambda b, m: ("", False)
L.subprocess.run = lambda *a, **k: R()
L.time.sleep = lambda s: None
L.loaded_model_ids = lambda b: list(served)
L.LOAD_SERVED_WAIT_S = 0.0

ok, msg = L.load_model_exclusive("http://x", "gemma")
check("exit 0 with nothing served is a FAILURE", ok is False, (ok, msg))
check("...that names the model", "gemma" in msg and "not being served" in msg, msg)

served.append("gemma")
ok, msg = L.load_model_exclusive("http://x", "gemma")
check("exit 0 with the model listed is success", ok is True, (ok, msg))

# The poll keeps looking until the deadline, not just once.
seen = {"n": 0}
def _late(b):
    seen["n"] += 1
    return ["gemma"] if seen["n"] >= 3 else []
L.loaded_model_ids = _late
ticks = iter([0, 1, 2, 3, 4, 5])
L.time.monotonic = lambda: next(ticks)
check("a model that appears on the third poll counts", L._wait_served("http://x", "gemma", timeout=5) is True, seen)

# The revive path does not cache a false "ok".
import llm
llm.LLM_REVIVE_MIN_INTERVAL_S = 0
llm._revive_last = 0.0
calls = []
L.ensure_exclusive = lambda base, model: calls.append(model) or (False, "not being served")
import comfy_client
comfy_client.card_is_exclusive = lambda: ""
got = llm._try_revive_model("No models loaded", {"model": "gemma"})
check("a revive whose load never served is reported as failed", got is False and calls == ["gemma"], (got, calls))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
