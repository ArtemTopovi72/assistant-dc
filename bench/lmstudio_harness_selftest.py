"""
Self-test for tests/lmstudio_harness.py. Proves each guard catches the exact confound
that produced the wrong vision conclusion (docs/vision_pipeline_forensics.md).

Run: ./venv/Scripts/python.exe tests/test_lmstudio_harness.py
Requires LM Studio running with a vlm loaded. Does NOT evict it. Which model that is
comes from MODEL_NAME (the same knob config.py reads), so live runs can be pointed at
whatever is actually loaded instead of assuming the 9B.
"""
import sys, os
# The harness it exercises still lives in tests/ (offline_guard imports it too);
# this SUITE moved to bench/ because it needs a live LM Studio.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lmstudio_harness import (
    chat, ensure_loaded, serial_lock, list_models, model_state,
    ModelNotFound, ModelUnavailable, ModelSubstitution, HarnessError,
)

PHANTOM = "qwen3.6-35b-a3b-uncensored-heretic-i1"   # NOT in catalog (the original phantom)
def _real_model() -> str:
    """A model id that is ACTUALLY in the catalog right now.

    The old hardcoded default carried an "@q8_0" quant suffix that LM Studio no
    longer publishes, so two checks failed against a perfectly healthy server —
    caught, fittingly, by this harness's own phantom-id guard. Resolve against
    the live catalog instead: an explicit MODEL_NAME still wins, then the
    configured house model, then any non-embedding entry.
    """
    env = os.getenv("MODEL_NAME")
    if env:
        return env
    try:
        catalog = [m for m in list_models() if "embed" not in m.lower()]
    except Exception:
        return "qwen3.5-9b-uncensored-hauhaucs-aggressive"
    if not catalog:
        return "qwen3.5-9b-uncensored-hauhaucs-aggressive"
    try:
        import config as _c
        if getattr(_c, "MODEL_NAME", "") in catalog:
            return _c.MODEL_NAME
    except Exception:
        pass
    return catalog[0]

NINEB = _real_model()

# Every guard below is about what LM Studio does when a model IS resident: the
# substitution check needs something for it to substitute, and the state check
# needs something to be in the "loaded" state. With an empty catalog the server
# answers "No models loaded" to everything, and two guards report failures that
# say nothing about the code. That is not hypothetical -- it happened in a full
# run, because LM Studio's idle TTL evicted the model roughly half an hour in.
try:
    _loaded = [m for m in list_models() if "embed" not in m.lower()]
except Exception as _exc:
    print("CANNOT RUN: LM Studio is not reachable (%s)." % _exc)
    sys.exit(2)
if not _loaded:
    print("CANNOT RUN: LM Studio has no model loaded, so there is nothing for it "
          "to silently substitute and nothing in the 'loaded' state. Load one "
          "(`lms load`) and run again.")
    sys.exit(2)

results = []
def check(name, fn):
    try:
        fn(); results.append((name, True, "")); print(f"PASS  {name}")
    except AssertionError as e:
        results.append((name, False, str(e))); print(f"FAIL  {name}: {e}")
    except Exception as e:
        results.append((name, False, repr(e))); print(f"ERROR {name}: {e!r}")

def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# 1. Phantom id is rejected at preflight (never silently substituted).
check("phantom id -> ModelNotFound",
      lambda: expect_raises(ModelNotFound, lambda: ensure_loaded(PHANTOM)))

# 2. Silent substitution is caught: request the phantom WITHOUT preflight; LM Studio
#    serves the loaded 9B; the served-id assertion must stop the test.
check("silent substitution -> ModelSubstitution",
      lambda: expect_raises(ModelSubstitution,
                            lambda: chat(PHANTOM, [{"role": "user", "content": "hi"}],
                                         preflight=False, max_tokens=8)))

# 3. A correct request returns the requested model with content, served-id verified.
def valid_served():
    r = chat(NINEB, [{"role": "user", "content": "Reply with the single word OK."}],
             max_tokens=8)
    assert r.served_model == NINEB, f"served {r.served_model!r}"
    assert not r.empty, "unexpected empty content"
check("valid request -> served id matches + content", valid_served)

# 4. Global serial lock: it cannot be acquired twice at once.
def serial_enforced():
    with serial_lock():
        expect_raises(HarnessError, lambda: serial_lock(timeout=1).__enter__())
check("serial lock prevents concurrency", serial_enforced)

# 5. Catalog/state introspection works (preflight's foundation).
def state_introspection():
    cat = list_models()
    assert NINEB in cat, "9B missing from catalog"
    assert model_state(NINEB) == "loaded", f"9B state={model_state(NINEB)}"
    assert model_state(PHANTOM) is None, "phantom should have no state"
check("catalog + load-state introspection", state_introspection)

print("\n" + "=" * 60)
ok = sum(1 for _, p, _ in results if p)
print(f"{ok}/{len(results)} guards verified")
sys.exit(0 if ok == len(results) else 1)
