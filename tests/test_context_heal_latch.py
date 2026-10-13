"""The context self-heal must stay armed: the fault comes back.

LM Studio can hold the same model resident twice (`<id>` at a small context and
`<id>:2` at the model default) and answer from either, so a prompt that did not
fit succeeds or fails depending on which copy picks it up. heal_context is a
rate limit (cooldown plus attempt cap), not a one-shot latch: a second heal at
the same size must still run, a permanently broken server must stop after the
cap, and a reload asks for ONE instance that is big enough without multiplying
the context by the PARALLEL slot count.

Run: venv/Scripts/python.exe tests/test_context_heal_latch.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import lmstudio as L
from types import SimpleNamespace as NS

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


MODEL = "test/model"
reloads = []


def fake_reload(model_id, tokens, gpu, parallel=0, ctx=None):
    reloads.append((model_id, tokens, parallel))
    return True, "ok"


def reset(cooldown=0.0, attempts=5):
    """A fresh latch, and the context that hands heal_context a fake CLI reload and
    the limits under test."""
    global C
    reloads.clear()
    L._heal_state.clear()
    C = NS(deps=NS(_HEAL_COOLDOWN_S=cooldown, _HEAL_MAX_ATTEMPTS=attempts,
                   reload_via_cli=fake_reload,
                   loaded_instances=lambda m="", ctx=None: [MODEL, MODEL + ":2"]))


try:
    # ── the fault comes back: a second heal at the SAME size must still run ──
    print("=" * 66)
    print("THE HEAL MUST NOT LATCH OFF")
    print("=" * 66)
    reset(cooldown=0.0)
    ok1, _ = L.heal_context(MODEL, 11073, ctx=C)
    ok2, _ = L.heal_context(MODEL, 11073, ctx=C)      # the phantom reappeared
    check("the first repair runs", ok1)
    check("a LATER repair at the same size runs too", ok2)
    check("both reloads actually happened", len(reloads) == 2, reloads)

    # ── but it must not spin ─────────────────────────────────────────────────
    print("=" * 66)
    print("…AND MUST NOT SPIN")
    print("=" * 66)
    reset(cooldown=600.0)
    ok1, _ = L.heal_context(MODEL, 11073, ctx=C)
    ok2, msg = L.heal_context(MODEL, 11073, ctx=C)
    check("a repair inside the cooldown is refused", ok1 and not ok2, msg)
    check("only one reload was issued", len(reloads) == 1, reloads)
    check("the refusal explains the cooldown", "cooldown" in msg.lower(), msg)

    reset(cooldown=0.0, attempts=3)
    for _ in range(5):
        L.heal_context(MODEL, 11073, ctx=C)
    check("a permanently broken server stops after the attempt cap",
          len(reloads) == 3, reloads)

    # ── it reloads ONE instance, big enough, without multiplying by slots ────
    print("=" * 66)
    print("WHAT IT ASKS FOR")
    print("=" * 66)
    reset(cooldown=0.0)
    L.heal_context(MODEL, 11073, ctx=C)
    model, tokens, parallel = reloads[0]
    check("it asks for at least the needed tokens", tokens >= 11073 + 2048, tokens)
    check("it does NOT multiply the context by the parallel slot count",
          tokens <= 65536, tokens)
    check("it reloads the model that failed", model == MODEL, model)

    # a bigger requirement must not be blocked by a smaller earlier repair
    reset(cooldown=0.0)
    L.heal_context(MODEL, 11073, ctx=C)
    L.heal_context(MODEL, 60000, ctx=C)
    check("a much larger requirement is honoured", len(reloads) == 2, reloads)
    check("…and asks for more than the first", reloads[1][1] > reloads[0][1],
          reloads)
finally:
    L._heal_state.clear()

# ── the phantom detector itself ─────────────────────────────────────────────
print("=" * 66)
print("PHANTOM DETECTION")
print("=" * 66)
PS = ("IDENTIFIER                    MODEL      STATUS  SIZE     CONTEXT  PARALLEL\n"
      "test/model                    test/model IDLE    15.63 GB 8192     4\n"
      "test/model:2                  test/model IDLE    15.63 GB 32768    4\n")


class _R:
    returncode = 0
    stdout = PS
    stderr = ""


P = NS(deps=NS(subprocess=NS(run=lambda *a, **k: _R())))
inst = L.loaded_instances("test/model", ctx=P)
check("both resident instances are seen",
      set(inst) == {"test/model", "test/model:2"}, inst)
check("the parallel column is read", L.loaded_parallel("test/model", ctx=P) == 4,
      L.loaded_parallel("test/model", ctx=P))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
