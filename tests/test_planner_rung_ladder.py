"""The layout planner burnt a guaranteed-miss attempt on the house model.

_ask_planner climbs a ladder of asks: rung 0 is a narrow 900-token squeeze,
rung 1 a bit wider, and the last rung a minimal prompt with a 6000-token budget.
That ladder is right for a small non-Gemma model, which really does answer at
rung 0.

Gemma 4 thinks on EVERY turn and cannot be told not to — the <think></think>
prefill is a Qwen lever and llm.py drops it for Gemma. Measured on
gemma-4-26b-a4b-qat, rung 0 spends its entire budget reasoning (8.7k characters),
returns nothing, and the ladder then jumps to the wide rung anyway. The recovery
worked; the wasted attempt was pure cost — about 32 s of GPU on EVERY drawing,
measured back to back on the same box:

    lettering prompt   74.1s -> 41.9s
    plain scene        51.7s -> 19.5s

with byte-identical layouts out the other side.

send_to_lm_studio is stubbed: this suite records which budgets the ladder asks
for and never touches a model.

Run: venv/Scripts/python.exe tests/test_planner_rung_ladder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import ideogram as I
import llm

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


GOOD_LAYOUT = {
    "high_level_description": "a poster", "background": "a wall",
    "medium": "photography", "aesthetics": "clean", "lighting": "soft",
    "photo": "50mm", "art_style": "",
    "elements": [{"desc": "poster", "text": "Добро пожаловать",
                  "x": 0.1, "y": 0.4, "w": 0.8, "h": 0.1}],
}


class Ctx:
    no_think = False        # thinking on: the case the wide-rung jump exists for
    def __init__(self, model):
        self.model_name = model
    def set_stage(self, *_a, **_k): pass


def _drive(model, *, answers):
    """Run the real ladder; `answers` is one reply per rung, in order.

    A reply of None models "spent the budget thinking and never wrote JSON",
    which is what Gemma does at a narrow budget.
    """
    asks = []
    seq = list(answers)

    def _fake(ctx, messages, **kw):
        asks.append({"max_tokens": kw.get("max_tokens"),
                     "prefill": kw.get("prefill"),
                     "temperature": kw.get("temperature")})
        reply = seq.pop(0) if seq else None
        if reply is None:
            return {"content": "", "reasoning_only": True, "reasoning_chars": 8700}
        import json
        return {"content": json.dumps(reply, ensure_ascii=False)}

    real = llm.send_to_lm_studio
    llm.send_to_lm_studio = _fake
    try:
        out = I._ask_planner(Ctx(model), "draw a poster")
    finally:
        llm.send_to_lm_studio = real
    return asks, out


print("=" * 70)
print("1. On Gemma the ladder starts at the WIDE rung")
print("=" * 70)
asks, out = _drive("google/gemma-4-26b-a4b-qat", answers=[GOOD_LAYOUT])
check("exactly one call was made", len(asks) == 1, asks)
check("...with the wide budget, not the 900-token squeeze",
      asks and asks[0]["max_tokens"] == 6000, asks)
check("...and no Qwen prefill (it is junk on Gemma)",
      asks and asks[0]["prefill"] is None, asks)
check("the layout came back", I._looks_like_layout(out), out)
check("the lettering survived", out["elements"][0]["text"] == "Добро пожаловать", out)

for model in ("google/gemma-4-12b-qat", "google/gemma-4-31b-qat"):
    asks, _ = _drive(model, answers=[GOOD_LAYOUT])
    check(f"{model}: one wide call", len(asks) == 1 and asks[0]["max_tokens"] == 6000,
          asks)

print()
print("=" * 70)
print("2. Every other model keeps the full ladder")
print("=" * 70)
# The narrow rung is genuinely cheaper for a model that can answer there —
# starting these at 6000 would be slower, not faster.
asks, out = _drive("qwen3.5-9b-uncensored", answers=[GOOD_LAYOUT])
check("a non-Gemma model tries the narrow rung first",
      len(asks) == 1 and asks[0]["max_tokens"] == 900, asks)
check("...and it is allowed its Qwen prefill",
      asks and asks[0]["prefill"] == "<think></think>", asks)

# ...and the escape hatch still works for a non-Gemma model that reasons away
# its budget: that recovery is what this change makes unnecessary on Gemma, not
# something it removes.
asks, out = _drive("qwen3.5-9b-uncensored", answers=[None, GOOD_LAYOUT])
check("a reasoning-only miss still jumps straight to the wide rung",
      [a["max_tokens"] for a in asks] == [900, 6000],
      [a["max_tokens"] for a in asks])
check("...and recovers the layout", I._looks_like_layout(out), out)

# A plain malformed answer (not reasoning-only) walks the ladder normally.
asks, out = _drive("qwen3.5-9b-uncensored", answers=[{"junk": 1}, GOOD_LAYOUT])
check("a malformed answer advances one rung at a time",
      [a["max_tokens"] for a in asks] == [900, 1600],
      [a["max_tokens"] for a in asks])

print()
print("=" * 70)
print("3. Failure is still reported, not silently faked")
print("=" * 70)
asks, out = _drive("google/gemma-4-26b-a4b-qat", answers=[None])
check("Gemma failing its one rung returns no layout",
      not I._looks_like_layout(out), out)
asks, out = _drive("qwen3.5-9b-uncensored", answers=[None, None, None])
check("a non-Gemma model exhausts the ladder and reports failure",
      not I._looks_like_layout(out), out)

print()
print("=" * 70)
print("4. An unknown/blank model keeps the safe full ladder")
print("=" * 70)
for model in ("", "some-future-model"):
    asks, _ = _drive(model, answers=[GOOD_LAYOUT])
    check(f"{model!r}: starts narrow", asks and asks[0]["max_tokens"] == 900, asks)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
