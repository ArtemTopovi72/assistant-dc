"""A bare acknowledgement skips the tool loop even with a picture in the chat.

Live 2026-09-13 (mega journey, step 56): '👍' after the twelfth picture took
53 s — ctx.last_image_path was set, so the fast path was refused and the
full tool loop (schemas, history, reasoning) produced "glad you liked it".
"""
import os, sys, types
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_fastpath as F

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

import intent
# What is a bare ack is the model's read (bench/intent_live.py has the phrases).
intent.STUB = lambda t: ({"needs_tool": False, "bare_ack": True} if t in ("👍", "спасибо")
                         else {"needs_tool": True, "wants": ["redraw_image"]})

class _Ctx:
    last_image_path = "C:/pics/ideogram_00599_.png"
    def is_cancelled(self): return False
F._sandbox_has_files = lambda ctx: False
ctx = _Ctx()
check("'👍' with a picture in the chat takes the fast path",
      F._fast_path_allowed(ctx, {"image_data": None}, "👍", "👍"))
check("'спасибо' with a picture in the chat takes the fast path",
      F._fast_path_allowed(ctx, {"image_data": None}, "thanks", "спасибо"))
check("an ack WITH an attached picture still goes to the loop",
      not F._fast_path_allowed(ctx, {"image_data": b"x"}, "👍", "👍"))
check("a follow-up edit with a picture in the chat still goes to the loop",
      not F._fast_path_allowed(ctx, {"image_data": None}, "make it nighttime", "сделай её ночной"))
ctx2 = _Ctx(); ctx2.is_cancelled = lambda: True
check("a cancelled ack turn is not answered on the fast path",
      not F._fast_path_allowed(ctx2, {"image_data": None}, "👍", "👍"))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
