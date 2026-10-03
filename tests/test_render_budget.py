"""One turn cannot spend unlimited full renders.

Live (2026-07-29, ~21:39-21:45): a single turn ran redraw → inspect → redraw →
inspect → draw → draw, each one a full image generation, holding ComfyUI for
minutes while another user sat in the queue. The agent was doing what it was
told — the system prompt says to inspect the result and fix what is wrong "until
it is right" — but nothing attached a cost to that loop.

The loop is kept (it genuinely fixes pictures); it gets a budget. On exhaustion
the agent is told to deliver what it has and say plainly what it could not fix,
which is exactly what the anti-false-success rules already demand elsewhere.

Run: venv/Scripts/python.exe tests/test_render_budget.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import config
import tools as T

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


TMP = tempfile.mkdtemp()
_drawn = []


class Ctx:
    def __init__(self, last=None):
        self.last_image_path = last
        self.last_image_prompt = ""
        self.reference_person_mode = False
        self.web_search_enabled = True
    def set_stage(self, *a, **k): pass
    def remember(self, *a, **k): pass
    def memory_text(self): return ""
    def is_cancelled(self): return False


def _fake_generate(*a, **kw):
    _drawn.append(1)
    p = os.path.join(TMP, f"img_{len(_drawn)}.png")
    open(p, "wb").write(b"render")
    return {"path": p, "status": "success", "score": 9, "attempts": 1}


print("\nTHE BUDGET EXISTS AND IS CONFIGURABLE")
check("a per-turn render cap is defined",
      isinstance(getattr(config, "IMAGE_MAX_RENDERS_PER_TURN", None), int),
      getattr(config, "IMAGE_MAX_RENDERS_PER_TURN", None))
check("it leaves room for the self-correction loop to work",
      config.IMAGE_MAX_RENDERS_PER_TURN >= 2, config.IMAGE_MAX_RENDERS_PER_TURN)
check("but is not effectively unlimited",
      config.IMAGE_MAX_RENDERS_PER_TURN <= 5, config.IMAGE_MAX_RENDERS_PER_TURN)

CAP = config.IMAGE_MAX_RENDERS_PER_TURN

print("\nA TURN SPENDS ITS BUDGET AND THEN STOPS")
_real = T.generate_image_with_refinement
T.generate_image_with_refinement = _fake_generate
try:
    _drawn.clear()
    state = {}
    ctx = Ctx()
    msgs = []
    for i in range(CAP + 3):
        msgs.append(T._handle_generate_image(ctx, state, {"description": f"scene {i}"}))
    check(f"exactly {CAP} renders happened, not {CAP + 3}", len(_drawn) == CAP,
          f"{len(_drawn)} renders")
    check("the first calls succeeded",
          all(not m.startswith("[TOOL ERROR]") for m in msgs[:CAP]),
          msgs[0][:90])
    check("the calls past the cap are refused",
          all(m.startswith("[TOOL ERROR]") for m in msgs[CAP:]), msgs[CAP][:90])
    over = msgs[CAP]
    check("the refusal tells it to deliver what it has",
          "Deliver the best picture you already have" in over, over)
    check("and to say what it could not fix, rather than claim success",
          "could not get right" in over, over)
    check("and not to render again", "Do NOT render again" in over, over)

    print("\nEACH TURN GETS ITS OWN BUDGET")
    _drawn.clear()
    fresh = {}
    msg = T._handle_generate_image(ctx, fresh, {"description": "a new turn"})
    check("a new state means a fresh budget",
          len(_drawn) == 1 and not msg.startswith("[TOOL ERROR]"), msg[:90])

    print("\nTHE COUNTER IS SHARED WITH redraw_image")
    # The live runaway alternated redraw and generate; a per-tool counter would
    # have let it spend the budget twice over.
    _drawn.clear()
    shared = {}
    T._handle_generate_image(ctx, shared, {"description": "one"})
    used_after_generate = shared.get("_renders_used")
    check("generate_image records against the shared counter",
          used_after_generate == 1, shared)
    budget_msg = T._render_budget_exhausted(shared, "redraw_image")
    check("redraw_image continues the SAME count, it does not start over",
          shared.get("_renders_used") == 2 and budget_msg is None, shared)
finally:
    T.generate_image_with_refinement = _real

print("\nLOCALIZED EDITS ARE NOT CHARGED TO THE BUDGET")
import inspect
inp = inspect.getsource(T._handle_inpaint_image)
check("inpaint_image does not spend a render — the user asked for it explicitly",
      "_render_budget_exhausted" not in inp)

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
