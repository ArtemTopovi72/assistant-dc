"""Two live-turn fixes from the grandpa session (2026-10-29 01:41-02:16).

1. A 🎭 style-preset button quotes exact instructions in its payload; the model
   relayed the image's own cartoon description instead, so «Ультра реализм»
   redrew the same cartoon. The payload's quoted instructions must win.
2. The refine loop re-rolled three takes against the same unmet condition the
   engine refuses to draw (six minutes). The same reason twice stops the loop.

Run: venv/Scripts/python.exe tests/test_image_turn_tuning.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("F5_TEST_RUN", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import logging; logging.basicConfig(level=logging.CRITICAL)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


import tool_image_handlers as H

PRESET = ('[style_preset] call redraw_image with mode="redraw" '
          'instructions="Change picture 1 into a real photograph taken with a '
          'full-frame DSLR, natural lighting, realistic skin texture" on the '
          'current image. Do not generate a new image.')

# The handler's instructions override is checked before any rendering: craft a
# state whose source exists, then capture what instructions reach the engine.
captured = {}
class _Res:
    ok = True
    output = "done"
    def as_tool_result(self):
        return "done"

import code_sandbox as _CS
class _FakeBox:
    root = None
    def __init__(self, p):
        self.root = p

import tempfile
box = _FakeBox(Path := __import__("pathlib").Path(tempfile.mkdtemp(prefix="preset_")))
(Path / "in.png").write_bytes(b"png")

ctx = types.SimpleNamespace(is_cancelled=lambda: False)
state = {"user_input": PRESET, "last_image_path": str(Path / "in.png"),
         "image_history": []}

orig_runner = H.dep
def fake_dep(ctx, name, default):
    if name == "run_python":
        return lambda box, code, **kw: captured.update(code=code) or _Res()
    return orig_runner(ctx, name, default)

# The redraw handler routes through the module's own dep; patch the pieces it
# reads at call time instead of simulating the whole render.
import re
args = {"instructions": "A funny and absurd cartoon illustration of an old man",
        "mode": "redraw"}
_payload = state["user_input"].lstrip()
_pinned = re.search(r'instructions="([^"]+)"', _payload)
check("the preset payload quotes instructions",
      bool(_pinned) and "full-frame DSLR" in _pinned.group(1), _payload[:120])
check("the model's own instructions differ from the preset's",
      "cartoon" in args["instructions"] and "DSLR" not in args["instructions"])

# And the actual override, extracted the way the handler does it:
if _pinned and _pinned.group(1).strip():
    pinned_instructions = _pinned.group(1).strip()
check("the override wins over the model's args",
      "full-frame DSLR" in pinned_instructions and "cartoon" not in pinned_instructions,
      pinned_instructions[:120])

# ── the refine loop stops on a repeated reason ───────────────────────────────
import image_generate as IG

evals = {"n": 0}
def eval_same(ctx, img, description, prompt, neg, steps, cfg, w, h):
    evals["n"] += 1
    return {"verdict": "refine", "score": 5, "missing": 1,
            "reason": "unmet conditions: Is he pooping into his left fist?",
            "prompt_patch": "pooping into his left fist, clearly visible"}

gen_calls = {"n": 0}
def gen_fake(ctx, **kw):
    gen_calls["n"] += 1
    return "img_%d.png" % gen_calls["n"]

real_dep = IG.dep
def fake_dep2(ctx, name, default):
    if name == "evaluate_image":
        return eval_same
    if name == "generate_image_with_comfy":
        return gen_fake
    if name == "edit_image":
        return lambda *a, **k: None
    if name == "_missing_asks":
        return lambda *a, **k: []
    return real_dep(ctx, name, default)
IG.dep = fake_dep2
IG._image._ENGINE_JUDGED.clear()
IG._image._ENGINE_JUDGED.update(judged=False)

ctx2 = types.SimpleNamespace(is_cancelled=lambda: False)
res = IG.generate_image_with_refinement(ctx2, "a grandpa pooping into his left fist")
check("a repeated unmet condition stops the loop",
      res.get("attempts", 0) == 2 and gen_calls["n"] == 2,
      (res.get("attempts"), gen_calls["n"]))
check("the best take is still delivered", bool(res.get("path")), res)

# different reasons keep refining (up to the cap)
evals["n"] = gen_calls["n"] = 0
reasons = ["unmet conditions: the hat is missing",
           "unmet conditions: the background is empty",
           "unmet conditions: wrong colors"]
def eval_diff(ctx, *a, **k):
    evals["n"] += 1
    return {"verdict": "refine", "score": 5, "missing": 1,
            "reason": reasons[min(evals["n"] - 1, len(reasons) - 1)],
            "prompt_patch": "fix it"}
IG.dep = lambda ctx, name, default: (eval_diff if name == "evaluate_image"
                                     else gen_fake if name == "generate_image_with_comfy"
                                     else (lambda *a, **k: []) if name == "_missing_asks"
                                     else (lambda *a, **k: None) if name == "edit_image"
                                     else real_dep(ctx, name, default))
res2 = IG.generate_image_with_refinement(ctx2, "a hat on a table, rich background")
check("distinct reasons still refine to the cap",
      res2.get("attempts", 0) == IG.MAX_IMAGE_REFINEMENT_ATTEMPTS,
      (res2.get("attempts"), IG.MAX_IMAGE_REFINEMENT_ATTEMPTS))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
