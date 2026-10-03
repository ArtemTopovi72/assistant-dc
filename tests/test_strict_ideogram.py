"""A FRESH picture is drawn by Ideogram or not at all.

The old model fallback looked like a safety net — never leave the user with
nothing — but it hands back a picture with NO layout. The user gets an image, so
nothing looks wrong; then "move that box" or "fix the sign" silently drops to a
pixel pipeline half an hour later. Observed live: the delivered file was
old-model-inpaint_00010_.png with no sidecar, and the box edit had nothing to work
with. Failing loudly is the better trade — the user can just ask again.

The old model is now gone from the product entirely (2026-09-23); this suite keeps
the contract that a failed or refused Ideogram render is reported, never
substituted, and that an edit is refused here rather than redrawn.

Run: venv/Scripts/python.exe tests/test_strict_ideogram.py
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import config
import image as I
# _ideogram_draw is a bare local name inside image_generate, so image's copy
# is NOT the one generate_image_with_comfy calls.
import image_generate as IG
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
_oldmodel_calls = []


def _fake_oldmodel(*a, **kw):
    """Stands in for the whole the old model path below the Ideogram branch.

    Records the workflow LABEL, which is what this seam actually carries —
    `previous_image_path` never reaches `_submit_and_poll`, so spying for it
    recorded None and made a passing case look broken.
    """
    _oldmodel_calls.append(kw.get("label") or "?")
    p = os.path.join(TMP, f"old-model_{len(_oldmodel_calls)}.png")
    open(p, "wb").write(b"oldmodel")
    return p


print("\nA FAILED IDEOGRAM RENDER IS NOT SUBSTITUTED WITH THE OLD MODEL")
_prev_engine = I._config.IMAGE_ENGINE
_prev_draw = IG._ideogram_draw
_prev_submit = I._submit_and_poll
I._config.IMAGE_ENGINE = "ideogram4"
IG._ideogram_draw = lambda *a, **kw: None          # the render failed
I._submit_and_poll = _fake_oldmodel

_oldmodel_calls.clear()
out = I.generate_image_with_comfy(None, "a police car", width=512, height=512)
check("nothing is returned", out is None, out)
check("and the old model was never asked to draw it", _oldmodel_calls == [], _oldmodel_calls)
check("the reason says the engine failed",
      I._GENERATE_FAILURE.get("reason") == "engine_failed", I._GENERATE_FAILURE)

print("\nA REFUSAL IS REPORTED AS A REFUSAL, NOT A SERVER ERROR")
import ideogram as G


def _refuse(*a, **kw):
    raise G.ContentRefused("declined")


IG._ideogram_draw = _refuse
_oldmodel_calls.clear()
out = I.generate_image_with_comfy(None, "something declined", width=512, height=512)
check("nothing is returned", out is None, out)
check("the old model is not used to route around the refusal", _oldmodel_calls == [], _oldmodel_calls)
check("the reason says refused", I._GENERATE_FAILURE.get("reason") == "refused",
      I._GENERATE_FAILURE)

print("\nTHE AGENT IS TOLD WHICH FAILURE IT WAS")
msgs = {}
for reason in ("refused", "engine_failed", "server_error"):
    I._GENERATE_FAILURE["reason"] = reason
    # Reach the failure branch: no image produced.
    state = {}

    class _Ctx:
        last_image_path = None
        last_image_prompt = ""
        def set_stage(self, *a, **k): pass
        def remember(self, *a, **k): pass
        def memory_text(self): return ""
        def is_cancelled(self): return False

    _real = T.generate_image_with_refinement
    T.generate_image_with_refinement = lambda *a, **kw: {"path": None, "status": "fail",
                                                        "score": 0, "attempts": 1}
    try:
        msg = T._handle_generate_image(_Ctx(), state, {"description": "a cat"})
    finally:
        T.generate_image_with_refinement = _real
    msgs[reason] = msg
    check(f"{reason}: flagged as a tool error", msg.startswith("[TOOL ERROR]"), msg[:90])
    check(f"{reason}: the model is told not to claim success",
          "do NOT imply or claim that an image was created" in msg, msg[-120:])

check("a refusal tells the user to rephrase, not to retry the same words",
      "rephrase" in msgs["refused"] and "declined again" in msgs["refused"],
      msgs["refused"])
check("an engine failure says asking again is worth it",
      "try again" in msgs["engine_failed"], msgs["engine_failed"])
check("the three messages are actually different",
      len({msgs["refused"], msgs["engine_failed"], msgs["server_error"]}) == 3)

print("\nAN EDIT IS NOT THIS FUNCTION'S JOB")
# Editing goes through FireRed (a user's photo) or the layout boxes (our own
# Ideogram picture). The old img2img-by-the old model branch is gone; handing this
# function a previous image must refuse, not quietly draw something new.
photo = os.path.join(TMP, "photo.png")
open(photo, "wb").write(b"a photo")
_oldmodel_calls.clear()
IG._ideogram_draw = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("drew"))
out = I.generate_image_with_comfy(None, "make it brighter", width=512, height=512,
                                  previous_image_path=photo)
check("nothing is returned for an edit", out is None, out)
check("nothing was submitted to ComfyUI", _oldmodel_calls == [], _oldmodel_calls)
check("the reason is reported", I._GENERATE_FAILURE.get("reason") == "engine_failed",
      I._GENERATE_FAILURE)

I._config.IMAGE_ENGINE = _prev_engine
IG._ideogram_draw = _prev_draw
I._submit_and_poll = _prev_submit

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
