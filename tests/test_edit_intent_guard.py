"""Pressing ✏️ and describing a change must not produce a brand-new picture.

With the ✏️ prefix intact the agent used to call generate_image anyway and
deliver a completely different picture: the user's image was discarded and the
change they asked for never happened. The system prompt already forbids this
("you MUST edit THAT image ... NEVER call generate_image in that case"); the rule
is enforced in the tool layer, where it cannot be talked out of.

Run: venv/Scripts/python.exe tests/test_edit_intent_guard.py
"""
import os, sys, types, tempfile, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tools as T
import graph as G

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
IMG = os.path.join(TMP, "working.png")
open(IMG, "wb").write(b"the user's picture")

_drawn = []


class Ctx:
    def __init__(self, last=None):
        self.last_image_path = last
        self.last_image_prompt = ""
        self.reference_person_mode = False
        self.web_search_enabled = True
        self.deps = types.SimpleNamespace(
            generate_image_with_refinement=lambda *a, **k: _fake_generate(*a, **k))
    def set_stage(self, *a, **k): pass
    def remember(self, *a, **k): pass
    def memory_text(self): return ""
    def is_cancelled(self): return False


def _fake_generate(*a, **kw):
    _drawn.append(kw.get("prompt") or (a[1] if len(a) > 1 else "?"))
    p = os.path.join(TMP, f"new_{len(_drawn)}.png")
    open(p, "wb").write(b"a brand new picture")
    return {"path": p, "status": "success", "score": 9, "attempts": 1}


# ------------------------------------------------- the prefix sets the flag
print("\nTHE EDIT BUTTON'S PREFIX IS RECORDED AS INTENT")
check("the prefix constant matches what the bot sends",
      G._EDIT_INTENT_PREFIX == "edit the image:", G._EDIT_INTENT_PREFIX)

# Glob the whole tg_* family so the anchor follows the ✏️ handler wherever it lives.
import glob as _glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = "\n".join(open(_p, encoding="utf-8").read()
                for _p in sorted(_glob.glob(os.path.join(_ROOT, "bot", "tg_*.py"))))
check("and the bot really sends that prefix",
      'pending_prefix = "edit the image: "' in src)

# This check is about the ORDER of two statements inside translate_node;
# .index() raises if either moves away, which is the loud failure we want.
tsrc = inspect.getsource(G.translate_node)
check("intent is taken from the RAW text, before translation rewrites it",
      tsrc.index("edit_intent") < tsrc.index("_translate_to_english"),
      "the flag is set after translation — a translated prefix may not match")

# ------------------------------------------------------------- the guard
print("\nAN EDIT TURN IS NOT ANSWERED WITH A NEW PICTURE")
_drawn.clear()
state = {"edit_intent": True, "image_path": IMG}
msg = T._handle_generate_image(Ctx(IMG), state, {"description": "a jar of honey"})
check("generate_image refuses", msg.startswith("[TOOL ERROR]"), msg[:100])
check("and nothing was drawn", _drawn == [], _drawn)
check("it names the tool to use instead", "inpaint_image" in msg, msg)
check("it explains that generating would discard the user's picture",
      "discard" in msg, msg)
check("and forbids retrying the same call", "Do not call generate_image again" in msg,
      msg)

# ---- the cases that must STILL draw --------------------------------
print("\nBUT AN ORDINARY REQUEST FOR A NEW PICTURE STILL DRAWS")
_drawn.clear()
state = {"image_path": IMG}          # a picture is loaded, but no edit intent
msg = T._handle_generate_image(Ctx(IMG), state, {"description": "a ship in a storm"})
check("no edit intent -> the picture is drawn", len(_drawn) == 1, msg[:120])

_drawn.clear()
state = {"edit_intent": True}        # edit intent but nothing to edit
msg = T._handle_generate_image(Ctx(None), state, {"description": "a ship"})
check("edit intent with NO working image -> still drawn (nothing to protect)",
      len(_drawn) == 1, msg[:120])

_drawn.clear()
state = {"edit_intent": True, "image_path": os.path.join(TMP, "gone.png")}
msg = T._handle_generate_image(Ctx(os.path.join(TMP, "gone.png")), state,
                               {"description": "a ship"})
check("a stale path that no longer exists does not block drawing",
      len(_drawn) == 1, msg[:120])

# ---- a photo of a person sent WITH «нарисуй …» ------------------------
# «Нарисуй кибер-Ленина … душит демона» with Lenin's photo came back drawn from
# scratch: a stranger's face and invented signs (live 2026-10-21).
print("\nTHE PEOPLE OF A PHOTO SENT WITH THE REQUEST ARE DRAWN FROM IT")
from models import AgentState
check("the sent photo survives the graph (an undeclared key is dropped by StateGraph)",
      "sent_photo" in AgentState.__annotations__)
H_calls = []


def _photo_ctx(faces, about_them):
    c = Ctx(IMG)
    c.deps.face_boxes = lambda path, **k: faces
    c.deps._handle_redraw_image = lambda ctx, state, args: H_calls.append(args) or "redrawn"
    c.yes_no = lambda q, text, **k: about_them
    return c


_drawn.clear()
state = {"image_path": IMG, "sent_photo": IMG, "user_input": "нарисуй его в аду, душит демона"}
msg = T._handle_generate_image(_photo_ctx([(1, 1, 50, 50)], True), state, {"description": "Lenin in hell"})
check("a face in the sent photo, the request is about them -> redrawn from the photo",
      msg == "redrawn" and not _drawn and H_calls[-1]["instructions"] == "Lenin in hell", (msg, _drawn))
_drawn.clear()
msg = T._handle_generate_image(_photo_ctx([(1, 1, 50, 50)], False), dict(state), {"description": "a cat"})
check("a request with nobody from the photo is drawn new", len(_drawn) == 1, msg[:120])
_drawn.clear()
msg = T._handle_generate_image(_photo_ctx([], True), dict(state), {"description": "a ship"})
check("a sent photo with no face is not a person to carry over", len(_drawn) == 1, msg[:120])
_drawn.clear()
msg = T._handle_generate_image(_photo_ctx([(1, 1, 50, 50)], True),
                               {"image_path": IMG, "user_input": "нарисуй его"}, {"description": "x"})
check("a picture from an earlier turn is not 'sent with the request'", len(_drawn) == 1, msg[:120])

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
