"""Pressing ✏️ and describing a change must not produce a brand-new picture.

Live failure (2026-07-29 21:43, Telegram): the user tapped ✏️, typed "добавь ему
надпись на футболку имя польского философа", the prefix arrived intact — and the
agent called generate_image anyway, replied "Я создал изображение банки меда с
фекалиями внутри" and delivered a completely different picture. Their image was
discarded and the change they asked for never happened.

The system prompt already forbids this ("you MUST edit THAT image ... NEVER call
generate_image in that case"). The prompt lost. So the rule is enforced in the
tool layer, where it cannot be talked out of.

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

# The ✏️ handler keeps moving (tg_bot.py → tg_dispatch.py → tg_callbacks.py),
# and a hardcoded module list goes stale on every move — which is how this
# check ended up reading a pair of files the handler had already left. Glob
# the whole tg_* family so the anchor follows the code instead of chasing it.
import glob as _glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = "\n".join(open(_p, encoding="utf-8").read()
                for _p in sorted(_glob.glob(os.path.join(_ROOT, "bot", "tg_*.py"))))
check("and the bot really sends that prefix",
      'pending_prefix = "edit the image: "' in src)

# translate_node is a module-level function now, not a closure inside
# build_graph. This check is about the ORDER of two statements inside that
# node, so it must read that node; .index() raises if either moves away,
# which is the loud failure we want.
tsrc = inspect.getsource(G.translate_node)
check("intent is taken from the RAW text, before translation rewrites it",
      tsrc.index("edit_intent") < tsrc.index("_translate_to_english"),
      "the flag is set after translation — a translated prefix may not match")

# ------------------------------------------------------------- the guard
print("\nAN EDIT TURN IS NOT ANSWERED WITH A NEW PICTURE")
_real = T.generate_image_with_refinement
T.generate_image_with_refinement = _fake_generate
try:
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
finally:
    T.generate_image_with_refinement = _real

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
