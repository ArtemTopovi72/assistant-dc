"""END-TO-END on the REAL model: draw a scene, then edit it, and prove the edit
went through the BOXES rather than through a pixel re-render.

The offline suite pins the wiring with a stubbed draw_agent. This one does not
stub the decisions at all:

  * `ideogram.plan_layout` really asks the model for a layout;
  * `draw_agent.edit_layout` really asks the model to turn "make the man blond"
    into operations on those boxes;
  * `image.classify_edit_intent` / `route_edit_request` are the shipping code.

Only ONE thing is faked: `image._submit_and_poll`, the HTTP call that hands a
workflow to ComfyUI and waits for pixels. That is a GPU render, not a decision —
faking it lets this run without ComfyUI while leaving every judgement real. The
render is where the picture comes from; the boxes are where the bugs are.

What it proves, in order:
  1. drawing records the layout AND the seed beside the image;
  2. a hair-colour request classifies as a contained face edit, not a whole-frame
     subject_edit (the live bug: it blended a blonde woman's face over the man);
  3. the edit is executed by rewording a BOX — the model's own output is
     inspected for the change;
  4. the seed is reused, so an edit is not a re-roll;
  5. the edited image carries its own layout, so it can be edited again.

Needs LM Studio. Does NOT need ComfyUI.
Run: venv/Scripts/python.exe tests/test_layout_edit_live.py
"""
import os, sys, json, copy, threading, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")

import requests
import config
import image as I
import ideogram as G
import draw_agent as D

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


def die(msg):
    print(f"\nCANNOT RUN: {msg}")
    sys.exit(2)


# ---------------------------------------------------------------- environment
try:
    served = [m["id"] for m in requests.get(
        config.LM_STUDIO_URL.replace("/chat/completions", "/models"), timeout=10
    ).json()["data"]]
except Exception as exc:
    die(f"LM Studio is not answering ({exc}). This test measures real model "
        f"decisions and is meaningless without it.")

MODEL = config.MODEL_NAME if config.MODEL_NAME in served else None
if MODEL is None:
    MODEL = next((m for m in served if "gemma" in m or "qwen" in m), None)
if MODEL is None:
    die(f"no chat model served; LM Studio has {served}")
print(f"model under test: {MODEL}")


class Ctx:
    def __init__(self):
        self.model_name = MODEL
        self.no_think = False
        self.reasoning_effort = "high"
        self.api_lock = threading.Lock()
        self.last_api_call_time = 0.0
        self.api_min_interval = 0.2
    def is_cancelled(self): return False
    def set_stage(self, *_a, **_k): pass


ctx = Ctx()
TMP = tempfile.mkdtemp(prefix="layout_edit_live_")

# ---------------------------------------------- fake ONLY the ComfyUI render
from PIL import Image as PILImage

_rendered = []            # every workflow that reached the "GPU", in order


def fake_submit(_ctx, workflow, *, timeout=1900, label="", **kw):
    """Write a real PNG and record what was submitted. No GPU, no decisions.

    Noise, not a flat fill: Ideogram returns a uniform safety card when it
    declines a prompt, and `is_refusal_card` detects exactly that (low pixel
    variance, one colour). A flat grey stand-in is indistinguishable from a
    refusal, so the shipping code correctly threw the render away.
    """
    path = os.path.join(TMP, f"render_{len(_rendered):02d}.png")
    PILImage.frombytes("RGB", (256, 256), os.urandom(256 * 256 * 3)).save(path)
    _rendered.append({"label": label, "workflow": copy.deepcopy(workflow)})
    return path


# Patch the transport at BOTH bindings. The ComfyUI client now lives in
# comfy_client.py; ideogram calls it through the module (comfy_client._submit_
# and_poll), while image.py bound the name at import time with `from ... import`.
# Patching only image's copy used to be enough because ideogram reached the
# function THROUGH image — that indirection existed solely to dodge an import
# cycle and is gone. Missing this would silently send this suite at a real
# ComfyUI, and it advertises that it needs none.
import comfy_client as C
C._submit_and_poll = fake_submit
I._submit_and_poll = fake_submit

# The read-back loop would run a vision model over a flat grey square and try to
# transcribe lettering that isn't there. This scene has no text, so the loop is
# off anyway — but pin it so a stray default cannot turn it on and hang the run.
config.IDEOGRAM_TEXT_ROUNDS = 0
config.IMAGE_ENGINE = "ideogram4"
I._config.IDEOGRAM_TEXT_ROUNDS = 0
I._config.IMAGE_ENGINE = "ideogram4"

PROMPT = ("мужчина в грязном комбинезоне копает лопатой у стены дома, "
          "рядом стоит полицейская машина, двое полицейских смотрят на него")

# ------------------------------------------------------------- 1. draw it
print("\n1. DRAW THE SCENE (real planner)")
img = I.generate_image_with_comfy(ctx, PROMPT, width=1024, height=1024, seed=4242)
check("an image came back", bool(img) and os.path.exists(img or ""), img)
if not img:
    die("generation failed — nothing to edit")

rec = I.load_layout_for(img)
check("a layout was recorded beside it", isinstance(rec, dict) and rec.get("layout"),
      repr(rec)[:200])
if not rec:
    die("no layout recorded — the edit path cannot engage")

els = rec["layout"].get("elements") or []
check("the layout has boxes", len(els) >= 1, f"{len(els)} elements")
check("each box has a description and a position",
      all(str(e.get("desc") or "").strip() and "x" in e and "w" in e for e in els),
      json.dumps(els, ensure_ascii=False)[:300])
check("the seed was recorded too — without it an edit is a re-roll",
      rec.get("seed") == 4242, rec.get("seed"))
print("   boxes the model planned:")
for i, e in enumerate(els):
    print(f"     {i}: {e.get('desc','')[:64]!r} "
          f"({e.get('x'):.2f},{e.get('y'):.2f}) {e.get('w'):.2f}x{e.get('h'):.2f}")

person = [e for e in els if any(w in (e.get("desc") or "").lower()
                                for w in ("мужчин", "man", "человек", "рабоч",
                                          "комбинезон", "digger", "worker"))]
check("one of the boxes is the man we are about to recolour", bool(person),
      "no element describes the man — the edit has nothing to reword")

renders_after_draw = len(_rendered)

# ------------------------------------------------------ 2. classify the edit
print("\n2. CLASSIFY THE EDIT")
INSTRUCTION = "пусть мужик будет блондин"
cat = I.classify_edit_intent(INSTRUCTION)
check(f"{INSTRUCTION!r} is a contained face edit, not a whole-frame subject_edit",
      cat == "face_edit", cat)
check("and it is a category the boxes can express", cat in I._LAYOUT_EDITABLE, cat)

# ------------------------------------------------- 3. edit through the boxes
print("\n3. EDIT IT (real box editor)")
before = copy.deepcopy(rec["layout"])
cat2, out = I.route_edit_request(ctx, img, INSTRUCTION, seed=None)
check("routed as a face edit", cat2 == "face_edit", cat2)
check("an edited image came back", bool(out) and os.path.exists(out or ""), out)
if not out:
    die("the layout edit produced no image")

check("it went to the renderer through the layout, not a pixel pipeline",
      len(_rendered) > renders_after_draw,
      "nothing new was submitted — no re-render happened")

rec2 = I.load_layout_for(out)
check("the edited image carries its own layout, so it stays editable",
      isinstance(rec2, dict) and rec2.get("layout"), repr(rec2)[:200])

after = (rec2 or {}).get("layout") or {}
check("the layout actually changed — the model edited a box",
      after.get("elements") != before.get("elements"),
      "the boxes came back identical; the edit did not reach them")

blond_words = ("блонд", "blond", "светлы", "fair-hair", "fair hair", "золотист")
touched = [e.get("desc", "") for e in (after.get("elements") or [])
           if any(w in (e.get("desc") or "").lower() for w in blond_words)]
check("a box now describes blond hair — the edit is IN the layout",
      bool(touched), json.dumps(after.get("elements"), ensure_ascii=False)[:400])
if touched:
    print(f"   reworded box: {touched[0][:90]!r}")

check("the same number of boxes — a recolour must not add or drop elements",
      len(after.get("elements") or []) == len(before.get("elements") or []),
      f"{len(before.get('elements') or [])} -> {len(after.get('elements') or [])}")

check("the seed carried over, so the edit is not a fresh roll",
      (rec2 or {}).get("seed") == 4242, (rec2 or {}).get("seed"))

# ------------------------------------------------------ 4. the negative case
print("\n4. AN UPLOADED PHOTO HAS NO BOXES AND MUST NOT USE THIS PATH")
photo = os.path.join(TMP, "uploaded.png")
PILImage.new("RGB", (256, 256), (200, 180, 160)).save(photo)
check("no layout for a photo we did not compose", I.load_layout_for(photo) is None)
n_before = len(_rendered)
cat3, out3 = I.route_edit_request(ctx, photo, "исправь надписи на корректные")
check("a lettering edit on it is refused rather than re-rendered blind",
      cat3 == "text_edit" and out3 is None, f"{cat3} {out3}")
check("and nothing was submitted for it", len(_rendered) == n_before,
      "something was rendered for an image with no boxes")

print(f"\n{_n - _bad}/{_n} checks passed")
print(f"artifacts: {TMP}")
sys.exit(1 if _bad else 0)
