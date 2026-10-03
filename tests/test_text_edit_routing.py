"""Lettering edits go to the layout boxes, never to a whole-frame re-render.

Live failure this pins (2026-07-29, Telegram): "исправь надписи на корректные" on
a generated picture matched no intent rule, fell through to the default
`subject_edit`, and was handed to FireRed — which re-rendered the frame, wiped
the lettering, and reported "Я успешно удалил все надписи". A re-render cannot
spell. The only thing that can is the layout the picture was composed from.

Run: venv/Scripts/python.exe tests/test_text_edit_routing.py
"""
import os, sys, json, types, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import image as I
import image_router as _R
# Which kind of edit a phrase is is the model's read (image_router.edit_plan);
# the phrases run live in bench/edit_intent_live.py. These are its answers.
_R.EDIT_STUB = {
    "исправь надписи на корректные": "text_edit", "пусть мужик будет блондин": "face_edit",
    "сделай рубашку зелёной": "clothing_edit", "добавь скамейку": "object_insert",
    "замени фон на пляж": "background_replace", "сделай в стиле аниме": "style_transfer",
    "убери человека справа": "person_remove", "увеличь разрешение": "upscale",
    "убери фон": "background_remove", "восстанови старое фото": "restore", "расширь кадр": "outpaint",
}.get

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}  {detail}")


# ---------------------------------------------------- the classifier sees it
# ------------------------------------------------- the layout survives a render
print("\nTHE LAYOUT THE PICTURE WAS DRAWN FROM IS REMEMBERED")
tmp = tempfile.mkdtemp()
img = os.path.join(tmp, "render.png")
open(img, "wb").write(b"not really a png")
LAY = {"background": "a street", "elements": [
    {"desc": "the door of the car", "text": "POLICE", "x": .2, "y": .5, "w": .36, "h": .10}]}

check("nothing recorded yet -> no layout", I.load_layout_for(img) is None)
I.save_layout_for(img, "a police car", LAY, width=1024, height=1024)
rec = I.load_layout_for(img)
check("saved and read back", isinstance(rec, dict) and rec.get("layout") == LAY, repr(rec)[:120])
check("the prompt is kept too — the repair re-renders the SAME scene",
      (rec or {}).get("prompt") == "a police car")
check("an unknown image has no layout", I.load_layout_for(os.path.join(tmp, "nope.png")) is None)
check("a corrupt record is not a crash",
      (open(img + ".layout.json", "w").write("{{{not json"), I.load_layout_for(img))[1] is None)

# ------------------------------------------------------------- the dispatch
print("\nWHERE THE REQUEST ACTUALLY GOES")
I.save_layout_for(img, "a police car", LAY, width=1024, height=1024)

calls = []
fake = types.ModuleType("draw_agent")
fake.edit_layout = lambda ctx, lay, instr: (calls.append(("edit_layout", instr)) or
                                            (lay, ["moved the sign"]))
_seed_seen = {}
fake.run = lambda ctx, req, **kw: (calls.append(("run", req, kw.get("layout"))) or
                                   _seed_seen.update(seed=kw.get("seed")) or
                                   {"image": os.path.join(tmp, "fixed.png"),
                                    "layout": kw.get("layout")})
sys.modules["draw_agent"] = fake

cat, out = I.route_edit_request(None, img, "исправь надписи на корректные")
check("routed as text_edit", cat == "text_edit", cat)
check("the drawing agent was asked to edit the boxes",
      any(c[0] == "edit_layout" for c in calls), calls)
check("and to re-render from that layout", any(c[0] == "run" for c in calls), calls)
check("it re-renders the ORIGINAL scene, not the bare instruction",
      any(c[0] == "run" and c[1] == "a police car" for c in calls), calls)
check("a repaired image comes back", out and out.endswith("fixed.png"), out)
# The repaired picture must carry its own layout, or the SECOND "fix the text"
# lands on an image with no boxes and gets refused as if it were an upload.
again = I.load_layout_for(out)
check("the repaired image keeps a layout of its own, so it can be fixed again",
      isinstance(again, dict) and again.get("layout") is not None, repr(again)[:120])
check("and that record still names the original scene",
      (again or {}).get("prompt") == "a police car", repr(again)[:120])

# No layout -> refuse. This is the whole point: the old behaviour was to hand it
# to FireRed, which is how the user got "I successfully removed all the text".
plain = os.path.join(tmp, "uploaded_photo.png")
open(plain, "wb").write(b"a photo the user sent")
calls.clear()
cat, out = I.route_edit_request(None, plain, "исправь надписи на корректные")
check("still classified as a lettering edit", cat == "text_edit", cat)
check("but nothing is rendered when there are no boxes to drag", out is None, out)
check("and the drawing agent is not invoked at all", calls == [], calls)

# ------------------------------------------------------- the tool-layer wiring
print("\nA FRESH RENDER LEAVES A LAYOUT BEHIND")
# The repair above is only reachable if generation recorded the boxes. Without
# this, every picture looks like an uploaded photo the moment it is finished.
drawn = os.path.join(tmp, "drawn.png")
open(drawn, "wb").write(b"render")
fake_ideo = types.ModuleType("ideogram")
fake_ideo.ContentRefused = type("ContentRefused", (Exception,), {})
fake_ideo.plan_layout = lambda ctx, prompt: LAY
fake_ideo.layout_to_caption = lambda lay: "caption"
fake_ideo.generate = lambda *a, **k: drawn
sys.modules["ideogram"] = fake_ideo
# The plan is sketch-checked before the render (draw_preview.preflight) and
# that needs the real ideogram module; here the plan passes through untouched.
fake_pre = types.ModuleType("draw_preview")
fake_pre.preflight = lambda ctx, layout, prompt, **kw: (layout, {})
sys.modules["draw_preview"] = fake_pre
fake.run = lambda ctx, req, **kw: {"image": drawn, "layout": kw.get("layout")}
_prev_engine = getattr(I._config, "IDEOGRAM_TEXT_ROUNDS", 1)
I._config.IDEOGRAM_TEXT_ROUNDS = 1
try:
    os.remove(drawn + ".layout.json")
except OSError:
    pass
res = I._ideogram_draw(None, "a police car", width=1024, height=1024, seed=1, timeout=60)
check("the render came back", res == drawn, res)
rec2 = I.load_layout_for(drawn)
check("and its layout was recorded, so lettering stays fixable",
      isinstance(rec2, dict) and rec2.get("layout") is not None, repr(rec2)[:120])
I._config.IDEOGRAM_TEXT_ROUNDS = _prev_engine

print("\nTHE TOOL LAYER KEEPS IT OFF FIRERED")
import inspect
import tools as T
src = inspect.getsource(T._handle_inpaint_image)
check("text_edit is dispatched before the whole-frame branch",
      src.index('category in ("text_edit", "text_add")') < src.index("elif manual_whole"))
check("the whole-frame toggle does NOT capture lettering",
      'if category == "text_edit"' in src)
check("text_edit is not in the FireRed category list",
      # the lettering dispatch tuple itself is ("text_edit", "text_add"); every
      # OTHER `elif category in (...)` list is a pixel pipeline
      not any('"text_edit"' in part.split(")")[0] and '"text_add"' not in part.split(")")[0]
              for part in src.split("elif category in (")[1:]))
check("a refusal explains itself instead of reporting a server error",
      "_NO_LAYOUT_NOTE" in src)
# Grepping for the string is not enough: it stayed present when the branch that
# reaches it was disabled. The note must be what a text_edit failure RETURNS.
_after = src.split('elif category == "text_edit":', 1)
check("the note is returned from the text_edit failure branch itself",
      len(_after) == 2 and "return image_mod._NO_LAYOUT_NOTE" in
      _after[1].split("else:")[0], "the branch no longer reaches the note")
check("the refusal tells the model not to reach for another edit tool",
      "Do not call another edit tool." in I._NO_LAYOUT_NOTE)
check("and it is flagged as a tool error so the agent cannot read it as success",
      I._NO_LAYOUT_NOTE.startswith("[TOOL ERROR]"))

print("\nANY CONTENT EDIT ON AN IMAGE WE COMPOSED GOES THROUGH THE BOXES")
# The live failure: "пусть мужик будет блондин" was a whole-frame FireRed
# re-render, which re-imagined the man and blended a blonde woman's face over
# his. On a picture we composed there IS a layout, and a recolour is a reworded
# box plus a redraw on the same seed — nothing repainted over the top.
fake.run = lambda ctx, req, **kw: (calls.append(("run", req, kw.get("seed"))) or
                                   {"image": os.path.join(tmp, "edited.png"),
                                    "layout": kw.get("layout"), "seed": kw.get("seed")})
I.save_layout_for(img, "a police car", LAY, width=1024, height=1024, seed=4242)
for instruction, want_cat in [
        ("пусть мужик будет блондин", "face_edit"),
        ("сделай рубашку зелёной",    "clothing_edit"),
        ("добавь скамейку",           "object_insert"),
        ("замени фон на пляж",        "background_replace"),
        ("сделай в стиле аниме",      "style_transfer"),
]:
    calls.clear()
    I.save_layout_for(img, "a police car", LAY, width=1024, height=1024, seed=4242)
    cat, out = I.route_edit_request(None, img, instruction)
    check(f"{instruction!r} -> {want_cat}, drawn from the boxes",
          cat == want_cat and any(c[0] == "run" for c in calls), f"{cat} {calls}")

# Removal is pixel FireRed and never re-renders the layout (user verdict, eraser-and-removal).
calls.clear()
I.save_layout_for(img, "a police car", LAY, width=1024, height=1024, seed=4242)
_cat, _ = I.route_edit_request(None, img, "убери человека справа")
check("'убери человека справа' -> person_remove, NOT redrawn from the boxes",
      _cat == "person_remove" and not any(c[0] == "run" for c in calls), f"{_cat} {calls}")

calls.clear()
I.save_layout_for(img, "a police car", LAY, width=1024, height=1024, seed=4242)
I.route_edit_request(None, img, "пусть мужик будет блондин")
check("the stored seed is reused, so an edit is not a re-roll",
      any(c[0] == "run" and c[2] == 4242 for c in calls), calls)

print("\nPIXEL OPERATIONS THE BOXES CANNOT EXPRESS STAY WHERE THEY BELONG")
for instruction in ("увеличь разрешение", "убери фон", "восстанови старое фото",
                    "расширь кадр"):
    cat = I.classify_edit_intent(instruction)
    check(f"{instruction!r} ({cat}) is not a layout edit",
          cat not in I._LAYOUT_EDITABLE, cat)

print("\nAN UPLOADED PHOTO STILL REACHES THE PIXEL PIPELINES")
calls.clear()
check("a photo with no boxes is not handed to the drawing agent",
      I.load_layout_for(plain) is None)
print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
