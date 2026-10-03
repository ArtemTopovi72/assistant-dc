"""Ideogram never silently changes the KIND of picture the user asked for.

Ideogram 4 is fed a structured JSON caption, not prose. If `style_description`
comes out empty the model is NOT neutral — it falls back to its own stylised house
look. So every path that loses the style turns "draw it realistically" into a
cartoon, silently, with no error anywhere.

The live failure this suite was written from: the planner emitted malformed JSON
(a stray "and" spliced in front of a key, then a duplicate ```json block).
`_extract_json` salvaged the first thing that happened to parse — one of the INNER
element objects — and every downstream guard accepted it because it was a dict.

Run: venv/Scripts/python.exe tests/test_ideogram_style.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
import _look_stub  # noqa: F401  the picture's look, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import ideogram as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


print("=" * 66)
print("JSON EXTRACTION NEVER RETURNS A FRAGMENT")
print("=" * 66)

# Verbatim shape of the reply that caused the bug.
BROKEN = '''{
  "high_level_description": "A realistic photo of a ginger cat.",
  "aesthetics": "photorealistic, sharp focus",
  "lighting": "soft natural light",
 and "photo": "85mm lens, f/1.8",
  "medium": "photography",
  "background": "a blurred living room",
  "elements": [
    {"desc": "a fluffy ginger tabby cat", "x": 0.3, "y": 0.3, "w": 0.4, "h": 0.6},
    {"desc": "a polished wood floor", "x": 0, "y": 0.7, "w": 1.0, "h": 0.3}
  ]
}
```json
{"high_level_description": "duplicate copy"}
'''

got = I._extract_json(BROKEN)
check("a malformed reply does not yield an element fragment",
      got is not None and "desc" not in got, json.dumps(got, ensure_ascii=False)[:160])
check("the repaired object keeps its style",
      (got or {}).get("medium") == "photography"
      and "85mm" in str((got or {}).get("photo")),
      json.dumps(got, ensure_ascii=False)[:200])
check("the repaired object keeps its elements",
      len((got or {}).get("elements") or []) == 2,
      str(len((got or {}).get("elements") or [])))

# A duplicated second object must not win over the real first one.
DOUBLE = ('{"high_level_description": "first", "medium": "photography", '
          '"elements": [{"desc": "a"}]}\n```json\n'
          '{"high_level_description": "second"}\n```')
got = I._extract_json(DOUBLE)
check("the first complete object wins over a duplicate",
      (got or {}).get("high_level_description") == "first",
      json.dumps(got, ensure_ascii=False)[:120])

# A stub that appears BEFORE the real object: order alone would pick the stub, so
# the object with the most layout keys has to win.
STUB_FIRST = ('{"high_level_description": "ok"}\n'
              '{"high_level_description": "the real one", "medium": "photography",'
              ' "photo": "35mm", "aesthetics": "gritty", "background": "a field",'
              ' "elements": [{"desc": "a tank"}]}')
got = I._extract_json(STUB_FIRST)
check("the richest object wins over an earlier stub",
      (got or {}).get("high_level_description") == "the real one",
      json.dumps(got, ensure_ascii=False)[:140])

# An UNBALANCED brace inside a string is what breaks naive brace counting: the
# depth never returns to zero and the object is never yielded at all.
# Trailing prose is what forces the brace scanner to do the work: without it the
# whole reply parses in one go and the scanner is never consulted.
BRACEY = ('{"background": "a sign reading {OPEN", "medium": "photography",'
          ' "elements": []}\nHope that helps!')
got = I._extract_json(BRACEY)
check("an unbalanced brace inside a string does not swallow the object",
      (got or {}).get("background") == "a sign reading {OPEN",
      json.dumps(got, ensure_ascii=False)[:120])

check("prose with no JSON at all yields None",
      I._extract_json("Sure! I can draw that for you.") is None)
check("an empty reply yields None", I._extract_json("") is None)


print()
print("=" * 66)
print("A REALISM REQUEST IS NEVER LEFT STYLELESS")
print("=" * 66)

REALISM = ["a realistic photo of a tank", "photorealistic portrait",
           "нарисуй реалистичное фото кота", "a hyperrealistic DSLR shot",
           "make it lifelike"]
STYLED = ["a cartoon cat", "anime girl in the rain", "an oil painting of a harbour",
          "мультяшный кот", "a watercolor sketch",
          "a realistic cartoon"]          # explicit style wins over "realistic"

for p in REALISM:
    check(f"realism detected: {p[:34]}", I.wants_realism(p))
for p in STYLED:
    check(f"explicit style respected: {p[:34]}", not I.wants_realism(p))

# The exact path that produced the cartoon: planner fails -> blank layout.
lay = I.apply_style_floor(I.blank_layout("a realistic photo of a tank"),
                          "a realistic photo of a tank")
check("a failed planner still yields a photographic style",
      lay["medium"] == "photography" and lay["photo"] and lay["aesthetics"],
      json.dumps(lay, ensure_ascii=False)[:200])
check("the floor leaves art_style empty (it would discard the photo cues)",
      not lay.get("art_style"), repr(lay.get("art_style")))

cap = I.layout_to_caption(lay)
style = cap.get("style_description") or {}
check("the floor survives into the caption Ideogram is fed",
      style.get("medium") == "photography" and style.get("photo"),
      json.dumps(style, ensure_ascii=False)[:200])

# The floor must not overrule a style the planner DID choose.
styled = I.normalize_layout({"medium": "oil painting", "art_style": "impressionist",
                             "elements": [{"desc": "a harbour"}]}, "an oil painting")
before = dict(styled)
after = I.apply_style_floor(styled, "an oil painting of a harbour")
check("an existing style is left alone", after["medium"] == before["medium"]
      and after["art_style"] == before["art_style"],
      json.dumps(after, ensure_ascii=False)[:160])

# And it must not force photography onto someone who asked for a cartoon.
toon = I.apply_style_floor(I.blank_layout("a cartoon cat"), "a cartoon cat")
check("a cartoon request is not forced into photography",
      toon["medium"] != "photography", repr(toon["medium"]))

# ── the floor must cover the COMMON case, not just explicit realism ──────────
# The original fix only applied when the prompt said "realistic", so an ordinary
# "нарисуй машину во дворе" still shipped an EMPTY style_description — and empty
# is not neutral, it hands the choice to Ideogram, whose house look is cartoon.
# That is why every picture came back a cartoon unless realism was asked for.
def _styleless(layout):
    return not any(str(layout.get(k) or "").strip()
                   for k in ("aesthetics", "lighting", "photo", "medium", "art_style"))

NEUTRAL = ("нарисуй машину во дворе", "draw a cat sitting on a fence",
           "картинка: закат над городом", "a police car in a narrow yard")
for p in NEUTRAL:
    lay = I.apply_style_floor(I.blank_layout(p), p)
    check(f"a neutral prompt is never styleless: {p[:30]}", not _styleless(lay),
          json.dumps(lay, ensure_ascii=False)[:120])
    check(f"and defaults to photography: {p[:30]}",
          lay.get("medium") == "photography" and not lay.get("art_style"),
          repr(lay.get("medium")))

# A named drawn style must survive a planner failure as ITSELF — forcing
# photography onto "нарисуй карикатуру" is the same bug in the other direction.
NAMED = {
    "нарисуй карикатуру на полицейских": "caricature",
    "сделай шарж на кота":               "caricature",
    "нарисуй мультяшную машину":         "cartoon",
    "аниме девушка":                     "anime",
    "an oil painting of a harbour":      "oil painting",
    "нарисуй акварелью мост":            "watercolour",
    "комикс про кота":                   "comic strip",
}
for p, want in NAMED.items():
    lay = I.apply_style_floor(I.blank_layout(p), p)
    check(f"named style survives a planner failure: {p[:28]}",
          lay.get("art_style") == want, repr(lay.get("art_style")))
    check(f"and is not turned into a photograph: {p[:28]}",
          lay.get("medium") != "photography" and not lay.get("photo"),
          repr(lay.get("medium")))
    check(f"named_style() reads it back: {p[:28]}",
          I.named_style(p) == want, repr(I.named_style(p)))

check("a neutral prompt names no style", I.named_style("нарисуй машину") == "")

# A style word that is also a NOUN FOR A PROP names no style. Live, 2026-09-12:
# "hangs a painting depicting a calm sea" made the floor call the whole scene
# "a painting" and the render came back as a framed canvas with an inset border.
PROPS = ["On the wall behind the sofa hangs a painting depicting a calm sea.",
         "a living room with a painting of the sea above the sofa",
         "a boy holding a drawing of a dog",
         "a desk with a sketch and a pencil",
         "a girl reading a comic on the sofa",
         "a realistic photo of a wall with a framed painting"]
for p in PROPS:
    check(f"a prop is not a style: {p[:34]}", I.named_style(p) == "", repr(I.named_style(p)))
check("a prop does not cancel a realism request",
      I.wants_realism("a realistic photo of a wall with a framed painting"))
for p, want in {"draw a painting of a harbour": "painting",
                "a painting of a cat": "painting",
                "make it look like a sketch": "sketch",
                "нарисуй кота в стиле живописи": "oil painting",
                "children's drawing style, a dog": "illustration"}.items():
    check(f"a style named as a style still counts: {p[:30]}", I.named_style(p) == want, repr(I.named_style(p)))
check("named_style tolerates None", I.named_style(None) == "")
# A named style has to WIN over a realism word in the same sentence, which is the
# only way to tell "caricature is in _STYLE_WORDS" from "this prompt happens to
# contain no realism word at all".
for p in ("нарисуй карикатуру, реалистично прорисованную",
          "сделай шарж, но фотореалистичный",
          "комикс в реалистичном стиле",
          "акварельный рисунок, как в жизни"):
    check(f"a named style beats a realism word: {p[:34]}",
          not I.wants_realism(p), f"wants_realism={I.wants_realism(p)}")
    lay = I.apply_style_floor(I.blank_layout(p), p)
    check(f"and the floor keeps it drawn: {p[:34]}",
          lay.get("medium") != "photography", repr(lay.get("medium")))

# The floor is a FLOOR, not an override: when the planner already chose specific
# camera work for a realism prompt, the generic default must not replace it.
rich = I.normalize_layout({"medium": "photography", "photo": "85mm lens, f/1.4",
                           "aesthetics": "cinematic, moody",
                           "lighting": "golden hour backlight",
                           "elements": [{"desc": "a tank"}]},
                          "a realistic photo of a tank")
after = I.apply_style_floor(rich, "a realistic photo of a tank")
check("the planner's own camera work is not replaced by the default",
      after["photo"] == "85mm lens, f/1.4"
      and after["lighting"] == "golden hour backlight",
      json.dumps(after, ensure_ascii=False)[:200])


print()
print("=" * 66)
print("PLANNER FALLBACKS")
print("=" * 66)

# ctx=None is the offline fallback path both planners take.
lay = I.plan_layout(None, "a realistic photo of a tank")
check("plan_layout fallback carries a style",
      lay["medium"] == "photography", repr(lay["medium"]))

cap = I.plan_caption(None, "a realistic photo of a tank")
style = (cap.get("style_description") or {})
check("plan_caption fallback carries a style",
      style.get("medium") == "photography",
      json.dumps(style, ensure_ascii=False)[:160])
check("plan_caption fallback still has a scene",
      (cap.get("compositional_deconstruction") or {}).get("elements"),
      json.dumps(cap, ensure_ascii=False)[:200])


print()
print("=" * 66)
print("THE PLANNER RETRIES INSTEAD OF ACCEPTING A FRAGMENT")
print("=" * 66)

import types as _types
_real_llm = sys.modules.get("llm")


class _FakeLLM:
    """Stands in for llm.send_to_lm_studio so the retry can be driven exactly."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
    def send_to_lm_studio(self, ctx, messages, **kw):
        self.calls += 1
        return {"content": self.replies.pop(0) if self.replies else ""}


def _with_llm(fake, fn):
    sys.modules["llm"] = fake
    try:
        return fn()
    finally:
        if _real_llm is not None:
            sys.modules["llm"] = _real_llm
        else:
            sys.modules.pop("llm", None)


GOOD = ('{"high_level_description": "a tank", "medium": "photography",'
        ' "photo": "35mm", "background": "a field",'
        ' "elements": [{"desc": "a tank", "x": 0.2, "y": 0.3, "w": 0.6, "h": 0.5}]}')
FRAGMENT = '{"desc": "a fluffy ginger cat", "x": 0.3, "y": 0.3, "w": 0.4, "h": 0.6}'

fake = _FakeLLM([FRAGMENT, GOOD])
got = _with_llm(fake, lambda: I._ask_planner(object(), "a realistic photo of a tank"))
check("a fragment reply is rejected and the planner is asked again",
      fake.calls == 2, f"called {fake.calls}x")
check("the retry's good answer is the one used",
      (got or {}).get("medium") == "photography",
      json.dumps(got, ensure_ascii=False)[:140])

fake = _FakeLLM([GOOD, GOOD])
_with_llm(fake, lambda: I._ask_planner(object(), "a realistic photo of a tank"))
check("a good first answer costs only one call", fake.calls == 1,
      f"called {fake.calls}x")

fake = _FakeLLM([FRAGMENT, FRAGMENT])
got = _with_llm(fake, lambda: I.plan_layout(object(), "a realistic photo of a tank"))
check("two bad answers fall back rather than using a fragment",
      got["medium"] == "photography" and got["elements"],
      json.dumps(got, ensure_ascii=False)[:180])


print()
print("=" * 66)
print("REPAIR IS NARROW")
print("=" * 66)

# The repair must only touch a conjunction spliced before a KEY, never text.
keep = '{"background": "a boy and a dog", "medium": "photography"}'
check("a legitimate 'and' inside a value is untouched",
      (I._extract_json(keep) or {}).get("background") == "a boy and a dog",
      json.dumps(I._extract_json(keep), ensure_ascii=False))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
