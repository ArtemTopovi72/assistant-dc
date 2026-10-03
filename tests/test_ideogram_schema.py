"""Our captions must match the published Ideogram 4 caption schema exactly.

The model is trained ONLY on structured JSON captions, and the pipeline runs a
CaptionVerifier that warns on unknown keys, missing required keys, and
out-of-order keys. Anything we get wrong is a train/inference mismatch that
costs quality silently — there is no error, just a worse picture (or the grey
safety card, see [ideogram4-drawing-mode]).

Rules encoded here, straight from the official prompting guide:
  · compositional_deconstruction is REQUIRED; background before elements
  · style_description key order is strict and differs per caption type:
      photo:     aesthetics, lighting, photo, medium, color_palette
      non-photo: aesthetics, lighting, medium, art_style, color_palette
  · exactly one of photo / art_style
  · color_palette is optional but must be LAST; <=16 overall, <=5 per element
  · hex colours are uppercase #RRGGBB — no shorthand, no lowercase
  · element key order: obj  -> type, bbox, desc, color_palette
                       text -> type, bbox, text, desc, color_palette
  · bbox is [ymin, xmin, ymax, xmax] on a 0-1000 grid
  · serialise with separators=(",",":") and ensure_ascii=False
  · resolution: multiples of 16, 256-2048 per side, aspect ratio <= 6:1

Run: venv/Scripts/python.exe tests/test_ideogram_schema.py
"""
import os, sys, json, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import ideogram as I

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


print("=" * 70)
print("STYLE_DESCRIPTION KEY ORDER")
print("=" * 70)

photo = I.build_caption("a street", [I.element("a cat")],
                        high_level="A cat on a street.",
                        aesthetics="moody", lighting="golden hour",
                        photo="35mm, f/1.4", medium="photograph",
                        palette=["#1B1B2F", "#E43F5A"])
check("photo caption key order",
      list(photo["style_description"]) ==
      ["aesthetics", "lighting", "photo", "medium", "color_palette"],
      str(list(photo["style_description"])))
check("photo caption has no art_style", "art_style" not in photo["style_description"])

art = I.build_caption("a street", [I.element("a cat")],
                      aesthetics="playful", lighting="flat",
                      medium="illustration", art_style="flat vector",
                      palette=["#FFFFFF"])
check("non-photo caption key order",
      list(art["style_description"]) ==
      ["aesthetics", "lighting", "medium", "art_style", "color_palette"],
      str(list(art["style_description"])))
check("non-photo caption has no photo key", "photo" not in art["style_description"])

nopal = I.build_caption("bg", [I.element("x")], aesthetics="a", lighting="b",
                        photo="c", medium="photograph")
check("color_palette is omitted when empty, not sent blank",
      "color_palette" not in nopal["style_description"])

check("top-level order: description, style, composition",
      list(photo) == ["high_level_description", "style_description",
                      "compositional_deconstruction"], str(list(photo)))
check("compositional_deconstruction is always present",
      "compositional_deconstruction" in
      I.build_caption("bg", [I.element("x")]))
check("background comes before elements",
      list(photo["compositional_deconstruction"]) == ["background", "elements"])


print()
print("=" * 70)
print("ELEMENT KEY ORDER AND BBOX")
print("=" * 70)

obj = I.element("a red car", I.bbox(0.1, 0.2, 0.3, 0.4), palette=["#FF0000"])
check("obj element key order",
      list(obj) == ["type", "bbox", "desc", "color_palette"], str(list(obj)))
txt = I.element("bold title", I.bbox(0, 0, 1, 0.2), text="ACME",
                palette=["#000000"])
check("text element key order",
      list(txt) == ["type", "bbox", "text", "desc", "color_palette"], str(list(txt)))
check("text presence sets type=text", txt["type"] == "text")
check("no text means type=obj", obj["type"] == "obj")
check("a boxless element omits bbox entirely", "bbox" not in I.element("x"))

# [ymin, xmin, ymax, xmax] on 0-1000 — getting this backwards mirrors the layout.
b = I.bbox(0.2, 0.1, 0.4, 0.6)      # x=0.2 y=0.1 w=0.4 h=0.6
check("bbox is [ymin, xmin, ymax, xmax] on a 0-1000 grid",
      b == [100, 200, 700, 600], str(b))
check("bbox clamps to the grid", I.bbox(-1, -1, 5, 5) == [0, 0, 1000, 1000],
      str(I.bbox(-1, -1, 5, 5)))


print()
print("=" * 70)
print("HEX COLOUR NORMALISATION")
print("=" * 70)

check("lowercase is upcased", I.hex_palette(["#1b1b2f"], 16) == ["#1B1B2F"])
check("shorthand is expanded", I.hex_palette(["#fa0"], 16) == ["#FFAA00"])
check("a missing # is added", I.hex_palette(["1B1B2F"], 16) == ["#1B1B2F"])
check("duplicates collapse", I.hex_palette(["#fff", "#FFFFFF"], 16) == ["#FFFFFF"])
check("junk is dropped, not guessed",
      I.hex_palette(["red", "#12345", "", None, "#00FF00"], 16) == ["#00FF00"])
check("style palette caps at 16",
      len(I.hex_palette([f"#{i:02X}0000" for i in range(30)], 16)) == 16)
check("element palette caps at 5",
      len(I.element("x", palette=[f"#{i:02X}0000" for i in range(30)])
          ["color_palette"]) == 5)
check("an all-junk palette omits the key",
      "color_palette" not in I.element("x", palette=["nope"]))
check("every emitted colour matches #RRGGBB",
      all(len(c) == 7 and c[0] == "#" and c[1:].upper() == c[1:]
          for c in I.hex_palette(["#fa0", "abcdef", "#123456"], 16)))


print()
print("=" * 70)
print("SERIALISATION")
print("=" * 70)

# Comments mention the old spelling, so look at CODE only — an earlier version
# of this check matched the explanatory comment and failed on a correct fix.
src = "\n".join(line.split("#", 1)[0]
                for line in inspect.getsource(I.generate).splitlines())
_dump = [l for l in src.splitlines() if "json.dumps(caption" in l]
check("generate() serialises the caption exactly once", len(_dump) == 1, str(_dump))
_dump = _dump[0] if _dump else ""
check("the caption is serialised with compact separators",
      'separators=(",", ":")' in _dump, _dump.strip())
check("ensure_ascii=False (no \\uXXXX escapes)", "ensure_ascii=False" in _dump)
check("no indent= pretty-printing on the submitted caption",
      "indent" not in _dump, _dump.strip())

# Cyrillic must survive as literal characters, which is the whole point of
# ensure_ascii=False — the verifier warns on escapes.
ru = json.dumps(I.build_caption("площадь", [I.element("кот")]),
                separators=(",", ":"), ensure_ascii=False)
check("Cyrillic stays literal", "кот" in ru and "\\u" not in ru)
check("compact form has no spaces after separators",
      '", "' not in ru and '": "' not in ru)


print()
print("=" * 70)
print("RESOLUTION ENVELOPE")
print("=" * 70)

check("multiples of 16 are preserved", I._snap16(1024) == 1024)
check("odd sizes snap to 16", I._snap16(1020) % 16 == 0)
check("below the floor clamps up to 256", I._snap16(64) == 256)
check("above the ceiling clamps down to 2048", I._snap16(4096) == 2048)
check("the documented max is 2048, not the old 4096", I.IDEOGRAM_MAX_SIDE == 2048)

for w, h in [(1024, 1024), (1536, 1024), (1024, 1536), (1920, 1088),
             (2048, 768), (1024, 1792), (1600, 400)]:
    ow, oh = I.ideogram_size(w, h)
    check(f"documented size {w}x{h} passes through unchanged",
          (ow, oh) == (w, h), f"got {ow}x{oh}")

for w, h in [(4000, 300), (300, 4000), (100, 100), (5000, 5000)]:
    ow, oh = I.ideogram_size(w, h)
    ratio = max(ow, oh) / min(ow, oh)
    check(f"{w}x{h} lands inside the envelope",
          256 <= ow <= 2048 and 256 <= oh <= 2048
          and ow % 16 == 0 and oh % 16 == 0 and ratio <= 6.0 + 1e-9,
          f"got {ow}x{oh} ratio={ratio:.2f}")

# Over-wide requests keep their orientation rather than being squared off.
ow, oh = I.ideogram_size(4000, 300)
check("an over-wide banner stays landscape", ow > oh, f"{ow}x{oh}")
ow, oh = I.ideogram_size(300, 4000)
check("an over-tall banner stays portrait", oh > ow, f"{ow}x{oh}")


print()
print("=" * 70)
print("MUTATION: BREAK EACH RULE, THE SUITE MUST NOTICE")
print("=" * 70)

_saved_hex = I.hex_palette
try:
    I.hex_palette = lambda c, limit: [str(x) for x in (c or [])][:limit]  # no normalising
    bad = I.build_caption("bg", [I.element("x")], aesthetics="a", lighting="b",
                          photo="c", medium="photograph", palette=["#1b1b2f"])
    check("un-normalised hex would be caught",
          bad["style_description"]["color_palette"] != ["#1B1B2F"])
finally:
    I.hex_palette = _saved_hex

check("and normalisation is restored",
      I.build_caption("bg", [I.element("x")], palette=["#1b1b2f"])
      ["style_description"]["color_palette"] == ["#1B1B2F"])


print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
