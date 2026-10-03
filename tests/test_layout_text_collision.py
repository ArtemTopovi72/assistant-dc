"""Regression: lettering painted across a subject.

Found in a real render. outputs/ideogram_00008_.png came back with a
"General Gingerbread" banner laid straight over the portrait's face, and
draw_geometry.geometry_report returned [] for that layout -- it saw no problem
at all.

The cause is the metric. Overlap was scored with IoU, and IoU is structurally
blind to this shape of collision: the sign is a thin strip (7% of the frame)
and the subject fills half of it, so the union swamps the intersection and the
pair scores 5% while 41% of the sign sits on the man. What matters is how much
of the LETTERING is ruined, which is intersection over the TEXT box's own area.

The opposite error is just as easy to make: ideogram.ensure_text_elements
deliberately attaches a sign to the surface the user named -- "a neon sign
above the door" becomes an element described "lettering on a small corner cafe
storefront", which SHOULD overlap the storefront. So an overlap is only a
collision when the sign is NOT attached to the thing it covers, and the words
are what tell them apart.

Run: .\venv\Scripts\python.exe tests\test_layout_text_collision.py
"""
import sys
sys.path.insert(0, ".")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import draw_geometry as G
import ideogram as I

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if not ok else ""))


def kinds(layout):
    return [p["kind"] for p in G.geometry_report(layout)]


# --- the exact layout that produced the bad render -------------------------
GINGERBREAD = {
    "background": "a dark, moody oil painting",
    "elements": [
        {"desc": "General Gingerbread, a humanoid gingerbread man with golden-brown "
                 "dough skin and white frosting mustache",
         "text": "", "x": 0.2, "y": 0.15, "w": 0.6, "h": 0.85},
        {"desc": "Intricate icing medals and chocolate epaulettes on the chest",
         "text": "", "x": 0.35, "y": 0.45, "w": 0.3, "h": 0.25},
        {"desc": "A tall, decorated gingerbread officer's shako hat with royal icing",
         "text": "", "x": 0.3, "y": 0.05, "w": 0.4, "h": 0.25},
        {"desc": 'a sign carrying the lettering "General Gingerbread"',
         "text": "General Gingerbread",
         "x": 0.044, "y": 0.12, "w": 0.912, "h": 0.08},
    ],
}

check("the banner across the face is reported at all",
      "text_collision" in kinds(GINGERBREAD), kinds(GINGERBREAD))

# IoU is what missed it -- pin that, so nobody "simplifies" the metric back.
_sign, _man = GINGERBREAD["elements"][3], GINGERBREAD["elements"][0]
check("IoU alone cannot see this collision", G._iou(_sign, _man) < 0.10,
      f"IoU={G._iou(_sign, _man):.3f}")
check("coverage of the text box does see it", G._covered(_sign, _man) >= 0.25,
      f"covered={G._covered(_sign, _man):.3f}")

fixed, notes = G.auto_fix_geometry(GINGERBREAD)
check("the repair clears it", "text_collision" not in kinds(fixed), kinds(fixed))
check("the repair says what it did",
      any("lettering" in n for n in notes), str(notes))

_sign_after = fixed["elements"][3]
check("the lettering keeps the size its string needs",
      abs(_sign_after["w"] - 0.912) < 1e-6 and abs(_sign_after["h"] - 0.08) < 1e-6,
      f'{_sign_after["w"]}x{_sign_after["h"]}')
check("and it actually moved off the head",
      _sign_after["y"] + _sign_after["h"] <= 0.15 + 1e-9,
      f'y={_sign_after["y"]}')

# --- the false positive this must NOT create -------------------------------
_base = I.normalize_layout({"background": "a street at night", "elements": [
    {"desc": "a small corner cafe storefront", "x": .1, "y": .2, "w": .6, "h": .5},
    {"desc": "people at outdoor tables", "x": .1, "y": .6, "w": .8, "h": .3}]})
_cafe = I.ensure_text_elements(
    _base, 'a neon sign above the door reading "CAFE ROSA"')
check("a sign attached to the surface it names is NOT a collision",
      "text_collision" not in kinds(_cafe), kinds(_cafe))
check("...and auto_fix leaves it where it belongs",
      "text_collision" not in kinds(G.auto_fix_geometry(_cafe)[0]))

# The word-overlap test has to ignore the lettering string itself: the sign
# spells the general's NAME, so the raw descriptions share "General" and
# "Gingerbread" and the collision would look intentional.
check("the lettering string does not count as attachment evidence",
      not G._attached_to(_sign, _man),
      str(G._attach_tokens(_sign["desc"], _sign["text"])))
check("a real surface reference does count",
      G._attached_to({"desc": "lettering on a small corner cafe storefront",
                      "text": "CAFE ROSA"},
                     {"desc": "a small corner cafe storefront"}))

# --- a clean layout must stay untouched (idempotence) ----------------------
CLEAN = {"background": "a field", "elements": [
    {"desc": "a red barn", "text": "", "x": 0.05, "y": 0.55, "w": 0.5, "h": 0.4},
    {"desc": "a wooden signpost", "text": "FARM",
     "x": 0.62, "y": 0.1, "w": 0.32, "h": 0.16}]}
check("a clean layout reports nothing", kinds(CLEAN) == [], kinds(CLEAN))
_once, _ = G.auto_fix_geometry(CLEAN)
_twice, _ = G.auto_fix_geometry(_once)
check("auto_fix is idempotent on it",
      [dict(e) for e in _once["elements"]] == [dict(e) for e in _twice["elements"]])

# --- separating a pile keeps each subject's shape and floor ---------------
_cats = {"elements": [{"desc": f"a {c} cat", "x": x, "y": .4, "w": .6, "h": .6}
                      for c, x in (("ginger", .4), ("grey", .2), ("black", .25))], "background": "sill"}
_f, _ = G.auto_fix_geometry(_cats)
check("separated subjects keep their shape and their floor",
      all(abs(e["w"] - e["h"]) < 0.02 and abs(e["y"] + e["h"] - 1.0) < 0.02 for e in _f["elements"]),
      _f["elements"])

ok = sum(1 for _, o, _ in checks if o)
print(f"\n{ok}/{len(checks)} checks passed")
sys.exit(0 if ok == len(checks) else 1)
