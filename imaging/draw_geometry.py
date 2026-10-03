"""Layout geometry: overlap repair, min-area growth and spreading.

Lifted out of draw_agent.py. Pure geometry -- boxes in, boxes out, no LLM call
and no module state. auto_fix_geometry and geometry_report are the entry points;
the rest are its helpers.

The callers (run, edit_layout) stay in draw_agent, so any suite that patches
draw_agent.auto_fix_geometry still reaches the code that runs.
"""
import copy
import json
import random
import re
import ideogram
import text_layout as _text_layout
from text_layout import text_geometry, _norm_text
from utils import safe_json_from_llm
import draw_text as _draw_text

from draw_text import _clamp_box, _els

# The geometry tunables. Defined HERE and nowhere else: draw_agent had the only
# other copy and no longer uses them, and two copies of a tunable are how a knob
# silently stops applying to half the code.
# Measured, not assumed (bench/draw_minarea_probe.py): one kitchen, one seed,
# the mug swept across 0.6 / 1.1 / 2.0 / 3.0 % of the frame. It came out clearly
# drawn and the scene stayed coherent at EVERY size, including the smallest.
#
# The old floor was 4% with the repair aiming at 6.4%, and that overshoot is
# what several bad renders were made of: a mug at 6% of a 1024px frame is the
# size of a hero product shot, and the model duly composed one -- a giant mug
# floating over the table, an inset panel with a white border, an advertising
# banner with a logo and bullet points. The floor was buying a vanishing prop
# that does not vanish, at the price of a photograph that turns into an advert.
MIN_AREA = 0.012         # below ~1.2% of the frame an element risks being lost
MAX_ELEMENTS = 6         # more than this and the renderer stops honouring boxes
MERGE_IOU = 0.45         # two boxes overlapping this much come out as one thing
FULL_FRAME = 0.92        # an element this wide AND tall competes with the background
# How much of a LETTERING box may sit on top of a non-text element before the
# sign is judged to be painted across the subject rather than placed in the
# scene. Measured as a fraction of the TEXT box's own area, not IoU -- see the
# text_collision branch in geometry_report for why IoU cannot see this.
TEXT_CLEARANCE = 0.25


# --------------------------------------------------------------------------- #
# The arrangement <-> result relationship (deterministic)
# --------------------------------------------------------------------------- #
def _inter(a, b) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return ix * iy


_ATTACH_STOPWORDS = frozenset("""
a an the of on in at to and or with its his her their is are that this for by from
sign signs lettering letters letter text word words reading says saying spelled
written label banner caption clean bold block capitals every correctly formed
evenly spaced large small big
надпись надписью вывеска вывеской слово слова текст буквы буква на с и в
""".split())


def _attach_tokens(desc: str, text: str) -> set:
    """Content words of a description, minus the lettering string it quotes.

    The string itself has to come out or nothing works: a sign that reads
    "General Gingerbread" shares those two words with the general it is painted
    over, and would look "attached" to him purely because it spells his name.
    """
    strip = set(re.findall(r"\w+", (text or "").lower()))
    # Three letters is enough: "car", "cup", "jar", "mug", "box", "van" are
    # exactly the things labels sit on, and at >= 4 "a sign on the roof of the
    # car" never attached to "a dark patrol car" -- so the sign was pushed off
    # its own host as a collision (found re-anchoring tests/test_draw_ops.py).
    return {w for w in re.findall(r"\w+", (desc or "").lower())
            if len(w) >= 3 and w not in _ATTACH_STOPWORDS and w not in strip}


def _attached_to(text_el, other) -> bool:
    """Whether a lettering box was deliberately put ON this element.

    A sign belongs on the surface the user named -- "a neon sign above the door"
    becomes an element described "lettering on a small corner cafe storefront",
    which SHOULD overlap the storefront. That is the difference between a label
    and a banner across someone's face, and it is visible in the words: the
    attached box quotes the surface it sits on.
    """
    return bool(_attach_tokens(text_el.get("desc"), text_el.get("text"))
                & _attach_tokens(other.get("desc"), ""))


# Words that say a box was PUT on/under/inside another one. Deliberately a
# preposition list and not a noun matcher: the editor answers in English about a
# layout written in Russian ("a ginger cat sleeping on the sofa" next to "синий
# диван у стены"), so matching the two by their nouns cannot work, while the
# preposition survives either language.
_SUPPORT_RE = re.compile(
    r"\b(?:on|onto|atop|upon|under|underneath|beneath|below|inside|within)\b"
    r"|\bin\s+the\b|\bon\s+top\b"
    r"|(?:^|\s)(?:на|над|под|внутри|в)\s", re.IGNORECASE)


def _stacked_on(small, big) -> bool:
    """`small` was deliberately placed ON (or under, or in) `big`.

    Two subjects on the same spot normally fuse into one thing, which is what
    the merge repair exists to prevent. But "посади кота на диван" IS an
    instruction to put one box on another, and separating them afterwards threw
    the edit away: the user asked for a cat on the sofa, the layout got a cat on
    the sofa, and the repair moved it back onto the floor -- the picture came
    back without the change and the agent reported success.

    Narrow on purpose: the description has to SAY it sits on something, the box
    has to be the smaller one, and it has to be mostly swallowed by the bigger.
    A cat merely overlapping a table by half does not qualify, so ordinary
    collisions are still repaired.
    """
    if small["w"] * small["h"] > big["w"] * big["h"] * 0.85:
        return False
    if not _SUPPORT_RE.search(str(small.get("desc") or "")):
        return False
    return _covered(small, big) >= 0.6


def _scale_to_area(el, target):
    """Grow a box about its centre until it covers `target` of the frame."""
    f = (target / max(1e-9, el["w"] * el["h"])) ** 0.5
    cx, cy = el["x"] + el["w"] / 2, el["y"] + el["h"] / 2
    el["w"], el["h"] = min(1.0, el["w"] * f), min(1.0, el["h"] * f)
    el["x"], el["y"] = cx - el["w"] / 2, cy - el["h"] / 2
    _clamp_box(el)


def _nest_on(small, big):
    """Shrink `small` so it sits INSIDE `big` instead of coinciding with it.

    "Передвинь кота на диван" comes back as the cat's box set to the sofa's box,
    exactly. Two identical boxes are the worst case for the renderer -- it fuses
    them into one cat-shaped sofa -- so the merge repair separated them, which
    threw the instruction away: the cat went back to the floor and the user was
    told it had been moved. Nesting keeps the instruction (it lands on the sofa)
    and gives the renderer two distinguishable regions.
    """
    # The host has to be big enough to hold a child that clears MIN_AREA --
    # otherwise nesting shrinks the child under the "renders as a smudge" bar,
    # the next pass grows it back, and the two repairs fight until the merge
    # separator wins and the instruction is lost. Grow the host first.
    if big["w"] * big["h"] < MIN_AREA * 2.8:
        _scale_to_area(big, MIN_AREA * 2.8)
    w = max(0.02, min(small["w"], big["w"] * 0.62))
    h = max(0.02, min(small["h"], big["h"] * 0.62))
    if w * h < MIN_AREA:                       # keep the child renderable
        f = (MIN_AREA / max(1e-9, w * h)) ** 0.5
        w, h = min(w * f, big["w"] * 0.85), min(h * f, big["h"] * 0.85)
    small["w"], small["h"] = w, h
    small["x"] = big["x"] + (big["w"] - w) / 2
    # Sat on top of, not centred in: the thing placed on a surface belongs in
    # its upper half, which is also what keeps the two boxes visually distinct.
    small["y"] = big["y"] + max(0.0, (big["h"] - h) * 0.25)
    _clamp_box(small)


def _deliberate_stack(a, b) -> bool:
    """Either box deliberately placed on the other."""
    return _stacked_on(a, b) or _stacked_on(b, a)


def _nestable(a, b):
    """(small, big) when one was PUT on the other but is not small enough to
    read as a separate object; None otherwise."""
    if _deliberate_stack(a, b):
        return None                     # already nested, leave it alone
    for small, big in ((a, b), (b, a)):
        if _SUPPORT_RE.search(str(small.get("desc") or "")) and big["w"] * big["h"] >= small["w"] * small["h"] * 0.9:
            return small, big
    return None


def _covered(el, other) -> float:
    """How much of EL's own area `other` sits on. Asymmetric on purpose."""
    return _inter(el, other) / max(1e-6, el["w"] * el["h"])


def _iou(a, b) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def geometry_report(layout: dict) -> list:
    """Problems in the ARRANGEMENT itself, before anything is rendered.

    Each entry is {"kind", "why", "targets"} — the same vocabulary the fixer and the
    UI use, so a problem can be explained to the user in the words that caused it.
    """
    els = (ideogram.normalize_layout(layout).get("elements") or [])
    out = []
    for i, el in enumerate(els):
        area = el["w"] * el["h"]
        if area < MIN_AREA:
            out.append({"kind": "too_small", "targets": [i],
                        "why": f'“{el["desc"][:40]}” covers {area * 100:.1f}% of the '
                               f"frame — that small, it renders as a smudge or vanishes"})
        if el["w"] >= FULL_FRAME and el["h"] >= FULL_FRAME and len(els) > 1:
            out.append({"kind": "full_frame", "targets": [i],
                        "why": f'“{el["desc"][:40]}” fills the whole frame, so it '
                               f"competes with the background and the other elements"})
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            iou = _iou(els[i], els[j])
            if iou >= MERGE_IOU and _nestable(els[i], els[j]) is not None:
                small, big = _nestable(els[i], els[j])
                out.append({"kind": "coincident",
                            "targets": [els.index(small), els.index(big)],
                            "why": f'“{small["desc"][:28]}” is placed ON '
                                   f'“{big["desc"][:28]}” but covers the same box — '
                                   f"the renderer fuses them into one object"})
                continue
            if iou >= MERGE_IOU and not _deliberate_stack(els[i], els[j]):
                out.append({"kind": "merged", "targets": [i, j],
                            "why": f'“{els[i]["desc"][:28]}” and “{els[j]["desc"][:28]}” '
                                   f"overlap by {iou * 100:.0f}% — the renderer fuses "
                                   f"boxes that sit on top of each other into one thing"})
    # Lettering laid across a subject. IoU is structurally blind to this: a sign
    # is a thin strip and a person fills half the frame, so the union swamps the
    # intersection and the pair scores ~5% while the sign is painted straight
    # over the face. Observed exactly that way -- a "General Gingerbread" banner
    # at y 0.12-0.20 across a portrait whose head starts at y 0.15, and
    # geometry_report returned []. Measure against the TEXT box's own area
    # instead, which is the thing being ruined.
    _text_ix = {i for i, _ in _draw_text._text_els({"elements": els})}
    for i in sorted(_text_ix):
        hits = [(_covered(els[i], o), j) for j, o in enumerate(els)
                if j != i and not _attached_to(els[i], o)]
        # Summed: a title spread over two musicians' heads, 13% on each, was
        # clear by every single-pair measure (live 2026-09-29).
        if hits and sum(c for c, _ in hits) >= TEXT_CLEARANCE:
            j = max(hits)[1]
            other = els[j]
            out.append({"kind": "text_collision", "targets": [i, j],
                        "why": f'the lettering “{str(els[i].get("text") or "")[:20]}” '
                               f'sits {sum(c for c, _ in hits) * 100:.0f}% on top of '
                               f'“{other["desc"][:28]}” — a sign painted across a '
                               f"subject comes out fused into it, not read as text"})
    if len(els) > MAX_ELEMENTS:
        out.append({"kind": "crowded", "targets": list(range(MAX_ELEMENTS, len(els))),
                    "why": f"{len(els)} elements — past about {MAX_ELEMENTS} the model "
                           f"stops honouring the boxes and the scene turns to soup"})
    return out


def _grow_to_min_area(el):
    """Scale a box up until it is big enough to survive the render.

    Scaling both sides equally is not enough on its own: a full-width sliver has no
    room to grow sideways, the clamp eats the growth and the element stays under
    MIN_AREA — the repair would report a fix it did not make. So whatever the frame
    refuses to give one axis is paid back on the other.
    """
    # 1.25, not 1.6: the overshoot has to clear the threshold, not sail past it.
    target = MIN_AREA * 1.25
    w, h = max(el["w"], 0.02), max(el["h"], 0.02)
    f = (target / max(1e-6, w * h)) ** 0.5
    w, h = min(w * f, 1.0), min(h * f, 1.0)
    if w * h < target:
        h = min(1.0, target / w)
    if w * h < target:
        w = min(1.0, target / h)
    # Grown from the BOTTOM edge, not from the centre. Things in a scene rest
    # on other things, and growing about the centre lifts them off: measured on
    # a render, a mug at y 0.28-0.42 on a table whose top edge is 0.45 came out
    # of the MIN_AREA repair at 0.21-0.49 and drew as a giant mug hovering in
    # mid-air above the table. Keeping the bottom edge puts the extra size where
    # a bigger object would actually go -- upward -- and the mug stays on the
    # table. The horizontal centre is kept for the same reason it always was.
    cx, bottom = el["x"] + el["w"] / 2, el["y"] + el["h"]
    el["w"], el["h"] = w, h
    el["x"], el["y"] = cx - w / 2, bottom - h
    _clamp_box(el)


def _merge_groups(pairs, n) -> list:
    """The reported merge PAIRS, joined into piles (union-find)."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _spread(els, group):
    """Lay a pile of overlapping boxes out into a row of disjoint slots.

    Pulling apart only the two boxes of one reported pair does not converge: with
    three boxes on the same spot, separating (a,b) and then (a,c) puts b back on top
    of c, and the repair oscillates forever. Distributing the whole pile at once ends
    it in a single pass. Narrowing a box to fit its slot would also shrink it under
    MIN_AREA, so the free axis grows back up to MIN_AREA and no further: giving
    back ALL the lost area made three cats in a row 0.31x1.0 poles.
    """
    cx = [el["x"] + el["w"] / 2 for el in els]
    cy = [el["y"] + el["h"] / 2 for el in els]
    span = lambda v: max(v[i] for i in group) - min(v[i] for i in group)
    horiz = span(cx) >= span(cy)        # keep the axis they are already spread on
    order = sorted(group, key=lambda i: (cx[i] if horiz else cy[i], i))
    n = len(order)
    gutter = min(0.02, 0.4 / (n + 1))
    slot = (1.0 - gutter * (n + 1)) / n
    for k, i in enumerate(order):
        el = els[i]
        start = gutter + k * (slot + gutter)
        if horiz:
            new = max(0.02, min(el["w"], slot))
            bottom = el["y"] + el["h"]
            el["h"] = min(1.0, max(el["h"] * new / el["w"], MIN_AREA / new))
            el["w"], el["x"], el["y"] = new, start + (slot - new) / 2, bottom - el["h"]
        else:
            new = max(0.02, min(el["h"], slot))
            el["w"] = min(1.0, max(el["w"] * new / el["h"], MIN_AREA / new))
            el["h"], el["y"] = new, start + (slot - new) / 2
        _clamp_box(el)


def _place_text_clear(els, i, text_ix) -> bool:
    """Slide a lettering box to the clearest spot at its current size.

    Deliberately NOT _spread: that narrows the boxes it separates, and a
    lettering box narrowed away from the aspect its string needs is the very
    thing auto_fix_text exists to prevent -- the repair would trade a sign
    across the face for an illegible smear. Size is preserved; only x/y move.

    Scans a grid and keeps the position that puts the least subject under the
    text, tie-broken towards where the model originally asked for it. Returns
    whether it actually moved.
    """
    el = els[i]
    others = [o for j, o in enumerate(els)
              if j != i and not _attached_to(el, o)]
    if not others:
        return False
    w, h = el["w"], el["h"]
    x0, y0 = el["x"], el["y"]
    span_x, span_y = max(0.0, 1.0 - w), max(0.0, 1.0 - h)
    best, best_key = None, None
    for sy in range(21):
        for sx in range(21):
            cand = {"x": span_x * sx / 20, "y": span_y * sy / 20, "w": w, "h": h}
            cov = sum(_covered(cand, o) for o in others)
            # Distance from the requested spot, so a layout that was already fine
            # everywhere keeps its composition instead of snapping to a corner.
            drift = abs(cand["x"] - x0) + abs(cand["y"] - y0)
            key = (round(cov, 4), round(drift, 4))
            if best_key is None or key < best_key:
                best, best_key = cand, key
    # No spot is clear at this size (a full-width title over two heads):
    # shrink it about its anchor edge, aspect kept, until it is.
    while (sum(_covered(best, o) for o in others) >= TEXT_CLEARANCE
           and best["h"] * 0.9 >= _text_layout.MIN_TEXT_H):
        cx = best["x"] + best["w"] / 2
        best["w"], best["h"] = best["w"] * 0.9, best["h"] * 0.9
        best["x"] = cx - best["w"] / 2
        if best["y"] > 0.5:
            best["y"] = min(1.0, best["y"] + best["h"] / 0.9) - best["h"]
    if abs(best["x"] - x0) < 1e-6 and abs(best["y"] - y0) < 1e-6 and best["w"] == w:
        return False
    el.update(best)
    _clamp_box(el)
    return True


# Things that are SUPPOSED to hang with nothing under them. Without this list a
# ceiling lamp gets dropped onto the table below it.
_AIRBORNE_RE = re.compile(
    r"\b(?:lamp|chandelier|pendant|light|bulb|picture|painting|poster|photo|"
    r"frame|clock|mirror|shelf|curtain|window|sky|cloud|sun|moon|star|bird|"
    r"balloon|kite|smoke|steam|banner|sign)\b"
    r"|люстр|лампа|светильник|картин|постер|плакат|фото|часы|зеркал|полк|"
    r"штор|окн|небо|облак|солнц|лун|звезд|птиц|шарик|дым|пар|вывеск|таблич",
    re.IGNORECASE)

# How far a prop may hang above a surface and still count as "meant to be on
# it". 0.42 vs a table top at 0.45 is a planner slip; 0.10 of the frame away is
# a different composition and is left alone.
_SETTLE_GAP = 0.10


def _settle_onto_surfaces(els) -> list:
    """Drop a prop that hangs just above the surface it is standing on.

    Measured 2026-08-29 on the bench renders: the planner put the mug's box at
    y 0.28-0.42 with the table at 0.45-0.75, so the mug ended ABOVE the table
    top with nothing under it. Ideogram drew the sensible picture -- a mug on
    the table -- and the critic then condemned the render for disobeying the
    layout, in seven of fourteen cases. The layout was wrong, not the picture.

    Deliberately narrow. The prop has to be small next to the surface, sit
    horizontally over it, hang by less than a tenth of the frame, and not be
    one of the things that hang on purpose.
    """
    moved = []
    for el in els:
        if _AIRBORNE_RE.search(str(el.get("desc") or "")):
            continue
        if str(el.get("text") or "").strip():
            continue                       # lettering is placed by its own repair
        bottom = el["y"] + el["h"]
        best = None
        for other in els:
            if other is el:
                continue
            if el["w"] * el["h"] > other["w"] * other["h"] * 0.6:
                continue                       # not a prop next to that surface
            top = other["y"]
            gap = top - bottom
            if not (0 < gap <= _SETTLE_GAP):
                continue
            # It has to be standing OVER the surface, not beside it.
            overlap = max(0.0, min(el["x"] + el["w"], other["x"] + other["w"])
                               - max(el["x"], other["x"]))
            if overlap < el["w"] * 0.5:
                continue
            if best is None or gap < best[0]:
                best = (gap, other)
        if best:
            el["y"] += best[0]
            _clamp_box(el)
            moved.append((el, best[1]))
    return moved


def auto_fix_geometry(layout: dict) -> tuple:
    """Repair the arrangement problems `geometry_report` found. Returns (layout, notes).

    Deliberately conservative: grow what is too small, spread merged boxes apart,
    shrink a full-frame element back to a subject, and say (without deleting
    anything) when the scene is overcrowded.

    Runs in passes, because one repair can expose the next (a shrunken full-frame box
    stops overlapping its neighbour). It stops when a pass changes nothing, when the
    arrangement comes out clean, or when it recognises a state it has already been in
    — a pile of boxes can be circular (freeing a from b drops a onto c, freeing a
    from c drops it back onto b), and in that case the best pass is kept. All three
    exits land on the same arrangement every time, so calling this twice is the same
    as calling it once.
    """
    layout = ideogram.normalize_layout(copy.deepcopy(layout))
    els = _els(layout)
    notes, said = [], set()

    def note(text):
        if text not in said:            # passes repeat; say each thing once
            said.add(text)
            notes.append(text)

    snapshot = lambda: [dict(el) for el in els]
    state = lambda: repr([(round(el["x"], 6), round(el["y"], 6),
                           round(el["w"], 6), round(el["h"], 6)) for el in els])
    best, best_n, seen = snapshot(), None, set()
    for _ in range(12):
        probs = [p for p in geometry_report(layout) if p["kind"] != "crowded"]
        if best_n is None or len(probs) < best_n:
            best, best_n = snapshot(), len(probs)
        if not probs or state() in seen:
            break
        seen.add(state())
        before = snapshot()
        for prob in probs:
            if prob["kind"] != "full_frame":
                continue
            el = els[prob["targets"][0]]
            el["w"], el["h"] = el["w"] * 0.62, el["h"] * 0.62
            el["x"], el["y"] = (1 - el["w"]) / 2, (1 - el["h"]) / 2
            _clamp_box(el)
            note(f'pulled “{el["desc"][:30]}” back off the frame edges — a '
                 f"full-frame box leaves the background nothing to be")
        for prob in probs:
            if prob["kind"] != "too_small":
                continue
            el = els[prob["targets"][0]]
            _grow_to_min_area(el)
            note(f'grew “{el["desc"][:30]}” to {el["w"] * el["h"] * 100:.0f}% '
                 f"of the frame so it survives the render")
        _text_ix = {k for k, _ in _draw_text._text_els({"elements": els})}
        for prob in probs:
            if prob["kind"] != "text_collision":
                continue
            i = prob["targets"][0]
            if _place_text_clear(els, i, _text_ix):
                note(f'moved the lettering “{str(els[i].get("text") or "")[:20]}” off '
                     f"the subject — a sign drawn across a person fuses into them "
                     f"instead of reading as text")
        for prob in probs:
            if prob["kind"] != "coincident":
                continue
            small, big = els[prob["targets"][0]], els[prob["targets"][1]]
            _nest_on(small, big)
            note(f'sat “{small["desc"][:24]}” inside “{big["desc"][:24]}” instead '
                 f"of on the same box — identical boxes render as one object")
        for group in _merge_groups([tuple(p["targets"]) for p in probs
                                    if p["kind"] == "merged"], len(els)):
            _spread(els, group)
            note("separated " + " and ".join(f'“{els[i]["desc"][:24]}”' for i in group)
                 + " so they stop merging into one object")
        if snapshot() == before:
            break                       # nothing moved, so nothing will next pass
    if best_n is not None and \
       len([p for p in geometry_report(layout) if p["kind"] != "crowded"]) > best_n:
        els[:] = best                   # the passes went in a circle — keep the best
    # Last, once nothing else is going to move a box: rest the props on what
    # they are standing on.
    for _el, _surface in _settle_onto_surfaces(els):
        note(f'rested “{_el["desc"][:24]}” on “{_surface["desc"][:24]}” instead '
             f"of leaving it hanging just above it")

    if len(els) > MAX_ELEMENTS:
        notes.append(f"{len(els)} elements is more than the renderer honours — "
                     f"consider deleting the least important ones")
    return layout, notes
