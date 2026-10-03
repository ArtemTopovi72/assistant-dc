"""Ten drawing scenarios, scored on the LAYOUT and traced step by step.

The product rule this measures: pictures are Ideogram only. The model plans a
LAYOUT -- boxes carrying a position and a description -- and every later change
(add, remove, move, reword, replace) is made by EDITING THAT LAYOUT and
repainting the whole canvas from it. No inpainting, no detailers, no masks.

So the layout is what is scored. It is machine-checkable ("is the cat's box
below the table's?"); the pixels are not, and a bench that judged pixels would
be measuring the renderer instead of the agent. Nothing here renders, for the
same reason and because a real render is a GPU job that would contend with
everything else on this machine.

What IS real: the planner call, the edit call, the op parser, the geometry
repair. Those are where the reported bugs live -- the agent drew the wrong
thing to begin with, or was asked to change one thing and changed another,
dropped a box, or said it had done something it had not.

Every step is traced (`--trace`) so a failure can be read as a sequence: what
was planned, what ops came back, which landed, which were rejected and why, and
what the geometry pass then did to the boxes.

Usage:
    venv/Scripts/python.exe bench/draw_scenarios.py [--reps 2] [--case move_cat]
                                                    [--trace] [--json out.jsonl]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tc_run import setup                       # noqa: E402  (shared bench boot)
import draw_agent as DA                        # noqa: E402
import ideogram as IG                          # noqa: E402


# -- reading a layout --------------------------------------------------------
# Matching is by WORD STEM, not by equality: the model writes "рыжий кот спит
# на диване", never "кот". Russian inflects, so "кота"/"коту"/"кошка" all have
# to hit the same box -- a check that missed them would report a defect the
# agent never committed (this bench family has done exactly that before).

def _norm(s) -> str:
    return str(s or "").lower().replace("ё", "е")


def find(layout, *stems):
    """Indices of every element whose desc/text carries any of `stems`."""
    out = []
    for i, el in enumerate(layout.get("elements") or []):
        blob = _norm(el.get("desc")) + " " + _norm(el.get("text"))
        if any(st in blob for st in stems):
            out.append(i)
    return out


def one(layout, *stems):
    ix = find(layout, *stems)
    return (layout["elements"][ix[0]] if ix else None)


def box(el):
    return tuple(round(float(el[k]), 3) for k in "xywh")


def cx(el): return float(el["x"]) + float(el["w"]) / 2
def cy(el): return float(el["y"]) + float(el["h"]) / 2


def below(a, b, slack=0.02):
    """`a` sits lower in the frame than `b` (y grows downward)."""
    return cy(a) > cy(b) - slack


def right_of(a, b, slack=0.02):
    return cx(a) > cx(b) - slack


def overlaps(a, b) -> bool:
    return (abs(cx(a) - cx(b)) < (float(a["w"]) + float(b["w"])) / 2 and
            abs(cy(a) - cy(b)) < (float(a["h"]) + float(b["h"])) / 2)


def same_box(a, b, eps=1e-6):
    return all(abs(float(a[k]) - float(b[k])) <= eps for k in "xywh")


def _stems_of(el):
    """A couple of content words from an element, to find it again afterwards."""
    words = [w for w in _norm(el.get("desc")).split() if len(w) > 4]
    return tuple(w[:5] for w in words[:3]) or (_norm(el.get("desc"))[:8],)


def untouched(before, after, *stems):
    """Every element NOT matching `stems` survived with its box intact.

    This is the collateral-damage check: an edit that quietly renumbers,
    reshapes or drops the boxes it was not asked about is the defect, even when
    the box it WAS asked about ends up right.
    """
    skip = set(find(before, *stems))
    for i, el in enumerate(before.get("elements") or []):
        if i in skip:
            continue
        mate = one(after, *_stems_of(el))
        if mate is None:
            return False, "lost %r" % (el.get("desc", "")[:40],)
        if not same_box(el, mate):
            return False, "moved %r: %s -> %s" % (el.get("desc", "")[:30],
                                                  box(el), box(mate))
    return True, "the target changed and nothing else did"


# The vocabulary the checks match on. Both languages for every object: the
# editor answers in English ("a ginger cat sleeping on the sofa") about a layout
# written in Russian, because English is the caption language Ideogram wants. A
# Russian-only matcher reported "the cat was never added" for a cat that was
# added -- the scorer inventing the defect, not the agent committing one.
CAT   = ("кот", "кош", "котён", "котен", "cat", "kitten")
SOFA  = ("диван", "sofa", "couch")
TABLE = ("стол", "table", "desk")
MUG   = ("кружк", "чашк", "mug", "cup")
BOX   = ("коробк", "box", "carton")
GLASS = ("бокал", "вино", "винн", "wine", "glass")
LAMP  = ("лампа", "лампу", "лампой", "светильник", "торшер", "lamp")


# -- the scenarios -----------------------------------------------------------

def _kitchen():
    """A hand-built starting layout. Hand-built on purpose for the EDIT cases:
    if the start came from the planner, a failed edit and a bad plan would be
    indistinguishable, and the bench would report the wrong bug."""
    return IG.normalize_layout({
        "background": "уютная кухня, тёплый свет",
        "elements": [
            {"desc": "деревянный стол",         "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": "белая кружка с чаем",     "x": 0.34, "y": 0.28, "w": 0.12, "h": 0.14},
            {"desc": "серый кот сидит на полу", "x": 0.30, "y": 0.76, "w": 0.20, "h": 0.20},
            {"desc": "картонная коробка",       "x": 0.78, "y": 0.55, "w": 0.18, "h": 0.22},
            {"desc": "синий диван у стены",     "x": 0.02, "y": 0.50, "w": 0.16, "h": 0.30},
        ],
    })


def c_initial_scene(before, after, trace):
    """Did the FIRST plan honour the spatial words in the request?"""
    cat = one(after, *CAT)
    tab = one(after, *TABLE)
    mug = one(after, *MUG)
    bx = one(after, *BOX)
    missing = [n for n, el in (("кот", cat), ("стол", tab),
                               ("кружка", mug), ("коробка", bx)) if el is None]
    if missing:
        return False, "the plan never drew: " + ", ".join(missing)
    if not below(cat, tab):
        return False, "the cat is not under the table (cat y=%.2f, table y=%.2f)" % (
            cy(cat), cy(tab))
    if below(mug, tab):
        return False, "the mug is not on the table (mug y=%.2f, table y=%.2f)" % (
            cy(mug), cy(tab))
    if not right_of(bx, tab):
        return False, "the box is not right of the table (%.2f vs %.2f)" % (
            cx(bx), cx(tab))
    return True, "the plan matches every stated relation"


def c_add_cat(before, after, trace):
    if len(find(after, *CAT)) <= len(find(before, *CAT)):
        return False, "no new cat element appeared"
    if len(after["elements"]) != len(before["elements"]) + 1:
        return False, "adding one thing changed the element count by %d" % (
            len(after["elements"]) - len(before["elements"]))
    # The sofa is the surface the new cat was put on, so it may be resized to
    # host it. Every OTHER box has to be exactly where it was.
    return untouched(before, after, *(CAT + SOFA))


def c_remove_box(before, after, trace):
    if find(after, *BOX):
        return False, "the box is still in the layout"
    if len(after.get("elements") or []) != len(before["elements"]) - 1:
        return False, "element count went %d -> %d: something else went with it" % (
            len(before["elements"]), len(after["elements"]))
    return untouched(before, after, *BOX)


def c_move_cat(before, after, trace):
    cat, sofa = one(after, *CAT), one(after, *SOFA)
    if cat is None:
        return False, "the cat is gone -- a move must not delete"
    if sofa is None:
        return False, "the sofa is gone"
    if same_box(cat, one(before, *CAT)):
        return False, "the cat's box never changed"
    if not overlaps(cat, sofa):
        return False, "the cat did not land on the sofa: cat=%s sofa=%s" % (
            box(cat), box(sofa))
    # The sofa is a PARTICIPANT in this move, not a bystander: seating the cat
    # on it legitimately resizes it (a host too small to hold a renderable
    # child is what made the repair fight itself). Everything else must hold.
    return untouched(before, after, *(CAT + SOFA))


def c_recolour_cat(before, after, trace):
    cat = one(after, *CAT)
    if cat is None:
        return False, "the cat is gone -- a reword must not delete"
    if not any(w in _norm(cat.get("desc")) for w in ("чёрн", "черн")):
        return False, "the cat is still described as %r" % (cat.get("desc"),)
    return untouched(before, after, *CAT)


def c_replace_mug(before, after, trace):
    if find(after, *MUG):
        return False, "the mug is still there"
    glass = one(after, *GLASS)
    if glass is None:
        return False, "no wine glass in the layout"
    old = one(before, *MUG)
    if abs(cx(glass) - cx(old)) > 0.3 or abs(cy(glass) - cy(old)) > 0.3:
        return False, "the glass was put somewhere else entirely: %s vs %s" % (
            box(glass), box(old))
    return untouched(before, after, *(MUG + GLASS))


def c_resize_box_only(before, after, trace):
    b0, b1 = one(before, *BOX), one(after, *BOX)
    if b1 is None:
        return False, "the box is gone -- 'make it bigger' is not 'remove it'"
    if float(b1["w"]) * float(b1["h"]) <= float(b0["w"]) * float(b0["h"]) + 1e-6:
        return False, "the box did not grow: %.3f -> %.3f" % (
            float(b0["w"]) * float(b0["h"]), float(b1["w"]) * float(b1["h"]))
    return untouched(before, after, *BOX)


def c_two_edits(before, after, trace):
    """Second edit applied on top of the first: both must be visible."""
    if find(after, *BOX):
        return False, "the first edit (remove the box) was undone by the second"
    cat = one(after, *CAT)
    if cat is None:
        return False, "the cat vanished somewhere between the two edits"
    sofa = one(after, *SOFA)
    if sofa is None or not overlaps(cat, sofa):
        return False, "the second edit (cat onto the sofa) did not land"
    return True, "both edits survive in the final layout"


def c_unsatisfiable(before, after, trace):
    """Nothing in the layout matches, so nothing may change.

    The failure this catches is the expensive one: rather than saying "there is
    no giraffe here", the editor picks the nearest box and rewrites or deletes
    it, and the user gets a picture with a random object missing.
    """
    if len(after.get("elements") or []) != len(before["elements"]):
        return False, "it deleted something to satisfy an impossible instruction"
    for i, el in enumerate(before["elements"]):
        if not same_box(el, after["elements"][i]):
            return False, "it moved %r instead of saying no" % (el["desc"][:30],)
        if _norm(el.get("desc")) != _norm(after["elements"][i].get("desc")):
            return False, "it rewrote %r into %r" % (
                el["desc"][:30], after["elements"][i]["desc"][:30])
    said_no = any(("no element matching" in n) or ("proposed no change" in n)
                  or ("skipped" in n)
                  for n in trace.get("notes") or [])
    if not said_no:
        return False, "it changed nothing but never said why: %s" % (trace.get("notes"),)
    return True, "refused the impossible edit and left the layout alone"


def c_self_omission(before, after, trace):
    """The agent notices its own gap and repairs the layout.

    The starting layout is deliberately missing an object the scene needs. A
    repair must ADD it, not reshuffle what is already there.
    """
    if one(after, *LAMP) is None:
        return False, "the missing lamp was still missing after the repair"
    return untouched(before, after, *LAMP)


def c_geometry_kept_sane(before, after, trace):
    """An edit may not leave geometry the renderer is known to mangle."""
    bad = DA.geometry_report(after)
    if bad:
        return False, "; ".join(p["why"] for p in bad)[:160]
    if len(find(after, *CAT)) <= len(find(before, *CAT)):
        return False, "the added kitten did not survive the geometry pass"
    return True, "the layout is renderable and the addition survived"


def c_four_edits(before, after, trace):
    """Four instructions in a row on one canvas.

    One edit landing is not the same as four landing: each round hands the next
    a layout the model has to re-read, and the failure this catches is drift --
    edit three quietly undoing edit one, or the boxes creeping until nothing is
    where it was put.
    """
    if find(after, *BOX):
        return False, "edit 1 (remove the box) did not survive"
    cat = one(after, *CAT)
    if cat is None:
        return False, "the cat was lost somewhere in the sequence"
    if not any(w in _norm(cat.get("desc")) for w in ("чёрн", "черн", "black")):
        return False, "edit 3 (make the cat black) is not in the final layout"
    sofa = one(after, *SOFA)
    if sofa is None or not overlaps(cat, sofa):
        return False, "edit 2 (cat onto the sofa) did not survive"
    if one(after, *LAMP) is None:
        return False, "edit 4 (add a lamp) is missing"
    return True, "all four edits are in the final layout"


def c_undo_the_last_edit(before, after, trace):
    """Asked to put something back. The layout is the state, so this is just
    another edit -- but a model that treats "верни" as a new object adds a
    SECOND box instead of restoring the one it removed."""
    boxes = find(after, *BOX)
    if not boxes:
        return False, "the box was not put back"
    if len(boxes) > 1:
        return False, "it added a second box instead of restoring the one"
    if len(after["elements"]) != len(before["elements"]):
        return False, "the element count did not come back to where it started"
    return True, "the removed object came back, once"


def c_two_of_the_same(before, after, trace):
    """Two cats in the layout; only the GREY one was named.

    The editor resolves a target by matching words, so a scene with two similar
    objects is where it quietly edits the wrong one.

    Tracked by BOX, not by the word "серый": a correct recolour rewrites the
    description to "чёрный кот", and the first version of this check called that
    a lost cat -- the scorer inventing the defect, not the agent committing one.
    """
    grey0 = one(before, "серый", "grey", "gray")
    ginger0 = one(before, "рыж", "ginger")
    if grey0 is None or ginger0 is None:
        return False, "the fixture lost a cat before the edit even ran"

    def at(box_of):
        for el in after.get("elements") or []:
            if same_box(el, box_of):
                return el
        return None

    grey, ginger = at(grey0), at(ginger0)
    if grey is None:
        return False, "the cat that was named is gone or was moved"
    if ginger is None:
        return False, "the ginger cat was moved or removed -- it was not named"
    black = lambda el: any(w in _norm(el.get("desc"))
                           for w in ("чёрн", "черн", "black"))
    if not black(grey):
        return False, "the named cat was not recoloured: %r" % (grey.get("desc"),)
    if black(ginger):
        return False, "it recoloured the ginger cat too"
    return True, "only the cat that was named changed"


CASES = [
    dict(id="initial_scene", kind="plan",
         request="нарисуй уютную кухню: кот под столом, белая кружка на столе, "
                 "картонная коробка справа от стола",
         check=c_initial_scene),
    dict(id="add_cat", kind="edit", start=_kitchen,
         instruction="добавь рыжего кота, который спит на диване",
         check=c_add_cat),
    dict(id="remove_box", kind="edit", start=_kitchen,
         instruction="убери картонную коробку",
         check=c_remove_box),
    dict(id="move_cat", kind="edit", start=_kitchen,
         instruction="передвинь серого кота на диван",
         check=c_move_cat),
    dict(id="recolour_cat", kind="edit", start=_kitchen,
         instruction="пусть серый кот будет чёрным",
         check=c_recolour_cat),
    dict(id="replace_mug", kind="edit", start=_kitchen,
         instruction="замени кружку на бокал вина",
         check=c_replace_mug),
    dict(id="resize_box_only", kind="edit", start=_kitchen,
         instruction="сделай картонную коробку больше",
         check=c_resize_box_only),
    dict(id="two_edits", kind="edit", start=_kitchen,
         instruction=["убери картонную коробку",
                      "теперь посади серого кота на диван"],
         check=c_two_edits),
    dict(id="unsatisfiable", kind="edit", start=_kitchen,
         instruction="убери жирафа",
         check=c_unsatisfiable),
    dict(id="self_omission", kind="edit",
         start=lambda: IG.normalize_layout({
             "background": "гостиная вечером",
             "elements": [
                 {"desc": "кресло у окна", "x": 0.15, "y": 0.45, "w": 0.30, "h": 0.35},
                 {"desc": "книжная полка", "x": 0.60, "y": 0.30, "w": 0.30, "h": 0.45},
             ]}),
         instruction="в сцене должна быть напольная лампа рядом с креслом — добавь её",
         check=c_self_omission),
    dict(id="four_edits", kind="edit", start=_kitchen,
         instruction=["убери картонную коробку",
                      "посади серого кота на диван",
                      "пусть кот будет чёрным",
                      "добавь напольную лампу рядом с диваном"],
         check=c_four_edits),
    dict(id="undo_the_last_edit", kind="edit", start=_kitchen,
         instruction=["убери картонную коробку",
                      "верни коробку обратно"],
         check=c_undo_the_last_edit),
    dict(id="two_of_the_same", kind="edit",
         start=lambda: DA.auto_fix_geometry(IG.normalize_layout({
             "background": "комната",
             "elements": [
                 {"desc": "серый кот сидит на полу",  "x": 0.10, "y": 0.60,
                  "w": 0.22, "h": 0.28},
                 {"desc": "рыжий кот спит на кресле", "x": 0.60, "y": 0.55,
                  "w": 0.26, "h": 0.30},
                 {"desc": "деревянный стол",          "x": 0.30, "y": 0.20,
                  "w": 0.34, "h": 0.28},
             ]}))[0],
         instruction="пусть серый кот будет чёрным",
         check=c_two_of_the_same),
    dict(id="geometry_kept_sane", kind="edit", start=_kitchen,
         instruction="добавь маленького котёнка на подоконник",
         check=c_geometry_kept_sane),
]


def _changed_descs(pre, post):
    """Descriptions whose box changed, or that are new, between two layouts."""
    was = {_norm(e.get("desc")): e for e in pre.get("elements") or []}
    out = []
    for e in post.get("elements") or []:
        old = was.get(_norm(e.get("desc")))
        if old is None:
            out.append("+ " + str(e.get("desc", ""))[:32])
        elif not same_box(e, old):
            out.append(str(e.get("desc", ""))[:32] + " %s->%s" % (box(old), box(e)))
    for k, e in was.items():
        if not any(_norm(p.get("desc")) == k for p in post.get("elements") or []):
            out.append("- " + str(e.get("desc", ""))[:32])
    return out


# -- the runner --------------------------------------------------------------

def run_case(ctx, case):
    """One scenario. Returns a row with the full trace attached."""
    trace = {"steps": [], "notes": []}
    t0 = time.perf_counter()

    def step(what, **data):
        trace["steps"].append(dict(t=round(time.perf_counter() - t0, 2),
                                   step=what, **data))

    if case["kind"] == "plan":
        step("plan", request=case["request"])
        after = IG.plan_layout(ctx, case["request"])
        before = {"elements": []}
        step("planned",
             elements=[e.get("desc", "")[:48] for e in after.get("elements") or []],
             boxes=[box(e) for e in after.get("elements") or []])
    else:
        # Settle the fixture through the SAME deterministic geometry pass the
        # edited layout goes through. Without this the first run reported
        # "moved the mug" for every case: the hand-written boxes are under
        # MIN_AREA, so the repair legitimately grew them, and the bench was
        # comparing a repaired layout against an unrepaired fixture -- the
        # scorer manufacturing the defect it claims to catch.
        before = DA.auto_fix_geometry(case["start"]())[0]
        after = copy.deepcopy(before)
        step("start", elements=[e["desc"] for e in before["elements"]])
        instructions = ([case["instruction"]]
                        if isinstance(case["instruction"], str)
                        else case["instruction"])
        for instruction in instructions:
            step("edit", instruction=instruction)
            pre = copy.deepcopy(after)
            after, notes = DA.edit_layout(ctx, after, instruction)
            trace["notes"].extend(notes or [])
            # Which ops LANDED is the thing worth seeing: the editor reports a
            # rejected op in the same list as an applied one, and "no element
            # matching" is the commonest way an edit does nothing while looking
            # like it did something.
            rejected = [n for n in (notes or [])
                        if "no element matching" in n or "unknown op" in n
                        or "skipped" in n or "ignored" in n
                        or "proposed no change" in n]
            landed = [n for n in (notes or []) if n not in rejected]
            step("edited", landed=landed, rejected=rejected,
                 elements=[e.get("desc", "")[:48] for e in after.get("elements") or []],
                 # By DESCRIPTION, not by index: a delete renumbers every
                 # element after it, and an index diff would then report every
                 # survivor as "changed" -- noise on exactly the trace someone
                 # is reading to find out what actually moved.
                 changed=_changed_descs(pre, after))
            fixed, fnotes = DA.auto_fix_geometry(after)
            if fnotes:
                after = fixed
                step("geometry", notes=fnotes)

    ok, why = case["check"](before, after, trace)
    return dict(id=case["id"], ok=ok, why=why, trace=trace,
                seconds=time.perf_counter() - t0, layout=after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--trace", action="store_true", help="print every step")
    ap.add_argument("--json", default="", help="append rows as JSONL here")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.case or c["id"] in args.case]
    _graph, ctx, _img = setup()

    rows = []
    for _rep in range(args.reps):
        for case in cases:
            r = run_case(ctx, case)
            rows.append(r)
            mark = "PASS" if r["ok"] else "FAIL"
            print("[%s] %-22s %5.1fs  %s" % (mark, r["id"], r["seconds"],
                                             r["why"][:100]))
            if args.trace or not r["ok"]:
                for s in r["trace"]["steps"]:
                    print("        " + json.dumps(s, ensure_ascii=False)[:300])
            if args.json:
                with open(args.json, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "layout"},
                                        ensure_ascii=False) + "\n")

    print("\n" + "=" * 72)
    for cid in dict.fromkeys(c["id"] for c in cases):
        got = [r for r in rows if r["id"] == cid]
        print("  %-22s %d/%d" % (cid, sum(1 for r in got if r["ok"]), len(got)))
    total = sum(1 for r in rows if r["ok"])
    print("-" * 72)
    print("  TOTAL %d/%d" % (total, len(rows)))
    return 0 if total == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
