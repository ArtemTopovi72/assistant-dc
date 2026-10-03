"""An agent that draws by arranging boxes, looks at what came out, and rearranges them.

Ideogram 4 composes from a structured caption whose elements carry bounding boxes
(see `ideogram.py`). That makes the layout a *state* the agent can hold, edit and
reason about — which is what this module is:

    plan  ->  draw  ->  look at the result  ->  fix the boxes  ->  draw again

Three things live here:

  * `apply_ops` / `edit_layout` — conversational editing of an existing layout
    ("drop the dog", "put the sign top-left", "make the tower bigger"). The model
    answers with OPERATIONS, not a rewritten scene, so an edit can never quietly
    lose the elements it was not asked to touch.
  * `geometry_report` / `auto_fix_geometry` — the deterministic half of "why did
    that come out as nonsense": box arrangements that are known to break the
    renderer (near-total overlap merges two subjects into one, a box under ~4% of
    the frame gets swallowed, more than ~6 elements turn into soup, a full-frame
    element fights the background). No model call, so it always runs.
  * `critique` — the vision half: look at the render and say which elements are
    missing, duplicated or in the wrong place, as operations to apply.

`run()` ties them together into the loop, reporting every step through `on_event`.
"""
import copy
import difflib
import json
import logging
import random
import re
from typing import Callable, Optional

import comfy_client
import ideogram
# Text-box arithmetic, shared with ideogram.py. Re-exported here so the many
# call sites below (and the suites that reach draw_agent.text_geometry /
# draw_agent.MIN_TEXT_H) keep working unchanged.
import text_layout as _text_layout
from text_layout import text_geometry, _norm_text
from utils import safe_json_from_llm

logger = logging.getLogger("assistant.draw_agent")

# The lettering half lives in draw_text.py. Re-exported by value so that both
# `draw_agent.<name>` call sites and the bare global calls inside run() keep
# resolving -- and so that patching draw_agent.<name> still steers run(), which
# is the seam the suites use.
import draw_text as _draw_text
# Layout geometry (overlap repair, min-area growth, spreading) and its four
# tunables live in draw_geometry. Re-exported by value: run/edit_layout stay in
# THIS module and resolve these as globals here, so draw_agent.auto_fix_geometry
# remains the patch point it has always been.
from draw_geometry import (  # noqa: F401,E402
    MIN_AREA, MAX_ELEMENTS, MERGE_IOU, FULL_FRAME,
    _iou, _covered, geometry_report, _grow_to_min_area, _merge_groups, _spread,
    auto_fix_geometry,
)

_els        = _draw_text._els
_num        = _draw_text._num
_clamp_box  = _draw_text._clamp_box
_text_els   = _draw_text._text_els
text_report = _draw_text.text_report
auto_fix_text = _draw_text.auto_fix_text
spell_out   = _draw_text.spell_out
repair_text = _draw_text.repair_text
read_text   = _draw_text.read_text
verify_text = _draw_text.verify_text
text_problems = _draw_text.text_problems
_reader     = _draw_text._reader
_ReaderCtx  = _draw_text._ReaderCtx
_crop_for_read = _draw_text._crop_for_read
_similarity = _draw_text._similarity
_TEXT_READ_PROMPT = _draw_text._TEXT_READ_PROMPT

# Text-box arithmetic lives in text_layout.py and is read as draw_agent.<NAME>
# by the suites; bind it here so that seam keeps resolving.
MIN_TEXT_H      = _text_layout.MIN_TEXT_H
CHAR_ASPECT     = _text_layout.CHAR_ASPECT
MAX_TEXT_CHARS  = _text_layout.MAX_TEXT_CHARS
SHAPE_TOLERANCE = _text_layout.SHAPE_TOLERANCE
TEXT_MATCH_OK   = _text_layout.TEXT_MATCH_OK

# Geometry thresholds. These are the arrangement facts that decide whether a
# layout renders as a scene or as a smear; kept in one place so they can be tuned
# from evidence rather than hunted through the code.


# Words that carry no identity: "the sign" must match "a sign", so they cannot be
# allowed to count against the overlap score (they were the reason the critic's own
# repair ops silently resolved to nothing).
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "with",
              "that", "this", "these", "those", "its", "it", "is", "are", "was",
              "for", "from", "into", "onto", "over", "under", "near", "by", "as",
              "some", "any", "all", "one", "two", "his", "her", "their", "there",
              # positions pick among twins (_pick_twin), they are not in any desc
              "left", "right", "middle", "center", "centre", "first", "second", "third"}


def _content_words(s: str) -> list:
    return [w for w in re.findall(r"\w+", s) if len(w) > 2 and w not in _STOPWORDS]


def _find(layout, target) -> int:
    """Resolve an op's target to an element index.

    The model refers to elements the way a person does — "the dog", "element 2",
    "the sign" — so accept an index or a text match on desc/text, scored by
    CONTENT-word overlap so "red car" finds "a red sports car" and "the dog" finds
    "a scruffy dog".
    """
    els = _els(layout)
    if not els:
        return -1
    if isinstance(target, bool):        # True is not "element 1"
        return -1
    if isinstance(target, (int, float)):
        i = int(target)
        # The model only ever sees 1-based numbers (`_for_model`, and both prompts
        # say "the element's number as shown"), so read the number the way it was
        # written; a raw 0-based index is only the fallback.
        if 1 <= i <= len(els):
            return i - 1
        if 0 <= i < len(els):
            return i
        return -1
    if not isinstance(target, str):     # a list/dict target is not a description
        return -1
    s = target.strip().lower()
    if not s:
        return -1
    m = re.fullmatch(r"(?:element\s*)?#?(\d+)", s)
    if m:
        return _find(layout, int(m.group(1)))
    best, best_score = -1, 0.0
    words = _content_words(s)
    for i, el in enumerate(els):
        hay = f'{el.get("desc", "")} {el.get("text", "")}'.lower()
        if s in hay:
            score = 1.0 + len(s) / 100.0
        elif words:
            # Near-misses count. The critic writes "the plane trees" for "a bare
            # plane tree" and "the cruiser" for "a police cruiser"; on exact
            # substrings alone those resolve to nothing, its whole repair is
            # dropped, and the loop reports "stalled" on a picture it knew how to
            # fix. Only a close match counts, so "a hot air balloon" still finds
            # nothing rather than landing on the nearest box.
            hay_words = _content_words(hay)
            hit = 0.0
            for w in words:
                if w in hay:
                    hit += 1.0
                    continue
                near = max((difflib.SequenceMatcher(None, w, h).ratio()
                            for h in hay_words), default=0.0)
                if near >= 0.8:
                    hit += near
            score = hit / len(words) * 0.9
        else:
            score = 0.0
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= 0.5 else -1


def _pick_twin(els, idx: int, cue: str) -> int:
    """Three "a grey cat" boxes: a description finds the first, but «кота справа»
    meant the right one (it recoloured the left, live 2026-09-29). Among
    identical twins the side/ordinal word in the target or instruction decides."""
    key = lambda e: (str(e.get("desc") or "").lower(), str(e.get("text") or ""))
    twins = sorted((k for k, e in enumerate(els) if key(e) == key(els[idx])),
                   key=lambda k: els[k]["x"] + els[k]["w"] / 2)
    if len(twins) < 2:
        return idx
    c = (cue or "").lower()
    # The EARLIEST word picks: «the left cat to the right» names the left one.
    hits = [(m.start(), pos) for pat, pos in (
                (r"\bright|справа|правы|правог|правом", -1), (r"\bleft|слева|левы|левог|левом", 0),
                (r"\bmiddle|\bcent|средн|посередин|в центре", len(twins) // 2),
                (r"\bsecond|втор", 1), (r"\bthird|трет", 2), (r"\bfirst|перв", 0))
            for m in [re.search(pat, c)] if m and -len(twins) <= pos < len(twins)]
    return twins[min(hits)[1]] if hits else idx


_ANCHORS = {
    "left": (0.03, None, 0.34, None), "right": (0.63, None, 0.34, None),
    "top": (None, 0.03, None, 0.34), "bottom": (None, 0.63, None, 0.34),
    "centre": (0.33, 0.33, 0.34, 0.34), "center": (0.33, 0.33, 0.34, 0.34),
    "middle": (0.33, 0.33, 0.34, 0.34),
    "top left": (0.03, 0.03, 0.34, 0.34), "top right": (0.63, 0.03, 0.34, 0.34),
    "bottom left": (0.03, 0.63, 0.34, 0.34), "bottom right": (0.63, 0.63, 0.34, 0.34),
    "sky": (0.15, 0.03, 0.7, 0.3), "foreground": (0.1, 0.55, 0.8, 0.42),
    "background": (0.05, 0.05, 0.9, 0.6),
}


def _anchor_box(where: str):
    """Turn a placement word ('top right', 'in the sky') into a box."""
    s = (where or "").strip().lower()
    for key in sorted(_ANCHORS, key=len, reverse=True):
        if key in s:
            x, y, w, h = _ANCHORS[key]
            return {"x": 0.33 if x is None else x, "y": 0.33 if y is None else y,
                    "w": 0.34 if w is None else w, "h": 0.34 if h is None else h}
    return None


def _op_list(ops) -> list:
    """Whatever the model answered, as a list of op dicts.

    A single op is often returned unwrapped, and a bare string would otherwise be
    iterated CHARACTER by character.
    """
    if isinstance(ops, dict):
        inner = ops.get("ops")
        return _op_list(inner) if isinstance(inner, (list, dict)) else [ops]
    if isinstance(ops, (list, tuple)):
        return list(ops)
    return []


# "верни коробку обратно" belongs here with "добавь": measured, the editor
# answered it with `replace` on an unrelated box and the user lost their mug
# to a cardboard box. Restoring something is adding it back, and the guard
# below turns exactly that kind of replace into an add.


def _additive_only(instruction: str) -> bool:
    """The instruction asks to ADD something and to remove nothing.

    Measured: told "добавь рыжего кота, который спит на диване", the editor
    answered with `replace` on the sofa plus `delete` on the existing cat -- the
    user asked for one more thing and lost two. Advice in the prompt does not
    reliably stop it (this project has measured that too), so the ops are
    filtered instead: on an add-only instruction, a delete or a replace cannot
    apply, and the note says so rather than the layout quietly losing a box.
    """
    import image_router   # the model's read of the edit: one insert, nothing else
    plan = image_router.edit_plan(instruction or "")
    return plan["kind"] == "object_insert" and not plan["steps"]


def _has_add_op(ops) -> bool:
    for raw in ops:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op") or raw.get("action") or "").strip().lower()
        if not op and len(raw) == 1:
            op = str(next(iter(raw))).strip().lower()      # {"add": {...}}
        if op in ("add", "insert", "new"):
            return True
    return False


_DESTRUCTIVE_OPS = frozenset({
    "delete", "remove", "drop",
    "replace", "swap", "change", "set_desc", "rename", "desc", "describe",
    "redescribe", "set_description", "update", "edit", "modify",
})


def _box_contains(outer: dict, inner: dict) -> bool:
    """Is the centre of `inner` inside `outer`?"""
    try:
        cx, cy = inner["x"] + inner["w"] / 2, inner["y"] + inner["h"] / 2
        return (outer["x"] <= cx <= outer["x"] + outer["w"]
                and outer["y"] <= cy <= outer["y"] + outer["h"])
    except (KeyError, TypeError):
        return False


def _instruction_removes(instruction: str, text: str, desc: str = "") -> bool:
    """Did the USER ask for this lettering to go? The critic's verdicts never
    come through `instruction`, so they never satisfy this. Read by the model:
    «убери второй шаг» takes the label "Text for step two" with it, «убери
    Вику» names the lettering «Вика», «убери среднюю банку» takes the jar's label."""
    if not (instruction or "").strip():
        return False
    import intent
    what = ("the lettering «%s»" % text) if text else "this lettering"
    if desc:
        what += " (%s)" % desc
    return intent.ask_yes(
        "A user asked to edit a picture: {text}. The picture has " + what + ". Does the "
        "instruction remove it -- by naming its words, the thing it labels or belongs to, "
        "or all the lettering?", instruction)


def _swaps_outermost(instruction: str) -> bool:
    import intent
    return intent.ask_yes("A user asked to swap things in a picture: {text}. Does the user "
                          "mean the two OUTERMOST ones (at both ends of the row)?", instruction)


def apply_ops(layout: dict, ops: list, *, additive_only: bool = False,
              instruction: str = "") -> tuple:
    """Apply edit operations to a layout. Returns (new_layout, notes).

    `additive_only` refuses the ops that can destroy a box (see
    `_additive_only`): set when the instruction only asked for something new.

    Every op is applied to a COPY, and an op that cannot be resolved is reported in
    the notes instead of throwing — a half-understood instruction must never destroy
    a layout the user spent time arranging.
    """
    layout = ideogram.normalize_layout(copy.deepcopy(layout))
    notes = []
    ops = _op_list(ops)
    if additive_only and _has_add_op(ops):
        # A vocabulary of removal words cannot be complete -- "lose the tree"
        # is a delete and matches none of them -- and a guard that fires on a
        # mixed instruction silently drops the half it did not understand.
        # So the words only raise the suspicion; the OPS settle it. An answer
        # that adds nothing at all while deleting or overwriting something is
        # the pathology; an answer that adds AND removes is a mixed
        # instruction being carried out, and is left alone.
        additive_only = False
    moved = set()
    touched = set()       # reworded in this batch
    numbered = list(_els(layout))     # the numbering the model was shown
    seen = set()
    for raw in ops:
        if not isinstance(raw, dict):
            notes.append(f"ignored a non-object op: {raw!r}")
            continue
        key = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue            # «в два раза больше» came as the same scale op twice: 4x
        seen.add(key)
        op = str(raw.get("op") or raw.get("action") or "").strip().lower()
        if not op and len(raw) == 1:
            # The op named by its KEY rather than by an "op" field:
            #   {"replace": {"target": 1, "desc": "a blonde man ..."}}
            # Measured live on gemma-4-12b: this is what it answers, and the
            # edit inside was exactly right — a single reworded box. We threw it
            # away as `unknown op ''`, so "make the man blond" silently did
            # nothing to the layout and the picture came back unchanged (or, on
            # another sampling, mangled). The model was never wrong here; the
            # parser only knew one of the two ways to write the same thing.
            only_key, only_val = next(iter(raw.items()))
            if isinstance(only_val, dict):
                op = str(only_key).strip().lower()
                raw = {**only_val, "op": op}
        if additive_only and op in _DESTRUCTIVE_OPS:
            # A `replace` under an add-only instruction is usually an ADD
            # written the wrong way round: told "добавь рыжего кота, который
            # спит на диване", the editor answers `replace the sofa with "a
            # ginger cat sleeping on a blue sofa"`. Refusing it outright
            # protects the sofa but leaves the user without the cat -- and the
            # description it wrote IS the thing to add.
            #
            # Unless the op is aimed at the very thing the user named. "Верни
            # фон белым" names the background and means change it; "верни
            # коробку обратно" named the box and the model replaced the MUG.
            # Whether the target matches the ask is the difference between an
            # ordinary reword and collateral damage.
            _new_desc = str(raw.get("desc") or raw.get("with") or raw.get("to")
                            or raw.get("description") or "").strip()
            _tgt = _find(layout, raw.get("target", raw.get("index")))
            _aimed_at_the_ask = False
            if _tgt >= 0 and instruction:
                # By STEM: Russian inflects, and "верни дивану зелёный цвет"
                # must recognise "синий диван" as the thing being talked about.
                # Whole-word matching missed it and turned a plain recolour
                # into a second sofa.
                _stem = lambda ws: {w[:5] for w in ws if len(w) > 3}
                _aimed_at_the_ask = bool(
                    _stem(_content_words(instruction.lower()))
                    & _stem(_content_words(
                        str(_els(layout)[_tgt].get("desc", "")).lower())))
            if _aimed_at_the_ask:
                pass                      # a reword of what was asked about
            elif op in ("delete", "remove", "drop") or not _new_desc:
                notes.append(f"skipped a '{op}' op: the instruction only asked "
                             f"to add something, so nothing may be removed")
                continue
            else:
                _anchor = (_els(layout)[_tgt] if _tgt >= 0 else None)
                raw = {"op": "add", "desc": _new_desc,
                       **({"x": _anchor["x"], "y": _anchor["y"],
                           "w": _anchor["w"], "h": _anchor["h"]} if _anchor else {})}
                notes.append(f"turned a '{op}' into an add: the instruction "
                             f"only asked for something new")
                op = "add"

        target = raw.get("target", raw.get("index", raw.get("desc")))
        els = _els(layout)

        def _resolve(label):
            """Element index for `target`, or -1 with a note appended ("label: no
            element matching ...") — the same "cannot find it, say so, move on"
            shape every op below needs before it can touch an element."""
            idx = _find(layout, target)
            if re.fullmatch(r"(?:element\s*)?#?\d+", str(target).strip().lower()):
                # «убери всё кроме девушки» = delete 2, delete 3: after the
                # first pop, 3 no longer existed and the lamppost stayed.
                n = _find({"elements": numbered}, target)
                idx = next((k for k, e in enumerate(els) if n >= 0 and e is numbered[n]), -1)
            elif idx >= 0:
                idx = _pick_twin(els, idx, f"{target} {instruction}")
            if idx < 0:
                notes.append(f"{label}: no element matching {target!r}")
            return idx

        try:
            if op in ("add", "insert", "new"):
                box = _anchor_box(str(raw.get("where") or raw.get("position") or "")) or \
                      {"x": 0.3, "y": 0.3, "w": 0.35, "h": 0.35}
                for k in ("x", "y", "w", "h"):      # a partial box still says something
                    if _num(raw.get(k)) is not None:
                        box[k] = _num(raw[k])
                # "target" as the thing to add is what the critic actually writes,
                # measured live: {"op": "add", "target": "a red bicycle leaning on
                # a wall", "x": ...} — three runs out of five, every one of them
                # skipped for "no description given" while the critic thought it
                # had asked for the missing element back.
                el = {"desc": str(raw.get("desc") or raw.get("what")
                                  or (target if isinstance(target, str) else "")).strip(),
                      "text": str(raw.get("text") or "").strip(), **box}
                if not el["desc"] and not el["text"]:
                    notes.append("add: no description given — skipped")
                    continue
                if el["text"] and not any(_num(raw.get(k)) is not None for k in ("w", "h")):
                    # A new lettering element with no box of its own gets the shape
                    # its string needs, not the generic 0.35 square — that square is
                    # exactly the geometry that renders as a smear.
                    box["w"], box["h"] = text_geometry(el["text"], min(box["h"], 0.14))
                    el.update(box)
                els.append(_clamp_box(el))
                notes.append(f'added “{el["desc"] or el["text"]}”')
            elif op in ("delete", "remove", "drop"):
                i = _resolve("delete")
                if i < 0:
                    continue
                if id(els[i]) in touched | moved:
                    # «посади кота на подоконник»: replace + move + delete of the same cat.
                    notes.append(f'kept “{str(els[i].get("desc") or "")[:30]}”: it was just '
                                 f"moved or reworded, deleting it too contradicts that")
                    continue
                # The requested lettering is never the critic's to delete. On
                # 2026-09-12 it saw stray text NEXT TO the label, answered with
                # "delete the label", and the next three renders had no words
                # to print at all. A user's own instruction to remove it is a
                # different path (edit_layout with that instruction).
                _txt = str(els[i].get("text") or "").strip()
                if _txt and not _instruction_removes(instruction, _txt, els[i].get("desc") or ""):
                    notes.append(f'kept “{_txt[:30]}”: the requested lettering is not '
                                 f"deleted on the critic's say-so")
                    continue
                # Nor is the thing the lettering is written ON: with the bottle
                # gone the label had no host, grew to half the frame, and the
                # render became a crop of a label (live, 2026-09-12).
                _hosted = [t for t in els if str(t.get("text") or "").strip()
                           and _box_contains(els[i], t)]
                if _hosted and not any(_instruction_removes(instruction, s, els[i].get("desc") or "") for s in
                                       [str(els[i].get("desc") or "")] + [str(t["text"]) for t in _hosted]):
                    notes.append(f'kept “{str(els[i].get("desc") or "")[:30]}”: it carries '
                                 f"the requested lettering")
                    continue
                notes.append(f'deleted “{els[i].get("desc", "")}”')
                els.pop(i)
            elif op in ("replace", "swap", "change", "set_desc", "rename",
                        # Rewording a box's description is its own intent — the
                        # model names it a dozen ways and a miss costs the whole op.
                        "desc", "describe", "redescribe", "set_description",
                        "update", "edit", "modify"):
                i = _resolve("replace")
                if i < 0:
                    continue
                new = str(raw.get("desc") or raw.get("with") or raw.get("to")
                          or raw.get("description") or "").strip()
                if new and new != els[i].get("desc"):
                    touched.add(id(els[i]))
                    notes.append(f'replaced “{els[i].get("desc", "")}” with “{new}”')
                    # "a dog looking towards the cat" after the cat became a parrot:
                    # other boxes that name the old subject now name the new one.
                    old_h, new_h = _head_noun(els[i].get("desc", "")), _head_noun(new)
                    if old_h and new_h and old_h.lower() != new_h.lower():
                        for o in els:
                            if o is not els[i]:
                                o["desc"] = re.sub(rf"\bthe {re.escape(old_h)}\b", "the " + new_h,
                                                   o.get("desc", ""), flags=re.I)
                    els[i]["desc"] = new
                if "text" in raw and str(raw.get("text") or "").strip() != els[i].get("text"):
                    els[i]["text"] = str(raw.get("text") or "").strip()
                    notes.append(f'lettering on “{els[i].get("desc", "")}” -> “{els[i]["text"]}”')
                before = {k: els[i][k] for k in ("x", "y", "w", "h")}
                for k in ("x", "y", "w", "h"):
                    if _num(raw.get(k)) is not None:
                        els[i][k] = _num(raw[k])
                _clamp_box(els[i])
                if not new and any(els[i][k] != before[k] for k in before):
                    # A box-only "replace" is a move/resize in disguise; without
                    # this it applied in silence and the loop could not tell it
                    # apart from an op that resolved to nothing.
                    notes.append(f'reshaped “{els[i].get("desc", "")[:30]}” to '
                                 f'{els[i]["w"]:.2f}×{els[i]["h"]:.2f} at '
                                 f'({els[i]["x"]:.2f}, {els[i]["y"]:.2f})')
            elif op == "swap_places":
                # Done by hand the model moved AND reworded both, swapping them twice.
                row = sorted((e for e in els if e["w"] < 0.6 and not e.get("text")),
                             key=lambda e: e["x"] + e["w"] / 2)
                if len(row) >= 2 and _swaps_outermost(instruction):
                    # the model picked the ice floe the penguins stand on as an "end" 2 times in 3
                    i, j = els.index(row[0]), els.index(row[-1])
                else:
                    target = raw.get("a"); i = _resolve("swap_places")
                    target = raw.get("b"); j = _resolve("swap_places")
                if i < 0 or j < 0 or i == j:
                    continue
                a, b = sorted((els[i], els[j]), key=lambda e: e["y"])
                if a["y"] + a["h"] <= b["y"] + 1e-6:
                    # One above the other (two lines of a poster): the big headline
                    # swapped by its feet landed on the small line. Top keeps top, bottom keeps bottom.
                    ay, by = b["y"] + b["h"] - a["h"], a["y"]
                else:
                    ay, by = b["y"] + b["h"] - a["h"], a["y"] + a["h"] - b["h"]
                (a["x"], a["y"]), (b["x"], b["y"]) = (
                    (b["x"] + b["w"] / 2 - a["w"] / 2, ay), (a["x"] + a["w"] / 2 - b["w"] / 2, by))
                _clamp_box(a); _clamp_box(b)
                moved.update((id(a), id(b)))
                notes.append(f'swapped “{a.get("desc", "")}” and “{b.get("desc", "")}”')
            elif op in ("move", "place", "position"):
                i = _resolve("move")
                if i < 0:
                    continue
                before = dict(els[i])
                box = _anchor_box(str(raw.get("where") or raw.get("position") or ""))
                if box:
                    els[i]["x"], els[i]["y"] = box["x"], box["y"]
                for k in ("x", "y"):
                    if _num(raw.get(k)) is not None:
                        els[i][k] = _num(raw[k])
                _clamp_box(els[i])
                _carry(els, els[i], before, moved)
                moved.add(id(els[i]))
                notes.append(f'moved “{els[i].get("desc", "")}” to '
                             f'({els[i]["x"]:.2f}, {els[i]["y"]:.2f})')
            elif op in ("resize", "scale", "size"):
                i = _resolve("resize")
                if i < 0:
                    continue
                before = dict(els[i])
                if "scale" in raw:
                    f = max(0.1, min(_num(raw["scale"], 1.0), 5.0))
                    cx = els[i]["x"] + els[i]["w"] / 2
                    bottom = els[i]["y"] + els[i]["h"]
                    els[i]["w"] *= f
                    els[i]["h"] *= f
                    # After a move in the same answer the move's x/y is where the
                    # model wants the grown box to start: "the biggest, in the
                    # centre" grew about (0.3, 0.2) into the corner.
                    if id(els[i]) not in moved:
                        # about the feet: "в два раза меньше" left the man floating mid-air
                        els[i]["x"] = cx - els[i]["w"] / 2
                        els[i]["y"] = bottom - els[i]["h"]
                for k in ("w", "h", "x", "y"):
                    if _num(raw.get(k)) is not None:
                        els[i][k] = _num(raw[k])
                _clamp_box(els[i])
                _carry(els, els[i], before, moved)
                notes.append(f'resized “{els[i].get("desc", "")}” to '
                             f'{els[i]["w"]:.2f}×{els[i]["h"]:.2f}')
            elif op in ("text", "retext", "set_text"):
                i = _resolve("text")
                if i < 0:
                    continue
                els[i]["text"] = str(raw.get("text") or "").strip()
                notes.append(f'lettering on “{els[i].get("desc", "")}” -> '
                             f'“{els[i]["text"]}”')
            elif op in ("background", "set_background"):
                layout["background"] = str(raw.get("desc") or raw.get("to") or "").strip()
                notes.append("background changed")
            elif op in ("style", "set_style"):
                for k in ("aesthetics", "lighting", "photo", "medium", "art_style",
                          "high_level_description"):
                    if raw.get(k):
                        layout[k] = str(raw[k]).strip()
                if (not raw.get("photo") and raw.get("medium")
                        and not re.search(r"photo", str(raw["medium"]), re.I)):
                    # «в стиле аниме» kept "standard lens, wide aperture".
                    layout.pop("photo", None)
                if raw.get("medium") and not raw.get("aesthetics"):
                    # «обратно как фото» kept "watercolor painting, bleeding colors".
                    layout["aesthetics"] = ", ".join(
                        p.strip() for p in str(layout.get("aesthetics") or "").split(",")
                        if not re.search(r"watercolou?r|oil|paint|brush|sketch|pencil|ink|vector|anime|"
                                         r"cartoon|comic|pixel|3d|render|photo|bleeding", p, re.I))
                notes.append("style changed")
            elif op in ("clear", "reset"):
                els.clear()
                notes.append("cleared every element")
            else:
                # An op name nobody anticipated ("shift", "enlarge", "reword") still
                # says what it wants in its own fields. Applying those to the element
                # it names beats throwing the adjustment away over its label.
                i = _find(layout, target)
                geom = [k for k in ("x", "y", "w", "h") if _num(raw.get(k)) is not None]
                new = str(raw.get("desc") or raw.get("to") or "").strip()
                if i >= 0 and (geom or new or "text" in raw):
                    for k in geom:
                        els[i][k] = _num(raw[k])
                    if new:
                        els[i]["desc"] = new
                    if "text" in raw:
                        els[i]["text"] = str(raw.get("text") or "").strip()
                    _clamp_box(els[i])
                    notes.append(f'op {op!r}: applied to “{els[i].get("desc", "")[:30]}”')
                else:
                    notes.append(f"unknown op {op!r} — skipped")
        except Exception as exc:
            # One malformed op must cost only that op, never the rest of the batch
            # and never the layout: the ops come from a model, not from code.
            logger.warning("op %r failed: %s", raw, exc)
            notes.append(f"op {op or '?'!r} could not be applied: {exc}")
    # «в центре и больше»: the move's x was for the old size, the grown cat sat at 0.38..0.76.
    mv = [e for e in _els(layout) if id(e) in moved]
    if len(mv) == 1 and __import__("intent").ask_yes(
            "A user asked to edit a picture: {text}\n\nDoes the instruction put something in "
            "the centre or middle of the picture?",
            instruction or ""):
        mv[0]["x"] = max(0.0, 0.5 - mv[0]["w"] / 2)
    _true_sides(_els(layout), notes)
    return ideogram.normalize_layout(layout), notes


_FRAME_RE = re.compile(r"\b(top|bottom|left|right)(?: side| edge| part)? of the (?:frame|picture|image|poster|canvas)", re.I)
_SIDE_RE = re.compile(r"\b(?:to|on) the (left|right)(?: side| edge)? of (?:the |a |an )?([\w-]+(?: [\w-]+)?)", re.I)


def _carry(els, el, before, moved) -> None:
    """What sits inside a box follows it: the speech bubble moved to the parrot
    and its words stayed over the dog; the man shrank and his t-shirt lettering
    stayed full width. The full-frame backdrop carries nothing."""
    if before["w"] >= FULL_FRAME and before["h"] >= FULL_FRAME:
        return
    fx, fy = el["w"] / max(1e-6, before["w"]), el["h"] / max(1e-6, before["h"])
    for o in els:
        if o is not el and id(o) not in moved and _covered(o, before) >= 0.9 \
                and o["w"] * o["h"] < 0.5 * before["w"] * before["h"]:
            o["x"] = el["x"] + (o["x"] - before["x"]) * fx
            o["y"] = el["y"] + (o["y"] - before["y"]) * fy
            o["w"] *= fx
            o["h"] *= fy
            _clamp_box(o)


def _head_noun(desc: str) -> str:
    """"a grumpy tabby cat with arched back facing the dog" -> "cat"."""
    words = re.split(r"\s(?:with|on|in|at|facing|holding|sitting|standing|lying|wearing|"
                     r"near|by|of|looking|that|which|who)\s", (desc or "").strip())[0].split()
    return re.sub(r"\W", "", words[-1]) if words else ""


def _true_sides(els, notes) -> None:
    """"to the right of the vase" after a swap put the cat on the left: the
    model rewords that 2 times in 3, the boxes are the truth every time."""
    for el in els:
        m = _SIDE_RE.search(str(el.get("desc") or ""))
        if not m:
            continue
        k = _find({"elements": [o for o in els if o is not el]}, m.group(2))
        if k < 0:
            continue
        anchor = [o for o in els if o is not el][k]
        side = "left" if el["x"] + el["w"] / 2 < anchor["x"] + anchor["w"] / 2 else "right"
        if side != m.group(1).lower():
            el["desc"] = el["desc"][:m.start(1)] + side + el["desc"][m.end(1):]
            notes.append(f'reworded “{el["desc"][:40]}” to match where it now stands')
    for el in els:
        m = _FRAME_RE.search(str(el.get("desc") or ""))
        if not m:
            continue
        c = el["y"] + el["h"] / 2 if m.group(1).lower() in ("top", "bottom") else el["x"] + el["w"] / 2
        now = {"top": c < 0.5, "bottom": c > 0.5, "left": c < 0.5, "right": c > 0.5}
        if not now[m.group(1).lower()]:
            flip = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[m.group(1).lower()]
            el["desc"] = el["desc"][:m.start(1)] + flip + el["desc"][m.end(1):]
            notes.append(f'reworded “{el["desc"][:40]}” to match where it now stands')


# --------------------------------------------------------------------------- #
# Editing by instruction
# --------------------------------------------------------------------------- #
_EDIT_PROMPT = """You edit the layout of a picture. The layout is a list of elements, \
each with a box: x, y (top-left corner) and w, h (size), all fractions of the frame \
from 0 to 1, with x=0,y=0 at the TOP-LEFT.

You will be given the current layout as JSON and one instruction from the user.
Answer with ONLY a JSON object listing the operations to apply — no prose, no fence:

{"ops": [
  {"op": "delete", "target": "the dog"},
  {"op": "add", "desc": "a red bicycle", "x": 0.1, "y": 0.55, "w": 0.3, "h": 0.35},
  {"op": "replace", "target": 2, "desc": "an oak tree"},
  {"op": "move", "target": "the sign", "x": 0.65, "y": 0.05},
  {"op": "swap_places", "a": 1, "b": 4},
  {"op": "resize", "target": "the tower", "scale": 1.4},
  {"op": "text", "target": "the sign", "text": "OPEN"},
  {"op": "background", "to": "a rainy street at night"},
  {"op": "style", "lighting": "harsh noon sun"}
]}

Rules:
- "target" is either the element's number as shown, or words from its description.
- Change ONLY what the instruction asks for. Do not re-list untouched elements.
- Use as many operations as the instruction needs: you are free to move a box, \
resize it, reword its description, add a new one and delete an old one in the same \
answer. One instruction often means several of those ("put the dog by the bench \
instead" is a move, and maybe a resize).
- "replace" reworks an element in place — a new "desc" changes what is drawn there \
without disturbing the rest of the scene. Prefer it over delete+add.
- A box carrying lettering must be roughly 0.6 x (number of characters) times as \
wide as it is high, and at least 0.08 of the frame high, or the letters come out as \
a smear.
- Keep x+w <= 1 and y+h <= 1.
- Give a new element a box that does not sit on top of an existing one unless the \
instruction says the two things overlap.
- Every countable subject gets its OWN box. "Make it two cats" keeps the existing \
cat and ADDS a second cat box beside it — never write "two cats" into one \
description, the renderer would draw them merged.
- A relative-position word ("next to", "beside", "in front of", "behind", "on top \
of", "under") names an EXISTING element as the anchor. Read that element's own \
box first, then place the new one from it: "next to"/"beside" -- same y (share \
its vertical range), x placed immediately to one side, touching or a small gap \
away, never at a fixed corner or the opposite edge of the frame. "in front of" -- \
larger and lower (bigger h, higher y+h) than the anchor. "behind" -- smaller and \
higher up (further from camera). Live failure this caused: "add a motorcyclist \
next to the car" put the rider's box at x=0.00 (the far edge) while the car sat \
center-frame — nowhere near "next to" — because nothing here said to read the \
anchor's box before choosing coordinates.
- When the instruction puts one thing ON, IN or RIDING another ("put the cat on \
the dog's back"), move its box AND reword its description to say so ("a small \
tabby cat sitting on the dog's back"): the renderer sees two overlapping boxes, \
not the relation.
- A move that makes a description's own position words false ("to the right \
of the vase" after a swap put it on the left) rewords them in the same answer.
- "first", "last", "outermost", "second from the right" count along the row of \
the things the user is talking about (the penguins), never a floor, table or \
surface they stand on. "Swap them" is ONE swap_places op and nothing else: no move, no reworded description.
- Other descriptions that name what you replaced or moved ("a dog looking towards \
the cat" after the cat became a parrot, "a bubble above the dog" moved to the \
parrot) get a "replace" in the same answer that names it correctly.
- Write every description in English. Lettering ("text") is in the language of the instruction: copied exactly when the user dictates it, and when you make it up for them («подпиши банки»), written in their language — «Клубника», not «STRAWBERRY»."""


_MISMATCH_RE = re.compile(
    r"(?:^|\b)(?:the |a |an )?(?P<what>[\w][\w' -]{0,40}?)\s+(?:is|are|was|were|came out|appears?|looks?|rendered)\s+"
    r"(?P<got>[\w][\w /-]{0,30}?)\s+(?:instead of|rather than|not)\s+(?P<want>[\w][\w /-]{0,40}?)(?:[.;,]|$)",
    re.IGNORECASE)


def insist_on_mismatch(layout: dict, problems: list) -> tuple:
    """Turn "the elephant is brown instead of bright pink" into an underlined
    description ("... elephant, bright pink -- not brown") on the element it
    names. Returns (layout, notes, problems left for the editor)."""
    notes, left = [], []
    els = layout.get("elements") or []
    for p in problems or []:
        m = _MISMATCH_RE.search(str(p or ""))
        if not m:
            left.append(p); continue
        what, got, want = (m.group("what").strip().lower(), m.group("got").strip(),
                           m.group("want").strip().rstrip("."))
        head = what.split()[-1] if what else ""
        hit = None
        for el in els:
            d = str(el.get("desc") or "").lower()
            if head and re.search(r"\b" + re.escape(head) + r"\b", d) and want.lower() in d:
                hit = el; break
        if hit is None:
            for el in els:
                if want.lower() in str(el.get("desc") or "").lower():
                    hit = el; break
        if hit is None:
            left.append(p); continue
        tag = f"{want} -- not {got}"
        if tag.lower() not in str(hit.get("desc") or "").lower():
            hit["desc"] = str(hit.get("desc") or "").rstrip(". ") + f", {tag}"
            notes.append(f"insisted: {head or what} must be {want}, not {got}")
        else:
            notes.append(f"insisted again: {head or what} must be {want}, not {got}")
    return layout, notes, left


_ILLUSTRATION_STYLE = {
    "aesthetics": "vivid saturated colours, clean stylised shapes",
    "lighting": "bright even light",
    "photo": "",
    "medium": "illustration",
    "art_style": "stylised digital illustration",
}


def unphoto_on_repeat_mismatch(layout: dict, notes: list, request: str = "") -> tuple:
    """Drop the photographic DEFAULT when the underlined attribute still did
    not come through.

    Live, 2026-09-12 (journey 4): "a blue elephant on a beach" was rendered as
    a photograph five times, grey every time -- the photo prior of a real
    animal beats any wording, and the caption had been insisting on "vibrant
    blue -- not grey" since attempt two. The photography was the floor the
    layout gets when the user named no style (ideogram_layout), not something
    the user asked for; a picture of the wrong colour is further from the
    request than an illustration of the right one. A user who asked for
    realism keeps it."""
    if not any(str(n).startswith("insisted again:") for n in notes or []):
        return layout, ""
    if str(layout.get("medium") or "") != "photography":
        return layout, ""
    try:
        import ideogram_layout as _il
        if _il.wants_realism(request or ""):
            return layout, ""
    except Exception:
        pass
    layout.update(_ILLUSTRATION_STYLE)
    return layout, "the photographic default kept overriding the colour -- drawing it as an illustration"


def edit_layout(ctx, layout: dict, instruction: str) -> tuple:
    """Apply a natural-language edit to `layout`. Returns (layout, notes)."""
    instruction = (instruction or "").strip()
    layout = ideogram.normalize_layout(layout)
    if not instruction:
        return layout, ["empty instruction"]
    if ctx is None:
        return layout, ["no model available for editing"]
    try:
        import llm
        res = llm.send_to_lm_studio(
            ctx,
            [{"role": "system", "content": _EDIT_PROMPT},
             {"role": "user", "content": json.dumps(_for_model(layout), ensure_ascii=False)
              + "\n\nInstruction: " + instruction}],
            tools=[], tool_choice="none", temperature=0.2, max_tokens=6000,
            prefill="<think></think>")
        # Layout-only extraction discarded {"ops": [...]} the same way it
        # discarded the critic's verdict — every edit came back "the model
        # proposed no change".
        data = safe_json_from_llm((res or {}).get("content") or "",
                                  required_keys=("ops", "op", "action"))
        ops = None
        if isinstance(data, list):
            ops = data
        elif isinstance(data, dict):
            ops = data.get("ops") or ([data] if data.get("op") or data.get("action")
                                      else None)      # a lone op, answered unwrapped
        if not ops:
            return layout, ["the model proposed no change"]
        layout, notes = apply_ops(layout, ops,
                                  additive_only=_additive_only(instruction),
                                  instruction=instruction)
        return split_counted_boxes(merge_text_column(layout), notes)
    except Exception as exc:
        logger.exception("layout edit failed")
        return layout, [f"edit failed: {exc}"]


# "let there be two cats" came back as ONE box described "two fluffy gray cats
# sitting together" — and the renderer drew one merged mass with extra limbs
# (live 2026-09-13, mega step 127; the same in the 12.09 run). A countable
# subject gets its own box: the box is split side by side, one subject each.
# Explicit numbers only: "a pair of glasses" is one object.
_COUNT_WORDS = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}
_COUNTED_DESC_RE = re.compile(
    r"^\s*(?P<n>two|three|four|[234])\s+(?P<rest>\S.*)$",
    re.IGNORECASE)
_TOGETHER_RE = re.compile(
    r"\s*,?\s*\b(?:sitting|standing|lying|walking|huddled|cuddling|snuggling|"
    r"playing|posing|sleeping)?\s*(?:together|side by side|next to each other|"
    r"beside each other)\b", re.IGNORECASE)


def _singular(phrase: str) -> str:
    """'fluffy gray cats sitting on a sill' -> 'fluffy gray cat sitting on a sill'.

    Only the FIRST plural-looking word is touched — the head noun of a counted
    phrase comes before its participles and prepositions."""
    words = phrase.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in ("is", "was", "has", "as", "his", "its", "this", "glasses", "jeans",
                  "trousers", "shorts", "pants", "scissors", "series", "species"):
            continue
        if lw.endswith("ies") and len(lw) > 4:
            words[i] = w[:-3] + "y"; return " ".join(words)
        if lw.endswith(("ches", "shes", "xes", "sses", "zes")):
            words[i] = w[:-2]; return " ".join(words)
        if lw.endswith("s") and not lw.endswith("ss") and len(lw) > 3:
            words[i] = w[:-1]; return " ".join(words)
    return phrase


def merge_text_column(layout: dict) -> dict:
    """Past MAX_ELEMENTS a list of same-styled lines (a menu) becomes one multi-line box."""
    els = list(layout.get("elements") or [])
    if len(els) <= MAX_ELEMENTS:
        return layout
    groups = {}
    for i, el in enumerate(els):
        if el.get("text"):
            groups.setdefault(str(el.get("desc") or ""), []).append(i)
    idx = max(groups.values(), key=len, default=[])
    if len(idx) < 3:
        return layout
    col = sorted((els[i] for i in idx), key=lambda e: (e["y"], e["x"]))
    x0, y0 = min(e["x"] for e in col), min(e["y"] for e in col)
    x1, y1 = max(e["x"] + e["w"] for e in col), max(e["y"] + e["h"] for e in col)
    merged = dict(col[0], x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                  text="\n".join(str(e["text"]) for e in col))
    keep = [e for i, e in enumerate(els) if i not in idx]
    keep.insert(idx[0], merged)
    return dict(layout, elements=keep)


def split_counted_boxes(layout: dict, notes: list) -> tuple:
    """One box per counted subject, side by side; stacked only when the words say so."""
    els = list(layout.get("elements") or [])
    out, changed = [], False
    for el in els:
        desc = str(el.get("desc") or "")
        m = _COUNTED_DESC_RE.match(desc)
        if not m or el.get("text"):
            out.append(el); continue
        n = _COUNT_WORDS.get(m.group("n").lower(), 0)
        if n < 2:
            out.append(el); continue
        rest = _TOGETHER_RE.sub("", m.group("rest")).strip(" ,")
        single = _singular(rest)
        x, y, w, h = float(el["x"]), float(el["y"]), float(el["w"]), float(el["h"])
        # "three palm trees" in a tall box were split into a totem, one on another's head.
        horizontal = not re.search(r"\bstack|\bpile|on top of each other|\btower", desc, re.I)
        if horizontal and w / n < h * 0.25:     # slivers: widen about the centre instead
            cx, w = x + w / 2, min(1.0, n * h * 0.25)
            x = min(max(0.0, cx - w / 2), 1.0 - w)
        for i in range(n):
            part = dict(el)
            part["desc"] = single
            if horizontal:
                part["x"], part["w"] = round(x + w * i / n, 4), round(w / n, 4)
            else:
                part["y"], part["h"] = round(y + h * i / n, 4), round(h / n, 4)
            out.append(part)
        changed = True
        notes = list(notes) + [f"split “{desc}” into {n} boxes of “{single}”"]
    if not changed:
        return layout, notes
    new = dict(layout); new["elements"] = out
    return ideogram.normalize_layout(new), notes


def _for_model(layout: dict) -> dict:
    """The layout as the model should see it: 1-based numbers, rounded boxes."""
    return {
        "background": layout.get("background", ""),
        "style": {k: layout.get(k, "") for k in ("aesthetics", "lighting", "photo",
                                                 "medium", "art_style") if layout.get(k)},
        "elements": [
            {"n": i + 1, "desc": el.get("desc", ""), "text": el.get("text", ""),
             "x": round(el["x"], 2), "y": round(el["y"], 2),
             "w": round(el["w"], 2), "h": round(el["h"], 2)}
            for i, el in enumerate(layout.get("elements") or [])
        ],
    }


# --------------------------------------------------------------------------- #
# Splitting an edit into region + content (for a pixel-preserving picture edit)
# --------------------------------------------------------------------------- #
_SPLIT_PROMPT = """A user wants to change one thing in an existing picture by editing \
ONLY that part and leaving everything else untouched. Turn their instruction into JSON:

{"region": "the thing in the picture to edit, as a short noun phrase",
 "content": "what that region should become",
 "removal": false}

Examples:
- "swap the grandpa for a grandma" -> {"region": "the old man", "content": "an elderly woman, a grandmother", "removal": false}
- "make the car red" -> {"region": "the car", "content": "a red car", "removal": false}
- "remove the sign" -> {"region": "the sign", "content": "", "removal": true}

Answer with ONLY the JSON object, no prose. Write both fields in English."""


def split_edit(ctx, instruction: str) -> dict:
    """Turn 'swap the grandpa for a grandma' into {region, content, removal}.

    This drives a CONTAINED picture edit (segment the region, redraw only it,
    composite back) instead of regenerating the whole caption — the only way to
    change a subject's identity while keeping the rest of the photo. Falls back to
    using the whole instruction as both fields so the edit still runs.
    """
    instruction = (instruction or "").strip()
    fallback = {"region": instruction, "content": instruction, "removal": False}
    if not instruction or ctx is None:
        return fallback
    try:
        import llm
        res = llm.send_to_lm_studio(
            ctx,
            [{"role": "system", "content": _SPLIT_PROMPT},
             {"role": "user", "content": instruction}],
            tools=[], tool_choice="none", temperature=0.1, max_tokens=200,
            prefill="<think></think>")
        data = safe_json_from_llm((res or {}).get("content") or "",
                                  required_keys=("region", "content", "removal"))
        if not isinstance(data, dict):
            return fallback
        region = str(data.get("region") or "").strip()
        content = str(data.get("content") or "").strip()
        removal = bool(data.get("removal"))
        if not region:
            return fallback
        return {"region": region, "content": content, "removal": removal}
    except Exception:
        logger.exception("splitting the edit instruction failed")
        return fallback


# --------------------------------------------------------------------------- #
# Lettering: the one thing the renderer gets wrong in a way you can MEASURE


# --------------------------------------------------------------------------- #
# Looking at the result
# --------------------------------------------------------------------------- #
# Kept SHORT on purpose. Measured 2026-07-29 on the reader model: the length of
# this prompt drives how long the model deliberates before answering, and the
# deliberation is what eats the token budget. A version with worked examples of
# every operation burned 1997 reasoning tokens and answered 1 time in 4 — the
# other three were cut off mid-thought and reported as "the critic could not
# see". This wording sits at ~600 and answered 4 times in 4. The full editing
# vocabulary lives in `_EDIT_PROMPT`, which `run` calls on the critic's own
# problem statements when its ops do not cover them.
_CRITIQUE_PROMPT = """You are checking whether a generated picture matches the layout \
it was drawn from. You will be given the layout (each element with the region of the \
frame it should occupy) and the picture.

Answer with ONLY a JSON object, no prose, no fence:
{"ok": true|false,
 "score": 0-10,
 "problems": ["short factual statements about what is wrong"],
 "ops": [{"op": "...", "target": "...", ...}]}

Judge only these things:
- Is each listed element actually there, once, recognisable?
- Is it roughly in the region the layout asked for?
- Did two elements merge into one object, or did one element appear twice?
- Is the picture incoherent (melted shapes, a smear, unreadable lettering)?

Set "ok" to false only if something on that list is really wrong. Style, taste and \
beauty are NOT problems. If everything listed is present and placed, say ok.

Repair what is wrong, do not just describe it: "ops" may move, resize, add, delete \
or replace a box, and replace may also reword a description that came out too vague. \
"target" is an element number as shown or words from its description; x, y are the \
TOP-LEFT corner and w, h the size, fractions of the frame, x+w and y+h at most 1. \
If a piece of lettering is reported misspelled, give that box more width and height. \
Leave "ops" empty when ok is true."""


def _regions(layout: dict) -> str:
    """The layout in words a vision model can check against a picture."""
    lines = []
    for i, el in enumerate(layout.get("elements") or []):
        cx, cy = el["x"] + el["w"] / 2, el["y"] + el["h"] / 2
        horiz = "left" if cx < 0.38 else "right" if cx > 0.62 else "centre"
        vert = "top" if cy < 0.38 else "bottom" if cy > 0.62 else "middle"
        lines.append(
            f'{i + 1}. {el.get("desc", "")}'
            + (f' with the text “{el["text"]}”' if el.get("text") else "")
            + f' — {vert} {horiz}, covering {el["w"] * el["h"] * 100:.0f}% of the frame '
              f'(x {el["x"]:.2f}-{el["x"] + el["w"]:.2f}, y {el["y"]:.2f}-{el["y"] + el["h"]:.2f})')
    bg = layout.get("background", "")
    return (f"Background: {bg}\nElements:\n" + "\n".join(lines)) if lines else f"Background: {bg}"


# One measurement, shared with the outer judge in image_generate (exposure.py).
import exposure as _exposure  # noqa: E402
from exposure import DARK_RE as _TOO_DARK_RE, measure as _brightness  # noqa: E402


# MEASURED, not chosen. bench/draw_seed_sweep.py rendered the one caption that
# ever collaged in the 14-scene bench at eight different seeds: 8 of 8 came back
# as collages. A fresh seed does not move this failure at all, so every re-roll
# was ~85s of the user's time spent to arrive at the same picture -- 170s per
# collaged render, on top of the rung that actually fixes it.
#
# The rungs below DO change something structural (the frame's aspect, the boxes'
# span, the number of elements), and one of them is what carried the 14/14 clean
# run: two_of_the_same finished it in 353s, which is four renders -- the first
# three being the initial draw and the two wasted re-rolls.
#
# Set back above zero only with a measurement. The sweep is the tool for it.
COLLAGE_REROLLS = 0
_SEAM_JUMP = 28          # a panel border changes this much, at once, along a line
_SEAM_FRAC = 0.85        # ...across at least this share of the line
_CHECKER_FRAC = 0.12     # transparency checkerboard, as a share of the picture
_CHECKER_CONCENTRATION = 0.30   # ...and how tightly its grey sits in a few tones


def _line_seams(gray, jump: int = _SEAM_JUMP, frac: float = _SEAM_FRAC) -> tuple:
    """Interior rows and columns where the picture changes ALL AT ONCE.

    A photograph is full of edges, but they belong to objects: they curve, they
    stop, they do not run the entire width of the frame at one exact y. A
    collage panel border does exactly that, which is what makes this
    measurable rather than a matter of taste.
    """
    import numpy as _np
    rows_cols = []
    for arr in (gray, gray.T):
        n = arr.shape[0]
        d = _np.abs(arr[2:, :].astype(int) - arr[:-2, :].astype(int))
        hit = [i + 1 for i, f in enumerate((d > jump).mean(axis=1)) if f >= frac]
        # A seam is a few pixels thick; collapse each run to one line.
        merged = []
        for i in hit:
            if not merged or i - merged[-1][-1] > 6:
                merged.append([i])
            else:
                merged[-1].append(i)
        # Interior only: the picture's own border is not a panel divider.
        rows_cols.append([int(sum(m) / len(m)) for m in merged
                          if 0.12 * n < sum(m) / len(m) < 0.88 * n])
    return rows_cols[0], rows_cols[1]


def _widened(width: int, height: int) -> tuple:
    """A 16:9 frame of about the same pixel count, or (0, 0) if already wide.

    A SQUARE frame is what a 2x2 grid fits into. Measured on the layout that
    collages on every seed: three boxes at 1024x1024 came back as a grid on
    three seeds running, and the SAME caption -- same boxes, same words -- at
    1280x720 came back as one coherent room on both seeds tried. Nothing about
    the request changed except the shape of the canvas.

    So this is a structural lever rather than another roll of the dice, and it
    keeps the boxes, which the merged-scene fallback gives up.
    """
    if not width or not height or width >= height * 1.3:
        return 0, 0
    area = float(width) * float(height)
    w = int(round((area * 16.0 / 9.0) ** 0.5 / 64.0)) * 64
    h = int(round(w * 9.0 / 16.0 / 64.0)) * 64
    return (w, h) if w > 0 and h > 0 else (0, 0)


def looks_like_collage(image_path: str) -> bool:
    """True when the render is a GRID OF PICTURES instead of one picture.

    The `two_of_the_same` layout -- a cat on the floor, a cat in an armchair, a
    table, over a thin background -- comes back as a 2x2 grid of separate stock
    photographs, often on a transparency checkerboard and with invented people
    or posters in the spare panels.

    Wording cannot fix it. Measured over eight renders at one fixed seed: a
    hand-written "this is one photograph of one room" sentence produced a single
    coherent room and reproduced pixel for pixel across runs, but every
    mechanically built version of that sentence collaged -- and changing ONE
    PREPOSITION in the sentence that worked ("в кресле" -> "на кресле") collapsed
    it into a collage too. So the outcome is chaotically sensitive to the exact
    string, and no template can be relied on. What CAN be relied on is looking
    at the pixels afterwards.

    Two independent signals, either of which is enough:
      * a full-width or full-height hard seam somewhere in the interior;
      * a transparency checkerboard covering a good share of the frame.

    Tuned on fourteen labelled renders of that layout (twelve collages, two
    single rooms) and correct on all of them. Deliberately conservative: the
    cost of a false positive is one more render, the cost of a miss is shipping
    a grid of stock photos as the user's picture.
    """
    try:
        import numpy as _np
        from PIL import Image as _Image
        im = _Image.open(image_path).convert("RGB")
        rgb = _np.asarray(im)
        rows, cols = _line_seams(_np.asarray(im.convert("L")))
        if rows or cols:
            return True
        sat = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)
        lum = rgb.mean(axis=2)
        grey = (sat <= 6) & (lum > 175) & (lum < 245)
        if grey.mean() <= _CHECKER_FRAC:
            return False
        # Greyness alone is not the checkerboard -- a white wall, an overcast
        # sky and a smooth light gradient are all desaturated and bright, and
        # condemning those would throw away good pictures. What separates them
        # is CONCENTRATION: the checkerboard is a synthetic backdrop of one or
        # two exact tones, while a real light surface is shaded across many.
        # Measured on the labelled renders: a collage backdrop puts 0.38-0.53
        # of its grey pixels into the top five luminance bins, a real room 0.13.
        # The threshold sits between, with margin on both sides.
        hist = _np.bincount(lum[grey].astype(int), minlength=256)
        total = int(hist.sum())
        if total <= 0:
            return False
        top5 = float(_np.sort(hist)[::-1][:5].sum()) / total
        return bool(top5 >= _CHECKER_CONCENTRATION)
    except Exception:
        # A detector that cannot read the file must not condemn the picture.
        logger.debug("collage check failed on %s", image_path, exc_info=True)
        return False


def render_without_collage(ctx, layout, *, width: int, height: int, seed: int,
                           steps: Optional[int] = None, cfg: Optional[float] = None,
                           on_progress=None, emit=None):
    """Render `layout`, and do not hand back a grid of separate pictures.

    Lives here rather than inside run() so that anything which draws -- the
    agent, the bench, a future caller -- gets the same defence. Measured over 53
    renders from earlier scenario runs, 15 came back as collages or cut-outs on
    a transparency checkerboard: this is not one layout misbehaving, it is about
    a quarter of everything this pipeline draws.

    Four rungs, each skipped when it cannot apply, none of which may cost the
    picture already in hand:

      1. draw it;
      2. re-roll the SEED, twice. The seed is otherwise held fixed on purpose so
         an edit moves only its box -- a collage is the one case where nothing
         in the picture is worth keeping stable. Measured: `add_cat` collaged in
         one run of two, so a fresh seed genuinely rescues some scenes;
      3. redraw WIDE. A 2x2 grid needs a squarish frame; the same caption that
         gridded three times at 1024x1024 came back as one room at 1280x720.
         Measured: `two_edits` collaged in all five runs it appears in, so for
         those scenes the seed is no lever and the canvas is;
      4. merge the boxes into ONE full-frame element. Last because it gives up
         the arrangement, and because that caption draws Ideogram's safety card
         about as often as it draws a room.

    `emit(kind, **fields)` is optional and used only for progress text.
    """
    def _say(text):
        if emit is not None:
            emit("stage", text=text)

    image = None
    # ONE claim for the whole repair chain, not one per attempt. Every retry
    # below decides on pixels alone -- no model is consulted between them --
    # so claiming per submit unloaded and reloaded a 20 GB model up to five
    # times for nothing. The layout critic runs AFTER this returns, with the
    # model back; it must never be moved inside.
    with comfy_client.card_session("drawing"):
        for _reroll in range(COLLAGE_REROLLS + 1):
            image = ideogram.generate(ctx, "", width=width, height=height, seed=seed,
                                      caption=ideogram.layout_to_caption(layout),
                                      steps=steps, cfg=cfg, on_progress=on_progress)
            if not image or _reroll >= COLLAGE_REROLLS or not looks_like_collage(image):
                break
            seed = random.randint(1, 999_999_999)
            _say("that came back as a collage of separate pictures — "
                 f"redrawing on seed {seed}")

        if image and looks_like_collage(image):
            ww, wh = _widened(width, height)
            if ww:
                _say("still a collage — redrawing wider, where a 2x2 grid does not fit")
                try:
                    alt = ideogram.generate(
                        ctx, "", width=ww, height=wh,
                        seed=random.randint(1, 999_999_999),
                        caption=ideogram.layout_to_caption(layout),
                        steps=steps, cfg=cfg, on_progress=on_progress)
                except ideogram.ContentRefused:
                    logger.info("wide redraw refused; falling through")
                    alt = None
                except Exception:
                    logger.warning("wide redraw failed", exc_info=True)
                    alt = None
                if alt and not looks_like_collage(alt):
                    image = alt

        if image and looks_like_collage(image):
            # A CUT-OUT is not a collage, and it has its own cause: boxes that leave
            # wide empty margins, which the renderer fills with transparency rather
            # than with the room. Stretching the same arrangement to the frame edges
            # fixed it at a fixed seed with nothing else changed. Tried before the
            # merge because it keeps the arrangement; the merge throws it away.
            filled = ideogram.filled_layout(layout)
            if filled.get("elements") != layout.get("elements"):
                _say("still not one picture — stretching the boxes to fill the frame")
                try:
                    alt = ideogram.generate(
                        ctx, "", width=width, height=height,
                        seed=random.randint(1, 999_999_999),
                        caption=ideogram.layout_to_caption(filled),
                        steps=steps, cfg=cfg, on_progress=on_progress)
                except ideogram.ContentRefused:
                    logger.info("filled redraw refused; falling through")
                    alt = None
                except Exception:
                    logger.warning("filled redraw failed", exc_info=True)
                    alt = None
                if alt and not looks_like_collage(alt):
                    image = alt

        if image and looks_like_collage(image):
            merged = ideogram.merged_layout(layout)
            if merged.get("elements") != layout.get("elements"):
                _say("still a collage — redrawing the scene as one picture instead "
                     "of separate boxes")
                for _ in range(2):
                    try:
                        alt = ideogram.generate(
                            ctx, "", width=width, height=height,
                            seed=random.randint(1, 999_999_999),
                            caption=ideogram.layout_to_caption(merged),
                            steps=steps, cfg=cfg, on_progress=on_progress)
                    except ideogram.ContentRefused:
                        # The merged caption draws the safety card on some seeds.
                        # That must cost a seed, never the picture already in hand.
                        logger.info("merged-scene fallback refused; trying another seed")
                        continue
                    except Exception:
                        logger.warning("merged-scene fallback failed", exc_info=True)
                        break
                    if alt and not looks_like_collage(alt):
                        image = alt
                        break
    return image


def _drop_unfounded(image_path: str, problems: list) -> list:
    """Throw out darkness complaints the pixels contradict.

    Measured 2026-08-29 over the fourteen bench renders: the critic called
    move_cat "extremely dark and incoherent, making most elements
    unidentifiable" and replace_mug "mostly black silhouettes". Both are
    ordinary, clearly lit interiors -- mean luminance 73 and 109 of 255, 95th
    percentile 162 and above. Asked to simply describe move_cat, the same model
    named the room, the table and the cat correctly and then said "very dark
    with high contrast". So it sees the picture and misjudges its exposure, in
    one direction, consistently.

    That matters because the verdict decides whether to redraw: a picture that
    is already right gets thrown away and drawn again, which is what a redraw
    request looks like from the outside. Brightness is one of the few things
    here that can be MEASURED, so it is measured, and the complaint is dropped
    when the pixels disagree. Everything else the critic says is left alone.
    """
    if not problems:
        return problems
    m = _brightness(image_path)
    if not _exposure.normally_exposed(m):      # genuinely dark: let it stand
        return problems
    return [p for p in problems if not _TOO_DARK_RE.search(str(p))]


# What a complaint has to mention to count as a layout problem rather than an
# opinion. The critic's own checklist, in the words it uses.
_LAYOUT_CONCERN_RE = re.compile(
    r"missing|absent|not (?:there|present|visible|shown|rendered|appear)|invisible|"
    r"no [a-z]+ (?:is|are|was) |instead of|wrong|incorrect|misspell|"
    r"twice|duplicate|second (?:copy|one)|merged|fused|melted|smear|incoheren|"
    r"unreadable|illegible|garbled|misplaced|out of (?:place|frame|position)|"
    r"cut off|cropped|collage|panel|split|blank|empty|placeholder|"
    r"does not match|doesn't match|looks like a|should read|but the picture shows|"
    r"lacks|lacking|too dark|entirely black|underexpos|"   # real on a picture that measures dark
    # Something EXTRA is as wrong as something missing: "a third cat visible
    # on the left which was not in the layout" was dropped as taste and a
    # two-cat request shipped with three (live, 2026-09-12, journey 21).
    r"\bextra\b|additional|\b(?:third|fourth|fifth)\b|more than (?:one|two|three|four)|"
    r"not in the (?:layout|plan|request)|not (?:asked|requested)|wasn't (?:in|part of)|"
    r"was not (?:in|part of|requested)|stray|unwanted|unrequested|"
    # The COLOUR of a named thing is content, not taste: "The elephant is not
    # pink as requested" was dropped as taste and a tan elephant shipped as a
    # successful recolour, twice (live 2026-09-18, journey 4).
    r"wrong colou?r|colou?r is (?:wrong|not|off)|"
    r"\b(?:is|are|looks?|appears?|remains?|still)\s+not\s+(?:red|pink|blue|green|yellow|"
    r"orange|purple|violet|black|white|brown|grey|gray|golden?|silver|turquoise|"
    r"beige|teal|navy|cyan|magenta)\b",
    re.IGNORECASE)


def critique(ctx, image_path: str, layout: dict, evidence: Optional[list] = None) -> dict:
    """Look at the render and report what the layout failed to produce.

    Returns {"ok", "score", "problems", "ops", "source"}. A vision failure is
    reported as ok=True with source="unavailable": a critic that cannot see must
    not be allowed to condemn a picture that may be perfectly good.

    `evidence` is what the deterministic checks already know is wrong — mainly the
    lettering read back off the picture. The critic cannot see a misspelling (it
    reads the word it expects, which is the whole reason `verify_text` exists), so
    without being told it would approve a garbled sign and propose no repair for it.
    """
    layout = ideogram.normalize_layout(layout)
    if image_path and ideogram.is_refusal_card(image_path):
        return {"ok": False, "score": 0, "source": "refusal",
                "problems": ["the model returned its safety card instead of a picture"],
                "ops": []}
    if ctx is None or not image_path:
        return {"ok": True, "score": 0, "problems": [], "ops": [], "source": "unavailable"}
    try:
        import llm
        # The measured exposure goes in as evidence too: told the numbers, the
        # critic invents fewer silhouettes; the ones it still invents are
        # dropped by _drop_unfounded.
        _evidence = list(evidence or [])
        _exp = _exposure.evidence(image_path)
        if _exp:
            _evidence.append(_exp)
        # No <think></think> prefill on vision calls — it breaks them (see
        # docs/vision_pipeline_forensics.md); the JSON instruction carries the format.
        # Same reader model as the transcription: on every live render measured
        # 2026-07-29 the chat model answered this with no JSON at all, so the
        # critic reported "unavailable" and had no vote on any real picture.
        raw = llm.analyze_image_with_llm(
            _reader(ctx), image_path=image_path,
            user_text="The layout this picture was drawn from:\n" + _regions(layout)
                      + ("\n\nAlready established about this picture:\n"
                         + "\n".join(f"- {e}" for e in _evidence) if _evidence else "")
                      + "\n\nCheck the picture against it.",
            # Generous on purpose. This model deliberates before answering and the
            # length varies enormously (849 to 2997 tokens on the SAME picture);
            # when it overruns, the reply is cut off before the JSON and the critic
            # reports "unavailable" — an invisibly disabled check. Measured over
            # six runs of one render: 1600 answered 1, 3000 answered 4, and
            # analyze_image_with_llm retries an empty reply once on top of that.
            # Nothing is spent when it stops early — this is a cap, not a target.
            system_prompt=_CRITIQUE_PROMPT, temperature=0.1, max_tokens=3000)
        # NOT ideogram._extract_json: that one only accepts LAYOUT-shaped objects,
        # so a perfectly good verdict — measured live, the model ends its reply
        # with {"ok": true, "score": 10, "problems": [], "ops": []} — scored zero
        # layout keys and was discarded. The critic has therefore reported
        # "unavailable" on every real picture it has ever been shown.
        data = safe_json_from_llm(raw or "",
                                  required_keys=("ok", "score", "problems", "ops"))
        if not isinstance(data, dict):
            logger.warning("critique: no JSON from the vision model")
            return {"ok": True, "score": 0, "problems": [], "ops": [],
                    "source": "unavailable"}
        problems = [str(p) for p in (data.get("problems") or []) if str(p).strip()]
        if _exposure.judge_is_blind(problems, image_path):
            # "The image is almost entirely black" about a twilight café that
            # measures 100+/255 (live, 2026-09-12, four verdicts in a row).
            # A critic that did not see the picture gets no vote on it -- not
            # on the darkness, and not on the "missing" things it lists as a
            # consequence of the darkness it imagined.
            logger.warning("critique: the critic called a normally exposed picture "
                           "black overall -- verdict discarded: %s", problems[:3])
            return {"ok": True, "score": 0, "problems": [], "ops": [],
                    "source": "blind"}
        _before = len(problems)
        problems = _drop_unfounded(image_path, problems)
        # Taste is not a problem. The prompt says so and the model ignores it:
        # "the background of the sign is messy" scored a correct café 2/10.
        # A complaint that names nothing on the checklist (missing, wrong,
        # misspelled, doubled, merged, misplaced, incoherent) is dropped.
        _kept = [p for p in problems if _LAYOUT_CONCERN_RE.search(p)]
        if len(_kept) != len(problems):
            logger.info("critique: dropped %d taste complaint(s): %s",
                        len(problems) - len(_kept),
                        [p for p in problems if p not in _kept][:3])
            problems = _kept
        if len(problems) != _before:
            logger.info("critique: dropped %d darkness complaint(s) the pixels "
                        "contradict", _before - len(problems))
        # A model writes "false" as often as false, and bool("false") is True — that
        # would turn every condemnation into an approval.
        ok = data.get("ok", None)
        if isinstance(ok, str):
            ok = ok.strip().lower() not in ("false", "no", "0", "none", "")
        ok = (not problems) if ok is None else bool(ok)
        if not ok and not problems and _before:
            # Every complaint it made was one the pixels contradict.
            ok = True
        score = _num(data.get("score"), 0.0)
        return {"ok": ok, "score": max(0, min(int(score), 10)), "problems": problems,
                "ops": [o for o in (data.get("ops") or []) if isinstance(o, dict)],
                "source": "vision"}
    except Exception as exc:
        # Same contract as an unparseable answer: the critic did not see anything,
        # so it gets no vote. `detail` keeps the reason for the log/UI.
        logger.exception("critique failed")
        return {"ok": True, "score": 0, "problems": [], "ops": [],
                "source": "unavailable", "detail": str(exc)}


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def _attempt_score(verdict) -> tuple:
    """How good an attempt was, smaller is better. Sorted on, so it is a tuple.

    The pieces, in order of how much they matter:
      * the critic's own verdict -- an `ok` beats anything not ok;
      * how many strings came out garbled (a sign reading AVCHKE is the single
        most visible way a picture is wrong);
      * how many problems the critic listed.
    """
    verdict = verdict or {}
    text = verdict.get("text") or {}
    return (0 if verdict.get("ok") else 1,
            len(text.get("failures") or []),
            len(verdict.get("problems") or []))


def _best_attempt(history):
    """The best attempt made so far, or None."""
    usable = [h for h in (history or []) if h.get("image")]
    if not usable:
        return None
    return min(usable, key=lambda h: _attempt_score(h.get("verdict")))


def run(ctx, request: str = "", *, layout: Optional[dict] = None,
        instruction: str = "", width: int = 1024, height: int = 1024,
        rounds: int = 2, auto_fix: bool = True, seed: Optional[int] = None,
        steps: Optional[int] = None, cfg: Optional[float] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Plan or edit a layout, draw it, judge the result, fix the boxes, draw again.

    `request`     — a new scene to plan from scratch.
    `layout`      — an existing layout to keep working on (takes precedence).
    `instruction` — a change to apply to that layout before drawing.
    `rounds`      — how many REPAIR passes are allowed after the first draw.

    Returns {"layout", "image", "history", "problems", "stopped"}. `history` holds
    every attempt: its layout, its image and what was wrong with it, so the caller
    can show the reasoning rather than just the last picture.
    """
    def emit(kind, **data):
        if on_event is not None:
            try:
                on_event(kind, data)
            except Exception:
                logger.exception("draw-agent event handler failed")

    cancelled = lambda: bool(ctx is not None and getattr(ctx, "is_cancelled", bool)())

    if layout is None:
        emit("stage", text="planning the layout")
        layout = ideogram.plan_layout(ctx, request)
        # Planned here, so checked here (image_generate checks its own plan
        # before handing it in): the boxes as a sketch, in front of the
        # vision model, next to the request -- see draw_preview.
        emit("stage", text="checking the plan as a sketch")
        import draw_preview
        layout, _pre = draw_preview.preflight(ctx, layout, request, width=width,
                                              height=height, emit=emit)
    layout = ideogram.normalize_layout(layout, request)
    if instruction:
        emit("stage", text=f"editing: {instruction}")
        layout, notes = edit_layout(ctx, layout, instruction)
        emit("edited", layout=layout, notes=notes)

    if auto_fix:
        # Lettering first: a text box gets the shape its STRING needs, and the
        # general geometry repair then works on boxes that are already sane.
        tprobs = text_report(layout)
        if tprobs:
            layout, notes = auto_fix_text(layout)
            if notes:
                emit("fixed", layout=layout, notes=notes,
                     problems=[p["why"] for p in tprobs])
        problems = geometry_report(layout)
        if problems:
            layout, notes = auto_fix_geometry(layout)
            emit("fixed", layout=layout, notes=notes,
                 problems=[p["why"] for p in problems])

    # ONE seed for the whole loop. This is what makes rearranging boxes meaningful:
    # with a fixed seed, moving/adding/removing a box changes only the part of the
    # picture that box governs, so the agent can actually see the arrangement->output
    # relationship. A new seed every attempt reshuffles the entire scene and an "edit"
    # becomes a brand-new, unrelated picture (the "it badly changes the picture" bug).
    if not seed or seed <= 0:
        seed = random.randint(1, 999_999_999)
    emit("stage", text=f"seed {seed} (held fixed so only the boxes change the picture)")

    history, image, last_problems = [], None, []
    _garbled_before: set = set()          # normalized texts already box-repaired once
    for attempt in range(max(1, 1 + max(0, rounds))):
        if cancelled():
            return {"layout": layout, "image": image, "history": history,
                    "seed": seed, "problems": last_problems, "stopped": "cancelled"}
        emit("stage", text=f"drawing (attempt {attempt + 1})", layout=layout)
        try:
            # The seed is held fixed across attempts ON PURPOSE (see above), so
            # that moving a box changes only that box. A COLLAGE is the one case
            # where that reasoning does not apply: the picture is not a worse
            # arrangement of the scene, it is not the scene at all, and there is
            # nothing in it worth keeping stable. Re-roll the seed and draw the
            # same layout again. Bounded, because a layout that collages on
            # every seed needs the caller told, not the card burned.
            image = render_without_collage(
                ctx, layout, width=width, height=height, seed=seed,
                steps=steps, cfg=cfg, on_progress=on_progress, emit=emit)
        except ideogram.ContentRefused as exc:
            emit("refused", text=str(exc))
            return {"layout": layout, "image": None, "history": history,
                    "seed": seed, "problems": [str(exc)], "stopped": "refused"}
        except Exception as exc:
            # ComfyUI drops the connection, a node is missing, the job vanishes —
            # the loop reports it like any other failed render instead of taking the
            # caller's turn down with it.
            logger.exception("draw-agent render failed")
            emit("failed", text=str(exc))
            return {"layout": layout, "image": None, "history": history,
                    "seed": seed, "problems": [f"the renderer failed: {exc}"], "stopped": "failed"}
        if not image:
            emit("failed", text="the renderer returned no image")
            return {"layout": layout, "image": None, "history": history,
                    "seed": seed, "problems": ["the renderer returned no image"],
                    "stopped": "failed"}
        emit("drawn", image=image, layout=layout)

        emit("stage", text="looking at the result")
        # Read the lettering back BEFORE the critic looks, and hand it the finding.
        # This is a separate, deterministic judgement: the critic is asked whether
        # the picture matches the layout and will happily call a sign "present and
        # correctly placed" while it reads AVCHKE. Comparing the transcription to
        # the string that was asked for is the only check that catches that, so it
        # OVERRIDES the critic's verdict — and, told about it, the critic can also
        # propose moving or enlarging the sign instead of leaving the repair to
        # geometry alone.
        text_check = verify_text(ctx, image, layout)
        tproblems = text_problems(text_check.get("checks"), text_check.get("stray"))
        verdict = critique(ctx, image, layout, evidence=tproblems)
        if tproblems:
            verdict = dict(verdict)
            verdict["ok"] = False
            verdict["problems"] = list(verdict.get("problems") or []) + [
                p for p in tproblems if p not in (verdict.get("problems") or [])]
        verdict["text"] = text_check

        history.append({"layout": copy.deepcopy(layout), "image": image,
                        "verdict": verdict})
        last_problems = verdict.get("problems") or []
        emit("judged", **verdict)
        # The verdict decides whether the user waits another minute; it has
        # to be in the log, or a rejected good picture is invisible afterwards.
        logger.info("draw-agent attempt %d: %s score=%s source=%s problems=%s text_failures=%s stray=%s",
                    attempt + 1, "OK" if verdict.get("ok") else "REJECTED",
                    verdict.get("score"), verdict.get("source"),
                    [str(p)[:120] for p in last_problems][:6],
                    text_check.get("failures"), (text_check.get("stray") or [])[:4])
        if verdict.get("ok"):
            return {"layout": layout, "image": image, "history": history,
                    "problems": last_problems, "text": text_check,
                    "stopped": "ok"}
        if attempt >= rounds:
            # Same rule as the bottom of the function: the last attempt is not
            # automatically the best one, so hand back whichever was.
            best = _best_attempt(history) or history[-1]
            return {"layout": best["layout"], "image": best["image"],
                    "history": history,
                    "problems": (best.get("verdict") or {}).get("problems") or [],
                    "text": (best.get("verdict") or {}).get("text") or text_check,
                    "stopped": ("out of rounds" if best["image"] == image else
                                "out of rounds — kept the best of %d attempts"
                                % len(history))}

        # Rearrange: garbled lettering first (its repair is geometric and known),
        # then the critic's own operations, then the deterministic geometry pass.
        drawn = copy.deepcopy(layout)
        garbled = {_norm_text((layout.get("elements") or [])[i].get("text"))
                   for i in (text_check.get("failures") or [])
                   if 0 <= i < len(layout.get("elements") or [])}
        garbled.discard("")
        if text_check.get("failures"):
            layout, notes = repair_text(layout, text_check["failures"], attempt)
            if garbled & _garbled_before:
                # Live 2026-09-18 (journey 25): "только до пятницы" came back
                # "только до пятниИЦЬ" on TWO attempts in a row, byte-identical,
                # because the seed never changed -- repair_text only enlarges the
                # box and restates the spelling, both geometric, so a fixed seed
                # reproduces the same glyph garble. Same remedy as stray text
                # surviving a wording fix: reroll.
                seed = random.randint(1, 999_999_999)
                notes.append(f"garbled lettering survived a box repair — new seed {seed}")
            _garbled_before |= garbled
            emit("retext", layout=layout, notes=notes, checks=text_check["checks"])
        if text_check.get("stray"):
            already = _draw_text.stray_suppressed(layout)
            layout, notes = _draw_text.suppress_stray_text(layout, text_check["stray"])
            if already:
                # The words were already in the caption and the decoration
                # came back: it is the seed's prior, not the wording. Re-roll.
                seed = random.randint(1, 999_999_999)
                notes.append(f"stray lettering survived the wording — new seed {seed}")
            emit("retext", layout=layout, notes=notes, checks=text_check["checks"])
        applied = []
        if verdict.get("ops"):
            layout, applied = apply_ops(layout, verdict["ops"])
            emit("rearranged", layout=layout, notes=applied)
        # The critic judges through a deliberately SHORT prompt (a long one makes
        # it deliberate until the token budget is gone — see `_CRITIQUE_PROMPT`),
        # so it often states a problem without an operation that fixes it, or
        # names a box in words that resolve to nothing. Its complaints then go to
        # the editor, which has the whole vocabulary and only has to turn words
        # into ops. This is what lets a complaint become an actual rearrangement
        # rather than the same picture drawn again on a new seed.
        landed = [n for n in applied
                  if "no element matching" not in n and "unknown op" not in n]
        unmet = [p for p in (verdict.get("problems") or []) if p not in tproblems]
        # "X is orange instead of bright pink" is a REPORT, not an instruction.
        # Handed to the editor as one, it rewrote the element to "an orange-red
        # elephant", the next render was orange, and the critic -- judging
        # against the rewritten layout -- scored it 10 (live, 2026-09-12; the
        # same again with brown). A wrong-attribute complaint keeps the layout's
        # wording, underlines it, and re-rolls the seed instead.
        layout, notes, unmet = insist_on_mismatch(layout, unmet)
        if notes:
            layout, _restyled = unphoto_on_repeat_mismatch(layout, notes, request)
            if _restyled:
                notes.append(_restyled)
            seed = random.randint(1, 999_999_999)
            notes.append(f"new seed {seed}")
            emit("rearranged", layout=layout, notes=notes)
        if unmet and not landed:
            layout, notes = edit_layout(
                ctx, layout,
                "The picture came back with these problems: " + "; ".join(unmet)
                + ". Change the layout so the NEXT render fixes them. Keep what every "
                  "element is supposed to be -- the descriptions already state the "
                  "intended look; never rewrite an element to match the wrong result.")
            if notes and notes != ["the model proposed no change"]:
                emit("rearranged", layout=layout, notes=notes)
        if verdict.get("ops") or unmet:
            # The critic may have reworded the surface a failed string sits on —
            # keep its wording, but put the letter-by-letter spelling back, or the
            # rewrite silently undoes the one repair that is known to work.
            # Matched by STRING, not by index: an op that deleted an element
            # renumbers everything after it, and re-spelling by stale index would
            # rewrite the description of an innocent box.
            for el in layout.get("elements") or []:
                if _norm_text(el.get("text")) in garbled:
                    el["desc"] = spell_out(el)
        fixed, notes = auto_fix_geometry(layout)
        if notes:
            layout = fixed
            emit("fixed", layout=layout, notes=notes, problems=[])
        # The critic's ops and the spreading pass can both reshape a box that was
        # just enlarged FOR its string, which would hand the next attempt the same
        # unreadable geometry. Re-deriving the width from the (now larger) height
        # is idempotent, so the repair survives without fighting the other fixers.
        layout, notes = auto_fix_text(layout)
        if notes:
            emit("fixed", layout=layout, notes=notes, problems=[])
        if layout == drawn:
            # The critic condemned the picture but nothing it asked for could be
            # applied (targets it named are not in the layout) and the geometry is
            # already sound. The next draw is a reroll on a new seed, NOT a repair —
            # say so, so the caller can show why the picture barely changed.
            emit("stalled", text="nothing in the layout could be changed — redrawing "
                                 "the same arrangement with a new seed",
                 problems=last_problems)
    # Out of rounds without the critic ever being satisfied. Handing back the
    # LAST attempt is what made the loop look like it was not looking at its own
    # work: five redraws, the second already good, and the fifth -- worse --
    # delivered. Each round re-rolls a scene the critic complained about, so
    # there is no reason the newest is the best. Give back the best one.
    best = _best_attempt(history)
    if best is not None and best.get("image") != image:
        return {"layout": best["layout"], "image": best["image"],
                "history": history,
                "problems": (best.get("verdict") or {}).get("problems") or [],
                "stopped": "out of rounds — kept the best of %d attempts" % len(history),
                "text": (best.get("verdict") or {}).get("text")}
    return {"layout": layout, "image": image, "history": history,
            "problems": last_problems, "stopped": "out of rounds",
            "text": (history[-1]["verdict"].get("text") if history else None)}
