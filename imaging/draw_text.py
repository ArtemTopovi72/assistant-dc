"""The lettering half of the drawing agent: reading, checking and repairing text.

Split out of draw_agent.py. Everything here is about the *strings inside the
picture* -- sizing a text box from the string it carries, cropping the render
and asking a vision model to transcribe it, comparing what came back against
what was asked for, and rewriting the boxes when it does not match.

The direction of dependency is one way: this module imports nothing from
draw_agent, and draw_agent re-exports every name below so that `draw_agent.<name>`
keeps resolving for call sites and suites alike.

`_els`, `_num` and `_clamp_box` came along because the repair functions need
them. They live here rather than in both places -- a second copy of `_num` would
be exactly the split-brain this refactor exists to avoid. draw_agent imports
them back.

One seam note. draw_agent.run() calls text_report / auto_fix_text / verify_text /
repair_text / text_problems / spell_out through draw_agent's own globals, so
patching `draw_agent.<name>` still redirects the loop, as before. Calls *between*
the functions in this file resolve in this module -- patch `draw_text.<name>` to
reach those.
"""
import copy
import difflib
import logging
import os
import re
from typing import Optional

import ideogram
from text_layout import (MIN_TEXT_H, MIN_LINE_H, CHAR_ASPECT, MAX_TEXT_CHARS,
                         SHAPE_TOLERANCE, TEXT_MATCH_OK,
                         text_geometry, text_geometry_lines, fit_text_in, lines_of,
                         _norm_text)
from utils import safe_json_from_llm

logger = logging.getLogger("assistant.draw_agent")


# --------------------------------------------------------------------------- #
# Layout operations
# --------------------------------------------------------------------------- #
def _els(layout):
    return layout.setdefault("elements", [])


def _num(v, default=None):
    """A coordinate the model wrote, or `default`.

    Every number in an op comes out of a language model, so "x": "left", "w": null
    and "h": {"value": 0.3} all arrive eventually. `float()` on those raises, and a
    raise here would throw away the whole edit — see `apply_ops`.
    """
    if isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f or f in (float("inf"), float("-inf")) else f


def _clamp_box(el):
    el["w"] = max(0.02, min(_num(el.get("w"), 0.3), 1.0))
    el["h"] = max(0.02, min(_num(el.get("h"), 0.3), 1.0))
    el["x"] = max(0.0, min(_num(el.get("x"), 0.0), 1.0 - el["w"]))
    el["y"] = max(0.0, min(_num(el.get("y"), 0.0), 1.0 - el["h"]))
    return el


def _text_els(layout: dict) -> list:
    """(index, element) for every element that carries lettering."""
    return [(i, el) for i, el in enumerate(layout.get("elements") or [])
            if str(el.get("text") or "").strip()]


def text_report(layout: dict) -> list:
    """Lettering boxes that cannot render, before anything is drawn.

    Same vocabulary as `geometry_report` ({"kind", "why", "targets"}) so the two
    can be shown together.
    """
    layout = ideogram.normalize_layout(layout)
    out = []
    for i, el in _text_els(layout):
        text = str(el["text"]).strip()
        # Per line: a wrapped label is judged by its longest line, since that
        # is what has to fit, and its box is as many line-heights tall.
        _lines = lines_of(text)
        n = max(len(ln) for ln in _lines)
        if n > MAX_TEXT_CHARS:
            out.append({"kind": "text_too_long", "targets": [i],
                        "why": f'“{text[:30]}” is {n} characters — past about '
                               f"{MAX_TEXT_CHARS} the model stops spelling and paints "
                               f"letter-shaped strokes"})
        _floor = MIN_LINE_H if len(_lines) > 1 else MIN_TEXT_H
        if el["h"] / len(_lines) < _floor - 1e-9:
            out.append({"kind": "text_too_small", "targets": [i],
                        "why": f'the box for “{text[:20]}” is {el["h"] * 100:.0f}% of the '
                               f"frame high — lettering needs at least "
                               f"{MIN_TEXT_H * 100:.0f}% per line to form readable glyphs"})
        want = CHAR_ASPECT * n / len(_lines)
        have = el["w"] / max(el["h"], 1e-6)
        if have < want / SHAPE_TOLERANCE or have > want * SHAPE_TOLERANCE:
            out.append({"kind": "text_wrong_shape", "targets": [i],
                        "why": f'the box for “{text[:20]}” is {have:.1f}:1 but {n} '
                               f"characters want about {want:.1f}:1 — the wrong shape "
                               f"is what squeezes letters into a smear"})
    return out


def auto_fix_text(layout: dict) -> tuple:
    """Give every lettering box the shape its string needs. Returns (layout, notes).

    Idempotent: the box is SET to the computed geometry, not nudged towards it, so
    a second call is a no-op and the loop cannot oscillate.
    """
    layout = ideogram.normalize_layout(copy.deepcopy(layout))
    els = _els(layout)
    notes = []
    for i, el in _text_els(layout):
        text = str(el["text"]).strip()
        cx, cy = el["x"] + el["w"] / 2, el["y"] + el["h"] / 2
        host = _host_of(layout, i)
        w, h = text_geometry_lines(text, el["h"] / max(1, len(lines_of(text))))
        if host is not None and w > host["w"] * 0.92:
            # Lettering that sits ON something -- a label on a bottle, a sign
            # on a shop -- must stay on it. The single-line rule stretched
            # 'Для гостей театра на Пролетарской' (34 chars) to 94% of the
            # frame and the label floated off its bottle as a banner across
            # the picture (live, 2026-09-12). Wrap it as a printer would; and
            # if even wrapped it needs more room than the host has at the
            # glyph floor, the SHOT moves closer -- the host grows around its
            # centre -- rather than the letters shrinking into scribble.
            wrapped, w, h = fit_text_in(text, host["w"] * 0.92, el["h"])
            if wrapped != el["text"]:
                el["text"] = wrapped
                text = wrapped
            need_w, need_h = w / 0.92, h / 0.6
            if need_w > host["w"] or need_h > host["h"]:
                # The host must stay WHOLE: a bottle that grows to the frame
                # edge comes back with its cap cut off (live, 2026-09-12),
                # so it is capped at IN_FRAME of the frame and told so.
                f = min(max(need_w / host["w"], need_h / host["h"]),
                        IN_FRAME / host["w"], IN_FRAME / host["h"])
                # Already as tall as the frame allows? Then the box widens on
                # its own: a box is a suggestion, and a bottle painted from a
                # wider box is still a bottle -- while a label that does not
                # fit is not a label.
                fw = f if f > 1.02 else min(need_w / host["w"], 0.96 / host["w"])
                fh = f if f > 1.02 else 1.0
                if fw > 1.02 or fh > 1.02:
                    f = max(fw, fh)
                    hx, hy = host["x"] + host["w"] / 2, host["y"] + host["h"] / 2
                    host["w"], host["h"] = host["w"] * fw, host["h"] * fh
                    host["x"] = min(max(hx - host["w"] / 2, (1 - IN_FRAME) / 2),
                                    1 - (1 - IN_FRAME) / 2 - host["w"])
                    # More room ABOVE than below: the model paints a bottle's
                    # neck and cap past the top of its box (the base stays
                    # put), and at an even margin the cap was still cut.
                    host["y"] = min(max(hy - host["h"] / 2, TOP_MARGIN),
                                    1 - (1 - IN_FRAME) / 2 - host["h"])
                    _clamp_box(host)
                    if WHOLE_NOTE not in str(host.get("desc") or ""):
                        host["desc"] = (str(host.get("desc") or "").rstrip(". ")
                                        + ", " + WHOLE_NOTE)
                    notes.append(f'moved in on “{str(host.get("desc") or "")[:30]}” '
                                 f"({f:.2f}x) so its lettering can be printed at a "
                                 "readable size")
            w = min(w, host["w"] * 0.92)
            h = min(h, host["h"] * 0.6)
        if host is not None:
            _give_headroom(layout, host, notes)
        if abs(w - el["w"]) < 0.005 and abs(h - el["h"]) < 0.005:
            continue
        el["w"], el["h"] = w, h
        el["x"], el["y"] = cx - w / 2, cy - h / 2
        if host is not None:
            # Inside the host, not merely inside the frame.
            el["x"] = max(host["x"], min(el["x"], host["x"] + host["w"] - w))
            el["y"] = max(host["y"], min(el["y"], host["y"] + host["h"] - h))
        _clamp_box(el)
        n_lines = len(lines_of(text))
        notes.append(f'sized the box for “{text[:20]}” to {el["w"]:.2f}×{el["h"]:.2f}'
                     + (f" in {n_lines} lines" if n_lines > 1 else "")
                     + f" — {len(text)} characters need that shape to stay readable")
        els[i] = el
    return layout, notes


TOP_MARGIN = 0.16        # what a grown host leaves above itself
IN_FRAME = 0.76          # a host that grows for its lettering stops here, whole -- the model draws it larger than its box, and at 0.88 the cap was still cut
WHOLE_NOTE = "shown whole from top to bottom, entirely inside the frame"
ONLY_NOTE = ("the ONLY lettering anywhere in the picture; no other writing, "
             "letters, numbers, brand names, logos or small print on anything")
STRAY_MIN_CHARS = 3      # shorter readings are ornament, not words


def suppress_stray_text(layout: dict, stray: list) -> tuple:
    """Tell the renderer, in every place it listens, that there is no other text.

    The model dresses a label with brand names, vintages and small print of
    its own -- live, the champagne came back with "КАДВТЕЛИРСТ ГОВ" under the
    requested line. There is no negative prompt in this caption schema, so
    the constraint is stated on the lettering element, on its host, and in
    the high-level description. Deterministic and idempotent.
    """
    layout = ideogram.normalize_layout(copy.deepcopy(layout))
    notes = []
    wanted = [str(el.get("text") or "").strip() for _, el in _text_els(layout)]
    for i, el in _text_els(layout):
        if ONLY_NOTE not in str(el.get("desc") or ""):
            el["desc"] = str(el.get("desc") or "").rstrip(". ") + " — " + ONLY_NOTE
        host = _host_of(layout, i)
        if host is not None and "plain and unmarked" not in str(host.get("desc") or ""):
            # Name the PARTS: "no other writing" alone left the foil, the neck
            # band and the monogram to the model's champagne prior. The critic
            # also needs it said this way, or it reads "no other lettering" as
            # "no lettering" and asks for the label to be deleted.
            host["desc"] = (str(host.get("desc") or "").rstrip(". ")
                            + "; apart from that one label every part of it -- foil, "
                              "cap, neck, shoulders, back and base -- is plain and "
                              "unmarked: no brand, no monogram, no vintage, no small print")
    hl = str(layout.get("high_level_description") or "").rstrip(". ")
    if wanted and "the one and only lettering" not in hl.lower():
        quoted = " and ".join("“%s”" % " ".join(w.split()) for w in wanted)
        layout["high_level_description"] = (
            hl + ". The label reads " + quoted + " -- that label is wanted and must "
            "be there, and it is the one and only lettering in the picture; every "
            "other surface is blank, with no other text, numbers, logos or monograms")
    if stray:
        notes.append("the picture carried lettering nobody asked for (%s) — every "
                     "surface is now described as blank apart from the requested words"
                     % ", ".join("“%s”" % s for s in stray[:4]))
    return layout, notes


def stray_suppressed(layout: dict) -> bool:
    """Has the 'no other lettering' wording already been written into this layout?

    Read from the description itself rather than a flag: normalize_layout
    rebuilds the dict and drops keys it does not know.
    """
    return "the one and only lettering" in str(layout.get("high_level_description") or "").lower()


HEADROOM_NOTE = "empty space above"


def _give_headroom(layout: dict, host: dict, notes: list) -> None:
    """Own the space above a lettered host, so the host stops growing into it.

    A box is a suggestion to the renderer, and a tall object ignores its top:
    a champagne bottle in a 0.62-high box was painted with its neck running
    off the frame in five renders out of five (2026-09-12). Given the strip
    above it as an element of its own -- out-of-focus room, empty air -- the
    same seed drew the bottle whole, cap and all. Added once, only when
    nothing else already owns that strip, and never when the host starts at
    the top of the frame.
    """
    top = host["y"]
    if top < 0.10:
        return
    els = _els(layout)
    for el in els:
        if el is host or str(el.get("text") or "").strip():
            continue
        if HEADROOM_NOTE in str(el.get("desc") or ""):
            return
        # something already covers the strip above the host
        if el["y"] <= 0.02 and el["h"] >= top * 0.6 and el["w"] >= 0.5:
            return
    room = str(layout.get("background") or "the scene behind it").strip().rstrip(".")
    # Appended, never inserted: the caller is iterating by index.
    els.append({"desc": f"{HEADROOM_NOTE} {str(host.get('desc') or 'it').split(',')[0][:40]}: "
                           f"{room}, out of focus, with nothing in front of it",
                   "text": "", "x": 0.0, "y": 0.0, "w": 1.0, "h": round(top * 0.8, 3)})
    notes.append("gave the space above “%s” its own element, so it is drawn whole "
                 "instead of running off the top" % str(host.get("desc") or "")[:30])


def _host_of(layout: dict, idx: int):
    """The non-lettering element the text box sits on, or None.

    The element whose box contains the text box's centre, smallest first, so
    'label on the bottle' hosts on the bottle and not on the table the bottle
    stands on. A host that is the whole frame is no host: lettering on the
    background is a banner, and a banner may be as wide as it likes.
    """
    els = layout.get("elements") or []
    t = els[idx]
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    best = None
    for j, el in enumerate(els):
        if j == idx or str(el.get("text") or "").strip():
            continue
        if el["w"] * el["h"] >= 0.85:
            continue
        if el["x"] <= cx <= el["x"] + el["w"] and el["y"] <= cy <= el["y"] + el["h"]:
            if best is None or el["w"] * el["h"] < best["w"] * best["h"]:
                best = el
    return best


def spell_out(el: dict) -> str:
    """The element's description, restated so the renderer cannot mis-hear the word.

    Repeating the string letter by letter inside the description is the one thing
    that reliably fixes a garbled render: the caption then carries the spelling
    twice, in two different forms, instead of once.
    """
    text = str(el.get("text") or "").strip()
    desc = str(el.get("desc") or "").strip()
    if not text:
        return desc
    # Idempotent: strip every earlier restatement, wherever the surface text
    # has grown around it since -- a repair round used to nest a new "(the
    # word ...)" inside the old one every pass, and the caption ballooned.
    surface = re.sub(r"\s*\((?:the (?:\d+-letter )?(?:word|lettering)|the (?:label|sign) (?:reads|says))[^()]*\)",
                     "", desc, flags=re.S).strip(" ,;—-")
    if not surface:
        surface = "a flat surface carrying lettering"
    lines = lines_of(text)
    if len(lines) == 1 and " " not in text:
        # One word: letter by letter is the one thing that reliably fixes a
        # garbled render (POLICE -> AVCHKE).
        # The spelling was "В-Е-Ч-Е-Р" and the sign came out "ВЕ-Ч-ЕР"
        # (live 2026-09-28): the hyphens are painted, so the letters are
        # counted and named instead, with separators ruled out explicitly.
        return (f'{surface} (the {len(text)}-letter word “{text}” as one solid word, in '
                f"clean bold block capitals, every letter correctly formed, no hyphens, "
                f"dots or gaps between the letters)")
    # Several words or lines: hyphenating the letters made the model PAINT the
    # hyphens ("ПРОЛЕТАР-СК-сой", live 2026-09-12). State the lines instead,
    # each as its own quoted phrase, in order.
    quoted = " / ".join("“%s”" % ln for ln in lines)
    return (f"{surface} (the lettering reads, in {len(lines)} line"
            f"{'s' if len(lines) > 1 else ''}, exactly: {quoted} — every word "
            f"spelled exactly like that, nothing added, every letter correctly formed)")


def repair_text(layout: dict, failures: list, escalation: int = 0) -> tuple:
    """Fix the lettering elements a read-back found wrong. Returns (layout, notes).

    `failures` are indices into the layout's elements. Deliberately deterministic:
    the failure mode is geometric, so asking a language model what to do about it
    just adds a way to get a worse box. Each round gives the string more room and
    states the spelling more explicitly.
    """
    layout = ideogram.normalize_layout(copy.deepcopy(layout))
    els = _els(layout)
    notes = []
    grow = 1.0 + 0.35 * (escalation + 1)
    for i in failures:
        if not (0 <= i < len(els)):
            continue
        el = els[i]
        text = str(el.get("text") or "").strip()
        if not text:
            continue
        n_lines = len(lines_of(text))
        w, h = text_geometry_lines(text, max(el["h"] / n_lines, MIN_TEXT_H) * grow)
        cx, cy = el["x"] + el["w"] / 2, el["y"] + el["h"] / 2
        host = _host_of(layout, i)
        if host is not None:
            # A repaired box still lives on its host; it does not escape it.
            w, h = min(w, host["w"] * 0.92), min(h, host["h"] * 0.6)
        el["w"], el["h"] = w, h
        el["x"], el["y"] = cx - w / 2, cy - h / 2
        if host is not None:
            el["x"] = max(host["x"], min(el["x"], host["x"] + host["w"] - w))
            el["y"] = max(host["y"], min(el["y"], host["y"] + host["h"] - h))
        _clamp_box(el)
        el["desc"] = spell_out(el)
        notes.append(f'“{text}” came out garbled — box enlarged to '
                     f'{el["w"]:.2f}×{el["h"]:.2f} and the spelling stated letter '
                     f"by letter in the caption")
    return layout, notes


# A vision model asked "what does the sign say" answers with the word it EXPECTS,
# not the one that is printed — which is precisely the failure being hunted. The
# prompt has to fight that: transcribe, do not read; nonsense is the answer, not a
# reason to guess.
_TEXT_READ_PROMPT = """You are transcribing the lettering that appears inside a picture.

Answer with ONLY a JSON object, no prose, no fence:
{"strings": ["each separate piece of lettering, exactly as printed"]}

Rules — these matter more than anything else:
- Copy the letters that are ACTUALLY painted, character for character.
- Generated pictures often contain misspelled or meaningless lettering. Write the \
characters you can see even when they spell nothing at all. Never correct a \
misspelling, never complete a partial word, and never write the word you think \
was intended.
- Copy ONLY from the picture. Nothing in these instructions is text to transcribe.
- If the shapes look like letters but you cannot make out any characters, write \
"???" for that piece.
- List every piece of lettering you can see, largest first. If there is none, \
answer {"strings": []}.
- Do not think out loud, do not explain, do not number the pieces. Your entire \
reply is the JSON object and nothing else. Repeated lettering is listed once."""


class _ReaderCtx:
    """`ctx` with a different model_name, for the one call that needs it.

    A copy would strip the rate limiter of its timestamp, and mutating the real
    ctx would race the other users the bot is serving — so writes are forwarded to
    the real context and only the model name is shadowed.
    """

    def __init__(self, ctx, model_name):
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "model_name", model_name)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ctx"), name)

    def __setattr__(self, name, value):
        if name == "model_name":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_ctx"), name, value)


def _reader(ctx):
    """The context the transcription runs under — see config.TEXT_READ_MODEL."""
    import config
    name = str(getattr(config, "TEXT_READ_MODEL", "") or "").strip()
    if not name or name == getattr(ctx, "model_name", ""):
        return ctx
    return _ReaderCtx(ctx, name)


def _crop_for_read(image_path: str, box: dict) -> Optional[str]:
    """The element's region of the render, written out for the vision call.

    Asking about the WHOLE picture does not work: on a real 1024x1024 street
    scene the reader enumerated "POLICE (partially visible)" forty times and ran
    out of tokens before emitting any JSON — which read as "could not look" and
    silently disabled the check on precisely the pictures it exists for. Given the
    element's own box it answers in one line. Small crops are enlarged, because
    the string being checked is often only a few dozen pixels tall.
    """
    try:
        import tempfile
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        W, H = im.size
        pad_x, pad_y = box["w"] * 0.3, box["h"] * 0.6
        x0 = max(0, int((box["x"] - pad_x) * W)); y0 = max(0, int((box["y"] - pad_y) * H))
        x1 = min(W, int((box["x"] + box["w"] + pad_x) * W))
        y1 = min(H, int((box["y"] + box["h"] + pad_y) * H))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        crop = im.crop((x0, y0, x1, y1))
        if max(crop.size) < 512:
            f = 512.0 / max(crop.size)
            crop = crop.resize((int(crop.width * f), int(crop.height * f)),
                               Image.LANCZOS)
        fh = tempfile.NamedTemporaryFile(prefix="textread_", suffix=".png", delete=False)
        fh.close()
        try:
            crop.save(fh.name)
        except Exception:
            # delete=False: a failed save leaves the empty scratch PNG behind,
            # and the caller only unlinks a region it was GIVEN. We return None.
            try: os.unlink(fh.name)
            except OSError: pass
            raise
        return fh.name
    except Exception:
        logger.warning("could not crop the lettering region", exc_info=True)
        return None


def read_text(ctx, image_path: str) -> Optional[list]:
    """Every string the vision model can actually read in the picture, or None.

    None means "could not look" (no model, no image, unparseable answer) — which
    must never be confused with "the picture has no text", or a vision outage would
    condemn every render that carries lettering.

    Runs on config.TEXT_READ_MODEL, not the active chat model: measured
    2026-07-29 on the same eight test frames, the 9B chat model read a clean
    84px "STOP" as "HIT" and scored garbled renders HIGHER than correct ones —
    using it here would redraw good pictures at random. gemma-4-12b transcribed
    all eight exactly (correct 1.00, garbled at most 0.50).
    """
    if ctx is None or not image_path:
        return None
    try:
        import llm
        ctx = _reader(ctx)
        # No <think></think> prefill on vision calls — it breaks them (see
        # docs/vision_pipeline_forensics.md).
        raw = llm.analyze_image_with_llm(
            ctx, image_path=image_path,
            user_text="Transcribe the lettering in this picture.",
            # 300 was not enough: on a busy render this model narrates its
            # reasoning first and the reply was CUT OFF before the JSON, which
            # read as "could not look" and disabled the check on exactly the
            # pictures it exists for.
            system_prompt=_TEXT_READ_PROMPT, temperature=0.0, max_tokens=800)
        # NOT ideogram._extract_json — that one only accepts LAYOUT-shaped objects
        # and throws a perfectly good {"strings": [...]} away as a fragment, which
        # reads as "could not look" and quietly disables the whole check.
        data = safe_json_from_llm(raw or "", required_keys=("strings",))
        if isinstance(data, list):
            data = {"strings": data}
        if not isinstance(data, dict) or "strings" not in data:
            logger.warning("read_text: no transcription JSON from the vision model")
            return None
        strings = data.get("strings")
        if isinstance(strings, str):
            strings = [strings]
        if not isinstance(strings, (list, tuple)):
            return None
        return [str(s) for s in strings if str(s).strip()]
    except Exception:
        logger.exception("reading the lettering failed")
        return None


# Glyph pairs a reader routinely confuses. A difference INSIDE this set is the
# reader's, not the picture's; anything else at the same position is a letter
# the renderer actually painted wrong.
_CONFUSABLE = {frozenset(p) for p in (
    "о0", "o0", "оo", "аa", "еe", "сc", "рp", "хx", "уy", "кk", "нh", "вb", "тt",
    "мm", "ий", "ьъ", "ее", "eё", "её", "il", "l1", "1i", "зэ", "3з", "0о", "ие",
    "шщ", "цч", "нп", "дл", "гт", "лх", "ьы", "ьб", "cg", "ce", "oq", "dq",
    "gq", "vy", "uv", "rn", "ft", "ае", "иu",
)}


def _letter_swapped(na: str, nb: str) -> bool:
    """Same length, same words, one or two letters that are not reader
    confusables -- "пятницы" painted as "пятниця". The renderer does this
    (it invents a Cyrillic letter that fits the shape), and at ratio 0.93 the
    old check called it a match; the poster went out misspelled (live,
    2026-09-12, journey 25)."""
    if len(na) != len(nb) or na == nb or len(na) < 4:
        return False
    diff = [(x, y) for x, y in zip(na, nb) if x != y]
    if not diff or len(diff) > 2:
        return False
    return any(frozenset((x, y)) not in _CONFUSABLE for x, y in diff)


def _similarity(a: str, b: str) -> float:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return 0.0
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    if _letter_swapped(na, nb):
        # A wrong letter is a failed label whatever the ratio says; keep the
        # score under the bar so the retry loop repairs it.
        r = min(r, TEXT_MATCH_OK - 0.01)
    return r


import ocr_reader as _ocr  # noqa: E402


def _match_lines(want: str, read: list) -> tuple:
    """(best reading, similarity) for a string that may be several lines.

    The reader lists each painted line as its own string, so a three-line
    label comes back as three candidates and none of them alone resembles the
    whole -- 'Пролетарской' scored 0.59 against a label that was printed
    perfectly (live, 2026-09-12). Each line is matched on its own and the
    score is the weakest line; one candidate may also carry the whole text.
    """
    lines = lines_of(want)
    if len(lines) <= 1:
        best, best_score = "", 0.0
        # A reader may split one line into pieces ("У", "ОЛЬГИ"); the pieces
        # joined are one more candidate.
        cands = list(read) + ([" ".join(read)] if len(read) > 1 else [])
        for cand in cands:
            score = _similarity(want, cand)
            if score > best_score:
                best, best_score = cand, score
        return best, best_score
    # the whole text in one candidate, or all of it in the joined reading
    joined = " ".join(read)
    if _norm_text(want) and _norm_text(want) in _norm_text(joined):
        return joined, 1.0
    picked, worst = [], 1.0
    for ln in lines:
        b, bs = "", 0.0
        for cand in read:
            sc = _similarity(ln, cand)
            if sc > bs:
                b, bs = cand, sc
        picked.append(b)
        worst = min(worst, bs)
    return " / ".join(p for p in picked if p), worst


def verify_text(ctx, image_path: str, layout: dict) -> dict:
    """Did the picture print the words the layout asked for?

    Returns {"ok", "checks", "failures", "source"}. `checks` carries what was asked
    for next to what was read, so the user can be shown the actual evidence
    ("POLICE" -> "AVCHKE") instead of a verdict.

    source="unavailable" means the read-back could not run; ok is True in that case
    on purpose — a check that cannot see gets no vote.
    """
    layout = ideogram.normalize_layout(layout)
    wanted = _text_els(layout)
    if not wanted:
        return {"ok": True, "checks": [], "failures": [], "source": "no_text"}

    pool = None                      # the whole-picture reading, read once if needed
    checks, failures, saw_any = [], [], False
    sources = set()
    for i, el in wanted:
        want = str(el["text"]).strip()
        read, region = None, _crop_for_read(image_path, el)
        best, best_score, src = "", 0.0, ""
        if region:
            try:
                # OCR first: deterministic, and right where the vision model
                # is deterministically wrong ("У ОЛЬГИ" -> "УАЗЬЛ", 4/4). When
                # it confirms the words there is nothing left to ask.
                ocr = _ocr.read(region)
                if ocr is not None:
                    saw_any = True
                    best, best_score = _match_lines(want, ocr)
                    src = "ocr"
                if best_score < TEXT_MATCH_OK:
                    read = read_text(ctx, region)
            finally:
                try:
                    os.unlink(region)
                except OSError:
                    pass
        if best_score < TEXT_MATCH_OK and read is None and not src:
            # no crop, or the reader could not answer on it
            if pool is None:
                pool = read_text(ctx, image_path)
            read = pool
        if read is not None:
            saw_any = True
            b, bs = _match_lines(want, read)
            if bs > best_score or not src:
                best, best_score, src = b, bs, "vision"
        if not src:
            checks.append({"index": i, "expected": want, "read": "",
                           "similarity": None, "ok": True})
            continue
        sources.add(src)
        ok = best_score >= TEXT_MATCH_OK
        if ok and pool is not None and best in pool:
            pool.remove(best)        # a shared reading answers one element only
        checks.append({"index": i, "expected": want, "read": best,
                       "similarity": round(best_score, 2), "ok": ok, "reader": src})
        if not ok:
            failures.append(i)
    if not saw_any:
        return {"ok": True, "checks": [], "failures": [], "source": "unavailable"}
    stray = stray_text(ctx, image_path, layout, wanted)
    return {"ok": not failures and not stray, "checks": checks,
            "failures": failures, "stray": stray,
            "source": "+".join(sorted(sources)) or "vision"}


def stray_text(ctx, image_path: str, layout: dict, wanted: list) -> list:
    """Lettering on a host that nobody asked for.

    Reads the HOST of each lettering element (the bottle, the shop front), not
    the whole picture: a whole-frame read enumerates every poster in a street
    and runs out of tokens. Anything read there that is not a line of the
    requested text is stray. Best effort: a reader that cannot answer reports
    nothing, and a check that cannot see gets no vote.
    """
    wanted_lines = set()
    for _, el in wanted:
        for ln in lines_of(str(el.get("text") or "")):
            wanted_lines.add(_norm_text(ln))
        wanted_lines.add(_norm_text(str(el.get("text") or "")))
    wanted_lines.discard("")
    seen_hosts, stray = set(), []
    for i, el in wanted:
        host = _host_of(layout, i)
        if host is None:
            continue
        key = (round(host["x"], 3), round(host["y"], 3), round(host["w"], 3), round(host["h"], 3))
        if key in seen_hosts:
            continue
        seen_hosts.add(key)
        region = _crop_for_read(image_path, host)
        if not region:
            continue
        try:
            # OCR when it can run: it does not invent lettering, and the
            # vision model does. The vision model is the fallback only.
            read = _ocr.read(region)
            if read is None:
                read = read_text(ctx, region)
        finally:
            try:
                os.unlink(region)
            except OSError:
                pass
        for cand in read or []:
            norm = _norm_text(cand)
            if len(norm) < STRAY_MIN_CHARS or norm == "???":
                continue
            if any(_similarity(norm, w) >= 0.6 or norm in w or w in norm
                   for w in wanted_lines):
                continue
            stray.append(cand.strip())
    return stray


def text_problems(checks: list, stray: list = None) -> list:
    """The failed checks as sentences, for the user and for the history."""
    out = []
    for s in stray or []:
        out.append(f'the picture carries lettering nobody asked for: “{s}”')
    for c in checks or []:
        if c.get("ok"):
            continue
        read = c.get("read") or ""
        out.append(f'the lettering should read “{c["expected"]}” but the picture '
                   + (f'shows “{read}”' if read else "shows no readable lettering there"))
    return out
