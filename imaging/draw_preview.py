"""Check the PLAN before the render: draw the layout's boxes as a schematic,
show that to the vision model next to the request, fix the boxes.

Why a picture and not the JSON. The planner is a text model; it writes boxes
it cannot see, and its mistakes are spatial: the cat it put next to the sofa
instead of on it, the table it forgot, the second dog the prompt never asked
for, the lamp the size of the room. A vision model given the boxes AS A
PICTURE sees those at a glance — for the cost of one small vision call
instead of a 90-second render that the critic then rejects.

What the checker is told, because it matters: the boxes are where things will
be PAINTED, not solid bodies. «A cat on the sofa» is a cat box INSIDE the
sofa box; «a woman holding a cup» is a cup box inside the woman's. Overlap
and nesting are how relations are drawn, and never a problem by themselves.
The checker is asked about three things only: is everything the request names
in the plan, is there anything the request did not ask for, and does the
arrangement contradict a relation the request states (on / under / left of /
far away / tiny / huge). Taste is not on the list.

Entry point: preflight(ctx, layout, request, ...) -> (layout, report).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import ideogram
from utils import safe_json_from_llm

logger = logging.getLogger("assistant.draw_preview")

# The checker's answer is applied through draw_agent.apply_ops, so its ops
# vocabulary is the critic's: add / delete / replace / move / resize.
PREFLIGHT_PROMPT = """You are checking a PLAN for a picture before it is drawn. \
You see a schematic: each numbered rectangle is where one element of the picture will \
be painted, on a frame of the final proportions. The request the picture must satisfy \
is given, and so is the list of elements with their regions.

READ THE SCHEMATIC CORRECTLY:
- A rectangle is the AREA an element will occupy, not a solid body. Rectangles may \
overlap and sit inside each other; that is how relations are drawn. "A cat on the sofa" \
is a cat rectangle inside the upper part of the sofa rectangle. "A woman holding a cup" \
is a cup rectangle inside the woman's. "A lamp behind the desk" is a lamp rectangle \
overlapping the desk's top edge. Overlap is NEVER a problem by itself.
- The background is painted everywhere behind the rectangles; things the request puts \
in the background (sky, a street, a forest) need no rectangle.

JUDGE ONLY THESE THREE THINGS:
1. MISSING: a thing, person, animal or piece of lettering the request names that has no \
rectangle (unless it is background scenery).
2. EXTRA: a rectangle for something the request did not ask for and does not imply.
3. WRONG ARRANGEMENT: the rectangles contradict a relation the request states — "on the \
table" but the rectangle floats far above the table; "to the left of" but it is on the \
right; "in the distance" but it fills the frame; a cup as large as the person; two \
things the request wants together placed at opposite edges; a thing that should stand \
on the ground cut off by the bottom edge.

Everything else — style, beauty, colours, whether the boxes look tidy — is NOT a problem.

Answer with ONLY a JSON object, no prose, no fence:
{"ok": true|false,
 "problems": ["short factual statements, one per problem"],
 "ops": [{"op": "add|delete|move|resize|replace", "target": "element number or words from its description", "desc": "...", "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}]}

x, y are the TOP-LEFT corner and w, h the size, as fractions of the frame, x+w and y+h \
at most 1. Fix what is wrong with the fewest ops; when a relation is wrong, MOVE the \
smaller thing to where the relation puts it (the cat onto the sofa, not the sofa under \
the cat). Leave "ops" empty and set ok to true when nothing on the list is wrong."""

_PALETTE = [(220, 60, 60), (40, 120, 220), (30, 160, 80), (230, 140, 20),
            (150, 60, 200), (0, 150, 160), (200, 40, 140), (110, 110, 30)]


def _font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def render_sketch(layout: dict, out_path: str, *, width: int = 1024, height: int = 1024,
                  size: int = 768) -> str:
    """Draw the layout as numbered translucent rectangles on a frame of the
    final proportions and save it as a JPEG. Returns out_path.

    Translucent on purpose: a rectangle inside another must stay visible, or
    the checker cannot see «the cat on the sofa» at all.
    """
    from PIL import Image, ImageDraw
    layout = ideogram.normalize_layout(layout)
    if width >= height:
        W, H = size, max(64, int(size * height / max(1, width)))
    else:
        H, W = size, max(64, int(size * width / max(1, height)))
    base = Image.new("RGBA", (W, H), (245, 245, 240, 255))
    draw = ImageDraw.Draw(base)
    # a faint thirds grid so "left / centre / right" is readable
    for f in (1 / 3, 2 / 3):
        draw.line([(W * f, 0), (W * f, H)], fill=(215, 215, 210, 255), width=1)
        draw.line([(0, H * f), (W, H * f)], fill=(215, 215, 210, 255), width=1)
    font, small = _font(max(12, W // 48)), _font(max(11, W // 60))
    els = layout.get("elements") or []
    # Larger boxes first so the small ones are painted on top and stay legible.
    order = sorted(range(len(els)), key=lambda i: -els[i]["w"] * els[i]["h"])
    for i in order:
        el = els[i]
        col = _PALETTE[i % len(_PALETTE)]
        x0, y0 = int(el["x"] * W), int(el["y"] * H)
        x1, y1 = int((el["x"] + el["w"]) * W), int((el["y"] + el["h"]) * H)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.rectangle([x0, y0, x1, y1], fill=col + (48,), outline=col + (255,), width=3)
        base = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(base)
    for i in order:
        el = els[i]
        col = _PALETTE[i % len(_PALETTE)]
        x0, y0 = int(el["x"] * W), int(el["y"] * H)
        x1 = int((el["x"] + el["w"]) * W)
        label = f'{i + 1}. {el.get("desc", "")}' + (f' “{el["text"]}”' if el.get("text") else "")
        lines = _wrap(draw, label, small, max(40, x1 - x0 - 10))[:3]
        ty = y0 + 4
        for ln in lines:
            tw = draw.textlength(ln, font=small)
            draw.rectangle([x0 + 4, ty, x0 + 4 + tw + 6, ty + small.size + 4], fill=(255, 255, 255, 230))
            draw.text((x0 + 7, ty + 2), ln, font=small, fill=col)
            ty += small.size + 6
    bg = str(layout.get("background") or "").strip()
    if bg:
        lines = _wrap(draw, "background: " + bg, font, W - 16)[:2]
        ty = H - (font.size + 6) * len(lines) - 6
        for ln in lines:
            tw = draw.textlength(ln, font=font)
            draw.rectangle([6, ty, 12 + tw, ty + font.size + 4], fill=(255, 255, 255, 235))
            draw.text((9, ty + 2), ln, font=font, fill=(40, 40, 40))
            ty += font.size + 6
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base.convert("RGB").save(out_path, "JPEG", quality=88)
    return out_path


def _sketch_path() -> str:
    import config
    d = os.path.join(str(getattr(config, "OUTPUT_DIR", "runtime")), "sketches")
    return os.path.join(d, f"sketch_{int(time.time() * 1000)}.jpg")


def _regions(layout: dict) -> str:
    import draw_agent
    return draw_agent._regions(layout)


def ask(ctx, sketch_path: str, layout: dict, request: str) -> dict:
    """One vision call: the schematic + the request + the regions in words.
    Returns {"ok", "problems", "ops", "source"}; a checker that could not
    answer is reported as ok with source="unavailable" — it gets no vote."""
    if ctx is None or not sketch_path:
        return {"ok": True, "problems": [], "ops": [], "source": "unavailable"}
    try:
        import llm
        import draw_text as _draw_text
        reader = _draw_text._reader(ctx)
        # Same zoomed-tile look every other picture check gets (vision_close):
        # a dense layout's small overlap or a label crowding another element
        # is exactly the kind of detail one full-frame glance misses.
        user_text = ("THE REQUEST:\n" + (request or "").strip()
                     + "\n\nTHE PLAN (what each rectangle is):\n" + _regions(layout)
                     + "\n\nCheck the plan against the request.")
        try:
            with open(sketch_path, "rb") as _fh:
                _sketch_bytes = _fh.read()
            import vision_close as _vclose
            _tile_notes = _vclose.tile_notes(reader, _sketch_bytes, label="the layout sketch")
            if _tile_notes:
                user_text += ("\n\nSmall details from zoomed-in parts of the sketch "
                              "(tiny overlaps, crowded labels, anything a full-frame "
                              "glance would miss):\n" + _tile_notes)
        except Exception:
            logger.warning("preflight: close look on the sketch failed", exc_info=True)
        raw = llm.analyze_image_with_llm(
            reader, image_path=sketch_path,
            user_text=user_text,
            system_prompt=PREFLIGHT_PROMPT, temperature=0.1, max_tokens=2500)
        data = safe_json_from_llm(raw or "", required_keys=("ok", "problems"))
        if not isinstance(data, dict):
            logger.warning("preflight: no JSON from the vision model")
            return {"ok": True, "problems": [], "ops": [], "source": "unavailable"}
        problems = [str(p).strip() for p in (data.get("problems") or []) if str(p).strip()]
        ok = data.get("ok", None)
        if isinstance(ok, str):
            ok = ok.strip().lower() not in ("false", "no", "0", "none", "")
        ok = (not problems) if ok is None else bool(ok)
        ops = [o for o in (data.get("ops") or []) if isinstance(o, dict)]
        return {"ok": ok and not ops, "problems": problems, "ops": ops, "source": "vision"}
    except Exception as exc:
        logger.exception("preflight failed")
        return {"ok": True, "problems": [], "ops": [], "source": "unavailable",
                "detail": str(exc)}


def _sane(before: dict, after: dict) -> bool:
    """A repair may not gut the plan: no more than one element lost, and never
    all of them. The checker fixes boxes; it does not redesign the scene."""
    nb, na = len(before.get("elements") or []), len(after.get("elements") or [])
    return na >= 1 and na >= nb - 1


def preflight(ctx, layout: dict, request: str, *, width: int = 1024, height: int = 1024,
              passes: int = 3, emit: Optional[Callable[[str, dict], None]] = None) -> tuple:
    """Sketch -> vision check -> apply its ops, up to `passes` times.

    Returns (layout, report) where report is {"sketch", "checked", "ok",
    "problems", "notes", "passes", "source"}. The layout comes back unchanged
    when the checker is unavailable or content — the render is never
    blocked on this step, and a checker that cannot see never edits.
    """
    import draw_agent
    layout = ideogram.normalize_layout(layout, request)
    report = {"sketch": "", "checked": False, "ok": True, "problems": [], "notes": [],
              "passes": 0, "source": "skipped"}
    if ctx is None or not (layout.get("elements") or []):
        return layout, report
    try:
        if hasattr(ctx, "set_stage"):
            ctx.set_stage("Checking the layout")
    except Exception:
        pass
    for n in range(max(1, passes)):
        try:
            sketch = render_sketch(layout, _sketch_path(), width=width, height=height)
        except Exception:
            logger.exception("preflight: could not draw the sketch")
            return layout, report
        report["sketch"] = sketch
        verdict = ask(ctx, sketch, layout, request)
        report.update(checked=verdict.get("source") == "vision", source=verdict.get("source"),
                      passes=n + 1)
        if emit is not None:
            try:
                emit("preflight", dict(sketch=sketch, layout=layout, **verdict))
            except Exception:
                logger.exception("preflight event handler failed")
        logger.info("preflight pass %d: %s source=%s problems=%s ops=%d", n + 1,
                    "OK" if verdict.get("ok") else "FIX", verdict.get("source"),
                    [p[:100] for p in verdict.get("problems") or []][:5],
                    len(verdict.get("ops") or []))
        if verdict.get("ok") or verdict.get("source") != "vision":
            report["ok"] = True
            return layout, report
        report["ok"] = False
        report["problems"] = list(verdict.get("problems") or [])
        if not verdict.get("ops"):
            return layout, report
        fixed, notes = draw_agent.apply_ops(layout, verdict["ops"])
        if not _sane(layout, fixed):
            report["notes"].append("the checker's repair would have gutted the plan — ignored")
            return layout, report
        # the geometry pass keeps the moved boxes inside the frame and apart
        # where they must be apart (lettering)
        try:
            fixed, gnotes = draw_agent.auto_fix_geometry(fixed)
            notes = list(notes) + list(gnotes or [])
        except Exception:
            logger.exception("preflight: geometry pass failed")
        report["notes"].extend(str(x) for x in notes)
        layout = fixed
    return layout, report
