"""Ideogram layout, style and caption model.

Extracted from ideogram.py. The pure, dependency-free layer of drawing mode:
the layout data model (bbox, element, blank_layout, normalize_layout), the
structured-JSON caption codec (build_caption, layout_to_caption,
caption_to_layout, hex_palette), the tolerant JSON reader the planner needs
(_repair_json, _extract_json, _iter_json_objects, _looks_like_layout) and the
style floor (named_style, wants_realism, apply_style_floor).

This layer touches nothing but the standard library -- no LLM, no ComfyUI, no
image module -- so unlike the image.py splits it needs no call-time proxy and
no seam plumbing. ideogram.py re-exports every name below, so existing
`ideogram.<name>` callers and suites are unaffected.

NOTE ON WHAT DID *NOT* MOVE: ideogram.generate declares
`global LAST_CAPTION, LAST_PROMPT, LAST_SERIAL, LAST_IMAGE`. Moving a function
that writes module globals would split that state in two -- writers on one side,
readers on the other -- so generate, the planner and those four globals all stay
in ideogram.py.
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger("assistant.ideogram")


class _IdeogramProxy:
    """Attribute proxy onto ideogram.py, which re-exports every name below.

    Sibling calls inside this layer go through it so that a suite mutating
    `ideogram.<name>` is still honoured after the split. Without it a caller
    that moved here would keep running the REAL function while the mutation
    test printed PASS -- tests/test_ideogram_schema.py does exactly that to
    hex_palette, and caught this. Resolution is deferred to call time, so there
    is no import cycle.
    """

    def __getattr__(self, name):
        import ideogram
        return getattr(ideogram, name)


_ideogram = _IdeogramProxy()


def bbox(x: float, y: float, w: float, h: float) -> list:
    """Fractions of the frame (0-1, origin top-left) -> Ideogram's bbox.

    Ideogram uses a 0-1000 grid in **[ymin, xmin, ymax, xmax]** order — y first,
    which is the opposite of the usual x1,y1,x2,y2 convention. Getting this wrong
    silently mirrors the whole layout across the diagonal.
    """
    c = lambda v: max(0, min(1000, round(v * 1000)))
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


_HEX_RE = re.compile(r"^#?([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


def hex_palette(colours, limit: int) -> list:
    """Normalise a colour list to Ideogram's format: uppercase #RRGGBB, deduped,
    capped. The verifier rejects shorthand (#fff) and lowercase, so expand and
    upcase rather than passing the user's spelling through. Junk entries are
    dropped, never guessed at."""
    out: list = []
    for raw in (colours or []):
        m = _HEX_RE.match(str(raw).strip())
        if not m:
            logger.warning("Ideogram: dropping malformed colour %r", raw)
            continue
        h = m.group(1).upper()
        if len(h) == 3:                       # #FA0 -> #FFAA00
            h = "".join(c * 2 for c in h)
        h = "#" + h
        if h not in out:
            out.append(h)
    return out[:limit]


_APERTURE_RE = re.compile(r"\bf\s*/\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_FOCAL_RE = re.compile(r"\b(\d{2,3})\s*mm(?:\s+lens)?\b", re.IGNORECASE)


def photo_in_words(photo: str) -> str:
    """Camera settings as words, never as figures.

    Ideogram treats a figure in the caption as something to print: the
    default "50mm lens, f/4" came out as "4/4" on a champagne label and as
    "//4" on another (2026-09-12). The setting still says what it said --
    depth of field and framing -- but in language the renderer cannot letter.
    """
    def _ap(m):
        try:
            n = float(m.group(1))
        except ValueError:
            return "moderate aperture"
        if n <= 2.0:
            return "very wide aperture, shallow depth of field"
        if n <= 4.0:
            return "wide aperture, soft background"
        return "narrow aperture, deep focus"

    def _fl(m):
        try:
            n = int(m.group(1))
        except ValueError:
            return "standard lens"
        if n < 35:
            return "wide-angle lens"
        if n <= 60:
            return "standard lens"
        return "telephoto lens"

    out = _APERTURE_RE.sub(_ap, str(photo or ""))
    out = _FOCAL_RE.sub(_fl, out)
    return re.sub(r"\s{2,}", " ", out).strip()


def build_caption(background: str, elements: list, *, high_level: str = "",
                  aesthetics: str = "", lighting: str = "", photo: str = "",
                  medium: str = "", art_style: str = "",
                  palette: Optional[list] = None) -> dict:
    """Assemble the caption dict. Key order matters to the model's verifier, so
    build it explicitly instead of merging dicts.

    Per the published schema the order is exactly:
      photo captions     aesthetics, lighting, photo, medium, color_palette
      non-photo captions aesthetics, lighting, medium, art_style, color_palette
    `color_palette` is the only optional member and must stay last.
    """
    caption = {}
    photo = photo_in_words(photo)
    if high_level.strip():
        caption["high_level_description"] = high_level.strip()
    pal = _ideogram.hex_palette(palette, 16)
    if pal or any(s.strip() for s in (aesthetics, lighting, photo, medium, art_style)):
        style = {"aesthetics": aesthetics, "lighting": lighting}
        if art_style.strip():                       # art_style and photo are exclusive
            style["medium"] = medium
            style["art_style"] = art_style
        else:
            style["photo"] = photo
            style["medium"] = medium
        if pal:
            style["color_palette"] = pal
        caption["style_description"] = style
    caption["compositional_deconstruction"] = {
        "background": background,
        "elements": elements,
    }
    return caption


# No instruction is appended to `background` to keep unlisted signage
# lettering-free. Tried in 4595f07 and measured 2026-09-23 (three scenes,
# fixed seeds, with/without; bench/signage_guard_ab.py): the pseudo-lettering
# on signs came out the same either way, so the clause bought nothing while
# putting renderer instructions into a field that is scene content.

def element(desc: str, box: Optional[list] = None, *, text: str = "",
            palette: Optional[list] = None) -> dict:
    """One element of the scene. `box` omitted => the model places it freely.
    `text` makes it a text element (Ideogram 4 renders legible lettering)."""
    el = {"type": "text" if text else "obj"}
    if box:
        el["bbox"] = box
    if text:
        el["text"] = text
    el["desc"] = desc
    pal = _ideogram.hex_palette(palette, 5)         # per-element cap is 5, not 16
    if pal:
        el["color_palette"] = pal
    return el


_LAYOUT_KEYS = ("high_level_description", "background", "elements", "medium",
                "aesthetics", "lighting", "photo", "art_style")


def _looks_like_layout(d) -> bool:
    """Does this dict plausibly answer the planner prompt?

    Without this check the salvage loop below happily returns the first thing that
    parses — which for a malformed reply is one of the INNER element objects,
    `{"desc": "a ginger cat", "x": 0.3, ...}`. That is a dict, so every downstream
    guard accepted it, and the layout came out with no medium, no photo and no
    aesthetics. A styleless caption is exactly when Ideogram falls back to its own
    house look, so "draw it realistically" quietly produced a cartoon.
    """
    return isinstance(d, dict) and any(k in d for k in _LAYOUT_KEYS)


def _repair_json(raw: str) -> str:
    """Undo the JSON slips this planner actually makes.

    Observed live: a stray conjunction spliced in front of a key —
    `..."lighting": "soft light",\\n and "photo": "85mm lens"` — which invalidates
    the whole object and sends the salvage loop hunting for inner fragments.
    """
    return re.sub(r'(,\s*)(?:and|или|and also|plus)\s+(?=")', r"\1", raw)


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the planner's JSON object out of a reply that may be fenced or chatty.

    Candidates are checked against _looks_like_layout, so a fragment can never
    stand in for the real object. Returns None when nothing layout-shaped is
    found — callers fall back deliberately rather than on garbage.
    """
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()

    # Score candidates instead of taking the first that parses. The reply that
    # caused the bug contained BOTH a malformed full object and a valid one-key
    # duplicate stub (`{"high_level_description": "duplicate copy"}`) — first-wins
    # returned the stub, which is layout-shaped but carries no style and no scene.
    # Richest-wins picks the repaired real object; ties go to the earlier one.
    candidates = []          # (n_layout_keys, order, obj)
    order = 0
    for text in (_ideogram._repair_json(raw), raw):
        chunks = [text]
        try:
            whole = json.loads(text)
        except Exception:
            pass
        else:
            if isinstance(whole, dict):
                candidates.append((sum(k in whole for k in _LAYOUT_KEYS), order, whole))
                order += 1
        for cand in _ideogram._iter_json_objects(chunks[0]):
            candidates.append((sum(k in cand for k in _LAYOUT_KEYS), order, cand))
            order += 1
    if not candidates:
        return None
    best_score, _, best = max(candidates, key=lambda c: (c[0], -c[1]))
    if best_score == 0:
        # Nothing layout-shaped at all — a fragment, not an answer. Say so, so the
        # caller falls back deliberately instead of on garbage.
        return None
    return best


def _iter_json_objects(raw: str):
    """Yield every complete top-level {...} object in `raw`, in order.

    Brace matching is string-aware — a `{` or `}` inside a description would
    otherwise cut the object in the wrong place.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    yield obj
                start = -1
            elif depth < 0:
                depth = 0


def blank_layout(prompt: str = "") -> dict:
    """An editable layout with one full-frame element — the storyboard's empty state."""
    prompt = (prompt or "").strip()
    return {"high_level_description": prompt, "aesthetics": "", "lighting": "",
            "photo": "", "medium": "", "background": prompt,
            "elements": ([{"desc": prompt, "text": "",
                           "x": 0.05, "y": 0.05, "w": 0.9, "h": 0.9}] if prompt else [])}


# Words that say "this must look like a photograph, not a drawing", in both the
# languages this bot is used in. The prompt reaches the planner untranslated when
# the user talks to the bot directly, so both matter.




# A named style, mapped to the English phrase that goes into `art_style`. Used
# only when the planner produced nothing — the user asked for a drawn look, so
# the floor must supply THAT, not photography.


_LOOKS = ("caricature", "cartoon", "anime", "comic strip", "watercolour",
          "oil painting", "pixel art", "3d render", "sketch", "illustration", "painting")


def _look(prompt: str) -> str:
    """The look the user asked the picture to have: one of _LOOKS, "photo", or
    "none". Read by the model: a painting HANGING on the wall in the scene is
    not a request for a painted picture (live 2026-09-12, the render came back
    as a framed canvas), and word lists could not tell the two apart."""
    import intent
    return intent.ask_choice(
        "A user asked for this picture: {text}. Which LOOK does the user ask the whole "
        "picture to have? caricature = карикатура / шарж; cartoon = мультик, мультяшный; "
        "anime = аниме; comic strip = комикс (even a realistic one); watercolour = "
        "акварель (also a watercolour sketch); oil painting = oil painting, масло, "
        "живопись; pixel art; 3d render; sketch = скетч, набросок; illustration = "
        "рисунок, a drawing, a children's drawing; painting = any other painting; "
        "photo = a realistic photograph; none = no look named. A named drawn look "
        "wins over a word like realistic. A thing inside the scene (a painting on the "
        "wall, a comic in a hand) is not the look.",
        prompt or "", _LOOKS + ("photo", "none"), "none")


def named_style(prompt: str) -> str:
    """The drawn/painted look the user actually named, or "" if they named none."""
    got = _look(prompt)
    return got if got in _LOOKS else ""


_DEFAULT_PHOTO_STYLE = {
    "aesthetics": "photorealistic, true-to-life detail, natural colour",
    "lighting": "natural light",
    "photo": "standard lens, wide aperture, soft background, sharp focus",
    "medium": "photography",
}


def wants_realism(prompt: str) -> bool:
    """True when the user asked for a photograph and did NOT ask for a style."""
    return _look(prompt) == "photo"


def apply_style_floor(layout: dict, prompt: str) -> dict:
    """Never hand Ideogram a caption with no style at all.

    An empty style_description is not neutral — it lets the model pick, and what it
    picks is its own stylised house look. That is how "draw it realistically" came
    back as a cartoon: the planner failed, the code fell back to a blank layout, and
    the blank layout carried no medium, no photo and no aesthetics. A planner
    failure must not silently change the KIND of picture the user asked for.
    """
    if not isinstance(layout, dict):
        return layout
    has_style = any(str(layout.get(k) or "").strip()
                    for k in ("aesthetics", "lighting", "photo", "medium", "art_style"))
    if has_style:
        return layout

    # The user named a drawn look — supply THAT, not photography. Otherwise a
    # planner failure on "нарисуй карикатуру" would come back as a photograph,
    # which is the same class of bug in the opposite direction.
    named = _ideogram.named_style(prompt)
    if named:
        layout["art_style"] = named
        layout["medium"] = "illustration"
        layout["photo"] = ""          # exclusive with art_style in build_caption
        logger.warning("Ideogram: caption had no style — applying the style the "
                       "user named (%s)", named)
        return layout

    # Nothing named. This used to return here unless the prompt explicitly said
    # "realistic", which left the COMMON case — "нарисуй машину во дворе" — with
    # an empty style_description. Empty is not neutral: Ideogram falls back to its
    # own stylised house look, so every ordinary request came back a cartoon and
    # you had to ask for realism to get a photograph. Photography is the default
    # the planner prompt already declares; the floor now enforces it.
    layout.update(_DEFAULT_PHOTO_STYLE)
    layout["art_style"] = ""          # exclusive with `photo` in build_caption
    logger.warning("Ideogram: caption had no style (%s) — applying the "
                   "photographic default",
                   "realism requested" if _ideogram.wants_realism(prompt) else "none requested")
    return layout


def _clamp01(v, default=0.0):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


# "a cat with no whiskers": the renderer has no negation, the named thing gets drawn.
_ABSENT_RE = re.compile(r"(?:,\s*|\s+(?:and|with|but)\s+)(?:no|without)\s+.*?(?=\s+and\b|[,.;]|$)", re.I)


def normalize_layout(data: dict, prompt: str = "") -> dict:
    """Coerce anything layout-shaped (planner output, an edited storyboard, a file
    the user hand-wrote) into the canonical editable layout dict."""
    out = _ideogram.blank_layout(prompt)
    if not isinstance(data, dict):
        return out
    for key in ("high_level_description", "aesthetics", "lighting", "photo",
                "medium", "art_style", "background"):
        if data.get(key) is not None:
            out[key] = str(data.get(key) or "").strip()
    elements = []
    for raw in (data.get("elements") or []):
        if not isinstance(raw, dict):
            continue
        desc = _ABSENT_RE.sub("", str(raw.get("desc") or "")).strip(" ,")
        text = str(raw.get("text") or "").strip()
        if not desc and not text:
            continue
        x, y = _clamp01(raw.get("x")), _clamp01(raw.get("y"))
        w = _clamp01(raw.get("w"), 1.0) or 1.0
        h = _clamp01(raw.get("h"), 1.0) or 1.0
        # keep the box inside the frame: Ideogram silently clips otherwise
        w, h = min(w, 1.0 - x), min(h, 1.0 - y)
        elements.append({"desc": desc or text, "text": text,
                         "x": x, "y": y, "w": max(w, 0.01), "h": max(h, 0.01)})
    out["elements"] = elements
    if not out["background"]:
        out["background"] = out["high_level_description"] or prompt.strip()
    return out


# REJECTED, kept as a warning. Ideogram 4 writes lettering whether or not it was
# asked to -- a plain "белая кружка с чаем" came back with invented words on the
# mug. The obvious fix was to append "no letters, words or logos on any object"
# to the caption's background, and it was measured twice:
#
#   * without the clause: a normal photograph of a kitchen;
#   * with it (two different phrasings): the scene rendered as cut-out objects
#     floating on a transparency checkerboard, and the invented lettering came
#     back anyway.
#
# An instruction of that shape reads to the renderer as a product-photography
# brief, and it delivers one. A real fix has to reach the renderer somewhere
# other than `background`, which is scene CONTENT; the workflow's negative
# conditioning is zeroed out (ConditioningZeroOut), so there is no negative
# prompt to put it in today. Do not re-add this to the caption text.

def _is_cyrillic(text: str) -> bool:
    """True when the layout is written in Russian, so the floor speaks its language.

    A synthesized sentence in the wrong language is a second subject in the
    caption, not a clarification.
    """
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    return cyr * 3 >= sum(1 for ch in text if ch.isalpha())


# The noun carries the work. "Одно изображение" was measured and REJECTED: it
# left the 2x2 collage standing and added a transparency checkerboard under it,
# because an "image" may be a cut-out. The hand-written sentence that fixed the
# render said ФОТОГРАФИЯ, and a photograph cannot be a cut-out or a grid.
_SCENE_FLOOR = {
    "photo": {
        "ru": ("Одна фотография одной сцены: {bg}. {els} — всё это в одном "
               "кадре, на одном снимке."),
        "en": ("A single photograph of one scene: {bg}. {els} — all of them in "
               "the SAME frame, in one shot."),
    },
    # A drawing is still ONE drawing; it just must not claim to be a photo.
    "art": {
        "ru": ("Одна цельная иллюстрация одной сцены: {bg}. {els} — всё это в "
               "одном кадре."),
        "en": ("One single illustration of one scene: {bg}. {els} — all of them "
               "in the SAME frame."),
    },
}


def synth_high_level(layout: dict) -> str:
    """A one-sentence "this is ONE picture" line, built from the layout itself.

    Measured, at one seed, on the layout that produced the worst render of a
    14-scene run: a black cat on the floor, a ginger cat in an armchair and a
    wooden table, over a one-word background. It came back as a 2x2 grid of four
    separate stock photographs -- with two invented people in it.

    The cause is not the thin background: rendering the same layout with a rich
    background produced the SAME 2x2 collage. It is the missing
    high_level_description. Every element names its own surface, nothing in the
    caption says they share a frame, and the model reconciles three settings the
    only way it can -- as three pictures. Adding this one sentence, and changing
    nothing else, produced a single coherent room containing all three.

    CORRECTION, measured afterwards and kept here because the first conclusion
    was wrong: this sentence does NOT reliably cure the collage. Eight further
    renders at the same seed showed every mechanically built version of it
    collaging, and changing ONE PREPOSITION inside the hand-written sentence
    that worked ("в кресле" -> "на кресле") collapsed that one into a collage
    too. The outcome is chaotically sensitive to the exact string, so no
    template can be trusted, and the first probe had found a lucky sentence
    rather than a mechanism.

    Two further measurements closed the question:

      * geometry is not the cause either. This layout's box statistics (three
        elements, 0.23 coverage, no mutual overlap) are indistinguishable from
        `kitchen` and `self_omission`, both of which render as single rooms, so
        any geometric predicate that fired here would fire on those too.
      * neither is the seed. bench/draw_seed_sweep.py rendered this caption at
        eight seeds and got eight collages.

    The sentence is kept because it costs nothing and states something true
    about the layout. The actual defence is downstream and deterministic:
    draw_agent.looks_like_collage() measures the render, and the ladder changes
    something STRUCTURAL -- the frame's aspect first, since a 2x2 grid does not
    fit a 16:9 frame. That rung is what produced the single coherent room.
    """
    els = [e.get("desc", "").strip() for e in (layout.get("elements") or [])
           if e.get("desc", "").strip()]
    bg = (layout.get("background") or "").strip()
    if not els:
        return ""
    text = bg + " " + " ".join(els)
    kind = "art" if str(layout.get("art_style") or "").strip() else "photo"
    tpl = _SCENE_FLOOR[kind]["ru" if _is_cyrillic(text) else "en"]
    return tpl.format(bg=bg or (els[0]), els=", ".join(els[:8]))


_MERGE_JOIN = {"ru": "%s, в которой находятся: %s",
               "en": "%s, containing: %s"}


def merged_layout(layout: dict) -> dict:
    """The same scene as ONE element filling the frame, instead of several.

    A last resort for a layout the renderer keeps returning as a collage. The
    boxes are what is given up -- and they are given up deliberately: a picture
    that puts the objects somewhere other than asked is still a picture of the
    scene, while a 2x2 grid of stock photographs is not.

    Measured on the layout that collages on every seed: as three boxes it came
    back as a grid three times running; merged into one full-frame element it
    came back as one room with both cats in it. (Elements with NO bbox at all
    are not an option -- Ideogram answered that caption with its safety card on
    three seeds out of three.)
    """
    layout = _ideogram.normalize_layout(layout)
    descs = [e["desc"].strip() for e in layout["elements"] if e.get("desc", "").strip()]
    if len(descs) < 2:
        return layout
    bg = (layout.get("background") or "").strip()
    lang = "ru" if _is_cyrillic(bg + " " + " ".join(descs)) else "en"
    desc = _MERGE_JOIN[lang] % (bg or descs[0], ", ".join(descs))
    out = dict(layout)
    out["elements"] = [{"desc": desc, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
                        "text": ""}]
    return _ideogram.normalize_layout(out)


FRAME_COVERAGE_MIN = 0.75


def filled_layout(layout: dict, margin: float = 0.02) -> dict:
    """The same arrangement, stretched so the elements reach the frame edges.

    Measured on the one scene that survived every collage rung: `two_edits`
    rendered as a coherent kitchen squeezed into a horizontal band, with a
    transparency checkerboard above and below it. Its caption was faultless --
    photographic style floor, "one photograph of one scene" sentence, sane
    boxes -- but every box sat inside x 310-800, y 20-750 of a 1000x1000 frame.
    The renderer filled what the boxes did not claim with nothing.

    Rendering that identical caption with the boxes rescaled to span the frame
    produced a full photograph, same seed, nothing else changed.

    Relative positions are preserved: this stretches the arrangement, it does
    not rearrange it. Returns the layout unchanged when the boxes already reach
    far enough, so a deliberate composition is never disturbed.
    """
    layout = _ideogram.normalize_layout(layout)
    els = layout.get("elements") or []
    if not els:
        return layout
    x0 = min(e["x"] for e in els)
    y0 = min(e["y"] for e in els)
    x1 = max(e["x"] + e["w"] for e in els)
    y1 = max(e["y"] + e["h"] for e in els)
    span_x, span_y = x1 - x0, y1 - y0
    if span_x <= 0 or span_y <= 0:
        return layout
    if span_x >= FRAME_COVERAGE_MIN and span_y >= FRAME_COVERAGE_MIN:
        return layout                       # already reaches the edges
    inner = 1.0 - 2.0 * margin
    sx, sy = inner / span_x, inner / span_y
    out = dict(layout)
    out["elements"] = []
    for e in els:
        n = dict(e)
        n["x"] = margin + (e["x"] - x0) * sx
        n["y"] = margin + (e["y"] - y0) * sy
        n["w"] = e["w"] * sx
        n["h"] = e["h"] * sy
        out["elements"].append(n)
    return _ideogram.normalize_layout(out)


def layout_to_caption(layout: dict) -> dict:
    """Editable layout -> the structured caption the model is fed."""
    layout = _ideogram.normalize_layout(layout)
    # The style floor belongs HERE, not only in plan_layout. Every render goes
    # through this function; only a FRESHLY PLANNED layout went through the
    # planner. A layout that was hand-built, edited, or restored from a
    # storyboard arrived with an empty style_description -- and an empty style
    # is not neutral, it lets the model pick. Measured on a render: four boxes
    # with no style came back as a collage of cut-out objects on a
    # transparency checkerboard, complete with an invented product packet.
    layout = apply_style_floor(layout, layout.get("background") or "")
    elements = [
        _ideogram.element(el["desc"], _ideogram.bbox(el["x"], el["y"], el["w"], el["h"]), text=el["text"])
        for el in layout["elements"]
    ]
    if not elements:                                  # never submit an empty scene
        elements = [_ideogram.element(layout["background"] or layout["high_level_description"])]
    background = layout["background"] or layout["high_level_description"]
    # The SCENE floor, beside the style floor and for the same reason: a layout
    # that arrives without a high_level_description does not get a neutral
    # renderer, it gets a collage. See synth_high_level().
    high_level = layout["high_level_description"].strip() or synth_high_level(layout)
    return _ideogram.build_caption(
        background, elements,
        high_level=high_level,
        aesthetics=layout["aesthetics"], lighting=layout["lighting"],
        photo=layout["photo"], medium=layout["medium"],
        art_style=layout.get("art_style", ""))


def caption_to_layout(caption: dict) -> dict:
    """Structured caption -> editable layout (the inverse of layout_to_caption).

    Lets the storyboard pick up a caption the agent already built and show it as
    draggable boxes, so "move that, reword this, redraw" works on the agent's own
    composition rather than only on layouts the user planned by hand.
    """
    if not isinstance(caption, dict):
        return _ideogram.blank_layout()
    comp = caption.get("compositional_deconstruction") or {}
    style = caption.get("style_description") or {}
    elements = []
    for el in (comp.get("elements") or []):
        if not isinstance(el, dict):
            continue
        box = el.get("bbox") or [0, 0, 1000, 1000]
        try:
            ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in box[:4])
        except (TypeError, ValueError):
            ymin, xmin, ymax, xmax = 0.0, 0.0, 1.0, 1.0
        elements.append({"desc": str(el.get("desc") or ""), "text": str(el.get("text") or ""),
                         "x": xmin, "y": ymin, "w": xmax - xmin, "h": ymax - ymin})
    return _ideogram.normalize_layout({
        "high_level_description": caption.get("high_level_description", ""),
        "aesthetics": style.get("aesthetics", ""), "lighting": style.get("lighting", ""),
        "photo": style.get("photo", ""), "medium": style.get("medium", ""),
        "art_style": style.get("art_style", ""),
        "background": comp.get("background", ""), "elements": elements})
