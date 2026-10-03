"""Build a real .pptx from a plan the model writes.

Two halves, deliberately separated:

  `plan_deck()`  — the LLM decides WHAT the deck says (title, sections, bullets,
                   speaker notes, and which slides want a picture).
  `build_pptx()` — deterministic assembly. No model involvement, so a deck can
                   always be produced from a plan, and a malformed plan degrades
                   into a plainer deck rather than into nothing.

Pictures are optional and never block the deck: each slide may name an image
query, the caller supplies whatever it managed to fetch or draw, and a slide with
no picture simply lays out as text.
"""
import logging
import os
import re
from typing import Callable, Optional

logger = logging.getLogger("assistant.slides")

# python-pptx' default 4:3 template is dated; 16:9 is what a projector expects.
_SLIDE_W_IN = 13.333
_SLIDE_H_IN = 7.5

MAX_SLIDES = 30
MAX_BULLETS = 7

# ── visual themes ─────────────────────────────────────────────────────────────
# The palette is chosen from the SUBJECT, not at random: a deck about farm
# machinery and a deck about a clinical trial should not look the same. Each
# theme is (dark panel, accent, heading, body, page tint) — the dark panel backs
# the cover and the section rule, the accent marks headings and bullets.
THEMES: dict = {
    "corporate": dict(panel=0x14273D, accent=0x2E86AB, head=0x14273D,
                      body=0x3A3A3A, tint=0xF4F7FA),
    "tech":      dict(panel=0x11151C, accent=0x00A6A6, head=0x11151C,
                      body=0x33383F, tint=0xF2F6F7),
    "nature":    dict(panel=0x1E3D2F, accent=0x4C9A62, head=0x1E3D2F,
                      body=0x36423B, tint=0xF3F8F4),
    "history":   dict(panel=0x3B2C1E, accent=0xA9743B, head=0x3B2C1E,
                      body=0x41382F, tint=0xFAF6F0),
    "energy":    dict(panel=0x2B1A0E, accent=0xD2691E, head=0x2B1A0E,
                      body=0x3D342C, tint=0xFCF5EE),
    "academic":  dict(panel=0x2A2438, accent=0x6C63A6, head=0x2A2438,
                      body=0x3A3646, tint=0xF6F4FA),
    "warm":      dict(panel=0x3A1F2B, accent=0xC1566B, head=0x3A1F2B,
                      body=0x40353A, tint=0xFDF4F6),
    "light":     dict(panel=0xEEF2F6, accent=0x2E86AB, head=0x14273D,
                      body=0x3A3A3A, tint=0xF7F9FB),
}
_DEFAULT_THEME = "corporate"

# Keywords that pick a theme when the planner did not (or picked a name we do not
# have). Deterministic, so the same topic always looks the same.
_THEME_HINTS = (
    ("light",    ("светлая тема", "светлую тему", "светлой теме", "светлом стиле", "белом фоне",
                  "light theme", "white background")),
    ("energy",   ("трактор", "машин", "завод", "двигател", "промышл", "нефт",
                  "энерг", "tractor", "engine", "factory", "industrial", "machinery")),
    ("nature",   ("природ", "сельск", "climate", "клима", "агро", "лес", "farm",
                  "nature", "ecolog", "эколог", "urozh", "урожай")),
    ("tech",     ("технолог", "цифров", "software", "ai", "нейросет", "compute",
                  "robot", "робот", "data", "данн")),
    ("history",  ("истори", "war", "войн", "век", "century", "герой", "biograph",
                  "биограф", "эпоха")),
    ("academic", ("исследован", "наук", "research", "study", "science", "теор",
                  "медицин", "clinical")),
    ("warm",     ("культур", "искусств", "музык", "art", "music", "food", "кухн")),
)


_S, _N = {"type": "string"}, {"type": "number"}
_ARR = lambda item: {"type": "array", "items": item}
_OBJ = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}
# The planner's JSON shape as a grammar: Gemma fenced it or dropped keys often enough
# that plan_deck kept a three-rung retry ladder for it.
DECK_SCHEMA = _OBJ({
    "title": _S, "subtitle": _S, "theme": {"type": "string", "enum": list(THEMES)}, "cover_image": _S,
    "slides": _ARR(_OBJ({
        "heading": _S, "bullets": _ARR(_S),
        "stats": _ARR(_OBJ({"value": _S, "label": _S}, ("value", "label"))),
        "chart": _OBJ({"title": _S, "kind": _S, "labels": _ARR(_S), "values": _ARR(_N),
                       "unit": _S, "source": _S}),
        "table": _OBJ({"columns": _ARR(_S), "rows": _ARR(_ARR(_S))}),
        "notes": _S, "image": _S}, ("heading", "bullets"))),
    "sources": _ARR(_OBJ({"title": _S, "url": _S}, ("url",)))}, ("title", "slides"))


def pick_theme(deck: dict, topic: str = "") -> dict:
    """The palette for this deck: what the planner asked for, else the subject."""
    name = (deck.get("theme") or "").strip().lower()
    if name not in THEMES:
        hay = f"{deck.get('title', '')} {topic}".lower()
        name = next((t for t, words in _THEME_HINTS if any(w in hay for w in words)),
                    _DEFAULT_THEME)
    return dict(THEMES[name], name=name)


_PLAN_PROMPT = """You plan presentation decks. Given a topic, produce the deck as JSON.

Answer with ONLY a JSON object, no prose, no code fence:
{"title": "deck title",
 "subtitle": "one line under the title",
 "theme": "one of: corporate, tech, nature, history, energy, academic, warm, light",
 "cover_image": "a short image search query for the FULL-BLEED title background",
 "slides": [
   {"heading": "slide title",
    "bullets": ["a point that carries a fact", "another one"],
    "stats": [{"value": "1962", "label": "first production year"},
              {"value": "220 hp", "label": "engine output"}],
    "chart": {"title": "Units built per year", "kind": "bar",
              "labels": ["1965", "1975", "1985"], "values": [3200, 9400, 12100],
              "unit": "units", "source": "https://example.org/report"},
    "table": {"columns": ["", "Model A", "Model B"], "rows": [["Price", "$799", "$899"]]},
    "notes": "what the speaker says on this slide",
    "image": "a short image search query, or empty string for no picture"}
 ],
 "sources": [{"title": "article title", "url": "https://..."}]}

Rules:
- Between 5 and 12 slides unless the request names a number; then use that number.
- Bullets are PHRASES, not paragraphs: 4 to 16 words each, at most 6 per slide.
  A slide crammed with sentences is the single most common bad deck. A slide of
  two-word labels ("Modern era", "The future") is the second most common, and it
  is worse: it says nothing. Every bullet must carry a FACT -- a number, a date,
  a name, a place, a consequence -- not a category heading.
- FIGURES ARE THE POINT. At least half the content slides must contain a concrete
  number with its unit and its year. "Production grew" is not a fact; "output rose
  from 3,200 to 12,100 units between 1965 and 1985" is.
- `stats` is the headline figure band: two to four SHORT value/label pairs shown
  large above the bullets. Use it on slides where the numbers are the message.
  Omit the key entirely when the slide has no figures.
- `chart` draws a real chart. Give it on AT LEAST TWO slides of the deck when the
  subject has any quantity that changes -- over time, between categories, between
  competitors. `kind` is "bar", "line" or "pie". `labels` and `values` must be the
  same length, 2 to 12 entries, and `values` must be plain numbers with no units
  or spaces (put the unit in `unit`). Omit the key when there is nothing to plot.
- `table` is for side-by-side comparisons (specs, plans, pros and cons): 2 to 5
  columns, 2 to 8 rows of short cells. A slide that promises a table carries one.
  Omit the key otherwise.
- `sources` is deck-level: 3 to 8 genuinely useful articles, each a real full URL
  you are confident exists. Prefer sources that a curious listener would actually
  want to open. Never invent a URL; if you are unsure of one, leave it out.
- `notes` carries the detail that does NOT fit on the slide. Two or three sentences.
- Give `image` only where a picture genuinely helps (a place, an object, a person,
  a process). Leave it "" for slides that are pure argument or numbers.
- Write in the SAME LANGUAGE as the request.
- When the request names an audience (a school grade, children, beginners, "простыми
  словами"), write every bullet for THEM: everyday words, no jargon, no "Label: value"
  bullets -- «Морфология: сходство с ящерицами» for a fifth-grader is a failed slide;
  «Они были похожи на больших ящериц» is the fact they can use.
- Open with a title/agenda slide and close with a conclusion or takeaways slide.
- `theme` sets the colour palette, so pick the one that FITS the subject: machinery
  and industry -> energy or tech; countryside, farming, climate -> nature; a company
  or a market -> corporate; a period or a biography -> history; research -> academic.
- `cover_image` is the single most visible picture in the deck: it fills the whole
  title slide behind the text. Ask for a wide, uncluttered, photographic scene of
  the subject itself — not a logo, not a diagram, not a portrait crop."""


# Last rung of the planner ladder: a worked example and nothing else. A model
# that has already burned two turns reasoning about the schema does better when
# shown the answer shape than when told about it again.
_PLAN_MINIMAL = """Output ONE JSON object for a slide deck. No prose. No code fence.

Example of the exact shape:
{"title": "The Trans-Siberian Railway", "subtitle": "Building the longest line",
 "theme": "history", "cover_image": "trans-siberian railway train",
 "slides": [
  {"heading": "Why it was built", "bullets": ["Linking east and west",
    "Moving troops and grain"], "notes": "Context for the decision.",
   "image": "old railway construction"},
  {"heading": "What it cost", "bullets": ["Decades of work", "Enormous expense"],
   "notes": "Scale of the effort.", "image": ""}
 ]}

Now do the same for the topic below. Same keys, same types, 5-8 slides,
bullets of at most 12 words, written in the language of the topic."""


_FACT_HITS = 8


def gather_facts(ctx, topic: str) -> tuple:
    """Search the web for the topic and return (digest, sources).

    The planner used to write decks out of the model's memory alone, and a model
    writing from memory will not risk a number -- which is exactly why the decks
    came back as lists of two-word labels with no figures and no links in them.
    Giving it real snippets to quote from is what makes "at least half the slides
    carry a figure" a possible instruction rather than an invitation to invent
    one.

    Never raises and never blocks the deck: no network, no results, or a slow
    search all degrade to ("", []) and the planner runs exactly as before.
    """
    topic = (topic or "").strip()
    if not topic:
        return "", []
    try:
        from search import raw_search_results
        hits = raw_search_results(ctx, topic, max_results=_FACT_HITS) or []
    except Exception:
        logger.warning("deck fact-gathering search failed for %r", topic, exc_info=True)
        return "", []
    lines, sources = [], []
    for h in hits:
        body = (h.get("body") or "").strip()
        if body:
            lines.append("- %s (%s): %s" % (h.get("title", ""), h.get("domain", ""),
                                            body[:320]))
        sources.append({"title": h.get("title") or h.get("href", ""),
                        "url": h.get("href", "")})
    digest = "\n".join(lines[:_FACT_HITS])
    logger.info("deck facts: %d hits for %r", len(hits), topic[:60])
    return digest, _clean_sources(sources)


_REPAIR_PROMPT = """The deck below is TOO THIN. It reads as a list of headings: it
states categories instead of facts.

Rewrite it as ONE JSON object with the same keys and the same slide count. Keep
the headings and the language. Change the CONTENT:
- every bullet must carry a concrete fact -- a number with its unit and year, a
  name, a place, a measured consequence. Delete any bullet you cannot make
  concrete and replace it with one you can.
- put a `stats` band (2-4 value/label pairs) on the slides whose point is numeric.
- add a `chart` to at least two slides: {"title","kind":"bar|line|pie","labels":
  [...],"values":[plain numbers],"unit"}. labels and values must be the same
  length, 2 to 12 entries.
- keep or extend `sources`: 3 to 8 real full URLs.
Use ONLY figures supported by the research notes you were given. No prose, no code
fence, JSON only."""


# «итоги года нашей кофейни: Q1 1,2 млн…» -- the user's own figures. A web search
# on it filled the deck with other cafés' profits presented as theirs.
_OWN_DATA_RE = re.compile(r"\b(?:наш\w*|мо[йяеи]\w*|свое\w*|our|my)\b", re.I)


def _own_data(topic: str) -> bool:
    return bool(_OWN_DATA_RE.search(topic or "")) and len(re.findall(r"\d+(?:[.,]\d+)?", topic or "")) >= 3


_STRUCTURAL_EDIT_RE = re.compile(
    r"поменя\w* местами|переставь|перенеси|удали|убери|swap|reorder|move|delete|remove", re.I)


_DROP_SLIDE_RE = re.compile(
    r"^\W*(?:удали|убери|delete|remove)\s+(?:the\s+)?(последн\w*|перв\w*|last|first|(\d+))(?:-?\w{0,3})?\s+(?:слайд|slide)\w*\W*$"
    r"|^\W*(?:удали|убери|delete|remove)\s+(?:слайд|slide)\s+(?:№\s*)?(\d+)\W*$", re.I)


def _dropped_slide(previous: dict, change: str):
    """«удали последний слайд» came back unchanged: the model keeps every
    slide it can. A bare delete by position needs no model."""
    m = _DROP_SLIDE_RE.match((change or "").strip())
    sl = list(previous.get("slides") or [])
    if not m or len(sl) < 2:
        return None
    num = m.group(2) or m.group(3)
    k = int(num) - 1 if num else (0 if m.group(1).lower().startswith(("перв", "first")) else -1)
    if not -len(sl) <= k < len(sl):
        return None
    del sl[k]
    return dict(previous, slides=sl, edited=True)


def _edit_deck(ctx, previous: dict, change: str, out_lang: str) -> dict:
    """Apply `change` to `previous` (a normalized deck) and return the new deck.

    "добавь слайд про спутники Марса" used to re-plan the whole deck from a
    merged topic: a different theme, different slides, different title (live,
    2026-09-12, journey 19). The model now gets the existing plan and the
    change; a plan it cannot produce leaves the previous deck untouched with
    `edit_failed` set so the caller can say so.
    """
    dropped = _dropped_slide(previous, change)
    if dropped:
        return dropped
    import json as _json
    from llm import call_llm_simple
    from utils import safe_json_from_llm

    base = {k: previous.get(k) for k in ("title", "subtitle", "theme", "cover_image",
                                          "slides", "sources") if previous.get(k) is not None}
    # A reorder/delete needs no research: searching «поменяй местами 2 и 3
    # слайд» only brought junk sources, whose closing slide then ate a content one.
    facts, web_sources = (("", []) if _STRUCTURAL_EDIT_RE.search(change or "")
                          else gather_facts(ctx, change))
    ask = "Current deck:\n" + _json.dumps(base, ensure_ascii=False)[:12000]
    ask += "\n\nChange request: " + (change or "").strip()
    if facts:
        ask += ("\n\nResearch notes for the change (by web search, so some may be "
                "off-topic — ignore any note that turns out to be about a "
                "different subject than the change request). From the notes that "
                "ARE on-topic, take any new figures, dates and names and quote "
                "them exactly:\n" + facts)
    if out_lang:
        from deep_research import _lang_directive
        ask += _lang_directive(out_lang)
    data = None
    for temp in (0.3, 0.6):
        raw = call_llm_simple(ctx, _EDIT_PROMPT, ask, temperature=temp,
                              max_tokens=8000, force_think=False, json_schema=DECK_SCHEMA) or ""
        got = safe_json_from_llm(raw, required_keys=("slides",))
        if isinstance(got, dict) and got.get("slides"):
            data = got
            break
        logger.warning("deck edit attempt produced no usable JSON (%d chars)", len(raw))
        if ctx is not None and getattr(ctx, "is_cancelled", bool)():
            break
    if not data:
        logger.error("deck edit failed — keeping the previous deck")
        out = dict(previous)
        out["edit_failed"] = True
        return out
    # The edit keeps the previous look and title unless the change is about them.
    data.setdefault("theme", previous.get("theme") or "")
    data.setdefault("title", previous.get("title") or "")
    # The count was the first plan's; an edit may add or drop slides.
    deck = normalize_deck(data, previous.get("title") or change)
    # 'dark_green_accent' is no theme: the renderer fell back by topic, to brown.
    deck["theme"] = next((t for t in (data.get("theme"), previous.get("theme")) if t in THEMES), "")
    deck["sources"] = _clean_sources((deck.get("sources") or []) + (previous.get("sources") or []) + list(web_sources or []))
    deck["edited"] = True
    return deck


_EDIT_PROMPT = """You edit an existing presentation deck. You get the current deck as JSON and one change request.

Apply ONLY the requested change. Keep the title, subtitle, theme, cover_image, sources and every slide the request does not touch — same text, same order. A new slide follows the same shape as the others (heading, bullets, stats, chart, table, notes, image) and carries real facts; a slide that promises a table carries a "table" {"columns": [...], "rows": [[...]]}. Never renumber or shorten what you were not asked to change.

"theme" is one of these names only: corporate (dark navy, blue accent), tech (near-black, teal accent), nature (dark green, green accent), history (dark brown, bronze accent), energy (dark brown, orange accent), academic (dark purple, violet accent), warm (dark plum, rose accent), light (white cover, blue accent). A colour wish picks the closest one.

Answer with ONLY the full updated deck as one JSON object in exactly the same shape as the input, no prose, no code fence."""


def plan_deck(ctx, topic: str, n_slides: int = 0, out_lang: str = "",
              previous: Optional[dict] = None) -> dict:
    """Ask the model for a deck plan. Returns a normalized dict, never raises.

    `out_lang` ("en"/"ru"/"") is the language the DECK ITSELF must be written
    in — deliberately separate from `topic`'s own language. By the time this
    runs, `topic` has usually already passed through the assistant's
    English-first entry translation (every non-English message is translated
    to English before tool-calling), so "written in the language of the topic"
    (the planner prompt's own instruction) silently means English regardless
    of what the user actually typed. The caller detects the language from the
    user's ORIGINAL, untranslated message and passes it here explicitly;
    "" falls back to the soft topic-language instruction alone.
    """
    from llm import call_llm_simple
    from utils import safe_json_from_llm

    if previous and previous.get("slides"):
        return _edit_deck(ctx, previous, topic, out_lang)
    ask = (topic or "").strip()
    own = _own_data(topic)
    facts, web_sources = ("", []) if own else gather_facts(ctx, topic)
    if own:
        ask += ("\n\nThese are the user's own figures. Use exactly these numbers "
                "and no others; a chart plots them.")
    if facts:
        # A literal web search on the topic string can surface hits that share
        # a WORD with it but not its actual subject -- live, 2026-09-19: a
        # joke deck about slacking off at work ("idle correctly at work")
        # came back with car-carburetor and Python-IDE snippets, because the
        # search matched "idle" rather than the intent, and the planner was
        # told to quote its notes verbatim with no relevance check at all.
        # The notes are still worth having (dates, names, real numbers beat
        # invented ones) -- the fix is not trusting them blindly.
        ask += ("\n\nResearch notes gathered for this topic (by web search, so "
                "some may be off-topic — a search for one word in the topic can "
                "return results about a completely different subject that "
                "happens to share that word). Use ONLY the notes that are "
                "actually about THIS topic for your figures, dates and names, "
                "and quote those exactly. Ignore and do not mention any note "
                "that turns out to be about something else. Anything you "
                "cannot support from an on-topic note, leave out rather than "
                "invent:\n" + facts)
    if n_slides:
        ask += f"\n\nThe deck must have exactly {n_slides} slides."
    if out_lang:
        from deep_research import _lang_directive
        ask += _lang_directive(out_lang)

    # A ladder, not a single shot. Gemma regularly spends a whole turn in its
    # reasoning channel and returns no JSON at all; with one attempt that
    # silently produced a ONE-SLIDE stub deck which the tool then announced as a
    # finished presentation. Each rung varies the ASK — not just the temperature —
    # because repeating an identical prompt to a model that just failed it mostly
    # fails again.
    attempts = (
        dict(prompt=_PLAN_PROMPT, temperature=0.4, max_tokens=6000, force_think=False),   # 2026-09-25: no reasoning anywhere
        dict(prompt=_PLAN_PROMPT, temperature=0.7, max_tokens=8000, force_think=False),
        dict(prompt=_PLAN_MINIMAL, temperature=0.3, max_tokens=8000, force_think=False),
    )
    data = None
    for i, cfg in enumerate(attempts, start=1):
        raw = call_llm_simple(ctx, cfg["prompt"], ask,
                              temperature=cfg["temperature"],
                              max_tokens=cfg["max_tokens"],
                              force_think=cfg["force_think"], json_schema=DECK_SCHEMA) or ""
        got = safe_json_from_llm(raw, required_keys=("slides",))
        if isinstance(got, dict) and got.get("slides"):
            if i > 1:
                logger.info("deck planner recovered on attempt %d", i)
            data = got
            break
        logger.warning("deck planner attempt %d produced no usable JSON (%d chars)",
                       i, len(raw))
        if ctx is not None and getattr(ctx, "is_cancelled", bool)():
            break
    planner_failed = not (isinstance(data, dict) and data.get("slides"))
    if planner_failed:
        logger.error("deck planner failed on every attempt — no deck content")
        data = data if isinstance(data, dict) else {}
    deck = normalize_deck(data, topic, n_slides=n_slides)

    # The prompt ASKS for figures, charts and links; the audit checks whether it
    # got them, because on this house's models an unchecked instruction is a
    # suggestion. One repair round only -- it costs a turn, and a model that
    # ignored the floor twice will ignore it a third time.
    audit = deck_facts_audit(deck)
    logger.info("deck audit: %s", audit)
    if not audit["ok"] and deck.get("slides") and (
            ctx is None or not getattr(ctx, "is_cancelled", bool)()):
        import json as _json
        try:
            payload = _json.dumps(deck, ensure_ascii=False)[:9000]
        except Exception:
            payload = ""
        if payload:
            raw = call_llm_simple(ctx, _REPAIR_PROMPT,
                                  (facts and ("Research notes:\n" + facts + "\n\n") or "")
                                  + "Deck to rewrite:\n" + payload,
                                  temperature=0.5, max_tokens=8000,
                                  force_think=False, json_schema=DECK_SCHEMA) or ""
            got = safe_json_from_llm(raw, required_keys=("slides",))
            if isinstance(got, dict) and got.get("slides"):
                repaired = normalize_deck(got, topic, n_slides=n_slides)
                after = deck_facts_audit(repaired)
                # Keep the repair only if it is genuinely richer. A rewrite that
                # loses slides or figures is a regression, and the model does
                # sometimes return a shorter, emptier deck.
                if (len(repaired["slides"]) >= len(deck["slides"])
                        and after["slides_with_figures"] >= audit["slides_with_figures"]
                        and after["charts"] >= audit["charts"]):
                    repaired["sources"] = repaired["sources"] or deck["sources"]
                    deck, audit = repaired, after
                    logger.info("deck repair accepted: %s", after)
                else:
                    logger.info("deck repair rejected as thinner: %s", after)

    _drop_ungrounded_charts(deck, facts + " " + (topic or ""))
    if facts:
        _drop_ungrounded_figures(deck, facts + " " + (topic or ""))
        _drop_unsupported_claims(ctx, deck, facts)

    # Last resort for the links: the search hits themselves are real URLs on the
    # topic, which beats a sources slide that is simply absent.
    # A bare domain is not a citation. Point each source at the page the search
    # actually found on it, when there is exactly one.
    deck["sources"], _fixed = repair_sources(deck.get("sources") or [], web_sources)
    if _fixed:
        logger.info("deck sources: %d bare domains repaired to the page found", _fixed)
    if len(deck.get("sources") or []) < 2 and web_sources:
        deck["sources"] = _clean_sources((deck.get("sources") or []) + web_sources)
        logger.info("deck sources backfilled from search (%d)", len(deck["sources"]))
    # Set LAST: the repair round above replaces the deck object wholesale, and
    # a flag written before it was silently dropped. Remembered rather than
    # inferred -- a ONE-slide deck can be exactly what was asked for, while a
    # deck whose planner never answered is not a deck at all, and only the
    # planner knows which happened.
    deck["planner_failed"] = planner_failed
    return deck


def _slug_title(s: str) -> str:
    """Loose title match: punctuation and case must not hide a duplicate."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


_URL_RE = re.compile(r"^https?://[^\s<>\"']{6,}$", re.I)
_NUM_RE = re.compile(r"\d")

MAX_STATS = 4
MAX_CHART_POINTS = 12
MAX_SOURCES = 8


def _is_figure(value: str) -> bool:
    """True when a stat VALUE is a measurement rather than a label.

    "a digit somewhere" is too weak a test. Measured on a live deck: the
    planner, asked for a stat band, produced `1 — первый колесный гигант` --
    an ordinal crowbarred into the value slot to satisfy the rule, sitting on
    the slide in 26pt beside a real "1961". A figure carries either a UNIT
    (%, ₽, л.с., т, км) or enough digits to be a quantity or a year.
    """
    v = (value or "").strip()
    # Filler dressed as a figure: a sleep deck led with «24/7 цикл
    # восстановления» and «100% необходимость для мозга» (live 2026-09-29).
    if v.replace(" ", "").lower() in {"24/7", "24/7/365", "100%", "360°", "365", "№1", "#1"}:
        return False
    digits = sum(1 for ch in v if ch.isdigit())
    if not digits:
        return False
    has_unit = any(ch.isalpha() or ch in "%°₽$€£" for ch in v)
    return bool(has_unit or digits >= 2)


def _clean_stats(raw) -> list:
    """Two to four short value/label pairs, or nothing.

    A stat band with one entry looks like a mistake and one with eight is a
    table, so the band is dropped rather than rendered badly. A `value` with no
    digit in it is not a figure -- it is a bullet that wandered into the wrong
    key -- and is discarded.
    """
    out = []
    for item in (raw or [])[:MAX_STATS + 2]:
        if isinstance(item, dict):
            value = str(item.get("value") or item.get("number") or "").strip()
            label = str(item.get("label") or item.get("name") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            value, label = str(item[0]).strip(), str(item[1]).strip()
        else:
            continue
        if not value or not _is_figure(value):
            continue
        out.append({"value": value[:14], "label": label[:44]})
        if len(out) >= MAX_STATS:
            break
    return out if len(out) >= 2 else []


def _to_number(v):
    """A plain float from whatever the model wrote, or None.

    Models return "12 100", "12,100", "12100 units" and "12.1" for the same
    quantity. Anything still ambiguous after this is refused, because a chart
    drawn from a misread number is worse than no chart.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("\u00a0", " ")
    s = re.sub(r"(?<=\d)[ ](?=\d)", "", s)            # thousands as spaces
    s = re.sub(r"(?<=\d),(?=\d\d\d\b)", "", s)        # thousands as commas
    s = s.replace(",", ".")                           # decimal comma
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _clean_chart(raw) -> dict:
    """A chart we can actually draw, or {}.

    Every check here has been earned by a way a plan can be wrong: labels and
    values of different lengths, values as strings with units glued on, a
    one-point "chart", a pie of negative numbers, and a series that is entirely
    zeros. A bad chart is dropped silently -- the slide keeps its text.
    """
    if not isinstance(raw, dict):
        return {}
    labels = [str(x).strip()[:28] for x in (raw.get("labels") or []) if str(x).strip()]
    values = [_to_number(x) for x in (raw.get("values") or [])]
    n = min(len(labels), len(values))
    labels, values = labels[:n], values[:n]
    keep = [i for i, v in enumerate(values) if v is not None]
    if len(keep) < 2:
        return {}
    labels = [labels[i] for i in keep][:MAX_CHART_POINTS]
    values = [values[i] for i in keep][:MAX_CHART_POINTS]
    if not any(v != 0 for v in values):
        return {}
    if (re.search(r"год|year", str(raw.get("unit") or ""), re.I)
            or re.search(r"дат|date|хронолог|timeline|летопис", str(raw.get("title") or ""), re.I)
            or any(lb == f"{v:g}" for lb, v in zip(labels, values))) \
            and all(float(v).is_integer() and 800 <= v <= 2100 for v in values):
        # Years are a timeline, not a quantity: "Даты гибели князя" drew 972 vs 973 as bars.
        return {}
    # "Оперативная память (ГБ)" vs "Ядра нейропроцессора (ед)" on one axis compares nothing.
    if len({u.lower() for u in re.findall(r"\(([^()]{1,12})\)\s*$", "\n".join(labels), re.M)}) > 1:
        return {}
    kind = str(raw.get("kind") or raw.get("type") or "bar").strip().lower()
    if kind not in ("bar", "line", "pie"):
        kind = "bar"
    if kind == "pie" and any(v <= 0 for v in values):
        kind = "bar"                                  # a pie of negatives is nonsense
    src_url = str(raw.get("source") or "").strip()
    return {"title": str(raw.get("title") or "").strip()[:80],
            "kind": kind, "labels": labels, "values": values,
            "unit": str(raw.get("unit") or "").strip()[:24],
            "source": src_url if _URL_RE.match(src_url) else ""}


def _clean_table(raw) -> dict:
    """{"columns": [...], "rows": [[...]]} cut to 5x8 short cells, or {}.
    «добавь слайд с таблицей характеристик» made a slide titled so, with no table."""
    if not isinstance(raw, dict):
        return {}
    cols = [str(c).strip()[:40] for c in (raw.get("columns") or [])][:5]
    rows = [[str(c).strip()[:60] for c in r][:len(cols)] + [""] * (len(cols) - len(r))
            for r in (raw.get("rows") or []) if isinstance(r, list)][:8]
    return {"columns": cols, "rows": rows} if len(cols) >= 2 and rows else {}


def _drop_ungrounded_charts(deck: dict, facts: str) -> None:
    """A chart is data or nothing. The audit's chart floor made the planner draw
    «100/100/100/100 % охвата маршрутов» and a 40/30/30 pie of Viking trade
    with no source: kept only when most of its numbers are in the notes."""
    text = re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", facts or "").lower()

    def _near(value: str, label: str) -> bool:
        # The number next to a word of its own label: small numbers alone are in any notes.
        word = r"\w{4,}|[a-zа-я]+\d+"          # «Q1» is a label word too
        stems = {w[:5] for w in re.findall(word, label.lower())}
        for m in re.finditer(r"(?<![\d.,])" + re.escape(value).replace(r"\.", "[.,]") + r"(?![\d])", text):
            if {w[:5] for w in re.findall(word, text[max(0, m.start() - 150):m.end() + 150])} & stems:
                return True
        return False

    for s in deck.get("slides") or []:
        ch = s.get("chart") or {}
        vals = ch.get("values") or []
        if not vals:
            continue
        shown = [f"{v:g}" for v in vals]
        labels = [str(x) for x in ch.get("labels") or []] + [""] * len(shown)
        if len(set(shown)) < 2 or sum(_near(v, lb) for v, lb in zip(shown, labels)) * 2 < len(shown):
            logger.info("deck: dropped a chart its notes do not support: %r", s["chart"].get("title"))
            s["chart"] = {}


def _blend(rgb: int, toward: int, t: float) -> int:
    """`rgb` mixed `t` of the way toward another colour. Used for tints of a
    theme colour, so a gridline is the body colour faded rather than a second
    grey chosen by hand."""
    t = max(0.0, min(1.0, float(t)))
    out = 0
    for shift in (16, 8, 0):
        a = (rgb >> shift) & 0xFF
        b = (toward >> shift) & 0xFF
        out |= int(round(a + (b - a) * t)) << shift
    return out


def _is_light(rgb: int) -> bool:
    """Whether dark text reads better on this colour than white does.

    Rec. 601 luma, thresholded at 0.62 rather than the usual 0.5: the shades
    here are warm and mid-tones read as lighter than their luma suggests.
    """
    r, g, b = (rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 > 0.62


def _on_panel(th: dict) -> int:
    return th["head"] if _is_light(th["panel"]) else 0xFFFFFF


def _pie_shades(accent: int, n: int) -> list:
    """`n` distinguishable shades of the deck's accent colour.

    A pie needs one colour per slice and the deck has one accent, so the accent
    is walked from darker to lighter. Blending toward white rather than
    rotating the hue keeps the chart recognisably part of THIS deck -- the
    point of the exercise was that PowerPoint's default blue/red/green looked
    like a chart pasted in from somewhere else.
    """
    n = max(1, int(n))
    r, g, b = (accent >> 16) & 0xFF, (accent >> 8) & 0xFF, accent & 0xFF
    out = []
    for i in range(n):
        # 0.0 for the first slice (a touch darker than the accent), rising to
        # 0.62 for the last -- beyond that the lightest slice loses its outline.
        t = (i / float(max(1, n - 1))) * 0.62 if n > 1 else 0.0
        dark = 0.82 + 0.18 * (1 - t)          # first slice slightly deepened
        cr = int(min(255, (r * dark) + (255 - r * dark) * t))
        cg = int(min(255, (g * dark) + (255 - g * dark) * t))
        cb = int(min(255, (b * dark) + (255 - b * dark) * t))
        out.append((cr << 16) | (cg << 8) | cb)
    return out


def _is_scene(path: str) -> bool:
    """Is this picture usable as a full-bleed COVER?

    A shop packshot is not: measured live, the best hit for a vitamin-D deck was
    one brand's carton on a white background, which reads as an advertisement
    and swallows the cover's white text. A plain themed cover is better than
    that, so a cut-out is refused and the deck falls back to it.

    Only the cover is judged this way -- a product photograph can be exactly the
    right illustration for a slide about that product.
    """
    try:
        import search
        if search.looks_like_cutout(str(path)):
            logger.info("cover image %r is a cut-out on white — using the plain "
                        "cover instead", os.path.basename(str(path)))
            return False
    except Exception:
        logger.debug("could not judge the cover image", exc_info=True)
    return True


def _chart_title_with_unit(chart: dict) -> str:
    """The chart's title, carrying its unit.

    The unit was collected, validated, and then shown nowhere: it becomes the
    SERIES name, and bar and line charts have their legend switched off, so a
    chart of percentages rendered as bars of 42, 55 and 71 with no "%" anywhere
    on the slide. Seen on an exported clinical deck.

    The title is where a reader looks for it. Skipped when the title already
    says the unit, so "Доля, %" never becomes "Доля, %, %".
    """
    title = str(chart.get("title") or "").strip()
    unit = str(chart.get("unit") or "").strip()
    if not unit:
        return title
    if not title:
        return ""            # "%" alone is not a title; leave the chart untitled
    low_t, low_u = title.lower(), unit.lower()
    if low_u in low_t:
        return title
    return "%s, %s" % (title.rstrip(" ,;:"), unit)


def _clean_sources(raw) -> list:
    """Real absolute http(s) links, deduplicated, titled.

    The planner cheerfully emits "see Wikipedia" and bare domains; neither is
    clickable, so neither survives.
    """
    out, seen = [], set()
    for item in (raw or []):
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or "").strip()
            title = str(item.get("title") or "").strip()
        else:
            url, title = str(item).strip(), ""
        if not _URL_RE.match(url) or url.lower() in seen:
            continue
        seen.add(url.lower())
        out.append({"title": (title or url)[:110], "url": url})
        if len(out) >= MAX_SOURCES:
            break
    return out


def _url_key(url: str) -> str:
    """Compare URLs the way a reader would: scheme, www and a trailing slash do
    not make it a different page."""
    u = re.sub(r"^https?://", "", (url or "").strip().rstrip("/"), flags=re.I)
    return re.sub(r"^www\.", "", u, flags=re.I).lower()


def repair_sources(sources: list, collected: list) -> tuple:
    """Point each source at the page we actually found on that domain.

    Observed on a live deck: the planner listed `news.ru`, `igrader.ru` and
    `kirovets-ptz.com` -- bare site roots -- while the search that fed it had
    returned the specific articles on exactly those domains. A root link is not
    a citation: the reader lands on a homepage and cannot tell which page
    carried the figure on the slide.

    Only substitutes when the domain has exactly ONE collected page, so an
    ambiguous domain keeps whatever the planner wrote. Titles are the
    planner's; only the address changes.

    Returns (sources, repaired_count).
    """
    if not sources or not collected:
        return sources, 0
    known = {_url_key(s["url"]) for s in collected}
    by_dom: dict = {}
    for s in collected:
        by_dom.setdefault(_url_key(s["url"]).split("/")[0], set()).add(s["url"])
    out, fixed, seen = [], 0, set()
    for s in sources:
        key = _url_key(s["url"])
        if key in known:
            if key not in seen:
                seen.add(key)
                out.append(s)                   # a page we actually read
            continue
        same = by_dom.get(key.split("/")[0]) or set()
        if len(same) == 1:
            url = next(iter(same))
            # Two planner entries can repair to the SAME page; print it once.
            if _url_key(url) in seen:
                continue
            seen.add(_url_key(url))
            out.append({"title": s["title"], "url": url})
            fixed += 1
        elif same and key not in seen:
            seen.add(key)
            out.append(s)                   # ambiguous domain: do not guess
        else:
            # Nothing was collected on this domain at all, so the address
            # cannot be checked. Dropped rather than printed: a deck whose
            # sources slide 404s is worse than one with fewer citations, and
            # the search results backfill the list below.
            fixed += 0
    return out, fixed


def _numbers(text: str) -> list:
    text = re.sub(r"(?<=\d)(?:[\s ]|,)(?=\d{3}\b)", "", text or "")   # 12 100 / 12,100
    return [float(m.replace(",", ".")) for m in re.findall(r"(?<![\w.,])\d+(?:[.,]\d+)?", text)]


def _close(n: float, k: float) -> bool:
    """A year is exact; an amount may be rounded."""
    return abs(n - k) <= (0.05 if 1000 <= k <= 2100 and k == int(k) else max(0.02 * abs(k), 0.05))


def _drop_ungrounded_figures(deck: dict, facts: str) -> None:
    """A bullet or stat whose figure is in none of the notes was made up by the planner
    (a deck on Tesla put its founding in 2008): out, not shown as fact. Rounding counts
    as found; a one-digit number is too common to judge."""
    known = _numbers(facts)
    if not known:
        return

    def grounded(text: str, small: bool = False) -> bool:
        return all(any(_close(n, k) for k in known)
                   for n in _numbers(text) if small or n >= 10 or n != int(n))

    for s in deck.get("slides") or []:
        bl = s.get("bullets") or []
        keep = [b for b in bl if grounded(b)]
        if len(keep) < len(bl):
            logger.info("deck: dropped %d bullet(s) with figures from no note: %r",
                        len(bl) - len(keep), [b for b in bl if b not in keep])
            s["bullets"] = keep
        if s.get("stats"):
            s["stats"] = [st for st in s["stats"] if grounded(str(st.get("value", "")), small=True)]


def _drop_unsupported_claims(ctx, deck: dict, facts: str) -> None:
    """The number is in the notes but the sentence around it is not («выручка 96,8 млрд
    в 2021» from a 2023 figure): MiniCheck reads each figure bullet against the notes."""
    import fact_check
    import json
    from llm import call_llm_simple
    refs = [(s, b) for s in deck.get("slides") or [] for b in s.get("bullets") or []
            if re.search(r"\d{2}", b)]
    if not refs:
        return
    claims = [b for _, b in refs]
    # MiniCheck reads English only: Russian notes against English claims scored an
    # invented «merged with Uber» 0.7. Both sides go through the same translation.
    notes = [ln for ln in facts.splitlines() if ln.strip()]
    texts = notes + claims
    if any(not t.isascii() for t in texts):
        raw = call_llm_simple(ctx, "Translate each string of the JSON list into English, literally. "
                                   "Write money as $N billion / $N million. Answer with the JSON "
                                   "list only, same length.",
                              json.dumps(texts, ensure_ascii=False), temperature=0.0,
                              max_tokens=12000, force_think=False,
                              json_schema={"type": "array", "items": {"type": "string"},
                                           "minItems": len(texts), "maxItems": len(texts)}) or ""
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            en = json.loads(m.group(0)) if m else None
        except ValueError:
            en = None
        if not (isinstance(en, list) and len(en) == len(texts)):
            return
        en = [str(x) for x in en]
        facts, claims = "\n".join(en[:len(notes)]), en[len(notes):]
    # The checker is literal: "about 1.8 million" against the notes' 1.81 scored 0.10,
    # with 1.81 it scored 0.94. The note's own figure goes in before the check.
    known = _numbers(facts)

    def exact(m):
        n = _numbers(m.group(0))
        k = next((k for k in known if n and _close(n[0], k)), None)
        return f"{k:g}" if k is not None else m.group(0)
    claims = [re.sub(r"\d[\d,]*(?:\.\d+)?", exact, c) for c in claims]
    scores = fact_check.support(claims, facts)
    if not scores:
        return
    logger.info("deck claims checked: %s", [(round(p, 2), c) for p, c in zip(scores, claims)])
    for (s, b), p in zip(refs, scores):
        # Measured: invented facts 0.01-0.04, true ones with the notes' figures 0.11+;
        # a paraphrase the 770M checker misses sits just above. Only the clear misses go.
        if p < 0.08 and b in s["bullets"]:
            logger.info("deck: dropped a bullet the notes do not support (%.2f): %r", p, b)
            s["bullets"].remove(b)


def deck_facts_audit(deck: dict) -> dict:
    """How informative the deck actually is, in counts rather than in hope.

    The planner is ASKED for figures, charts and links; this is what it DID.
    `plan_deck` uses the verdict to decide whether one repair round is worth a
    turn, and the numbers go into the log either way, so a deck that quietly got
    thinner is visible without opening the file.
    """
    slides = deck.get("slides") or []
    with_num = sum(1 for s in slides
                   if any(_NUM_RE.search(b) for b in (s.get("bullets") or []))
                   or s.get("stats"))
    charts = sum(1 for s in slides if s.get("chart"))
    n = len(slides)
    return {"slides": n, "slides_with_figures": with_num, "charts": charts,
            "sources": len(deck.get("sources") or []),
            "ok": bool(n) and with_num * 2 >= n and charts >= 1
            and len(deck.get("sources") or []) >= 2}


def normalize_deck(data: dict, topic: str = "", n_slides: int = 0) -> dict:
    """Coerce whatever the model produced into the shape build_pptx expects.

    Never rejects: a deck with one bad slide should lose that slide, not the deck.
    """
    out = {"title": "", "subtitle": "", "theme": "", "cover_image": "",
           "slides": [], "sources": []}
    if isinstance(data, dict):
        out["title"] = str(data.get("title") or "").strip()
        out["subtitle"] = str(data.get("subtitle") or "").strip()
        _th = str(data.get("theme") or "").strip().lower()
        out["theme"] = _th if _th in THEMES else ""
        out["cover_image"] = str(data.get("cover_image") or "").strip()
        out["sources"] = _clean_sources(data.get("sources"))
        for raw in (data.get("slides") or [])[:MAX_SLIDES]:
            if not isinstance(raw, dict):
                continue
            heading = str(raw.get("heading") or raw.get("title") or "").strip()
            bullets = raw.get("bullets") or raw.get("points") or []
            if isinstance(bullets, str):
                bullets = [b.strip(" -•\t") for b in bullets.splitlines() if b.strip()]
            bullets = [str(b).strip() for b in bullets if str(b).strip()][:MAX_BULLETS]
            if not heading and not bullets:
                continue
            out["slides"].append({
                "heading": heading or (topic or "Slide").strip()[:80],
                "bullets": bullets,
                "stats": _clean_stats(raw.get("stats")),
                "chart": _clean_chart(raw.get("chart")),
                "table": _clean_table(raw.get("table")),
                "notes": str(raw.get("notes") or "").strip(),
                "image": str(raw.get("image") or raw.get("image_query") or "").strip(),
            })
    # Planners often attach links per slide instead of at the deck level. They
    # are the same links; hoist them so the sources slide sees them.
    if not out["sources"]:
        lifted = []
        for raw in ((data.get("slides") or []) if isinstance(data, dict) else []):
            if isinstance(raw, dict):
                lifted.extend(raw.get("sources") or [])
        out["sources"] = _clean_sources(lifted)
    if not out["title"]:
        out["title"] = (topic or "Presentation").strip()[:120]
    # Drop a content slide that just repeats the title slide. The planner likes to
    # open with a slide whose heading IS the deck title, which reads as a
    # duplicate of the cover the builder already makes.
    _t_norm = _slug_title(out["title"])
    if len(out["slides"]) > 1 and _slug_title(out["slides"][0]["heading"]) == _t_norm:
        out["slides"].pop(0)
    # ...or is literally NAMED the title slide ("Титульный лист" as content
    # slide 1 ate half of a 4-slide deck, live 2026-09-28), and an edit that
    # repeats a slide instead of removing one ("убери последний слайд" came
    # back with "Древние истоки" twice).
    _seen = set()
    _kept = []
    for s in out["slides"]:
        h = _slug_title(s["heading"])
        if re.fullmatch(r"(?:титульный\s*(?:лист|слайд)|title\s*slide|обложка|cover)", s["heading"].strip().lower()) \
                or (h and h in _seen):
            continue
        _seen.add(h)
        _kept.append(s)
    if _kept:
        out["slides"] = _kept
    # "Exactly N slides" was only ever ASKED for in the prompt and never enforced:
    # a request for 6 came back as 10. The count the user gave is a requirement,
    # not a hint — honour it, counting the cover the builder adds.
    if n_slides and n_slides > 0:
        # The cover AND the closing sources slide both count: "4 slides" used
        # to come back as cover + 3 + sources = 5 (live, 2026-09-12).
        closing = 1 if out.get("sources") else 0
        want_content = min(int(n_slides), MAX_SLIDES) - 1 - closing
        if want_content < 2 and closing:
            # A tiny deck ("3 слайда") cannot spend a third of itself on a
            # sources slide: cover + ONE content slide + sources was refused
            # downstream as a stub (live, 2026-09-12, journey 19). The sources
            # move into the speaker notes of the last slide instead.
            closing = 0
            want_content = min(int(n_slides), MAX_SLIDES) - 1
            out["sources_in_notes"] = True
        want_content = max(2, want_content)
        if len(out["slides"]) > want_content:
            logger.info("deck planner returned %d slides for a request of %d — trimming",
                        len(out["slides"]) + 1, n_slides)
            out["slides"] = out["slides"][:want_content]
    if not out["slides"]:
        # An empty plan still has to produce a file the user can open and edit.
        out["slides"] = [{"heading": out["title"], "bullets": [], "stats": [],
                          "chart": {}, "notes": "", "image": ""}]
    return out


_SOURCES_HEADING = {"ru": "Источники", "en": "Sources"}


def _deck_lang(deck: dict) -> str:
    """"ru" if the deck is written in Cyrillic, else "en".

    The builder writes exactly one piece of text of its own -- the sources
    heading -- and an English word on an otherwise Russian deck looks like a
    bug. Measured off the deck's own text rather than passed in, so the caller
    cannot forget it.
    """
    text = " ".join([str(deck.get("title") or "")]
                    + [str(s.get("heading") or "") for s in (deck.get("slides") or [])])
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    return "ru" if cyr * 3 >= sum(1 for ch in text if ch.isalpha()) else "en"


def _short_url(url: str, limit: int = 72) -> str:
    """A URL that fits on a slide: no scheme, no www, elided in the middle."""
    s = re.sub(r"^https?://(www\.)?", "", (url or "").strip()).rstrip("/")
    if len(s) <= limit:
        return s
    return s[:limit - 14] + "..." + s[-11:]


# Words that identify nothing on their own. A query anchored with "история" is
# anchored to every topic there is.
_STOP_ANCHOR = {
    "история", "истории", "развитие", "обзор", "введение", "презентация", "тема",
    "the", "a", "an", "of", "and", "history", "overview", "introduction",
    "presentation", "about", "story", "modern", "future", "today",
}


def _anchor_terms(text: str) -> list:
    """The words in a title that actually identify the subject.

    A model designation ("К-700", "X5") identifies it best, so anything carrying
    a digit is kept whatever its length; otherwise a word has to be long enough
    to mean something and not be a filler like "история".
    """
    words = re.findall(r"[\w\-]+", (text or ""), flags=re.U)
    out = []
    for w in words:
        low = w.lower()
        if low in _STOP_ANCHOR:
            continue
        if any(ch.isdigit() for ch in w) or len(w) >= 5:
            out.append(w)
            continue
        # A one- or two-letter DESIGNATOR immediately after a kept word is part
        # of the subject's name, not filler: "Витамин D" searched as "Витамин"
        # is a different subject, and that is how a deck about vitamin D came
        # back with no cover picture at all. Only uppercase, only adjacent, so
        # a stray conjunction cannot slip through.
        if out and len(w) <= 2 and w.isupper() and w.isalpha():
            out.append(w)
    return out


# Words that describe a PHOTOGRAPH rather than its subject. A planner writes
# image queries like a prompt for an illustrator; an image search reads them as
# stock-photo captions.
_SCENE_WORDS = {
    "wide", "shot", "closeup", "close", "up", "view", "detail", "details",
    "photo", "photograph", "photography", "image", "picture", "scene",
    "during", "sunset", "sunrise", "golden", "hour", "dramatic", "cinematic",
    "beautiful", "stunning", "vast", "massive", "huge", "working", "background",
    "landscape", "portrait", "aerial", "modern", "vintage", "retro", "old",
    "фото", "фотография", "снимок", "вид", "крупный", "план", "закат",
    "красивый", "огромный", "широкий", "кадр", "кадре", "ракурс", "фон",
    "яркий", "детальный", "реалистичный", "профессиональный",
}

_IMAGE_QUERY_WORDS = 6


def _image_query(query: str, subject: str) -> str:
    """A SHORT, subject-led query, because a long one finds stock photography.

    Measured against the live image search with the same subject three ways:

      "…heavy duty agricultural tractor working in a vast field during sunset
       wide shot Кировец"          -> 3 hits, all vecteezy/freepik stock
      "Кировец …" (anchor moved to the front)  -> 3 hits, the same stock
      "Кировец К-700 трактор"      -> 6 hits, every one an actual Kirovets
                                      (the factory's own site among them)

    So the lever is not WHERE the subject sits, it is how much scene prose
    surrounds it: a descriptive phrase matches the captions stock agencies
    write, while a bare subject matches photographs of the thing. The planner
    writes prose because it is describing a picture it imagines; this keeps its
    two or three content words and drops the photography vocabulary.
    """
    query = (query or "").strip()
    terms = _anchor_terms(subject)
    strong = [t for t in terms if any(ch.isdigit() for ch in t) or t[:1].isupper()]
    lead = (strong or terms)[:2]
    seen = {w.lower() for w in lead}
    extra = []
    for w in re.findall(r"[\w\-]+", query, flags=re.U):
        low = w.lower()
        if low in _SCENE_WORDS or low in seen or len(w) < 3:
            continue
        seen.add(low)
        extra.append(w)
        if len(lead) + len(extra) >= _IMAGE_QUERY_WORDS:
            break
    out = " ".join(lead + extra).strip()
    return out or query


def _safe_name(text: str, fallback: str = "deck") -> str:
    name = re.sub(r"[^\w\- ]+", "", (text or "")).strip()[:60]
    return name or fallback


def _rgb(v):
    from pptx.dml.color import RGBColor
    return RGBColor((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)


def _fill_rect(slide, x, y, w, h, colour, alpha=None):
    """A flat rectangle. `alpha` 0..100 = percent OPAQUE (python-pptx has no
    transparency API, so the alpha element is written into the fill XML)."""
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Emu
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x)), Emu(int(y)),
                                 Emu(int(w)), Emu(int(h)))
    shp.shadow.inherit = False
    shp.line.fill.background()
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(colour)
    if alpha is not None:
        try:
            from lxml import etree
            _A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
            srgb = shp.fill.fore_color._xFill.find(_A + "srgbClr")
            if srgb is not None:
                el = etree.SubElement(srgb, _A + "alpha")
                el.set("val", str(int(alpha * 1000)))
        except Exception:
            logger.debug("could not set fill transparency", exc_info=True)
    return shp


def _resolve_image(path) -> str:
    """A usable local path for a slide/cover image, or "" if there is none.
    Shared existence check + format guard used by both the cover and the
    per-slide picture lookup below."""
    return _embeddable(str(path)) if path and os.path.exists(str(path)) else ""


def _embeddable(path: str) -> str:
    """A path python-pptx can actually embed.

    OOXML (and python-pptx) accept BMP/GIF/JPEG/PNG/TIFF/WMF only. Preferring
    Wikimedia and Unsplash for picture QUALITY started returning .webp, and the
    whole deck build then died with "unsupported image format ... got 'WEBP'".
    Transcode instead of losing the deck. Returns the original path when it is
    already fine, or "" if the file cannot be read at all.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
            if fmt in ("BMP", "GIF", "JPEG", "PNG", "TIFF"):
                return path
            rgb = im.convert("RGB")
            out = os.path.splitext(path)[0] + "_conv.png"
            rgb.save(out, "PNG")
        logger.info("transcoded %s image for the deck: %s", fmt or "?",
                    os.path.basename(out))
        return out
    except Exception:
        logger.warning("unusable image %s — the slide keeps its text", path,
                       exc_info=True)
        return ""


def _aspect_fill(pic, box_w, box_h, img_path):
    """Crop a placed picture so it FILLS its box instead of being stretched.

    Stretching a 4:3 photo across a 16:9 panel is the most obvious sign of an
    auto-generated deck, so the overflow is cropped away instead.
    """
    from PIL import Image
    with Image.open(str(img_path)) as im:
        iw, ih = im.size
    if not iw or not ih:
        return
    target, source = box_w / box_h, iw / ih
    if abs(target - source) < 0.01:
        return
    if source > target:                     # wider than the box: trim the sides
        cut = (1.0 - target / source) / 2
        pic.crop_left = cut
        pic.crop_right = cut
    else:                                   # taller: trim top and bottom
        cut = (1.0 - source / target) / 2
        pic.crop_top = cut
        pic.crop_bottom = cut


# The body column: from just under the title rule down to the footer. Both
# numbers are read by _fit_bullets, so the type can never be sized against a
# box the builder does not actually use.
_MIN_CHART_IN = 1.5
_PIC_FRAC = 0.40          # the photo panel's share of the slide width
_PIC_GAP_IN = 0.45        # breathing room between the text column and the photo
_BODY_TOP_IN = 2.05
_BODY_BOTTOM_IN = 6.55


def _fit_bullets(bullets: list, width_in: float, height_in: float) -> tuple:
    """(font size in pt, space-after in pt) that makes the bullets FILL the column.

    A fixed 17pt with a fixed 11pt gap is right for one slide and wrong for
    every other one: three short bullets leave four inches of nothing, and six
    long ones overflow into the footer. So the type is sized to the text there
    actually is.

    The estimate is deliberately crude -- Segoe UI averages about half its point
    size per character, which is enough to count wrapped lines to within one --
    and it only ever has to choose between six candidate sizes. It walks from
    largest to smallest and takes the first that fits, so a sparse slide gets big
    type and a dense one stays inside the column.
    """
    if not bullets:
        return 17, 11
    # The ceiling depends on HOW MANY bullets there are, not only on whether the
    # text fits. Measured twice:
    #   * at a flat ceiling of 24pt, two bullets filled the top fifth of a 4.5in
    #     column and left the rest of the slide blank -- the fitter could shrink
    #     for a dense slide but never grow for a sparse one;
    #   * at a flat ceiling of 30pt, SIX bullets also fit, and a wall of huge
    #     type is a worse slide than an empty one. Fitting is not the same as
    #     reading well.
    # So a sparse slide is allowed to grow and a dense one is not.
    ceiling = _BULLET_CEILING.get(len(bullets), _BULLET_CEILING["many"])
    for size in [p for p in _BULLET_PT if p <= ceiling]:
        gap = max(8, round(size * 0.62))
        if _bullets_height(bullets, width_in, size, gap) <= height_in * 0.98:
            return size, gap
    return 14, 8


_MIN_BULLET_PT = 14


_HEADING_PT = (27, 24, 21, 18)
_STAT_PT = (26, 23, 20, 18, 16, 14)


def _fit_stat(value: str, width_in: float) -> int:
    """The largest size at which a headline figure still fits its card on ONE
    line. Bold and wide-set, so ~0.60 of the point size per character."""
    text = (value or "").strip()
    if not text:
        return _STAT_PT[0]
    for size in _STAT_PT:
        if len(text) * size * 0.60 / 72.0 <= width_in:
            return size
    return _STAT_PT[-1]


_BULLET_PT = (30, 27, 24, 22, 20, 18, 17, 15, 14)
# Read off rendered slides: two bullets carry 30pt comfortably, six do not.
_BULLET_CEILING = {1: 30, 2: 30, 3: 27, 4: 24, 5: 22, "many": 22}


def _fit_heading(text: str, width_in: float, height_in: float) -> int:
    """The largest heading size whose wrapped lines fit the title band.

    A heading is bold, so it runs wider per character than body text; 0.56 of
    the point size per character matches the measured wrap of Segoe UI Semibold
    to within one line on the headings this deck produces.
    """
    text = (text or "").strip()
    if not text:
        return _HEADING_PT[0]
    for size in _HEADING_PT:
        chars_per_line = max(8, int((width_in * 72.0) / (size * 0.56)))
        lines = max(1, -(-len(text) // chars_per_line))
        if (lines * size * 1.16) / 72.0 <= height_in:
            return size
    return _HEADING_PT[-1]


def _bullets_height(bullets: list, width_in: float, size: int, gap: int) -> float:
    """Inches the bullets need at this size. Crude on purpose -- Segoe UI averages
    about half its point size per character, enough to count wrapped lines to
    within one."""
    lines = 0
    chars_per_line = max(12, int((width_in * 72.0) / (size * 0.5)))
    for b in bullets:
        text = len(b) + 3                          # the bullet glyph and its space
        lines += max(1, -(-text // chars_per_line))
    # 1.15 line spacing, plus the gap after every bullet but the last
    return (lines * size * 1.15 + (max(1, len(bullets)) - 1) * gap) / 72.0


def _theme_hyperlink_colour(prs, colour: int) -> bool:
    """Repaint the theme's hyperlink colours to the deck's accent.

    PowerPoint does not use the colour set on a hyperlink RUN: it takes
    `<a:hlink>` from the slide master's theme, which in python-pptx's default
    template is 0000FF, with 800080 for a followed link. So a sources slide on a
    warm brown deck came out in Office blue and, after one click, purple --
    measured on a rendered deck, not assumed.

    Returns True when the theme was repainted, False when the part could not be
    found or edited (in which case the links are merely off-palette, so this
    must never raise).
    """
    _REL = ("http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/theme")
    hex6 = "%06X" % (int(colour) & 0xFFFFFF)
    try:
        for master in prs.slide_masters:
            part = master.part.part_related_by(_REL)
            xml = part.blob.decode("utf-8")
            new = re.sub(r"(<a:(?:fol)?[Hh]link>\s*<a:srgbClr val=\")[0-9A-Fa-f]{6}",
                         lambda m: m.group(1) + hex6, xml)
            if new == xml:
                return False
            part._blob = new.encode("utf-8")
        return True
    except Exception:
        logger.debug("could not repaint the theme hyperlink colour", exc_info=True)
        return False


def build_pptx(deck: dict, out_path: str,
               images: Optional[dict] = None, topic: str = "") -> Optional[str]:
    """Assemble a DESIGNED .pptx.

    `images` maps slide index -> local image path, plus an optional "cover" key
    for the full-bleed title background.

    Returns the path written, or None if python-pptx is not installed. A slide
    whose picture fails to place keeps its text — a broken image must never cost
    the content.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
        from pptx.chart.data import CategoryChartData
        from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    except ImportError:
        logger.error("python-pptx is not installed — cannot build a deck")
        return None

    images = images or {}
    th = pick_theme(deck, topic)
    prs = Presentation()
    prs.slide_width = Inches(_SLIDE_W_IN)
    prs.slide_height = Inches(_SLIDE_H_IN)
    W, H = prs.slide_width, prs.slide_height
    blank = prs.slide_layouts[6]        # fully blank: we place everything ourselves
    # Links are painted by the THEME, not by the run, so the palette has to be
    # set here or the sources slide arrives in Office blue on every deck.
    _theme_hyperlink_colour(prs, th["accent"])
    FONT = "Segoe UI"
    BULLET = "•  "

    def _textbox(slide, left, top, width, height):
        box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                       Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        return tf

    def _hanging_indent(par, size):
        """Wrap the second line under the TEXT, not under the bullet glyph.

        Observed on a rendered deck: "История началась весной 1961 года во время
        визита Никиты Хрущева / в США" -- the continuation starts at the left
        margin, level with the bullet, and reads as a new item. The bullet is
        drawn as part of the text (the layout places every box itself, so there
        is no list style to inherit), which is exactly the case where the
        indent has to be set explicitly.
        """
        try:
            marL = int(Pt(size * 1.15).emu)
            pPr = par._p.get_or_add_pPr()
            pPr.set("marL", str(marL))
            pPr.set("indent", str(-marL))
        except Exception:
            logger.debug("could not set the hanging indent", exc_info=True)

    def _style(run, size, colour, bold=False):
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = _rgb(colour)

    _CHART_KINDS = {"bar": XL_CHART_TYPE.COLUMN_CLUSTERED,
                    "line": XL_CHART_TYPE.LINE_MARKERS,
                    "pie": XL_CHART_TYPE.PIE}

    def _stat_band(slide, stats, left, top, width):
        """The headline figures, big, in a row of tinted cards. Returns its height.

        This is the part of a slide a listener reads from the back of the room,
        so the figure is set nearly three times the size of its label and the
        label sits under it rather than beside it.
        """
        if not stats:
            return 0.0
        n = len(stats)
        gap, height = 0.18, 1.02
        card_w = (width - gap * (n - 1)) / n
        for j, st in enumerate(stats):
            x = left + j * (card_w + gap)
            _fill_rect(slide, Inches(x), Inches(top), Inches(card_w),
                       Inches(height), th["accent"], alpha=16)
            _fill_rect(slide, Inches(x), Inches(top), Emu(38100), Inches(height),
                       th["accent"])
            inner_w = card_w - 0.24
            tf = _textbox(slide, x + 0.16, top + 0.06, inner_w, height - 0.12)
            vp = tf.paragraphs[0]
            vp.text = st["value"]
            # The figure was fixed at 26pt however long it was and however
            # narrow the card. Caught by the file audit before it was ever
            # exported: "1 200 000 ₽" needed 1.19in of a 0.9in card on a slide
            # that also carried a photograph, so three stats shared 6.8in
            # instead of 11.5. Sized to the card, with a floor -- a figure
            # smaller than its own label is not a headline.
            _style(vp.runs[0], _fit_stat(st["value"], inner_w), th["head"],
                   bold=True)
            lp = tf.add_paragraph()
            lp.text = st["label"]
            lp.space_before = Pt(2)
            _style(lp.runs[0], 11, th["body"])
        return height + 0.22

    def _draw_table(slide, table, left, top, width, height):
        """A native table: dark header row, white/tinted body rows."""
        try:
            grid = [table["columns"]] + table["rows"]
            size = 12 if len(grid) <= 5 else 10
            t = slide.shapes.add_table(len(grid), len(table["columns"]), Inches(left), Inches(top),
                                       Inches(width), Inches(min(height, 0.42 * len(grid)))).table
            for r, cells in enumerate(grid):
                for c, val in enumerate(cells):
                    cell = t.cell(r, c)
                    cell.text = val or " "
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = _rgb(th["panel"] if r == 0 else
                                                    (th["tint"] if r % 2 == 0 else 0xFFFFFF))
                    _style(cell.text_frame.paragraphs[0].runs[0], size,
                           _on_panel(th) if r == 0 else th["body"], bold=r == 0 or c == 0)
            return True
        except Exception:
            logger.warning("deck: table could not be drawn", exc_info=True)
            return False

    def _draw_chart(slide, chart, left, top, width, height):
        """A real, editable pptx chart -- not a picture of one.

        Native charts keep their numbers, so the user can correct a value in
        PowerPoint instead of asking for the whole deck again. A chart that
        fails to build must not cost the slide its text, hence the catch.
        """
        try:
            data = CategoryChartData()
            data.categories = chart["labels"]
            series_name = chart.get("unit") or chart.get("title") or "value"
            data.add_series(series_name, tuple(chart["values"]))
            gf = slide.shapes.add_chart(
                _CHART_KINDS.get(chart["kind"], XL_CHART_TYPE.COLUMN_CLUSTERED),
                Inches(left), Inches(top), Inches(width), Inches(height), data)
            ch = gf.chart
            ch.font.name = FONT
            ch.font.size = Pt(11)
            ch.font.color.rgb = _rgb(th["body"])
            # The axes and gridlines were never styled, so PowerPoint drew them
            # in its default near-black while every word on the deck is a soft
            # grey. Same family as the line and marker colours above: the theme
            # reached the series and stopped there. A pie has no axes, hence the
            # guard rather than a bare try.
            if chart["kind"] != "pie":
                for _axis_name in ("category_axis", "value_axis"):
                    try:
                        _ax = getattr(ch, _axis_name)
                        _ax.format.line.color.rgb = _rgb(th["body"])
                        _ax.tick_labels.font.size = Pt(10)
                        _ax.tick_labels.font.color.rgb = _rgb(th["body"])
                        if getattr(_ax, "has_major_gridlines", False):
                            _ax.major_gridlines.format.line.color.rgb = _rgb(
                                _blend(th["body"], 0xFFFFFF, 0.72))
                    except Exception:
                        logger.debug("could not style the %s", _axis_name,
                                     exc_info=True)
            if chart["kind"] == "pie":
                ch.has_legend = True
                ch.legend.position = XL_LEGEND_POSITION.RIGHT
                ch.legend.include_in_layout = False
                # A pie was the one chart kind nobody had coloured: bars and
                # lines take the theme accent, the pie took PowerPoint's default
                # blue/red/green -- rendered on a warm brown deck and seen on an
                # exported slide. Same family as the line-colour bug: the theme
                # reached some chart kinds and not others.
                # The slices hang off the SERIES, not off the plot: PiePlot has
                # no .points, and reaching for it raised into the except below,
                # which is how the first attempt shipped a still-default pie
                # while every check passed. Verified against the exported PNG.
                try:
                    pts = list(ch.plots[0].series[0].points)
                    shades = _pie_shades(th["accent"], len(chart["values"]))
                    for pt, rgb in zip(pts, shades):
                        pt.format.fill.solid()
                        pt.format.fill.fore_color.rgb = _rgb(rgb)
                except Exception:
                    logger.warning("could not colour the pie slices", exc_info=True)
                # And a pie has no axis, so without labels the reader cannot
                # tell 41 from 22 except by eye.
                try:
                    plot = ch.plots[0]
                    plot.has_data_labels = True
                    plot.data_labels.number_format = "0"
                    plot.data_labels.number_format_is_linked = False
                    plot.data_labels.font.size = Pt(11)
                    plot.data_labels.font.bold = True
                    plot.data_labels.font.color.rgb = _rgb(0xFFFFFF)
                    # ...but white on the palest slice is barely there. Seen on
                    # the exported PNG: the lightest shade carried a white "22"
                    # that had to be hunted for. Per slice, by luminance.
                    for pt, rgb in zip(list(ch.plots[0].series[0].points),
                                       _pie_shades(th["accent"],
                                                   len(chart["values"]))):
                        pt.data_label.font.size = Pt(11)
                        pt.data_label.font.bold = True
                        pt.data_label.font.color.rgb = _rgb(
                            th["head"] if _is_light(rgb) else 0xFFFFFF)
                except Exception:
                    logger.warning("could not label the pie slices", exc_info=True)
            else:
                ch.has_legend = False
                try:
                    plot = ch.plots[0]
                    plot.vary_by_categories = False
                    ser = plot.series[0]
                    if chart["kind"] == "line":
                        # A line carries its colour on the LINE, not the fill.
                        # Colouring the fill left every line chart in PowerPoint
                        # default blue on a warm brown deck -- seen on a slide.
                        ser.format.line.color.rgb = _rgb(th["accent"])
                        ser.format.line.width = Pt(2.5)
                        # ...and the MARKERS are a separate object again: the
                        # line came out warm orange with default blue diamonds
                        # sitting on it. Seen on the same exported deck, one
                        # level down from the bug above.
                        try:
                            mk = ser.marker
                            mk.format.fill.solid()
                            mk.format.fill.fore_color.rgb = _rgb(th["accent"])
                            mk.format.line.color.rgb = _rgb(th["accent"])
                            mk.size = 6
                        except Exception:
                            logger.warning("could not colour the line markers",
                                           exc_info=True)
                    else:
                        ser.format.fill.solid()
                        ser.format.fill.fore_color.rgb = _rgb(th["accent"])
                except Exception:
                    logger.debug("could not colour the chart series", exc_info=True)
            title = _chart_title_with_unit(chart)
            if title:
                ch.has_title = True
                ch.chart_title.text_frame.text = title
                try:
                    _style(ch.chart_title.text_frame.paragraphs[0].runs[0],
                           12, th["head"], bold=True)
                except Exception:
                    logger.debug("could not style the chart title", exc_info=True)
            else:
                ch.has_title = False
            return True
        except Exception:
            logger.warning("could not draw the chart %r", chart.get("title"),
                           exc_info=True)
            return False

    # ── cover ────────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    cover = images.get("cover")
    placed = False
    cover = _resolve_image(cover)
    if cover:
        try:
            pic = s.shapes.add_picture(str(cover), 0, 0, width=W, height=H)
            _aspect_fill(pic, W, H, cover)
            # A scrim. Without it a light photograph swallows white text whole.
            _fill_rect(s, 0, 0, W, H, th["panel"], alpha=62)
            placed = True
        except Exception:
            logger.warning("cover image failed — falling back to a colour cover",
                           exc_info=True)
    if not placed:
        _fill_rect(s, 0, 0, W, H, th["panel"])
    _fill_rect(s, Inches(0.9), Inches(2.55), Inches(1.4), Emu(45720), th["accent"])

    tf = _textbox(s, 0.9, 2.78, _SLIDE_W_IN - 3.0, 2.3)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = deck.get("title", "")
    _style(p.runs[0], 40, _on_panel(th), bold=True)
    if deck.get("subtitle"):
        p2 = tf.add_paragraph()
        p2.text = deck["subtitle"]
        p2.space_before = Pt(10)
        _style(p2.runs[0], 17, th["body"] if _is_light(th["panel"]) else 0xE6E6E6)

    # ── content ──────────────────────────────────────────────────────────────
    total = len(deck.get("slides", []))
    for i, sl in enumerate(deck.get("slides", [])):
        s = prs.slides.add_slide(blank)
        img = images.get(i) or images.get(str(i))
        img = _resolve_image(img)
        has_img = bool(img)

        _fill_rect(s, 0, 0, W, H, th["tint"])          # page tint
        _fill_rect(s, 0, 0, Emu(96520), H, th["accent"])   # accent spine

        if has_img:
            panel_w = int(W * _PIC_FRAC)
            try:
                pic = s.shapes.add_picture(str(img), W - panel_w, 0,
                                           width=panel_w, height=H)
                _aspect_fill(pic, panel_w, H, img)
            except Exception:
                logger.warning("could not place the picture on slide %d", i + 1,
                               exc_info=True)
                has_img = False

        # Derived from where the photo actually starts, not guessed as a fraction
        # of the slide. 0.55 of the width put the text column's right edge 0.08in
        # INSIDE the picture panel -- invisible while every heading was short,
        # and a heading over the photograph the day one is not. The file audit
        # reported it on every picture slide of three decks.
        text_w = ((_SLIDE_W_IN * (1.0 - _PIC_FRAC)) - 0.75 - _PIC_GAP_IN
                  if has_img else _SLIDE_W_IN - 1.8)

        htf = _textbox(s, 0.75, 0.62, text_w, 1.15)
        hp = htf.paragraphs[0]
        hp.text = sl.get("heading", "")
        # Sized to fit, not fixed at 27pt. A long heading on a picture slide ran
        # three lines out of its 1.15in box, through the accent rule and across
        # the first bullet -- and the geometry audit could not see it, because
        # the BOX was the right size and only the text was too big for it.
        _style(hp.runs[0], _fit_heading(hp.text, text_w, 1.15), th["head"],
               bold=True)
        _fill_rect(s, Inches(0.78), Inches(1.66), Inches(0.9), Emu(38100),
                   th["accent"])

        # The body column is shared: figures on top, then bullets, then a chart
        # at the foot. Each element that is present takes its band and hands the
        # rest down, so a slide with all three still fits and a slide with only
        # bullets still fills.
        body_top = _BODY_TOP_IN
        body_bottom = _BODY_BOTTOM_IN
        body_top += _stat_band(s, sl.get("stats") or [], 0.75, body_top, text_w)

        chart = sl.get("chart") or {}
        if chart:
            _bl = [str(b) for b in sl.get("bullets", []) if str(b).strip()]
            # A chart is the argument on a slide that has one; the bullets are
            # the caption. So the chart takes the larger share -- but never the
            # room the text actually needs. Measured on a clinical deck: a stat
            # band plus four bullets plus a 3in chart left the bullets 0.35in,
            # and the last two ran underneath the chart's title.
            need = _bullets_height(_bl, text_w, _MIN_BULLET_PT,
                                   max(8, round(_MIN_BULLET_PT * 0.62))) if _bl else 0.0
            chart_h = 3.9 if len(_bl) <= 2 else 3.0
            chart_h = min(chart_h, body_bottom - body_top - need - 0.25)
            # 1.5in, not 1.8: measured on a clinical slide the chart could have
            # had 1.74in after the text was served and was dropped over 0.06in.
            # A short chart of two bars still reads; a missing one loses the
            # figure the slide exists to show.
            if chart_h >= _MIN_CHART_IN and _draw_chart(s, chart, 0.75,
                                              body_bottom - chart_h,
                                              text_w, chart_h):
                body_bottom -= chart_h + 0.25
                if chart.get("source"):
                    stf = _textbox(s, 0.75, _SLIDE_H_IN - 0.92, text_w - 0.9, 0.3)
                    sp = stf.paragraphs[0]
                    sp.text = _short_url(chart["source"])
                    _style(sp.runs[0], 8, th["body"])

        table = sl.get("table") or {}
        if table and not chart:
            _bl = [str(b) for b in sl.get("bullets", []) if str(b).strip()]
            need = _bullets_height(_bl, text_w, _MIN_BULLET_PT,
                                   max(8, round(_MIN_BULLET_PT * 0.62))) if _bl else 0.0
            table_h = min(0.42 * (len(table["rows"]) + 1), body_bottom - body_top - need - 0.25)
            if table_h >= _MIN_CHART_IN and _draw_table(s, table, 0.75, body_bottom - table_h,
                                                        text_w, table_h):
                body_bottom -= table_h + 0.25

        bullets = [str(b) for b in sl.get("bullets", []) if str(b).strip()]
        band = max(0.6, body_bottom - body_top)
        size, gap = _fit_bullets(bullets, text_w, band)
        btf = _textbox(s, 0.75, body_top, text_w, band)
        # Top-anchored, directly under the rule. This used to be MIDDLE-anchored
        # inside a box spanning almost the whole slide, to stop three bullets
        # leaving the lower half empty -- and it moved the emptiness rather than
        # removing it: measured on a rendered deck, the gap between the rule and
        # the first bullet was 2.2in and the gap below the last one was 2.0in, on
        # a 7.5in slide. Emptiness above the text reads worse than below it,
        # because nothing explains the gap. The size and leading below do the
        # actual filling.
        btf.vertical_anchor = MSO_ANCHOR.TOP
        first = True
        for b in bullets:
            p = btf.paragraphs[0] if first else btf.add_paragraph()
            first = False
            p.text = BULLET + b
            p.space_after = Pt(gap)
            p.line_spacing = 1.15
            _style(p.runs[0], size, th["body"])
            _hanging_indent(p, size)

        # Footer: deck title, then the page number. BOTH stay inside the text
        # column — pinning the number to the slide edge put it on top of the
        # photo panel, where dark grey on a dark image is unreadable.
        foot_w = text_w
        # foot_w - 0.9, not - 0.8: the title box used to end 0.05in INSIDE the
        # page-number box. Harmless while the title is short, and a collision the
        # day it is not; the file audit reported it on every slide.
        ftf = _textbox(s, 0.75, _SLIDE_H_IN - 0.62, foot_w - 0.9, 0.35)
        fp = ftf.paragraphs[0]
        fp.text = (deck.get("title") or "")[:60]
        _style(fp.runs[0], 9, th["body"])
        ntf = _textbox(s, 0.75 + foot_w - 0.85, _SLIDE_H_IN - 0.62, 0.8, 0.35)
        npar = ntf.paragraphs[0]
        npar.text = f"{i + 1}/{total}"
        npar.alignment = PP_ALIGN.RIGHT
        _style(npar.runs[0], 9, th["accent"], bold=True)

        if sl.get("notes"):
            try:
                s.notes_slide.notes_text_frame.text = sl["notes"]
            except Exception:
                logger.debug("could not attach speaker notes", exc_info=True)

    # ── sources ──────────────────────────────────────────────────────────────
    # A closing slide of real, clickable links. It is the cheapest way for the
    # deck to be checkable: whoever doubts a figure can open where it came from.
    sources = deck.get("sources") or []
    if sources and deck.get("sources_in_notes"):
        # A tiny deck keeps its slide count: the links go into the speaker
        # notes of the last content slide instead of a slide of their own.
        try:
            last = prs.slides[len(prs.slides) - 1]
            tf = last.notes_slide.notes_text_frame
            head = _SOURCES_HEADING.get(_deck_lang(deck), _SOURCES_HEADING["en"])
            lines = [f"{src_item['title']} — {src_item['url']}" for src_item in sources]
            tf.text = ((tf.text + "\n\n") if tf.text else "") + head + ":\n" + "\n".join(lines)
        except Exception:
            logger.debug("could not attach sources to notes", exc_info=True)
        sources = []
    if sources:
        s = prs.slides.add_slide(blank)
        _fill_rect(s, 0, 0, W, H, th["tint"])
        _fill_rect(s, 0, 0, Emu(96520), H, th["accent"])
        htf = _textbox(s, 0.75, 0.62, _SLIDE_W_IN - 1.8, 1.15)
        hp = htf.paragraphs[0]
        hp.text = _SOURCES_HEADING.get(_deck_lang(deck), _SOURCES_HEADING["en"])
        _style(hp.runs[0], 27, th["head"], bold=True)
        _fill_rect(s, Inches(0.78), Inches(1.66), Inches(0.9), Emu(38100),
                   th["accent"])

        stf = _textbox(s, 0.75, _BODY_TOP_IN, _SLIDE_W_IN - 1.8,
                       _BODY_BOTTOM_IN - _BODY_TOP_IN)
        stf.vertical_anchor = MSO_ANCHOR.TOP
        first = True
        for src_item in sources:
            para = stf.paragraphs[0] if first else stf.add_paragraph()
            first = False
            para.space_after = Pt(10)
            para.line_spacing = 1.1
            r_title = para.add_run()
            r_title.text = BULLET + src_item["title"]
            _style(r_title, 15, th["body"], bold=True)
            r_break = para.add_run()
            r_break.text = "   "
            _style(r_break, 15, th["body"])
            r_url = para.add_run()
            r_url.text = _short_url(src_item["url"])
            _style(r_url, 12, th["accent"])
            try:
                # The whole line links, title included: a listener clicks words,
                # not the grey URL after them.
                r_title.hyperlink.address = src_item["url"]
                r_url.hyperlink.address = src_item["url"]
            except Exception:
                logger.debug("could not attach a hyperlink", exc_info=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    prs.save(out_path)
    logger.info("deck written: %s (%d slides, theme=%s)", out_path,
                len(prs.slides._sldIdLst), th["name"])
    return out_path


def make_presentation(ctx, topic: str, out_dir: str, *, n_slides: int = 0,
                      image_fetcher: Optional[Callable[[str], Optional[str]]] = None,
                      max_images: int = 4, out_lang: str = "",
                      previous: Optional[dict] = None) -> dict:
    """Plan a deck, optionally illustrate it, and write the file.

    `image_fetcher(query) -> path|None` is supplied by the caller so this module
    never decides HOW a picture is obtained (web search, local generation) — it
    only decides which slides asked for one. Illustration is capped because a
    picture per slide is both slow and usually worse design.

    `out_lang` — see plan_deck(): the language the deck must be written in,
    detected by the caller from the user's original message.

    Returns {"path", "deck", "images", "error"}.
    """
    deck = plan_deck(ctx, topic, n_slides, out_lang=out_lang, previous=previous)
    if deck.get("edit_failed"):
        return {"path": "", "deck": deck, "images": {},
                "error": "the change could not be applied to the previous deck "
                         "(the model did not answer) — the file was not rewritten"}
    if deck.get("planner_failed"):
        # Measured with LM Studio down: the caller got a path and an empty error
        # string, and announced a finished presentation. The file was a cover, a
        # slide whose only text was the topic, and a sources list. Handing that
        # over as a deck is the same lie as describing a presentation that was
        # never built, which this project already guards against elsewhere.
        logger.error("no deck was planned for %r — refusing to call it a "
                     "presentation", topic[:60])
        return {"path": "", "deck": deck, "images": {},
                "error": "the deck planner returned no content (the model did "
                         "not answer) — no slides were written"}
    images: dict = {}
    if image_fetcher is not None:
        # The COVER first, and out of its own budget: it is the full-bleed
        # background of the title slide and by far the most visible picture in
        # the deck, so it must not lose its slot to a mid-deck illustration.
        _cq = _image_query(deck.get("cover_image") or "",
                           deck.get("title") or topic) \
            or deck.get("title") or topic
        if _cq:
            try:
                got = image_fetcher(_cq)
            except Exception:
                logger.warning("cover image fetch failed for %r", _cq, exc_info=True)
                got = None
            if got and os.path.exists(str(got)) and _is_scene(got):
                images["cover"] = got
        for i, sl in enumerate(deck["slides"]):
            # the cover does not count against the per-slide budget
            if len([k for k in images if k != "cover"]) >= max_images:
                break
            q = sl.get("image") or ""
            if not q:
                continue
            q = _image_query(q, deck.get("title") or topic)
            if ctx is not None and getattr(ctx, "is_cancelled", bool)():
                break
            try:
                got = image_fetcher(q)
            except Exception:
                logger.warning("image fetch failed for %r", q, exc_info=True)
                got = None
            if got and os.path.exists(str(got)):
                images[i] = got

    stem = _safe_name(deck.get("title") or topic)
    path = os.path.join(out_dir, stem + ".pptx")
    try:
        written = build_pptx(deck, path, images, topic=topic)
    except (PermissionError, OSError) as exc:
        # The previous deck on this topic is still OPEN in PowerPoint, so Windows
        # refuses to overwrite it. Losing the whole build — planning, four web
        # images, two minutes of work — over a locked filename is absurd; write a
        # numbered sibling instead.
        logger.warning("could not write %s (%s) — writing a new file instead",
                       path, exc)
        written = None
        for n in range(2, 12):
            alt = os.path.join(out_dir, f"{stem} ({n}).pptx")
            try:
                written = build_pptx(deck, alt, images, topic=topic)
                path = alt
                break
            except (PermissionError, OSError):
                continue
        if not written:
            logger.exception("deck assembly failed")
            return {"path": "", "deck": deck, "images": images, "error": str(exc)}
    except Exception as exc:
        logger.exception("deck assembly failed")
        return {"path": "", "deck": deck, "images": images, "error": str(exc)}
    if not written:
        return {"path": "", "deck": deck, "images": images,
                "error": "python-pptx is not installed"}
    # Audit the FILE, not the plan. The plan-side numbers said "two charts" on a
    # deck whose file had none; without this line the difference is invisible
    # until somebody opens the deck.
    try:
        built = pptx_audit(written)
        logger.info("deck built: %s", built)
        if built["empty"]:
            logger.warning("deck has slides with no content: %s", built["empty"])
        if built["overflow"]:
            logger.warning("deck has text overflowing its box: %s", built["overflow"])
        if built["offslide"]:
            logger.warning("deck has shapes off the canvas: %s", built["offslide"])
        if built["overlaps"]:
            logger.warning("deck has overlapping shapes: %s", built["overlaps"])
        planned = len([s for s in deck.get("slides") or [] if s.get("chart")])
        if planned and built["charts"] < planned:
            logger.warning("deck lost %d of %d planned charts in layout",
                           planned - built["charts"], planned)
    except Exception:
        logger.debug("could not audit the built deck", exc_info=True)
    return {"path": written, "deck": deck, "images": images, "error": ""}


def pptx_audit(path: str) -> dict:
    """What the BUILT FILE contains, as opposed to what the plan promised.

    deck_facts_audit reads the plan. On a clinical deck it reported two charts
    while the file it produced had none: the chart reservation had dropped both
    over a fraction of an inch, and the number that was supposed to catch a
    thin deck was reading the wrong side of the build. A report is not a
    detector. This opens the file.

    Overlaps are reported per slide as pairs of shape indices whose boxes
    intersect by more than a hairline, which is how the chart-over-bullets
    defect looked before anyone exported a PNG.
    """
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation(path)
    charts = pictures = 0
    overlaps = []
    offslide = []
    overflow = []
    empty = []
    for i, slide in enumerate(prs.slides, 1):
        boxes = []
        body_text = 0                  # content, as opposed to heading + footer
        has_visual = False
        for sh in slide.shapes:
            if getattr(sh, "has_chart", False):
                charts += 1
            if getattr(sh, "has_chart", False):
                has_visual = True
            if sh.shape_type == 13:                      # PICTURE
                pictures += 1
                has_visual = True
            # Only things that carry content can collide meaningfully; the
            # full-bleed background rectangles overlap everything by design.
            if sh.has_text_frame and not sh.text_frame.text.strip():
                continue
            if sh.width is None or sh.height is None:
                continue
            w, h = Emu(sh.width).inches, Emu(sh.height).inches
            if w >= _SLIDE_W_IN - 0.01 and h >= _SLIDE_H_IN - 0.01:
                continue
            label = (sh.text_frame.text.strip().splitlines()[0][:28]
                     if sh.has_text_frame else str(sh.shape_type))
            if sh.has_text_frame:
                _t = sh.text_frame.text.strip()
                # The heading is one short line and the footer is the deck title
                # and a page number; neither is what the slide is FOR.
                if "•" in _t or len(_t) > 60:
                    body_text += 1
            # Text that does not fit its box is invisible to a geometry check:
            # the box is the right size and only the words are too big. That is
            # exactly how a long heading came to lie across the first bullet.
            if sh.has_text_frame:
                need = 0.0
                for par in sh.text_frame.paragraphs:
                    txt = "".join(r.text for r in par.runs)
                    if not txt.strip():
                        continue
                    pt = max((r.font.size.pt for r in par.runs
                              if r.font.size is not None), default=18)
                    cpl = max(6, int((w * 72.0) / (pt * 0.52)))
                    need += max(1, -(-len(txt) // cpl)) * pt * 1.16 / 72.0
                if need > h + 0.06:
                    overflow.append((i, label, round(need, 2), round(h, 2)))
            L, T = Emu(sh.left).inches, Emu(sh.top).inches
            # A box that runs off the canvas loses whatever sits in the tail.
            if L < -0.01 or T < -0.01 or L + w > _SLIDE_W_IN + 0.01 or T + h > _SLIDE_H_IN + 0.01:
                offslide.append((i, label, round(L, 2), round(T, 2),
                                 round(L + w, 2), round(T + h, 2)))
            boxes.append((L, T, w, h, label))
        # A slide carrying nothing but its own heading is a defect the plan-side
        # numbers cannot see -- found in a real deck in runtime/, from a run
        # where the planner had failed. The refusal upstream covers that cause;
        # this names the symptom whatever the cause turns out to be next time.
        if i > 1 and not body_text and not has_visual:
            empty.append(i)
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ax, ay, aw, ah, alab = boxes[a]
                bx, by, bw, bh, blab = boxes[b]
                ox = min(ax + aw, bx + bw) - max(ax, bx)
                oy = min(ay + ah, by + bh) - max(ay, by)
                if ox > 0.02 and oy > 0.02:
                    # Named, not numbered: an index into a private list tells
                    # the reader nothing about which two things collided.
                    overlaps.append((i, alab, blab, round(ox, 2), round(oy, 2)))
    return {"slides": len(prs.slides._sldIdLst),
            "charts": charts, "pictures": pictures, "overlaps": overlaps,
            "offslide": offslide, "overflow": overflow, "empty": empty}
