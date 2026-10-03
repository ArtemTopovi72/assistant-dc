"""A real, named person in a drawing request gets their actual looks spelled out.

The image engine draws from the words it is given and does not know faces by
name: «кибернетический Ленин» came back with a full head of hair, then as
«какой-то урод», and a redraw «он лысый» did not fix it (live 10-03). The model
that plans the picture names the person; their looks are read from the web
(the model's own memory when the search finds nothing) and ride along with
the description.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

logger = logging.getLogger("assistant.image")

_SYSTEM = (
    "An image request may name a REAL, specific, well-known person (a politician, "
    "an actor, a historical figure). Return STRICT JSON only: {\"person\": \"<their "
    "name in English, or empty>\", \"look\": \"<what makes their FACE and HEAD "
    "recognisable, in plain visual English: hair (or baldness) and its colour, "
    "beard/moustache, forehead, eyes, face shape, typical age and clothing>\"}. "
    "Only physical appearance, 10-40 words, no story, no opinion. Empty strings "
    "for a generic subject (a cat, a knight, a woman) or a fictional character.")

_FROM_WEB = (
    "Below are web search results about {person}. From them (and what you know), "
    "describe what makes {person}'s FACE and HEAD recognisable, in plain visual "
    "English, 10-40 words: hair or baldness and its colour, beard/moustache, "
    "forehead, eyes, face shape, typical age and clothing. Only appearance. Output "
    "the description alone.")

_CACHE: "OrderedDict[str, str]" = OrderedDict()
_LOCK = threading.Lock()
STUB = None          # suites: STUB(text) -> (person, look)
SEARCH_STUB = None   # suites: SEARCH_STUB(query) -> search text


def _ask(text: str) -> tuple:
    if STUB is not None:
        return STUB(text)
    if os.getenv("F5_TEST_RUN"):
        return "", ""
    from llm import call_llm_simple
    from utils import safe_json_from_llm
    out = call_llm_simple(None, _SYSTEM, "Request: " + text[:1500], temperature=0.0,
                          max_tokens=200, prefill="<think></think>")
    data = safe_json_from_llm(out or "", ["person", "look"]) or {}
    return str(data.get("person") or "").strip(), str(data.get("look") or "").strip()


def _search(ctx, query: str) -> str:
    if SEARCH_STUB is not None:
        return SEARCH_STUB(query)
    if os.getenv("F5_TEST_RUN") or ctx is None or not getattr(ctx, "web_search_enabled", True):
        return ""
    import search
    txt = search.run_web_search(ctx, query) or ""
    return "" if txt in (search.NO_RESULTS, search.SEARCH_FAILED) else txt


def _look_from_web(ctx, person: str) -> str:
    """The person's looks as the web describes them (owner 10-03: «описание
    брать в интернете норм»); '' when the search finds nothing."""
    try:
        found = _search(ctx, f"{person} appearance face hair beard description")
        if not found.strip():
            return ""
        if STUB is not None:
            return STUB("WEB:" + found)[1]
        from llm import call_llm_simple
        return (call_llm_simple(None, _FROM_WEB.format(person=person), found[:4000],
                                temperature=0.0, max_tokens=160,
                                prefill="<think></think>") or "").strip()
    except Exception:
        logger.warning("person look: web read failed for %r", person, exc_info=True)
        return ""


def note(text: str, ctx=None) -> str:
    """'' or one line naming the person's real appearance, to add to a prompt."""
    text = (text or "").strip()
    if not text:
        return ""
    with _LOCK:
        if text in _CACHE:
            return _CACHE[text]
    try:
        person, look = _ask(text)
    except Exception:
        logger.warning("person look: model read failed", exc_info=True)
        return ""
    if person:
        look = _look_from_web(ctx, person) or look
    look = " ".join(look.split())[:400]
    line = (f"{person} must be recognisable as the real {person}: {look}."
            if person and len(look.split()) >= 3 else "")
    with _LOCK:
        _CACHE[text] = line
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)
    if line:
        logger.info("person look: %s", line[:200])
    return line


def with_look(description: str, about: str = "", ctx=None) -> str:
    """`description` with the person's looks appended (read from `about` too:
    a redraw's instruction alone may only say «he is bald»)."""
    line = note(" ".join(x for x in (description, about) if x), ctx)
    return f"{description}\n{line}" if line and line not in description else description
