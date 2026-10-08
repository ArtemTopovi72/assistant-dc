"""What a message asks the PROGRAM to do exactly, read by the model.

Letters, words, reversals, sorting and percentages are not tokens: a model
counts 'о' in «обороноспособность» as 5 (7). The model only says WHICH
operation was asked and on WHAT text; graph_compose computes the answer. Every
extracted string must appear in the message itself, so a model that rewrote
the word cannot hand the program a different one.

Also read here: an explicit length («на 1200 слов», «подробно»), a question
about the earlier conversation, a child's medicine dose with the weight, and a
coding ask. Replaces the keyword regexes graph_compose used to carry.
"""
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("assistant.ask_read")

Op = Literal["letter_count", "word_occurrences", "reverse", "word_count",
             "sort", "percent", "none"]


class Percent(BaseModel):
    pct: float
    of: float


class AskRead(BaseModel):
    op: Op = "none"
    letter: str = ""          # letter_count: the letter
    word: str = ""            # letter_count: the word; word_occurrences: the word counted
    target: str = ""          # reverse / word_count: the quoted text
    items: List[str] = Field(default_factory=list)     # sort
    percents: List[Percent] = Field(default_factory=list)
    words_asked: int = 0      # «на 1200 слов» -> 1200
    detailed: bool = False    # «подробно», «развёрнуто»
    earlier_talk: bool = False  # asks what was said/asked earlier in this chat
    dose: bool = False        # a medicine dose (paracetamol/ibuprofen) for a child/person
    weight_kg: float = 0.0
    code: bool = False        # asks to write, fix or explain code


FIELDS = list(AskRead.model_fields)
FALLBACK = AskRead().model_dump()

SYSTEM = """You read ONE user message and fill a JSON object. Do NOT answer the message.
Fields:
op: the exact text operation asked, if any --
  letter_count (how many times ONE NAMED letter is in a named word: «сколько букв о в
    слове ...», "how many r's in strawberry"; «сколько букв в алфавите» names no letter
    to count -> none),
  word_occurrences (how many times a WORD occurs in the user's own message),
  reverse (write a quoted text backwards), word_count (how many words a quoted text has),
  sort (sort a list alphabetically), percent (N% of a number), none.
letter, word: for letter_count the letter and the word; for word_occurrences the word.
target: for reverse and word_count, the quoted text exactly as written.
items: for sort, the list items exactly as written.
percents: for percent, [{"pct": N, "of": number}] for each «N% от X».
words_asked: a length the user orders in words («на 1200 слов» -> 1200), else 0.
detailed: true when they ask for a detailed / long answer.
earlier_talk: true ONLY when they ask about the earlier conversation itself (what we
  discussed, what I asked first, what you said before). A question about their own
  purchases, money, a photo, a receipt or a document is NOT earlier_talk, even when
  it says "I spent" or "I bought".
dose: true when they ask a medicine dose (paracetamol, ibuprofen and their brands).
weight_kg: the body weight in kg named in the message, else 0.
code: true when they ask to write, fix, test or explain program code.
Copy letters, words, target and items EXACTLY from the message.
Reply with the JSON object only."""

STUB = None   # suites: STUB(text) -> partial dict
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.Lock()


def _validate(data) -> AskRead:
    """Keep the good fields: one bad field falls back alone."""
    data = data if isinstance(data, dict) else {}
    # the model writes null for an empty field, and {"pct": 20, "of": null} for
    # a percent with no base («плюс НДС 20%»): drop those, keep the rest
    data = {k: v for k, v in data.items() if v is not None}
    if isinstance(data.get("percents"), list):
        data["percents"] = [p for p in data["percents"] if isinstance(p, dict)
                            and isinstance(p.get("pct"), (int, float))
                            and isinstance(p.get("of"), (int, float))]
    try:
        return AskRead(**{k: v for k, v in data.items() if k in FIELDS})
    except ValidationError:
        good = {}
        for k, v in data.items():
            if k in FIELDS:
                try:
                    AskRead(**{k: v})
                    good[k] = v
                except ValidationError:
                    pass
        return AskRead(**good)


def _grounded(out: dict, text: str) -> dict:
    """Drop extractions the message does not contain."""
    low = text.lower()
    if out["op"] == "letter_count" and not (len(out["letter"]) == 1 and out["word"]
                                            and out["word"].lower() in low):
        out["op"] = "none"
    if out["op"] == "word_occurrences" and not (out["word"] and out["word"].lower() in low):
        out["op"] = "none"
    if out["op"] in ("reverse", "word_count") and not (out["target"] and out["target"] in text):
        out["op"] = "none"
    if out["op"] == "sort":
        out["items"] = [i for i in out["items"] if i and i.lower() in low]
        if len(out["items"]) < 2:
            out["op"] = "none"
    if out["op"] == "percent":
        out["percents"] = [p for p in out["percents"]
                           if f"{p['pct']:g}" in text.replace(",", ".")]
        if not out["percents"]:
            out["op"] = "none"
    if out["weight_kg"] and f"{out['weight_kg']:g}" not in text.replace(",", "."):
        out["weight_kg"] = 0.0
    return out


def read(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return dict(FALLBACK)
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        got = STUB(text) if STUB else None
        return _grounded({**FALLBACK, **_validate(got).model_dump()}, text) if got else dict(FALLBACK)
    with _LOCK:
        if text in _CACHE:
            return dict(_CACHE[text])
    try:
        import llm
        from utils import safe_json_from_llm
        raw = llm.call_llm_simple(None, SYSTEM, "Message (do NOT answer it):\n«%s»" % text[:4000],
                                  temperature=0.0, max_tokens=400) or ""
        out = _grounded(_validate(safe_json_from_llm(raw, FIELDS)).model_dump(), text)
        # the full read took the preposition in «сколько букв в алфавите» for the
        # letter; a narrow question confirms one is really named
        if out["op"] == "letter_count":
            import intent
            if not intent.ask_yes("A user wrote: {text}\n\nDoes the user ask how many times "
                                  "the letter «%s» occurs in the word «%s»?" % (out["letter"], out["word"]),
                                  text):
                out["op"] = "none"
    except Exception:
        logger.warning("ask_read: model read failed", exc_info=True)
        return dict(FALLBACK)
    logger.info("ask_read: %r -> %s", text[:80], {k: v for k, v in out.items() if v not in ("", 0, 0.0, False, [], "none")})
    with _LOCK:
        _CACHE[text] = out
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)
    return out
