"""Silent-payload guard for tool results.

The PlanBench-XL stress test showed the model fabricates when a tool returns a
SUCCESS-shaped but unusable payload: empty (70% fabrication), malformed (46%),
truncated (92%), partial-missing-fields (100%), plausible-but-wrong (100%),
conflicting (58%). The control-flow hardening can't catch these — they are a
judgment failure inside one tool turn.

This guard adds two layers in front of the model:

  1. DETERMINISTIC BLOCK — for the unambiguous structural failures (empty,
     malformed/garbled bytes, truncated mid-word, too-thin). These are downgraded
     to an honest "no usable result" signal so the model retries or reports the
     failure instead of inventing content. This is the reliable part.

  2. SKEPTIC BANNER — appended to payloads that pass the structural checks but
     could still be partial / stale / conflicting. It cannot force the model to be
     honest, but it materially reduces "fill in the missing field" fabrication and
     flags staleness. World-knowledge errors (a plausible wrong number) are NOT
     detectable here and are explicitly out of scope.

`assess_payload(query, body)` is pure and unit-tested against the exact junk
payloads from the benchmark. `guard_search_result(query, raw)` is what the tool
handlers call: it returns the text to hand back to the model (either an honest
block message or the payload + banner).
"""
import re
from datetime import datetime
from collections import namedtuple

Verdict = namedtuple("Verdict", "usable status reason")

# Replacement char + NUL are unambiguous corruption markers.
_CORRUPT = ("\x00", "�")
# Sentence-final punctuation a well-formed result/snippet ends on.
_TERMINAL = tuple(".!?…»\"”)]}»")
# Minimum extractable text to be considered non-thin (markup/whitespace stripped).
_MIN_TEXT = 25

SKEPTIC_BANNER = (
    "\n[DATA-VALIDATION] Treat the results above as raw, possibly-incomplete data, "
    "not a finished answer. Rules: (1) state ONLY facts explicitly present in the "
    "results; (2) if the user asked for something the results do NOT contain — a "
    "specific field, number, coordinate, date, or name — say it was not found "
    "instead of inventing it; (3) if the data looks dated or sources give different "
    "values, say so and present all of them; (4) if the results do not actually "
    "answer the question, tell the user plainly rather than guessing; (5) keep the "
    "(source: domain) mark right after each fact you relay -- the chat turns them "
    "into footnotes and lists the pages underneath, so never write your own list "
    "of sources or links."
)


def _strip_markup(text: str) -> str:
    """Rough count of human-readable characters (drop json punctuation/markup)."""
    return re.sub(r"[\s{}\[\]\"':,]+", "", text)


def _looks_like_json(text: str) -> bool:
    # Only a payload that IS json: prose with a `{"` somewhere inside (a script
    # snippet scraped off a review page) was blocked whole as "broken JSON" and
    # the model said there was no data on the iPhone 16 vs S25 (live 2026-09-28).
    t = text.lstrip()
    return t.startswith("{") or (t.startswith("[") and t.rstrip().endswith("]")
                                 and not re.match(r"\[\d+\]\s*\w", t))


def _json_parses(text: str) -> bool:
    import json
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def assess_payload(query: str, body: str, now_year: int | None = None) -> Verdict:
    """Classify a raw tool payload. Pure function — no I/O."""
    now_year = now_year or datetime.now().year
    raw = body or ""
    text = raw.strip()

    # 1) EMPTY
    if not text:
        return Verdict(False, "empty", "the search returned an empty result (no content)")

    # 2) MALFORMED — corruption bytes, or JSON-shaped but unparseable
    if any(ch in raw for ch in _CORRUPT):
        return Verdict(False, "malformed", "the result contains corrupted/garbled bytes")
    if _looks_like_json(text) and not _json_parses(text):
        return Verdict(False, "malformed", "the result is broken/unparseable JSON")

    # 3) THIN — almost no readable content
    if len(_strip_markup(text)) < _MIN_TEXT:
        return Verdict(False, "thin", "the result has too little usable content to rely on")

    # 4) TRUNCATED — ends mid-word with no terminal punctuation (and not a bare URL)
    if (text[-1].isalpha() and not text.endswith(_TERMINAL)
            and not text.rstrip().endswith(("...", "…"))
            and not re.search(r"https?://\S+$", text)):
        # a real summary essentially always ends on punctuation; mid-letter = cut off
        return Verdict(False, "truncated", "the result appears cut off mid-sentence (truncated)")

    # 5) STALE — newest year mentioned is well in the past (flag, don't block)
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    if years and max(years) <= now_year - 2 and re.search(
            r"по состоянию|кэш|действующ|на момент|актуальн|сейчас|current|as of", text, re.I):
        return Verdict(True, "stale",
                       f"the data looks dated (newest year mentioned is {max(years)})")

    return Verdict(True, "ok", "")


def _block_message(v: Verdict, kind: str = "search") -> str:
    return (
        f"[TOOL ERROR] The {kind} did not return a usable result — {v.reason}. "
        "Do NOT present this as an answer and do NOT invent or complete the missing "
        "content. Either retry with a different query, try another tool, or tell the "
        "user plainly that the information could not be retrieved."
    )


def guard_search_result(query: str, raw: str, kind: str = "search") -> str:
    """Entry point for tool handlers. `raw` is the unwrapped result text.

    Returns the string to hand to the model: an honest block message for
    structurally-broken payloads, otherwise the payload plus the skeptic banner
    (with a staleness note prepended when relevant)."""
    v = assess_payload(query, raw)
    if not v.usable:
        return _block_message(v, kind)
    prefix = f"[DATA-VALIDATION] Note: {v.reason}.\n" if v.status == "stale" else ""
    return prefix + raw + SKEPTIC_BANNER
