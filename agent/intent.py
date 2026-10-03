"""What the user's message asks for, read ONCE per message by the model.

Replaces the keyword regexes that decided routing (graph._TOOL_TRIGGER_RE,
graph_fastpath._IMAGE_INTENT_RE, tool_retrieval cue words, the forced-tool
regexes in graph_personality): a word list cannot read a sentence --
«добавили» held «оба» and a question about a forwarded video got a transcript
(live 2026-10-02 16:03); paraphrases with no cue word took the tool-less fast
path and got invented answers.

read(ctx, text, previous="", attached="") -> dict, cached per input:
  needs_tool      a correct answer needs a tool
  wants           the agent's own tool names it needs, most likely first
  must_call       the ONE tool to force on the first round, or ""
  bare_ack        the message is only "ok / thanks"
  presents_photo  it only introduces an attached photo
  is_question     it asks something
  about_picture   it asks or talks about what is IN the attached picture/video
                  (answering needs a look at it), not an order to change it
  names_picture   which earlier picture it names by position: 1 2 3 from the
                  first, -1 the last, -2 the previous; 0 none
  reply_language  "en"/"ru" when it asks for THIS reply in that language, else ""
  language_mode   "en"/"ru" when it asks to use that language from now on, else ""
  translate       it asks to translate a text it gives
  redo            "same": a bare "again"; "changed": again, with its own changes; ""
  asks_for_file   it asks for a file (archive, deck, document, pdf) to be made or sent
  doc_question    the user's own saved documents could answer it
  song            "make" a song with music, "lyrics" words only, after a song: "sound"
                  (same words, new style/voice/tempo) or "words" (rewrite words/length)
  song_seconds    the length asked for, in seconds, 0 when none
  song_uses_previous_text  the song is to use the text just written («на него»)
  weather         {city, when} for a forecast question, else null
  undo            it asks to bring back the picture as it was before the last change
On any failure it returns FALLBACK: needs_tool True, no picks, nothing forced
-- the full agent loop with the default tool set, the safe direction.
bench/intent_live.py checks it against the real model.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from collections import OrderedDict

logger = logging.getLogger(__name__)

FIELDS = ("needs_tool", "wants", "must_call", "bare_ack", "presents_photo", "is_question",
          "about_picture", "names_picture", "reply_language", "language_mode", "translate",
          "redo", "asks_for_file", "doc_question",
          "song", "song_seconds", "song_uses_previous_text", "weather", "undo",
          "complaint", "collage", "dedupe")

SYSTEM = (
    "You route a chat message for an assistant that has these tools:\n{menu}\n\n"
    "Read what the user MEANS (any language, typos, slang), not single words.\n"
    "needs_tool: true when a correct answer needs a tool -- current or "
    "real-world facts (prices, rates, weather, news, places, addresses, "
    "results, time), making or changing a picture/video/song/deck/file, "
    "non-trivial arithmetic, running code, saving something to remember, "
    "setting a reminder, timer or alarm, the user's files or documents, a "
    "researched report, review or comparison (deep_research). false "
    "for chat, opinions, advice, jokes, explanations from general knowledge, thanks.\n"
    "wants: the tools it needs, most likely first (empty when needs_tool is false).\n"
    "must_call: the ONE tool that has to run before any answer, because "
    "answering without it would be wrong or a fake promise -- arithmetic with "
    "multi-digit numbers or percentages, or a sum/total over numbers in the "
    "chat or on an attached photo: calculate; current facts, prices, results, "
    "places, addresses: search (weather_forecast, local_time, exchange_rate "
    "when that is what is asked); a presentation/slides: create_presentation; "
    "'write code and run it / show the result': run_code. \"\" when no single "
    "tool is required (chat, a picture to draw, simple mental math, a "
    "question about a date that is already known). Attached photos and data "
    "are already read and in the chat: a sum over them is calculate, never "
    "open_image, inspect_image or recall_context. deep_research only when "
    "a report or a deep review is asked for by name.\n"
    "bare_ack: the whole message is only an acknowledgement (ok, thanks, got "
    "it, понял, спасибо).\n"
    "presents_photo: the message only introduces or captions an attached "
    "photo (\"вот мой кот\", a name) and asks nothing.\n"
    "is_question: the message asks something.\n"
    "about_picture: a photo or video is attached and the message asks or "
    "talks about what is in it (what is written, who is that, your opinion "
    "of it, the most expensive item on it) -- false for an order to change "
    "or redraw it, and false when nothing is attached.\n"
    "names_picture: when the message points at one of several pictures BY "
    "POSITION, which one: 1 the first, 2 the second, 3 the third, -1 the "
    "last/newest, -2 the previous one; 0 when it names none (\"this one\", "
    "no picture at all).\n"
    "reply_language: \"en\" or \"ru\" when the user asks for the answer in "
    "that language (\"и то же самое по-английски\", \"колыбельная на "
    "английском\"); \"\" otherwise. A question ABOUT a word -- how it is "
    "said or what it means in another language (\"как будет по-английски "
    "подоконник?\", \"что значит serendipity?\") -- is answered in the "
    "user's own language: reply_language \"\", translate false.\n"
    "language_mode: \"en\" or \"ru\" only when the user asks to keep using "
    "that language from now on (\"from now on\", \"дальше\", \"впредь\", "
    "\"всегда\"); a one-off \"ответь на английском\" is \"\".\n"
    "translate: true when the message asks to translate a text it contains "
    "or points at.\n"
    "redo: \"same\" when the message only asks to do the previous thing "
    "again (\"заново\", \"ещё раз\", \"again\"); \"changed\" when it asks "
    "to redo the SAME thing with its own changes (\"пересобери коллаж без "
    "дубликатов\"); a next, different request (\"а теперь коллаж\") is \"\"; "
    "\"\" otherwise.\n"
    "asks_for_file: true when the user asks for a file -- an archive, a "
    "deck, a document, a pdf, a report -- to be made or sent.\n"
    "doc_question: true when the user's own saved documents (a contract, notes, a "
    "receipt, a price list) could answer it -- a question about facts in their files, "
    "or a summary/list made from them. false for chat and reactions, making or "
    "changing something (a picture, a song, a deck), current web facts (weather, "
    "news), saving or forgetting facts, and follow-ups that transform the previous "
    "answer (shorter, translate, again).\n"
    "song: \"make\" when the user wants a song made -- music with vocals to listen to (\"сочини песню про кота\", \"спой колыбельную\", \"сделай гимн отдела\"); "
    "\"lyrics\" when only the words are wanted (\"напиши текст песни\"); right after the assistant sent a song: \"sound\" for the same words with a new style, voice, "
    "tempo or version, \"words\" to change its words, chorus or length; \"\" otherwise -- "
    "talking about songs or singing (\"как научиться петь\", \"расскажи про гимн России\") is \"\"; "
    "\"reset\" when the user asks to reset the song or music settings (\"сбрось "
    "настройки песни\", \"reset song settings\", \"обнули параметры музыки\").\n"
    "song_seconds: the length the user asks for, in seconds (\"на полторы минуты\" = 90), 0 when none.\n"
    "song_uses_previous_text: true when the song is to be made from the text just "
    "written (\"сочини на него песню\", \"спой это\", \"из этого стиха\").\n"
    "weather: for any message about the weather forecast {{\"city\": the place NAMED in "
    "the message or \"\", \"when\": now|today|tomorrow|day_after_tomorrow|week|weekend|monday..sunday "
    "or \"\", \"place_by_reference\": true when the message points at a place without "
    "naming it (\"там\", \"у них\"); a message that names no place at all is false}}; after a forecast a bare place or day (\"а в Сочи?\", \"Питер\", "
    "\"а послезавтра?\") is one too. null for climate in general, weather in a game, a "
    "picture of weather.\n"
    "undo: true when the user asks to bring back the picture as it was before the "
    "last change (\"верни как было\", \"отмени\", \"undo\", \"put it back\").\n"
    "complaint: the user says the previous answer was wrong or missing (\"не то\", "
    "\"опять не так\", \"где файл?\", \"я же просил\", \"that's wrong\").\n"
    "collage: the user asks for a collage or grid of photos.\n"
    "dedupe: the user asks to drop duplicate or repeated photos (re-shots of one scene).\n"
    "The previous assistant line, when given, is context for short follow-ups "
    "(\"а в евро?\" after a rate needs the same tool).\n"
    "Answer with ONE JSON object and nothing else: "
    "{{\"needs_tool\": bool, \"wants\": [tool names], \"must_call\": \"tool name or empty\", "
    "\"bare_ack\": bool, \"presents_photo\": bool, \"is_question\": bool, "
    "\"about_picture\": bool, \"names_picture\": int, \"reply_language\": "
    "\"en|ru|\", \"language_mode\": \"en|ru|\", \"translate\": bool, "
    "\"redo\": \"same|changed|\", \"asks_for_file\": bool, \"doc_question\": bool, "
    "\"song\": \"make|lyrics|sound|words|reset|\", \"song_seconds\": int, \"song_uses_previous_text\": bool, "
    "\"undo\": bool, \"complaint\": bool, \"collage\": bool, \"dedupe\": bool, \"weather\": {{\"place_by_reference\": bool, \"city\": str, \"when\": str}} or null}}")


class Weather(BaseModel):
    city: str = ""
    when: Literal["now", "today", "tomorrow", "day_after_tomorrow", "week", "weekend", "monday", "tuesday",
                  "wednesday", "thursday", "friday", "saturday", "sunday", ""] = ""
    place_by_reference: bool = False


class Intent(BaseModel):
    """The read, validated. Not sent as json_schema: Gemma's grammar rejects the
    no-think prefill, and without it the model thinks past any budget (1465
    chars of reasoning, no answer, 2026-10-02)."""
    needs_tool: bool = True
    wants: list[str] = []
    must_call: str = ""
    bare_ack: bool = False
    presents_photo: bool = False
    is_question: bool = False
    about_picture: bool = False
    names_picture: int = Field(0, ge=-2, le=3)
    reply_language: Literal["en", "ru", ""] = ""
    language_mode: Literal["en", "ru", ""] = ""
    translate: bool = False
    redo: Literal["same", "changed", ""] = ""
    asks_for_file: bool = False
    doc_question: bool = False
    song: Literal["make", "lyrics", "sound", "words", "reset", ""] = ""
    song_seconds: int = Field(0, ge=0, le=600)
    song_uses_previous_text: bool = False
    undo: bool = False
    complaint: bool = False
    collage: bool = False
    dedupe: bool = False
    weather: Optional[Weather] = None



def _validate(got: dict) -> Intent:
    """A field the model got wrong ("song": "rap", "names_picture": 7) takes
    its default; the rest of the read stands."""
    got = {k: v for k, v in (got or {}).items() if k in Intent.model_fields}
    while True:
        try:
            return Intent.model_validate(got)
        except ValidationError as e:
            bad = {err["loc"][0] for err in e.errors() if err["loc"]}
            if not bad & got.keys():
                return Intent()
            got = {k: v for k, v in got.items() if k not in bad}

FALLBACK = {"needs_tool": True, "wants": [], "must_call": "", "bare_ack": False,
            "presents_photo": False, "is_question": False, "about_picture": False, "names_picture": 0,
            "reply_language": "", "language_mode": "", "translate": False,
            "redo": "", "asks_for_file": False, "doc_question": False,
            "song": "", "song_seconds": 0, "song_uses_previous_text": False, "weather": None, "undo": False,
            "complaint": False, "collage": False, "dedupe": False, "ok": False}

YES_STUB = None   # suites: YES_STUB(question, text) -> bool
_YES_CACHE: "OrderedDict[tuple, bool]" = OrderedDict()


CHOICE_STUB = None   # suites: CHOICE_STUB(question, text) -> option
CITY_STUB = None     # suites: CITY_STUB(fact) -> city (bot/tg_weather._city_of_fact)


def ask_choice(question: str, text: str, options: tuple, default: str = "") -> str:
    """One word picked from `options` about `text` ({text} in the question).
    Steadier than a yes/no where the answer is a NAME: «reset song settings»
    -> yes/no said no, the pick said "song" (2026-10-02). Model down or an
    answer outside the options -> `default`."""
    text = (text or "").strip()
    if not text:
        return default
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return (CHOICE_STUB(question, text) if CHOICE_STUB else default) or default
    key = (question, text[:1500], options)
    with _LOCK:
        if key in _YES_CACHE:
            return _YES_CACHE[key]
    try:
        import llm
        ans = llm.call_llm_simple(None, "Answer with one word.",
                                  question.replace("{text}", "«%s»" % text[:1500])
                                  + "\nOne word: " + ", ".join(options) + ".",
                                  temperature=0.0, max_tokens=5) or ""
        # options can be several words ("oil painting"): the longest option the
        # answer starts with, not just its first word
        said = " ".join(re.sub(r"[^\w\s-]", " ", ans.lower()).split())
        hits = [o for o in options if said == o.lower() or said.startswith(o.lower() + " ")]
        got = max(hits, key=len) if hits else default
    except Exception:
        logger.warning("ask_choice: model read failed", exc_info=True)
        return default
    with _LOCK:
        _YES_CACHE[key] = got
    return got


def ask_yes(question: str, text: str, default: bool = False) -> bool:
    """One narrow yes/no about `text`, for checks that live inside one flow
    (a song topic, a video description, an edit instruction) rather than in
    the per-message read. Cached; model down -> `default`."""
    text = (text or "").strip()
    if not text:
        return default
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return bool(YES_STUB(question, text)) if YES_STUB else default
    key = (question, text[:1500])
    with _LOCK:
        if key in _YES_CACHE:
            return _YES_CACHE[key]
    try:
        import llm
        ans = llm.call_llm_simple(None, "Answer yes or no only.",
                                  # the question names the text in its own words: a bare
                                  # "Text:" label was read as the answer («the cat» ->
                                  # is it lettering? Yes)
                                  question.replace("{text}", "«%s»" % text[:1500]),
                                  temperature=0.0, max_tokens=5) or ""
        got = ans.strip().lower().startswith(("yes", "да"))
    except Exception:
        logger.warning("ask_yes: model read failed", exc_info=True)
        return default
    with _LOCK:
        _YES_CACHE[key] = got
        while len(_YES_CACHE) > 512:
            _YES_CACHE.popitem(last=False)
    return got

# Suites script the model's replies call by call; an extra intent call would
# eat one. Under F5_TEST_RUN the read is STUB(text) when a suite sets it, else
# FALLBACK (full loop, nothing forced).
STUB = None

_MENU = None
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()


def _menu():
    """(names, text): the agent's tools, one line each -- what `wants` picks from."""
    global _MENU
    if _MENU is None:
        import graph
        rows = []
        for sc in graph.TOOL_SCHEMAS:
            f = sc.get("function", {})
            first = re.split(r"(?<=[.!?])\s", (f.get("description") or "").strip(), maxsplit=1)[0]
            rows.append((f.get("name", ""), first[:140]))
        _MENU = ([n for n, _ in rows], "\n".join(f"  {n} -- {d}" for n, d in rows))
    return _MENU


def read(ctx, text: str, previous: str = "", attached: str = "") -> dict:
    """`previous`: the last assistant line. `attached`: what came with the
    message ("a photo", "a CSV with numbers"), "" when nothing did."""
    text = (text or "").strip()
    if not text:
        return dict(FALLBACK)
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        got = STUB(text) if STUB else None
        return {**FALLBACK, **got, "ok": True} if got else dict(FALLBACK)
    key = (text, (previous or "")[:300], attached)
    with _LOCK:
        if key in _CACHE:
            return dict(_CACHE[key])
    # The message is DATA to classify: bare, the model answered it instead
    # («объясни теорему Пифагора» came back as the theorem, 26/134).
    user = ""
    if previous:
        user += "Previous assistant line: «%s»\n\n" % previous[:300]
    if attached:
        user += "Attached to the message: %s\n\n" % attached
    user += ("Message to route (do NOT answer it):\n«%s»\n\n"
             "Reply with the JSON object only." % text[:2000])
    try:
        import llm
        from utils import safe_json_from_llm
        names, menu = _menu()
        raw = llm.call_llm_simple(ctx, SYSTEM.format(menu=menu), user,
                                  temperature=0.0, max_tokens=300) or ""
        got = _validate(safe_json_from_llm(raw, FIELDS))
        out = got.model_dump()
        out["wants"] = [w for w in out["wants"] if w in names]
        # deep_research costs minutes: offered when wanted, never forced
        # («что было с 2020 по 2024 год» was read as a report).
        out["must_call"] = (out["must_call"] if out["must_call"] in names
                            and out["must_call"] != "deep_research" else "")
        w = got.weather
        # other questions in the same message (other tools wanted) need the
        # full loop, not the forecast shortcut
        out["weather"] = ({"city": w.city.strip(), "when": w.when, "by_reference": w.place_by_reference}
                          if w and set(out["wants"]) <= {"weather_forecast"} else None)
        out["ok"] = True
    except Exception:
        logger.warning("intent: model read failed, taking the full loop", exc_info=True)
        return dict(FALLBACK)
    logger.info("intent: %r -> %s", text[:80], out)
    with _LOCK:
        _CACHE[key] = out
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)
    return dict(out)
