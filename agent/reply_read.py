"""What the assistant's own reply CLAIMS, read once per reply by the model.

Replaces the word lists that judged replies (turn_audit._CLAIMS, the fast-path
delivery/action/denial regexes, graph_finalize's action, promise, already-done,
stated-result, file-ready and admission patterns): «Ухожу в спящий режим»,
«Сделал вариант для обложки» and «проверим три причины» are claims or not by
what the sentence says, not by which verb stem it holds. The guards still
decide in code -- a claim with no successful tool behind it -- the model only
reads the sentence. bench/reply_read_eval.py scores it on labelled live replies.

read(ctx, reply, user="") -> dict, cached per input:
  did             what the reply says was DONE in this turn, from DID; the model quotes the
                  words of each claim and a quote not found in the reply is dropped
  delivers        it presents something as attached / shown / below right now
  promises        it announces doing something now or next instead of having done it (quoted
                  and checked against the reply like `did`)
  already         it says the thing was already done in an earlier turn
  computed        it states the numeric result of a calculation
  admits_failure  it says something failed or could not be done
  denies          it says the assistant cannot do or has no access to something
  tool_as_text    it writes a tool call out as text (JSON with an action/name)
  does_tool_job   it hands over a tool's input instead of the result: a prompt for
                  an image generator, what code "will print"
On any failure: NOTHING -- no claims, so no guard rewrites a reply on a guess.
"""
from __future__ import annotations
from seams import text_model

import logging
import os
import threading
from collections import OrderedDict
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("assistant." + __name__)

DID = ("draw", "video", "file", "song", "voice", "remember", "forget", "reminder", "reminder_cancel",
       "cart", "pickup_point", "ran_code", "edited_file", "other")

SYSTEM = (
    "You read a reply an AI assistant wrote to a user and report what the reply CLAIMS. "
    "Do not judge whether the claims are true; read what the words say.\n"
    "did: each action the reply says the assistant DID in this turn, as finished, with the "
    "reply's own words that say it: [{\"what\": kind, \"words\": \"exact words copied from the "
    "reply\"}]. An empty list when no sentence reports a finished action. Kinds, the most "
    "specific first: draw -- made, changed, fixed, recoloured, moved or replaced something in a "
    "picture or photo (\"перенёс его на кухню\", \"заменил человека\", \"убрал пятно\"); "
    "video -- made, continued or animated a video (\"оживил портрет\"); file -- a document, "
    "deck, archive or table file; song; voice -- a voice message; remember -- saved a fact "
    "to memory; forget -- deleted facts from memory; reminder -- set a reminder, timer or "
    "alarm, including \"напомню через минуту\"; reminder_cancel; cart -- added to a shopping "
    "cart or list; pickup_point -- set a delivery point; ran_code -- ran code or tests; "
    "edited_file -- changed a file on disk; other -- an action with an effect that fits none "
    "of these. A result said to be ready (\"коллаж готов\", \"the file is ready\") is a claim "
    "of the kind that made it. Not a claim: a plan, an offer, a conditional, an attempt that "
    "failed, an explanation, a figure of speech (\"сделаю вывод\"), reading or describing "
    "something, predicting what code prints, or what the USER did.\n"
    "delivers: the reply presents a separate result -- a picture, photo, video, file, voice "
    "message or plan card -- as attached, shown or placed below right now. You cannot see "
    "attachments: \"вот твой рыжий кот\", \"вот реальное фото...\", \"here is your image\", "
    "\"файл прикреплён\", \"план ниже\", \"вот так выглядит...\" all present one. Text the "
    "reply itself contains (a list, a plan written out, an answer, a number, a prompt) is not "
    "a delivery.\n"
    "promises: the reply's own words that announce the assistant doing something now or "
    "next instead of having done it, copied exactly: [\"сначала я распакую архив\", "
    "\"начинаю поиск\", \"сформирую новый коллаж\", \"попробую ещё раз\"]. Every such "
    "phrase, in any tense that means \"I am about to / I am starting to\". Not a promise: "
    "an offer or a question (\"могу сделать, хочешь?\"), a choice put to the user, a step "
    "the USER is asked to do, an assurance that a reminder will fire. [] when there is none.\n"
    "already: ONLY when the reply says the requested work was done in an EARLIER turn or "
    "message, as the reason not to do it now (\"я уже сделал это в прошлом ответе\"). "
    "\"Готово\", \"уже у тебя\", \"теперь\" about this turn's result are not already.\n"
    "computed: the reply states the numeric result of a calculation.\n"
    "admits_failure: the reply says an attempt failed, did not work or could not be "
    "completed. Saying the assistant cannot do something at all is denies, not this.\n"
    "denies: the reply says the assistant cannot do something, has no access to it, is "
    "only a text model, or sends the user to another service.\n"
    "tool_as_text: the reply writes a tool call out as text (JSON with \"action\", "
    "\"name\"/\"arguments\", <tool_call>).\n"
    "does_tool_job: the reply hands over what a tool would need instead of the result -- a "
    "prompt for an image generator, what a piece of code \"will print\".\n"
    "Answer with ONE JSON object and nothing else: {\"did\": [{\"what\": kind, \"words\": str}], \"delivers\": bool, "
    "\"promises\": [str], \"already\": bool, \"computed\": bool, \"admits_failure\": bool, "
    "\"denies\": bool, \"tool_as_text\": bool, \"does_tool_job\": bool}")


class Claim(BaseModel):
    what: Literal[DID]
    words: str


class Read(BaseModel):
    did: list[Claim] = []
    delivers: bool = False
    promises: list[str] = []
    already: bool = False
    computed: bool = False
    admits_failure: bool = False
    denies: bool = False
    tool_as_text: bool = False
    does_tool_job: bool = False


NOTHING = {**Read().model_dump(), "promises": False, "ok": False}
FIELDS = tuple(Read.model_fields)


def _validate(got: dict) -> Read:
    """A field the model got wrong takes its default; an unknown action in `did` is dropped."""
    got = {k: v for k, v in (got or {}).items() if k in Read.model_fields}
    if isinstance(got.get("did"), list):
        got["did"] = [d for d in got["did"] if isinstance(d, dict) and d.get("what") in DID]
    while True:
        try:
            return Read.model_validate(got)
        except ValidationError as e:
            bad = {err["loc"][0] for err in e.errors() if err["loc"]}
            if not bad & got.keys():
                return Read()
            got = {k: v for k, v in got.items() if k not in bad}


_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()


def read(ctx, reply: str, user: str = "") -> dict:
    """`user`: the message the reply answers -- context, not the thing read."""
    reply = (reply or "").strip()
    if not reply:
        return dict(NOTHING)
    # Suites script the model call by call; under F5_TEST_RUN they script this read on the
    # context too (seams.reply_read), so an unscripted one claims nothing.
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return dict(NOTHING)
    key = (reply[:3000], (user or "")[:500])
    with _LOCK:
        if key in _CACHE:
            return dict(_CACHE[key])
    msg = (("The user wrote: «%s»\n\n" % user[:500]) if user else "") + (
        "The assistant's reply to read (do NOT answer it):\n«%s»\n\nReply with the JSON object only."
        % reply[:3000])
    try:
        from utils import safe_json_from_llm
        raw = text_model(ctx)(ctx, SYSTEM, msg, temperature=0.0, max_tokens=400) or ""
        out = _validate(safe_json_from_llm(raw, FIELDS)).model_dump()
        # A claim is only what the reply says: the quoted words must be in it.
        flat = " ".join(reply.lower().split())
        said = lambda w: bool(w.strip()) and " ".join(w.lower().split()) in flat
        out["did"] = sorted({c["what"] for c in out["did"] if said(c["words"])})
        out["promises"] = any(said(w) for w in out["promises"])
        out["ok"] = True
    except Exception:
        logger.warning("reply_read: model read failed, no claims assumed", exc_info=True)
        return dict(NOTHING)
    logger.info("reply_read: %r -> %s", reply[:80], {k: v for k, v in out.items() if v})
    with _LOCK:
        _CACHE[key] = out
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)
    return dict(out)


def claims_result(r: dict) -> bool:
    """The reply says something was made or done, or hands it over."""
    return bool(r.get("did") or r.get("delivers"))
