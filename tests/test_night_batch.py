"""The batch of complaints from 2026-08-01, each pinned to a check.

Every one of these was a real report, not a hypothesis:

  1. "why are you feeding it all the tools when I only press one button" — the
     inline image buttons were missing from the intent map, so the single most
     unambiguous input in the product shipped all 14 tool schemas every round.
  2. "after clearing the chat it repeated all the steps back to me" — pinned facts
     survived clear_context and are injected into EVERY turn.
  3. "every time I ask for a redraw — no response" — an empty final_answer reached
     the delivery layer, which printed the literal placeholder "(no response)".
  4. "I asked for 5 jokes and got 'there's one about an aeroplane'" — the search
     distiller was hard-capped at "1 to 4 sentences" with a 400-token budget.
  5. "deep research spat out unstructured text" — my own retry nudge told the model
     to drop everything but the text, and it dropped the Markdown with it.
  6. forwarded voice notes: transcript or summary, by choice.
  7. PowerPoint decks as a real deliverable file.

Run: venv/Scripts/python.exe tests/test_night_batch.py
"""
import sys, os, types, tempfile, inspect, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_night_")
import tg_bot as T

# The 📋 retelling is a ONE-SHOT model call outside the agent since 1cdadf4
# (2026-09-14): it never enters the task queue. The suite records that call.
import llm as _llm_mod, time as _time
RETOLD = []
def _fake_retell(ctx, system, said, **k):
    RETOLD.append(said)
    return "Кратко: " + said[:40]
_llm_mod.call_llm_simple = _fake_retell
def retold_after(n_prev, timeout=4.0):
    """The transcripts retold since RETOLD had n_prev entries (thread joined)."""
    t0 = _time.time()
    while len(RETOLD) <= n_prev and _time.time() - t0 < timeout:
        _time.sleep(0.05)
    return RETOLD[n_prev:]
T.redirect_data_dir(_DATA_DIR)
import graph as G
import graph_personality as GP
import slides
import search as S
from tools import TOOL_SCHEMAS, TOOLS

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999931


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []; bot.pushed = []; bot.docs = []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append((text, kw.get("keyboard"))), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: bot._send_text(cid, text, **kw)
    bot._send_document = lambda cid, path, caption="": (
        bot.docs.append(path), True)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda t: bot.pushed.append(t), depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"OggS" + os.urandom(32)
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


def texts(bot):
    return " ".join(t for t, _ in bot.sent)


bot = make_bot()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. ONE BUTTON MUST NOT SHIP THE WHOLE TOOL LIST")
print("=" * 70)

total = len(TOOL_SCHEMAS)
check("there are enough tools for this to matter", total >= 10, total)
for key, payload in T._CB_CMDS.items():
    only = G._intent_tools(payload)
    check(f"the {key} button narrows the tool list",
          only is not None and 0 < len(only) < total,
          f"{key} -> {only if only is None else sorted(only)} of {total}")
check("regenerate asks for exactly one tool",
      G._intent_tools(T._CB_CMDS["regenerate"]) == frozenset({"generate_image"}),
      G._intent_tools(T._CB_CMDS["regenerate"]))
check("a typed sentence still gets the full set (no over-narrowing)",
      G._intent_tools("what do you think about this") is None)
# The prefixes must MATCH the payloads, not merely exist.
for prefix, names in G._INTENT_TOOL_MAP:
    check(f"the intent prefix {prefix[:26]!r} resolves",
          G._intent_tools(prefix + " x") == names, prefix)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. CLEAR MEANS CLEAR")
print("=" * 70)

s = bot._get_session(CID)
s.set_tg_facts([{"ts": 1, "text": "the user is building a tractor deck"}])
s.history = [{"role": "user", "content": "step one"}]
s.set_tg_memory(["generate: a tractor"])
s.last_image_prompt = "a tractor"
s.clear_context()
for field, val in (("pinned facts", s.get_tg_facts()), ("history", s.history),
                   ("session memory", s.get_tg_memory()),
                   ("image register", s.image_log)):
    check(f"clear drops the {field}", val == [], val)
check("clear drops the remembered image prompt", s.last_image_prompt == "")
# Facts are injected on EVERY turn — that is why leaving them behind read as
# "it repeated all my steps back to me".
check("facts really are per-turn context, so this mattered",
      "pinned_facts" in inspect.getsource(T))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. NEVER ANSWER WITH SILENCE")
print("=" * 70)

# _finalize_answer (the retry ladder + last-resort text) has moved twice now:
# out of graph.py into graph_personality.py, and from there into
# graph_finalize.py when the personality cluster was split into layers. Each
# move silently emptied this structural check, because `inspect.getsource(GP)`
# named a FILE rather than the code under test. Ask the function itself which
# module it lives in, so the next move cannot break the check either.
src = inspect.getsource(inspect.getmodule(GP._finalize_answer))
check("the closing reply is retried, not attempted once",
      "closing reply was empty — retry" in src)
check("the retry SHRINKS the budget (more room = more deliberation, not more answer)",
      "(1.0, 0.5, 0.3)" in src, "retry ladder not found")
check("the last-resort answer no longer requires that tools ran",
      "if not final_answer and not ctx.is_cancelled():" in src)
check("and it admits the fault instead of blaming the user",
      "not a refusal" in src)
# The delivery layer's placeholder is what the user actually SAW -- verbatim,
# in English, in a Russian chat. The guard still exists (voice must not be
# synthesised for an empty reply) but it is now a flag, and what goes out is a
# translated line saying a retry usually works.
_delivery = "".join(inspect.getsource(_m)
                    for _n, _m in inspect.getmembers(T.TelegramBot, inspect.isfunction))
check("the empty-reply guard still exists", "empty_reply" in _delivery)
check("and the placeholder is no longer what the user is shown",
      "(no response)" not in _delivery.replace("# ", "")
      or 'or "(no response)"' not in _delivery)
check("an artefact-only turn sends no 'ничего не ответила' line -- the picture "
      "IS the answer",
      'final.get(k) for k in (' in _delivery and 'Every reply goes out as text too' in _delivery and 'reply = "" if produced' in _delivery)
check("the shown text is localized",
      '"empty_reply"' in (ROOT / "bot/tg_strings.py").read_text(encoding="utf-8")
      if "ROOT" in dir() else '_t("empty_reply"' in _delivery)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. FIVE JOKES MEANS FIVE JOKES")
print("=" * 70)

# the model's count read is stubbed; the phrases run live in bench/intent_sweep_live.py
import intent as _intent
_COUNTS = {"5 анекдотов про Чапаева и Петьку": "5", "five jokes about Chapaev": "5",
           "три рецепта борща": "3", "list 10 models": "10", "give me 2 examples": "2",
           "500 examples": "500"}
_intent.CHOICE_STUB = lambda q, t: _COUNTS.get(t, "0") if "NUMBER of items" in q else None
for q, n in (("5 анекдотов про Чапаева и Петьку", 5),
             ("five jokes about Chapaev", 5),
             ("три рецепта борща", 3),
             ("list 10 models", 10),
             ("give me 2 examples", 2)):
    check(f"'{q[:30]}' is read as an order for {n}", S.wanted_count(q) == n,
          S.wanted_count(q))
for q in ("iphone 5 price", "погода в Москве", "what is the capital of Peru",
          "курс доллара"):
    check(f"'{q[:26]}' is NOT mistaken for a counted request",
          S.wanted_count(q) == 0, S.wanted_count(q))
check("a silly count is ignored", S.wanted_count("500 examples") == 0)

from prompts import SEARCHER_PROMPT
check("the prompt no longer hard-caps every answer at 4 sentences",
      "1 to 4 sentences maximum" not in SEARCHER_PROMPT)
check("it still keeps SHORT answers short",
      "1 to 4 sentences" in SEARCHER_PROMPT)
check("it names the failure it must not repeat",
      "there is one about an aeroplane" in SEARCHER_PROMPT)
check("it forbids inventing items to pad the count",
      "invented" in SEARCHER_PROMPT)
ssrc = inspect.getsource(S)
check("the token budget scales with the count", "220 * want + 300" in ssrc)
check("the FETCH widens too — you cannot extract 5 jokes from 5 snippets",
      "max(SEARCH_MAX_RESULTS, want * 2)" in ssrc)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. THE DEEP-RESEARCH RETRY MUST NOT COST THE MARKDOWN")
print("=" * 70)

import deep_research as D
tsrc = inspect.getsource(D._think_call)
check("the retry exists", "retry" in tsrc.lower())
check("and it demands the Markdown structure back",
      "heading" in tsrc and "Markdown" in tsrc, tsrc[-400:])
check("it does not just say 'output only the finished text'",
      "output only the finished text." not in tsrc)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. A FORWARDED VOICE NOTE: TRANSCRIPT OR SUMMARY, BY CHOICE")
print("=" * 70)

check("an empty forward_origin dict still counts as forwarded",
      T._is_forwarded({"forward_origin": {}}))
for k in ("forward_from", "forward_from_chat", "forward_sender_name", "forward_date"):
    check(f"{k} counts as forwarded", T._is_forwarded({k: 1}))
check("an ordinary message is not a forward",
      not T._is_forwarded({"text": "hi", "chat": {"id": 1}}))

bot.sent.clear(); bot.pushed.clear()
bot._transcribe = lambda ctx, fid, media="voice", **kw: "тут длинный монолог про доставку в четверг"
bot._get_ctx = lambda: object()
bot._resolve_and_push(CID, [{"type": "fwd_voice", "file_id": "V1",
                             "seconds": 95, "caption": ""}])
s = bot._get_session(CID)
check("a forwarded voice note is NOT fed in as an instruction", not bot.pushed,
      [p.user_text for p in bot.pushed])
check("the transcript is held", "доставку" in s.fwd_transcript, s.fwd_transcript)
check("the user is asked what to do with it",
      any("fwdv:" in json.dumps(kb or {}) for _t_, kb in bot.sent), texts(bot)[:120])
kb = [kb for _t_, kb in bot.sent if kb and "inline_keyboard" in kb][-1]
picks = [b["callback_data"] for r in kb["inline_keyboard"] for b in r]
# Each button carries the forward's id (fwdv:<choice>:<id>) so a stale button
# from an earlier, since-superseded forward can be told apart from the
# current one — see _Session.fwd_transcript_id.
check("all the choices are offered",
      {p.split(":", 2)[1] for p in picks} == {"text", "sum", "both", "own"}, picks)
check("every button carries the SAME forward id",
      len({p.split(":", 2)[2] for p in picks}) == 1
      and all(len(p.split(":", 2)) == 3 for p in picks), picks)
check("the length is quoted so the choice is informed",
      "1" in texts(bot) or "мин" in texts(bot) or "min" in texts(bot),
      texts(bot)[:120])


def press(data):
    bot.sent.clear(); bot.pushed.clear()
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": 9}}})


s.fwd_transcript = "тут длинный монолог про доставку в четверг"
bot._store.put(s)
press("fwdv:text")
check("the transcript is returned VERBATIM, not paraphrased by the model",
      "доставку" in texts(bot) and not bot.pushed,
      f"sent={texts(bot)[:80]} pushed={len(bot.pushed)}")

s.fwd_transcript = "тут длинный монолог про доставку в четверг"
bot._store.put(s)
_n = len(RETOLD)
press("fwdv:sum")
_got = retold_after(_n)
check("summary goes through the model (one-shot, not the agent queue)",
      len(_got) == 1 and not bot.pushed, (_got, len(bot.pushed)))
check("and the transcript travels with the request",
      _got and "доставку" in _got[0])

s.fwd_transcript = "x"
bot._store.put(s)
_n = len(RETOLD)
press("fwdv:both")
_got = retold_after(_n)
check("'both' sends the text AND retells it",
      bot.docs is not None and len(_got) == 1 and texts(bot),
      f"sent={len(bot.sent)} retold={len(_got)}")

# A finished choice no longer makes the keyboard stale: the transcript is kept
# so the other two buttons still work (that is the whole point -- the inline
# keyboard stays on screen, and wiping the transcript made the second press
# claim the note was gone). What IS stale is a button belonging to an EARLIER
# forward, which is what carries an id that no longer matches.
s.fwd_transcript = "тут длинный монолог про доставку в четверг"
s.fwd_transcript_id = "curr1234"
bot._store.put(s)
press("fwdv:text:oldid999")
check("a stale button says so instead of doing nothing",
      T._t("fwd_voice_gone", "ru")[:12] in texts(bot)
      or T._t("fwd_voice_gone", "en")[:12] in texts(bot), texts(bot)[:100])
check("the transcript is cleared by a context wipe",
      (bot._get_session(CID).clear_context() or True)
      and bot._get_session(CID).fwd_transcript == "")

# A second forward arriving before the first is answered used to silently
# overwrite the single pending transcript, so tapping the FIRST (still
# on-screen) "which do you want?" prompt delivered the SECOND voice's
# content under it -- reproduced live via forward A -> ignore -> forward B
# -> tap A's button.
s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
bot._transcribe = lambda ctx, fid, media="voice", **kw: {
    "VA": "voice A content", "VB": "voice B content"}.get(fid, "")
bot.sent.clear(); bot.pushed.clear()
bot._resolve_and_push(CID, [{"type": "fwd_voice", "file_id": "VA",
                             "seconds": 5, "caption": ""}])
kb_a = [kb for _t_, kb in bot.sent if kb and "inline_keyboard" in kb][-1]
data_a = kb_a["inline_keyboard"][0][0]["callback_data"]

bot._resolve_and_push(CID, [{"type": "fwd_voice", "file_id": "VB",
                             "seconds": 5, "caption": ""}])
kb_b = [kb for _t_, kb in bot.sent if kb and "inline_keyboard" in kb][-1]
data_b = kb_b["inline_keyboard"][0][0]["callback_data"]
check("the second forward's button carries a different id from the first's",
      data_a != data_b, (data_a, data_b))

press(data_a)
# live 2026-09-27: two forwards in a row -- the FIRST prompt must still deliver A
check("the first prompt's button delivers A's content, not B's and not 'gone'",
      "voice A content" in texts(bot) and "voice B content" not in texts(bot),
      texts(bot)[:120])

press(data_b)
check("the CURRENT (second) button still delivers B's content correctly",
      "voice B content" in texts(bot), texts(bot)[:120])

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. A REAL POWERPOINT FILE")
print("=" * 70)

check("the tool is registered", any(t.name == "create_presentation" for t in TOOLS))
check("and reaches the model's schema list",
      any(s_["function"]["name"] == "create_presentation" for s_ in TOOL_SCHEMAS))

deck = slides.normalize_deck({
    "title": "Тракторы", "subtitle": "краткий обзор",
    "slides": [
        {"heading": "История", "bullets": ["первый трактор", "паровая тяга"],
         "notes": "говорить про 1892", "image": "vintage tractor"},
        {"not": "a slide"},
        {"heading": "Сегодня", "bullets": "джон дир\nкировец"},
    ]}, "тракторы")
check("a malformed slide is dropped, not the deck", len(deck["slides"]) == 2,
      [s_["heading"] for s_ in deck["slides"]])
check("bullets given as one string are split",
      deck["slides"][1]["bullets"] == ["джон дир", "кировец"],
      deck["slides"][1]["bullets"])
check("an empty plan still yields an openable deck",
      len(slides.normalize_deck({}, "тема")["slides"]) == 1)
check("too many bullets are trimmed",
      len(slides.normalize_deck(
          {"slides": [{"heading": "h", "bullets": [str(i) for i in range(30)]}]}
      )["slides"][0]["bullets"]) == slides.MAX_BULLETS)

out = tempfile.mkdtemp(prefix="deck_")
path = slides.build_pptx(deck, os.path.join(out, "d.pptx"))
check("a real file is written", path and os.path.exists(path))
check("and it is a real OOXML package (starts with PK)",
      open(path, "rb").read(2) == b"PK", open(path, "rb").read(4))
from pptx import Presentation
prs = Presentation(path)
check("title slide + one per content slide", len(prs.slides._sldIdLst) == 3,
      len(prs.slides._sldIdLst))
check("it is 16:9, not the 4:3 default",
      round(prs.slide_width / prs.slide_height, 2) == 1.78,
      prs.slide_width / prs.slide_height)
check("speaker notes survive",
      "1892" in prs.slides[1].notes_slide.notes_text_frame.text)
allrun = " ".join(sh.text_frame.text for sl in prs.slides for sh in sl.shapes
                  if sh.has_text_frame)
check("the Russian text is intact", "Тракторы" in allrun and "Кировец" in allrun.title()
      or "кировец" in allrun, allrun[:120])
check("a missing picture costs the picture, not the slide",
      slides.build_pptx(deck, os.path.join(out, "e.pptx"),
                        images={0: os.path.join(out, "nope.png")}) is not None)

# The bot has to actually SEND it — a tool that reports success while the user
# receives nothing is the exact failure this project keeps re-learning.
# Follow the METHOD on the class, not the module file: the turn body moved into
# tg_tasks.py when the monolith was split, and a grep of tg_bot.py then reported
# document delivery missing while it worked perfectly.
bsrc = ""
for _n, _m in inspect.getmembers(T.TelegramBot, inspect.isfunction):
    _s = inspect.getsource(_m)
    if "[doc delivery]" in _s:
        bsrc = _s
        break
check("the turn has a document-delivery block", bool(bsrc))
check("the bot delivers document_path", 'final.get("document_path")' in bsrc)
check("it checks the file is really there first",
      "[doc delivery] path does not exist" in bsrc)
check("and it tells the user when the send fails",
      "doc_send_failed" in bsrc)
check("the failure message exists in both languages",
      set(T._MSG["doc_send_failed"]) >= {"en", "ru"})

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
