"""Two more live-chat defects: a promised deck that was never built, and a
forwarded voice note destroyed by an unusable button payload.

1. DECK NARRATED, NEVER BUILT. 📊 Presentation sends the literal prefix
   "create a presentation about:", so the round is offered exactly one tool —
   and the model still answered "План презентации включает историю создания…"
   without calling create_presentation. No file exists, and the user is told
   about a presentation they cannot open. Same fabrication family as the image
   and remember_fact guards, which is where the fix goes.

2. A FORWARDED NOTE ERASED BY AN UNKNOWN CHOICE. _do_fwd_voice cleared the held
   transcript BEFORE looking at `what`, and every branch is keyed on `what`. An
   unrecognised payload (a keyboard from an older build) therefore matched
   nothing, having already destroyed the only copy: the press produced total
   silence and the next one answered "I no longer have that voice message".

Run: venv/Scripts/python.exe tests/test_deck_and_fwdvoice_guards.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_deckguard_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as G
import graph_personality as GP

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


# ── 1. the deck guard ────────────────────────────────────────────────────────
print("=" * 66)
print("A PRESENTATION THAT WAS DESCRIBED BUT NEVER BUILT")
print("=" * 66)

import inspect
# These guards used to live in the personality_node CLOSURE inside build_graph,
# so this read the whole agent loop. The nodes are module-level functions now
# and build_graph is bare wiring; personality_node itself later moved out of
# graph.py entirely, to graph_personality.py (seam-blocked names read back
# through `graph` at call time, everything else moved outright). Read THAT
# module: a source-text anchor that silently finds nothing is worse than
# useless, and graph_personality is where the guard actually lives now.
src = inspect.getsource(GP)

check("the deck prefix is matched literally, not guessed",
      G._DECK_PREFIX == "create a presentation about:")
check("a draft with no create_presentation call forces a corrective round",
      "deck_corrective_sent" in src and
      '"create_presentation" not in tools_called_this_turn' in src)
check("the corrective tells the model an outline is not a file",
      "An outline in chat is not a file" in src or
      "outline in chat is not a file" in src)
check("there is a last resort that builds it from the topic",
      'execute_tool(ctx, state, "create_presentation"' in src)
check("the guard sits with the other fabrication guards",
      src.index("deck_corrective_sent") > src.index("memory_corrective_sent"))

# the button's prefix and the guard's prefix must be the same string
check("the Telegram button sends exactly the prefix the guard matches",
      T._PROMPT_KB["deck"].strip().lower().rstrip() == G._DECK_PREFIX,
      f"{T._PROMPT_KB['deck']!r} vs {G._DECK_PREFIX!r}")
# …and the intent map must offer that tool
_tools = G._intent_tools("create a presentation about: тракторы")
check("the intent map offers create_presentation for that prefix",
      _tools is not None and "create_presentation" in _tools, _tools)

# ── 2. the forwarded-voice choice ────────────────────────────────────────────
print("=" * 66)
print("AN UNKNOWN CHOICE MUST NOT DESTROY THE TRANSCRIPT")
print("=" * 66)

CID = 999871


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: bot.sent.append((t, kw.get("keyboard")))
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


SAID = "по итогам квартала выручка выросла на двенадцать процентов"

bot = make_bot()
sess = bot._get_session(CID)
sess.fwd_transcript = SAID
bot._store.put(sess)

bot._do_fwd_voice(CID, "summary", SAID)      # not one of text/sum/both
check("an unknown choice answers instead of going silent", bot.sent, bot.sent)
check("it KEEPS the transcript",
      bot._get_session(CID).fwd_transcript == SAID,
      bot._get_session(CID).fwd_transcript)
kb = bot.sent[-1][1] if bot.sent else None
data = [b.get("callback_data") for r in (kb or {}).get("inline_keyboard", [])
        for b in r]
check("it re-offers the real choices", set(data) == {"fwdv:text", "fwdv:sum",
                                                     "fwdv:both", "fwdv:own"}, data)

# the real payloads still work, and mark the note as acted on WITHOUT
# discarding it -- the keyboard is inline and stays clickable, so wiping the
# transcript here made every choice after the first answer 'that voice note
# is no longer in my hands' for a note still on screen (see test_voice_routing
# section F). Staleness is decided by the button's forward id, not by whether
# some earlier choice already ran.
for what, must in (("text", "выручка"), ("both", "выручка")):
    bot = make_bot()
    s = bot._get_session(CID)
    s.fwd_transcript = SAID
    bot._store.put(s)
    bot._resolve_and_push = lambda cid, batch: bot.sent.append((str(batch), None))
    bot._do_fwd_voice(CID, what, SAID)
    joined = " ".join(t for t, _ in bot.sent)
    check(f"choice {what!r} delivers the note", must in joined, joined[:120])
    check(f"choice {what!r} keeps the transcript for the other buttons",
          bot._get_session(CID).fwd_transcript == SAID)
    check(f"choice {what!r} marks the forward as acted on",
          bot._get_session(CID).fwd_transcript_done)

bot = make_bot()
s = bot._get_session(CID)
s.fwd_transcript = SAID
bot._store.put(s)
pushed = []
bot._resolve_and_push = lambda cid, batch: pushed.append(batch)
retold = []
bot._retell_transcript = lambda cid, said, lang, is_video: retold.append((said, is_video))
bot._do_fwd_voice(CID, "sum", SAID)
import time as _time
for _ in range(50):
    if retold: break
    _time.sleep(0.05)
# The retelling is a one-shot model call over the transcript, NOT an agent
# turn: through the agent it inherited the chat's last picture and noticed a
# second press as «тот же текст» (live 2026-09-14 23:02).
check("choice 'sum' retells the transcript one-shot, not through the agent",
      retold == [(SAID, False)] and not pushed, (retold, pushed))
retold.clear()
bot._do_fwd_voice(CID, "sum", SAID + "\n\n👁 Раскадровка видео:\n0:00 — лампа")
for _ in range(50):
    if retold: break
    _time.sleep(0.05)
check("a transcript with a storyboard is retold as a VIDEO",
      retold and retold[0][1] is True, retold)

# every callback_data the keyboard can produce must be handled
kb = T._fwd_voice_kb("ru", board=True)
produced = {b["callback_data"].split(":", 1)[1]
            for r in kb["inline_keyboard"] for b in r}
check("every button the keyboard offers is a choice _do_fwd_voice accepts",
      produced <= {"text", "sum", "both", "board", "own"}, produced)

# 3. THE VIDEO MUST BE IN THE HISTORY THE MODEL SEES. Live 15:39: the retelling
# ended with «Что требуется: дать экспертное мнение», the user typed «Дай
# экспертное мнение», and the model -- whose history had no trace of the
# video -- offered topics from days-old chats. The retelling is a one-shot call
# outside the agent, so the turn has to be committed by hand.
import llm as _llm
bot = make_bot()
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: "Основная тема: оценка ремонта кухни.\nЧто требуется:\n- дать экспертное мнение"
VIDEO_SAID = "оцени косяки на стенах\n\n👁 Раскадровка видео:\n0:09 — плакат на стене"
_s = bot._get_session(CID); _s.set_history([]); bot._store.put(_s)   # earlier sections wrote here too
bot._retell_transcript(CID, VIDEO_SAID, "ru", True)
hist = bot._get_session(CID).get_history()
check("a retold video lands in the chat history as a user turn + the retelling",
      len(hist) == 2 and hist[0]["role"] == "user" and "forwarded video message" in hist[0]["content"]
      and "Раскадровка" in hist[0]["content"] and hist[1]["role"] == "assistant"
      and "экспертное мнение" in hist[1]["content"], hist)
bot._retell_transcript(CID, VIDEO_SAID, "ru", True)
hist = bot._get_session(CID).get_history()
check("a second press adds only its answer, not the transcript again",
      len(hist) == 3 and hist[2]["role"] == "assistant", [m["role"] for m in hist])
bot._do_fwd_voice(CID, "text", SAID)
hist = bot._get_session(CID).get_history()
check("a transcript delivered verbatim is remembered too, as a voice message",
      len(hist) == 5 and "forwarded voice message" in hist[3]["content"] and SAID in hist[3]["content"], hist[3:])

# 4. A follow-up about a video looks at the FRAMES again. The keyframe sheet
# is the chat's current picture; "дай экспертное мнение" has no question word,
# but the model reads it as about the video (intent about_picture).
import graph as _graph, types as _types, os as _os, tempfile as _tempfile, intent as _intent
_intent.STUB = {"дай экспертное мнение": {"about_picture": True},
                "нарисуй кота": {"needs_tool": True, "wants": ["generate_image"]}}.get
_vd = _os.path.join(_tempfile.mkdtemp(), "video_ab12cd34"); _os.makedirs(_vd)
_sheet = _os.path.join(_vd, "sheet.jpg"); open(_sheet, "wb").close()
_ctx = _types.SimpleNamespace(last_image_path=_sheet)
check("the video sheet is recognised", _graph.is_video_sheet(_sheet) and not _graph.is_video_sheet(_os.path.join(_vd, "x.jpg")))
check("an opinion asked after a video re-looks at its frames",
      _graph.needs_relook(_ctx, {"user_input": "дай экспертное мнение"})
      and not _graph.needs_relook(_ctx, {"user_input": "нарисуй кота"}))
check("the history frames the video as WATCHED",
      "WATCHED" in hist[0]["content"] and "never say you cannot see" in hist[0]["content"], hist[0]["content"][:120])

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
