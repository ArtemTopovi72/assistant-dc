"""Two different things arrive as a voice note, and they must not be confused.

  A) The user SPEAKS to the bot. That is an instruction. "нарисуй кота" must draw.
  B) The user FORWARDS someone else's note. That is material. It must be
     transcribed and then handled as transcript / summary / both — never fed in
     as if the user had said it.

And the choice in (B) can arrive three ways: an inline button, a caption sent
WITH the note, or the next thing the user types. The caption was being captured
and then ignored, so the bot asked "what should I do with it?" one second after
the user had written "перескажи".

Run: venv/Scripts/python.exe tests/test_voice_routing.py
"""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="voice_routing_")
import tg_bot as T

# The 📋 retelling is a ONE-SHOT model call outside the agent since 1cdadf4
# (2026-09-14): it never enters the task queue. The suite records that call.
import llm as _llm_mod, time as _time
RETOLD = []
# The intent read is a model call too (_fwdv_intent); the routing below is
# what this suite checks, so the model's verdict is played by a table.
# The model's own reading is checked live: bench/fwdv_intent_live.py.
_INTENT = {"перескажи": "sum", "перескажи кратко": "sum", "расшифруй": "text",
           "и то, и другое": "both", "расшифруй и перескажи": "both"}
def _fake_retell(ctx, system, said, **k):
    if system == T._FWDV_SYSTEM:
        return '{"label": "%s"}' % _INTENT.get(said.strip().lower(), "other")
    RETOLD.append(said)
    return "Кратко: " + said[:40]
_llm_mod.call_llm_simple = _fake_retell
def retold_after(n_prev, timeout=4.0):
    """The transcripts retold since RETOLD had n_prev entries (thread joined)."""
    t0 = _time.time()
    while len(RETOLD) <= n_prev and _time.time() - t0 < timeout:
        _time.sleep(0.05)
    return RETOLD[n_prev:]
T.redirect_data_dir(_DATA)

OK = BAD = 0
CID = 999941


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot(transcript="нарисуй рыжего кота"):
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent, bot.pushed, bot.enqueued = [], [], []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append((text, kw.get("keyboard"))), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: bot._send_text(cid, text, **kw)
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._transcribe = lambda ctx, fid, media="voice", **kw: transcript
    bot._get_ctx = lambda: types.SimpleNamespace()
    # The real dispatch endpoint: _resolve_and_push ends at _backend.push(task).
    bot._backend = types.SimpleNamespace(
        push=lambda task: bot.pushed.append(task), depth=lambda: 0,
        chat_depth=lambda cid: 0, tasks_ahead=lambda tid: 0,
        drop_chat=lambda cid: 0, close=lambda: None, name=lambda: "stub")
    bot._enqueue_item = lambda cid, item: (bot.enqueued.append(item), None)[1]
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


def sent_text(bot):
    return " ".join(t for t, _ in bot.sent)


def msg(**kw):
    base = {"message_id": 7, "chat": {"id": CID}, "from": {"id": CID}}
    base.update(kw)
    return base


VOICE = {"file_id": "vfile", "duration": 95}
FWD = {"forward_origin": {}}          # a dict — and {} is FALSY, which is the trap

# ══════════════════════════════════════════════════ A. routing: mine vs forwarded
print("\n" + "=" * 70)
print("A. MY OWN VOICE IS AN INSTRUCTION; A FORWARDED ONE IS MATERIAL")
print("=" * 70)

b = make_bot()
b._dispatch({"message": msg(voice=dict(VOICE))})
check("my own voice note becomes a plain instruction",
      [i["type"] for i in b.enqueued] == ["voice"], [i["type"] for i in b.enqueued])

b = make_bot()
b._dispatch({"message": msg(voice=dict(VOICE), **FWD)})
check("a forwarded voice note is routed separately",
      [i["type"] for i in b.enqueued] == ["fwd_voice"], [i["type"] for i in b.enqueued])

check("an empty forward_origin dict still counts as forwarded",
      T._is_forwarded({"forward_origin": {}}))
check("a message with no forward marker is not forwarded",
      not T._is_forwarded({"text": "hi"}))
for k in ("forward_from", "forward_from_chat", "forward_sender_name", "forward_date"):
    check(f"{k} counts as forwarded", T._is_forwarded({k: {}}))

# ══════════════════════════════════════════════════════════ B. the intent reader
print("\n" + "=" * 70)
print("B. WHAT THE USER ASKED FOR A FORWARDED NOTE")
print("=" * 70)

for text, want in [("перескажи", "sum"), ("расшифруй и перескажи", "both"),
                   ("", ""), ("А если бы тебя добавили? что бы сказали твои друзья?", "")]:
    got = T._fwdv_intent(text)
    check(f"{text!r} -> {want!r}", got == want, f"got {got!r}")

# ═════════════════════════════════════════ C. the caption is honoured, not ignored
print("\n" + "=" * 70)
print("C. A CAPTION THAT ALREADY SAYS WHAT TO DO IS NOT RE-ASKED")
print("=" * 70)

SPEECH = "Привет, купи молока и позвони маме в среду."


_REAL_PUSH = T.TelegramBot._resolve_and_push


def run_fwd(caption):
    """Forward a note and let the REAL _resolve_and_push handle the item."""
    bot = make_bot(transcript=SPEECH)
    # The fwd_voice branch lives inside _resolve_and_push, so the queue hop is
    # what we short-circuit — not the handler under test.
    bot._enqueue_item = lambda cid, item: _REAL_PUSH(bot, cid, [item])
    bot._dispatch({"message": msg(voice=dict(VOICE), caption=caption, **FWD)})
    return bot


_n = len(RETOLD)
b = run_fwd("перескажи")
_got = retold_after(_n)
check("caption 'перескажи' summarizes instead of asking", bool(_got),
      sent_text(b)[:200])
check("the summary request carries the transcript",
      _got and SPEECH in _got[0], str(_got)[:200])
check("no 'what should I do with it?' question was sent",
      "What should I" not in sent_text(b) and "Что с ней" not in sent_text(b),
      sent_text(b)[:200])

b = run_fwd("расшифруй")
check("caption 'расшифруй' sends the transcript verbatim",
      SPEECH in sent_text(b), sent_text(b)[:200])
check("and does NOT round-trip it through the model",
      not b.pushed and not retold_after(len(RETOLD), timeout=0.3), str(b.pushed))

b = run_fwd("")
check("with no caption it DOES ask", "🎧" in sent_text(b), sent_text(b)[:200])
check("and offers all three choices",
      sum(k in str(b.sent) for k in ("fwdv:text", "fwdv:sum", "fwdv:both")) == 3,
      str(b.sent)[-300:])

b = run_fwd("нарисуй кота")
check("an unrelated caption still asks rather than guessing",
      "🎧" in sent_text(b) and not b.pushed, sent_text(b)[:200])

# ══════════════════════════ D. answering by TYPING after the note, not by button
print("\n" + "=" * 70)
print("D. TYPING THE ANSWER AFTER THE NOTE WORKS TOO")
print("=" * 70)

b = make_bot()
s = b._get_session(CID)
s.fwd_transcript = SPEECH
b._store.put(s)
_n = len(RETOLD)
_REAL_PUSH(b, CID, [{"type": "text", "text": "перескажи"}])
_got = retold_after(_n)
check("typing 'перескажи' after forwarding summarizes the note",
      _got and SPEECH in _got[0], str(_got)[:200])
# The transcript is NOT consumed any more -- it is marked done. Wiping it made
# the inline keyboard single-use, and since that keyboard stays on screen the
# second press answered "that voice note is no longer in my hands" for a note
# the user could still see. Retiring it now means only that the type-your-
# answer shortcut stops capturing text, so a later unrelated "перескажи"
# cannot reach back into a note already dealt with.
check("the transcript is KEPT so the inline keyboard still works",
      getattr(b._get_session(CID), "fwd_transcript", "") == SPEECH)
check("...but the forward is marked as already acted on",
      getattr(b._get_session(CID), "fwd_transcript_done", False))
_n_before = len(b.pushed)
_REAL_PUSH(b, CID, [{"type": "text", "text": "перескажи"}])
check("a LATER typed 'перескажи' is not re-captured by the finished note",
      not any(SPEECH in str(vars(t)) for t in b.pushed[_n_before:]),
      str(b.pushed[_n_before:])[:300])

b = make_bot()
s = b._get_session(CID); s.fwd_transcript = SPEECH; b._store.put(s)
_REAL_PUSH(b, CID, [{"type": "text", "text": "нарисуй кота"}])
check("an unrelated message is NOT swallowed by the pending note",
      b.pushed and "нарисуй кота" in str([vars(t) for t in b.pushed]),
      str(b.pushed)[:200])
check("and the note stays pending for the buttons",
      getattr(b._get_session(CID), "fwd_transcript", "") == SPEECH)

# ══════════════════════════════════════════════════════ E. clearing forgets it
print("\n" + "=" * 70)
print("E. CLEARING THE CHAT FORGETS A PENDING NOTE")
print("=" * 70)
b = make_bot()
s = b._get_session(CID); s.fwd_transcript = SPEECH; b._store.put(s)
s.clear_context()
check("clear_context drops the held transcript",
      not getattr(s, "fwd_transcript", ""), repr(getattr(s, "fwd_transcript", "")))

# =========================== F. every button on the keyboard keeps working
print()
print("=" * 70)
print("F. THE THREE CHOICES ARE NOT SINGLE-USE")
print("=" * 70)

# Reproduces the live report: 📝 Расшифровка delivered the transcript, then the
# next press on the SAME (still visible, still clickable) inline keyboard said
# "🎧 Этого голосового у меня больше нет — перешли его заново."
b = make_bot()
s_ = b._get_session(CID)
s_.fwd_transcript = SPEECH
s_.fwd_transcript_id = "abc12345"
b._store.put(s_)

def press(bot, choice, fwd_id="abc12345"):
    bot._dispatch({"callback_query": {"id": "1",
                                      "data": "fwdv:%s:%s" % (choice, fwd_id),
                                      "from": {"id": CID},
                                      "message": {"chat": {"id": CID},
                                                  "message_id": 5}}})

press(b, "text")
check("the first press delivers the transcript", SPEECH in sent_text(b),
      sent_text(b)[:200])

_before = sent_text(b)
_n = len(RETOLD)
press(b, "sum")
_got = retold_after(_n)
_after = sent_text(b)[len(_before):]
check("a SECOND press does not claim the note is gone",
      "больше нет" not in _after and "no longer in my hands" not in _after,
      _after[:300])
check("...and it actually summarizes, carrying the transcript",
      _got and SPEECH in _got[0], str(_got)[:200])

# A button belonging to an EARLIER forward must still be refused -- retaining
# the transcript must not resurrect the bug where a stale keyboard delivered
# whatever happens to be pending now.
b2 = make_bot()
s2 = b2._get_session(CID)
s2.fwd_transcript = SPEECH
s2.fwd_transcript_id = "newid999"
b2._store.put(s2)
press(b2, "text", fwd_id="staleid1")
check("a button from an earlier forward is still refused",
      "больше нет" in sent_text(b2) or "no longer in my hands" in sent_text(b2),
      sent_text(b2)[:200])
check("...and the refusal did not deliver the current transcript",
      SPEECH not in sent_text(b2), sent_text(b2)[:200])

# A fresh forward re-arms the typed-answer shortcut the finished one retired.
b3 = run_fwd("расшифруй")
check("acting on a caption marks the forward done",
      getattr(b3._get_session(CID), "fwd_transcript_done", False))
b3._enqueue_item = lambda cid, item: _REAL_PUSH(b3, cid, [item])
b3._dispatch({"message": msg(voice=dict(VOICE), caption="", **FWD)})
check("a NEW forward re-arms the typed-answer shortcut",
      not getattr(b3._get_session(CID), "fwd_transcript_done", True),
      repr(getattr(b3._get_session(CID), "fwd_transcript_done", None)))

# clear_context has to drop the flag too, or a cleared chat would come back with
# the shortcut still retired for a note it no longer holds.
s4 = make_bot()._get_session(CID)
s4.fwd_transcript = SPEECH
s4.fwd_transcript_done = True
s4.clear_context()
check("clear_context resets the done flag as well",
      not getattr(s4, "fwd_transcript_done", True))

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
