"""The Telegram mashup flow: menu, two tracks in turn, then the mix.

Menu-first by design (the user picked it): 🎚 Mashup arms the capture, then the
two tracks arrive one message at a time and the bot says which is which. The
engine itself is stubbed -- separating real stems is GPU work, and what is
under test here is the CAPTURE: which message becomes which track, what
happens to audio that arrives when nothing is armed, and whether an abandoned
flow can eat a later voice message.

Run: venv/Scripts/python.exe tests/test_tg_mashup_flow.py
"""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_mashup_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import mashup as M

OK = BAD = 0
CID = 999947


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  %s" % name)
    else:
        BAD += 1; print("FAIL  %s   %s" % (name, str(extra)[:200]))


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent, bot.pushed, bot.enqueued, bot.audio = [], [], [], []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append((text, kw.get("keyboard"))), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: bot._send_text(cid, text, **kw)
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._get_ctx = lambda: types.SimpleNamespace()
    bot._dl_bytes = lambda fid: b"RIFFfake-audio-bytes"
    bot._send_audio = lambda cid, path, caption="": (
        bot.audio.append((path, caption)), True)[1]
    bot._backend = types.SimpleNamespace(
        push=lambda task: bot.pushed.append(task), depth=lambda: 0,
        chat_depth=lambda cid: 0, tasks_ahead=lambda tid: 0,
        drop_chat=lambda cid: 0, close=lambda: None, name=lambda: "stub")
    bot._enqueue_item = lambda cid, item: (bot.enqueued.append(item), None)[1]
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    # The session store is a FILE shared by every bot built here, so without
    # this a section inherits the previous one's half-armed mashup and its
    # audio gets captured as a track instead of taking the path under test.
    _s = bot._get_session(CID)
    _s.mashup_state = ""
    _s.mashup_vocal_path = ""
    _s.mashup_vocal_speech = False
    _s.fwd_transcript = ""
    bot._store.put(_s)
    return bot


def sent_text(bot):
    return " ".join(t for t, _ in bot.sent)


def audio_msg(file_id="f1", kind="audio"):
    return {"message": {"chat": {"id": CID, "type": "private"},
                        "from": {"id": CID}, "message_id": 1,
                        kind: {"file_id": file_id, "duration": 30}}}


def doc_msg(file_id="f9", name="beat.mp3", mime="audio/mpeg"):
    return {"message": {"chat": {"id": CID, "type": "private"},
                        "from": {"id": CID}, "message_id": 1,
                        "document": {"file_id": file_id, "file_name": name,
                                     "mime_type": mime}}}


def arm(bot):
    """Press the menu button through the real resolver."""
    s = bot._get_session(CID)
    bot._start_mashup_flow(CID, s, "ru")
    return bot._get_session(CID)


# ── A. the button arms the capture ─────────────────────────────────────────
print("=" * 70); print("A. THE MENU BUTTON ARMS THE CAPTURE"); print("=" * 70)

bot = make_bot()
s = arm(bot)
check("pressing the button asks for track 1",
      T._t("mash_ask1", "ru")[:20] in sent_text(bot), sent_text(bot)[:150])
check("and arms the capture", s.mashup_state == "want1", s.mashup_state)
check("the button is on the creativity keyboard",
      T._b("mashup", "ru") in str(T._cr_music_kb("ru")),
      str(T._cr_music_kb("ru")))
check("its label resolves back to the key, whatever the language",
      T._LABEL2KEY.get(T._b("mashup", "ru")) == "mashup"
      and T._LABEL2KEY.get(T._b("mashup", "en")) == "mashup",
      T._b("mashup", "ru"))
check("the button label routes to the flow",
      T._DIRECT_KB.get("mashup") == "__mashup__", T._DIRECT_KB.get("mashup"))


# ── B. two tracks, in turn ─────────────────────────────────────────────────
print("\n" + "=" * 70); print("B. TWO TRACKS ARRIVE ONE AT A TIME"); print("=" * 70)

bot = make_bot()
arm(bot)
bot._dispatch(audio_msg("voc1"))
s = bot._get_session(CID)
check("the first audio is taken as the voice track", s.mashup_state == "want2",
      s.mashup_state)
check("...and stored on disk, not in the session",
      s.mashup_vocal_path and os.path.exists(s.mashup_vocal_path),
      s.mashup_vocal_path)
check("...under the redirected data dir, not the repo root",
      _DATA in s.mashup_vocal_path, s.mashup_vocal_path)
check("the bot confirms it and asks for track 2",
      T._t("mash_got1", "ru")[:12] in sent_text(bot)
      and T._t("mash_ask2", "ru")[:20] in sent_text(bot), sent_text(bot)[:300])
check("the track was NOT routed to the transcriber or the agent",
      not bot.enqueued and not bot.pushed, (bot.enqueued, bot.pushed))

# The second track runs the mashup. The engine is stubbed: this asserts the
# WIRING (which file is the vocal, which is the bed), not the mixing.
calls = {}
_real = M.make_mashup
M.make_mashup = lambda v, i, o, **kw: (
    calls.update(vocal=v, instr=i, out=o, kw=kw),
    open(o, "wb").write(b"x"),
    {"path": o, "bpm_vocal": 100.0, "bpm_instr": 120.0, "stretch": 1.2,
     "semitones": -2, "key_vocal": "A minor", "key_instr": "G major",
     "speech": kw.get("vocal_is_speech", False), "seconds": 30.0, "took": 9.0})[2]
try:
    vocal_path = s.mashup_vocal_path
    bot._dispatch(audio_msg("bed1"))
    for t in list(bot._live_threads()) if hasattr(bot, "_live_threads") else []:
        pass
    import time
    for _ in range(100):          # the mashup runs on its own thread
        if bot.audio:
            break
        time.sleep(0.05)
    check("the mashup ran", bool(calls), calls)
    check("track 1 is the VOCAL donor", calls.get("vocal") == vocal_path,
          (calls.get("vocal"), vocal_path))
    check("track 2 is the INSTRUMENTAL donor",
          calls.get("instr") and calls["instr"] != vocal_path, calls.get("instr"))
    check("the result is delivered as audio", len(bot.audio) == 1, bot.audio)
    check("the caption reports both tempos and the stretch",
          bot.audio and "100" in bot.audio[0][1] and "120" in bot.audio[0][1]
          and "1.2" in bot.audio[0][1], bot.audio[0][1] if bot.audio else "")
    s2 = bot._get_session(CID)
    check("the capture is disarmed once both tracks are in", s2.mashup_state == "",
          s2.mashup_state)
    check("...and the held track is released", s2.mashup_vocal_path == "",
          s2.mashup_vocal_path)
finally:
    M.make_mashup = _real


# ── C. a voice note is speech, and says so ─────────────────────────────────
print("\n" + "=" * 70); print("C. A VOICE NOTE IS MARKED AS SPEECH"); print("=" * 70)

bot = make_bot()
arm(bot)
bot._dispatch(audio_msg("v1", kind="voice"))
s = bot._get_session(CID)
check("a voice note is recorded as speech", s.mashup_vocal_speech is True,
      s.mashup_vocal_speech)

seen = {}
_real = M.make_mashup
M.make_mashup = lambda v, i, o, **kw: (
    seen.update(kw), open(o, "wb").write(b"x"),
    {"path": o, "bpm_vocal": 0.0, "bpm_instr": 120.0, "stretch": 1.0,
     "semitones": 0, "key_vocal": "", "key_instr": "G major",
     "speech": True, "seconds": 12.0, "took": 3.0})[2]
try:
    bot._dispatch(audio_msg("bed2"))
    import time
    for _ in range(100):
        if bot.audio:
            break
        time.sleep(0.05)
    check("speech is passed through to the engine as speech",
          seen.get("vocal_is_speech") is True, seen)
finally:
    M.make_mashup = _real

# A song sent as an audio message is NOT speech.
bot = make_bot()
arm(bot)
bot._dispatch(audio_msg("song1", kind="audio"))
check("an audio track is not treated as speech",
      bot._get_session(CID).mashup_vocal_speech is False)

# An mp3 sent as a FILE counts too -- that is how people share tracks.
bot = make_bot()
arm(bot)
bot._dispatch(doc_msg())
s = bot._get_session(CID)
check("an audio document is accepted as a track", s.mashup_state == "want2",
      s.mashup_state)
check("...and keeps its extension so the decoder knows the format",
      s.mashup_vocal_path.endswith(".mp3"), s.mashup_vocal_path)
check("...and was not sent to the document reader",
      not bot.enqueued, bot.enqueued)

# A non-audio document is left entirely alone.
bot = make_bot()
arm(bot)
bot._dispatch(doc_msg(name="notes.pdf", mime="application/pdf"))
check("a PDF still goes to the document reader while a mashup waits",
      bot.enqueued and bot.enqueued[0].get("type") == "document", bot.enqueued)


# ── D. nothing armed: audio behaves exactly as before ──────────────────────
print("\n" + "=" * 70); print("D. WITH NOTHING ARMED, AUDIO IS UNCHANGED"); print("=" * 70)

bot = make_bot()
bot._dispatch(audio_msg("v2", kind="voice"))
check("a voice note with no mashup armed still goes to the transcriber",
      bot.enqueued and bot.enqueued[0].get("type") == "voice", bot.enqueued)
check("...and no track was written", not bot._get_session(CID).mashup_vocal_path)


# ── E. an abandoned flow must not eat a later voice message ────────────────
print("\n" + "=" * 70); print("E. NAVIGATING AWAY ABANDONS THE CAPTURE"); print("=" * 70)

# This is the trap the forwarded-voice transcript fell into: state left armed,
# quietly swallowing the next thing that looked like an answer.
bot = make_bot()
arm(bot)
s = bot._get_session(CID)
check("armed before navigating away", s.mashup_state == "want1")
T_RESOLVE = bot._resolve_and_push
T_RESOLVE(CID, [{"type": "text", "text": T._b("weather", "ru")}])
s = bot._get_session(CID)
check("pressing another menu button disarms the capture", s.mashup_state == "",
      s.mashup_state)
bot.enqueued.clear()
bot._dispatch(audio_msg("v3", kind="voice"))
check("...so a later voice note is speech again, not a mashup track",
      bot.enqueued and bot.enqueued[0].get("type") == "voice", bot.enqueued)


# ── F. failures reach the user ─────────────────────────────────────────────
print("\n" + "=" * 70); print("F. FAILURES ARE REPORTED, NOT SWALLOWED"); print("=" * 70)

bot = make_bot()
arm(bot)
bot._dispatch(audio_msg("voc9"))
_real = M.make_mashup
M.make_mashup = lambda *a, **k: (_ for _ in ()).throw(
    M.MashupUnavailable("I could not find any singing in the first track"))
try:
    bot._dispatch(audio_msg("bed9"))
    import time
    for _ in range(100):
        if "singing" in sent_text(bot):
            break
        time.sleep(0.05)
    check("the engine's own sentence reaches the user",
          "singing" in sent_text(bot), sent_text(bot)[-250:])
    check("no audio was delivered", not bot.audio, bot.audio)
    check("the capture is disarmed after a failure too",
          bot._get_session(CID).mashup_state == "",
          bot._get_session(CID).mashup_state)
finally:
    M.make_mashup = _real

# A download that comes back empty must say so rather than writing a 0-byte
# track and failing much later inside the decoder.
bot = make_bot()
arm(bot)
bot._dl_bytes = lambda fid: None
bot._dispatch(audio_msg("gone"))
check("a failed download is reported at once",
      T._t("mash_need_audio", "ru")[:14] in sent_text(bot), sent_text(bot)[:200])
check("...and the capture stays armed for another try",
      bot._get_session(CID).mashup_state == "want1",
      bot._get_session(CID).mashup_state)


# ── G. clearing the chat forgets a half-built mashup ───────────────────────
print("\n" + "=" * 70); print("G. CLEARING FORGETS A HALF-BUILT MASHUP"); print("=" * 70)

bot = make_bot()
arm(bot)
bot._dispatch(audio_msg("voc0"))
s = bot._get_session(CID)
s.clear_context()
check("clear_context drops the pending state", s.mashup_state == "", s.mashup_state)
check("...and the held track", s.mashup_vocal_path == "", s.mashup_vocal_path)
check("...and the speech flag", s.mashup_vocal_speech is False)


print("\n%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)
