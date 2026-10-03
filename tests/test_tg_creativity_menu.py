"""The 🎨 Creativity submenu: main -> creativity -> back, and its 🎵 Songs
entry point end to end (topic prompt -> generation -> delivery), mirroring
tests/test_tg_weather_flow.py's pattern for the weather submenu.

Run: venv/Scripts/python.exe tests/test_tg_creativity_menu.py
"""
import sys, os, types, tempfile, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T
import tg_songs as tg_songs_mod
import music as M

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_creativity_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


class _SyncThread:
    """Drop-in for threading.Thread that runs target() immediately, inline,
    so a background song generation finishes before the assertion right
    after it runs (same trick test_tg_weather_flow.py uses)."""
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
    def start(self):
        self._target(*self._args, **self._kwargs)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot.sent_kb = []
    bot.audio_sent = []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append(text), bot.sent_kb.append(kw.get("keyboard")), 1)[-1]
    bot._send_audio = lambda cid, path, caption="": (
        bot.audio_sent.append((cid, path, caption)), True)[-1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    return bot


def cb_update(chat_id, data, msg_id=555):
    return {"callback_query": {
        "id": "cbq1", "data": data, "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}


def msg(cid, text, mid=1):
    return {"chat": {"id": cid}, "from": {"id": cid, "username": "u"},
            "message_id": mid, "text": text}


bot = make_bot()
threading.Thread = _SyncThread   # module-level patch: affects tg_accounts too.

# The song is now a QUEUED TASK (tg_songs.song_payload -> _run_task_inner), so
# the flow's contract is checked in two halves: the real _start_song_generation
# must enqueue the payload, and the generation itself is driven directly here.
_enq = []
_real_start = bot._start_song_generation
bot._enqueue_item = lambda cid, item: _enq.append((cid, item))
_real_start(999801, "a birthday song", "ru", 30)
check("a topic is queued as a [song] task, not run on a bare thread",
      _enq and _enq[0][1]["text"] == "[song:30] a birthday song", _enq)
check("the payload parses back", tg_songs_mod.parse_song_payload(_enq[0][1]["text"]) == ("a birthday song", 30))
bot._start_song_generation = lambda cid, topic, lang, duration=0: bot._generate_song(cid, topic, lang, duration)

CID = 999801
bot._user_store.put(T._User(chat_id=CID, name="U", status="approved", is_admin=False))

print("=" * 66)
print("MAIN MENU -> CREATIVITY SUBMENU -> BACK")
print("=" * 66)

bot._dispatch(cb_update(CID, "n/a"))  # warm the session
creativity_label = T._b("creativity", "en")
check("the 'creativity' key has a button label", bool(creativity_label))

bot._resolve_and_push(CID, [{"type": "text", "text": creativity_label}])
check("pressing 🎨 Creativity opens the creativity submenu",
      bot._get_session(CID).menu == "creativity")
check("...and shows a submenu title", any("🎨" in s for s in bot.sent[-1:]), bot.sent[-1:])

_kb = bot.sent_kb[-1] or {}
_labels = {b for row in _kb.get("keyboard", []) for b in row}
# The session's active language is the house default (ru) here -- nothing in
# this flow ever set sess.lang, and the keyboard is rendered in THAT
# language regardless of what language the button label we typed came from.
check("the creativity keyboard offers Images", T._b("cr_images", "ru") in _labels, _labels)
check("the creativity keyboard offers Presentation", T._b("deck", "ru") in _labels, _labels)
check("the creativity keyboard offers Music", T._b("cr_music", "ru") in _labels, _labels)

back_label = T._b("back", "en")
bot._resolve_and_push(CID, [{"type": "text", "text": back_label}])
check("⬅ Back from the creativity submenu returns to the main menu",
      bot._get_session(CID).menu == "")

print()
print("=" * 66)
print("MAIN MENU NO LONGER SHOWS DRAW/DECK DIRECTLY")
print("=" * 66)

_main = T._main_kb(True, False, "en")
_main_labels = {b for row in _main["keyboard"] for b in row}
check("draw is gone from the main keyboard", T._b("draw", "en") not in _main_labels, _main_labels)
check("deck is gone from the main keyboard", T._b("deck", "en") not in _main_labels, _main_labels)
check("creativity is on the main keyboard", T._b("creativity", "en") in _main_labels, _main_labels)

print()
print("=" * 66)
print("CREATIVITY -> SONGS -> TOPIC PROMPT -> GENERATION -> DELIVERY")
print("=" * 66)

songs_label = T._b("songs", "en")
bot.sent.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": songs_label}])
check("pressing 🎵 Songs arms song_topic",
      bot._get_session(CID).reg_state == "song_topic")
check("...and puts the session in the music tier",
      bot._get_session(CID).menu == "cr_music")
check("...and sends a topic prompt", any("🎵" in s for s in bot.sent[-1:]), bot.sent[-1:])

_orig_caption = M.build_structured_caption
_orig_generate = M.generate_music
_orig_engine = M.engine_available

# engine_available reaches the FILESYSTEM (weights) and the NETWORK (ComfyUI's
# /object_info) -- stubbed here so this suite stays offline and deterministic
# regardless of what is installed or running on the machine.
M.engine_available = lambda ctx=None, preset=None: (True, "")

def _stub_caption(ctx, topic, lang, **kw):
    return {"lyrics": f"lyrics about {topic}", "style": "pop"}

def _stub_generate(ctx, lyrics, style, *, duration_s=60, seed=None, preset=None, **kw):
    return "/tmp/fake_song.wav"

bot.sent.clear(); bot.audio_sent.clear()
M.build_structured_caption = _stub_caption
M.generate_music = _stub_generate
try:
    bot._user_gate(CID, msg(CID, "a birthday song for my sister"))
finally:
    M.build_structured_caption = _orig_caption
    M.generate_music = _orig_generate

check("a topic triggers generation and delivers audio",
      len(bot.audio_sent) == 1, bot.audio_sent)
check("the delivered path is what generate_music returned",
      bot.audio_sent and bot.audio_sent[0][1] == "/tmp/fake_song.wav", bot.audio_sent)
check("reg_state clears once the song is delivered",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("THE THREE FAILURE MODES SAY THREE DIFFERENT THINGS")
print("=" * 66)
print("""
Live bug: a healthy engine (weights present, ComfyUI up with the Music3
nodes, engine_available() == (True, "")) told the user "song generation
isn't set up yet" because the songwriter LLM burned its whole token budget
reasoning and returned nothing. MusicUnavailable covered three unrelated
failures -- engine missing, lyrics-writing failed, render produced no file
-- and all three mapped to the one "not set up" string, with NO logging on
that branch to tell them apart afterwards.
""")

# 1) engine genuinely unavailable -> "isn't set up yet" (the only truthful use)
sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot.sent.clear(); bot.audio_sent.clear()
M.engine_available = lambda ctx=None, preset=None: (False, "the Music3 weights are not downloaded yet")
try:
    bot._user_gate(CID, msg(CID, "another topic"))
finally:
    M.engine_available = lambda ctx=None, preset=None: (True, "")

check("a missing engine delivers no audio", bot.audio_sent == [])
check("...and says exactly 'isn't set up yet'",
      T._t("song_unavailable", "ru") in bot.sent, bot.sent)
check("reg_state clears after the failure (not left armed forever)",
      bot._get_session(CID).reg_state == "")

# 2) engine fine, writer fumbled -> a DIFFERENT, retry-flavoured message
sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot.sent.clear(); bot.audio_sent.clear()

def _raises_songwriting(ctx, topic, lang, **kw):
    raise M.SongwritingFailed("no usable lyrics/style pair")

M.build_structured_caption = _raises_songwriting
try:
    bot._user_gate(CID, msg(CID, "another topic"))
finally:
    M.build_structured_caption = _orig_caption

check("a writer failure delivers no audio", bot.audio_sent == [])
check("...and does NOT claim the engine is unconfigured",
      T._t("song_unavailable", "ru") not in bot.sent, bot.sent)
check("...it says the lyrics couldn't be written, so retrying makes sense",
      T._t("song_lyrics_failed", "ru") in bot.sent, bot.sent)
check("reg_state clears after a writer failure",
      bot._get_session(CID).reg_state == "")

# 3) engine fine, render produced nothing -> the generic error, still not "not set up"
sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot.sent.clear(); bot.audio_sent.clear()

def _raises_render(ctx, lyrics, style, *, duration_s=60, seed=None, preset=None):
    raise M.MusicUnavailable("the render produced no file")

M.build_structured_caption = _stub_caption
M.generate_music = _raises_render
try:
    bot._user_gate(CID, msg(CID, "another topic"))
finally:
    M.build_structured_caption = _orig_caption
    M.generate_music = _orig_generate

check("a failed render delivers no audio", bot.audio_sent == [])
check("...and does NOT claim the engine is unconfigured either",
      T._t("song_unavailable", "ru") not in bot.sent, bot.sent)
check("...it reports a generation error", T._t("song_error", "ru") in bot.sent, bot.sent)

print()
print("=" * 66)
print("A GENERIC EXCEPTION IS ALSO HANDLED GRACEFULLY")
print("=" * 66)

sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot.sent.clear(); bot.audio_sent.clear()

def _raises_generic(ctx, topic, lang, **kw):
    raise RuntimeError("comfyui down")

M.build_structured_caption = _raises_generic
try:
    bot._user_gate(CID, msg(CID, "yet another topic"))
finally:
    M.build_structured_caption = _orig_caption

check("a generic exception does not deliver audio", bot.audio_sent == [])
check("...and sends a translated error, not raising out of _user_gate",
      len(bot.sent) >= 1, bot.sent)
check("reg_state clears after the generic failure too",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("/cancel DURING song_topic ABANDONS THE FLOW CLEANLY")
print("=" * 66)

sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot.sent.clear()
allowed = bot._user_gate(CID, msg(CID, "/cancel"))
check("/cancel abandons the song topic prompt",
      bot._get_session(CID).reg_state == "")
check("/cancel returns to the main-menu state (not left processed further)",
      allowed is False)
check("a cancellation notice was sent", any(s for s in bot.sent), bot.sent)

sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
allowed = bot._user_gate(CID, msg(CID, "/broadcast hello"))
check("a non-/cancel command abandons song_topic instead of becoming the topic",
      bot._get_session(CID).reg_state == "" and allowed is True)

sess = bot._get_session(CID); sess.reg_state = "song_topic"; bot._store.put(sess)
bot._dispatch(cb_update(CID, "acct_lang"))
check("an unrelated inline button abandons song_topic too",
      bot._get_session(CID).reg_state == "")

M.engine_available = _orig_engine   # leave the module as we found it

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
