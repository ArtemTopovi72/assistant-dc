"""🎤 Cover: every way a song and its new words arrive reaches the render.

Live 10-03 («каверы наглухо убиты … не даёт текст вставить»): a song
forwarded from a music bot was dropped as "the bot's own message", an .m4a
sent as a file (application/octet-stream) went to the document reader, and
the new words were taken by whatever else reads text -- a forwarded lyric by
the chat, an old forwarded voice note as its answer, an armed song topic.
"""
import os, sys, time, threading, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="fwdtext_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)
T._DEBOUNCE_FWD_S, T._DEBOUNCE_FWD_MAX_S = 1.5, 3.0   # production waits longer for a typed instruction
import graph as graph_mod
import tg_tasks

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

INVOKES = []
class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event(); self.last_image_path = ""
        self.last_image_prompt = ""; self.stage_callback = None
        self.total_user_turns = 0; self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass
class StubGraph:
    def __init__(self, ctx): self.ctx = ctx
    def invoke(self, base):
        INVOKES.append(base.get("user_input", ""))
        return {"final_answer": "echo", "messages": base.get("messages", [])}
graph_mod.build_graph = lambda ctx: StubGraph(ctx)
SHARED = FakeCtx()

bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend(); bot.sent = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
bot._edit_text = lambda *a, **kw: None; bot._delete = lambda *a, **kw: None
bot._api_post = lambda *a, **k: {}; bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._running = True
thr = threading.Thread(target=bot._consumer_loop, daemon=True); thr.start()


import cover, remix
cover.available = lambda: True; remix.available = lambda: True
CID = 9_300_077
bot._user_store.put(T._User(chat_id=CID, name="Cv", status="approved", is_admin=False))
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"; sess.lang_chosen = True; sess.reg_state = ""; bot._store.put(sess)
bot._dl_bytes = lambda fid, *a, **k: b"ID3fake"
GO = []
bot._run_busy = lambda cid, fn, *a: GO.append(a)
_seq = [42_000_000]
def _nid(): _seq[0] += 1; return _seq[0]
def m(**kw):
    d = {"message_id": _nid(), "chat": {"id": CID, "type": "private"}, "from": {"id": CID}, "date": int(time.time())}
    d.update(kw); return {"update_id": _nid(), "message": d}
def wait(pred, t=8):
    e = time.monotonic() + t
    while time.monotonic() < e:
        if pred(): return True
        time.sleep(0.1)
    return pred()

A = {"file_id": "A1", "duration": 180, "mime_type": "audio/mpeg", "file_name": "s.mp3"}
BOTFWD = {"forward_origin": {"type": "user", "sender_user": {"id": 555, "is_bot": True, "first_name": "VK Music"}}, "forward_date": 1}
USRFWD = {"forward_origin": {"type": "user", "sender_user": {"id": 777, "is_bot": False, "first_name": "Друг"}}, "forward_date": 1}
CHFWD = {"forward_origin": {"type": "channel", "chat": {"id": -100, "title": "Music"}}, "forward_from_chat": {"id": -100, "type": "channel", "title": "Music"}, "forward_date": 1}
variants = {
  "audio": dict(audio=A), "audio+caption": dict(audio=A, caption="вот песня"),
  "voice": dict(voice={"file_id": "V", "duration": 30}),
  "video": dict(video={"file_id": "VD", "duration": 30, "mime_type": "video/mp4"}),
  "video_note": dict(video_note={"file_id": "VN", "duration": 30}),
  "doc mp3": dict(document={"file_id": "D", "mime_type": "audio/mpeg", "file_name": "x.mp3"}),
  "doc no mime": dict(document={"file_id": "D", "file_name": "x.mp3"}),
  "doc octet": dict(document={"file_id": "D", "mime_type": "application/octet-stream", "file_name": "x.m4a"}),
  "doc flac": dict(document={"file_id": "D", "mime_type": "audio/flac", "file_name": "x.flac"}),
  "fwd from bot": dict(audio=A, **BOTFWD), "fwd from user": dict(audio=A, **USRFWD),
  "fwd from channel": dict(audio=A, **CHFWD),
  "too big": dict(document={"file_id": "D", "mime_type": "audio/flac", "file_name": "x.flac", "file_size": 60*1024*1024}),
  "reply to prompt": dict(audio=A, reply_to_message={"message_id": 1, "from": {"id": 123, "is_bot": True}, "text": "пришли песню"}),
}
for name, kw in variants.items():
    s_ = bot._get_session(CID); s_.cover_state = "want_audio"; bot._store.put(s_)
    n = len(bot.sent)
    bot._dispatch(m(**kw))
    ok = wait(lambda: bot._get_session(CID).cover_state == "want_text", 3)
    if name == "too big":
        check("a song over the download limit is named as too big",
              not ok and any("МБ" in t for _, t in bot.sent[n:]), bot.sent[n:])
    else:
        check(f"the song arrives as {name}: asked for the new words", ok, [t[:60] for _, t in bot.sent[n:]])
import tg_bot as TB
TB._fwdv_intent = lambda t: "text"     # what a model may well answer for a lyric
LY = "Я иду по улице одна\nНадо мной горит луна\nИ весна, весна, весна\nНе даёт уснуть до дна"
USRFWD = {"forward_origin": {"type": "user", "sender_user": {"id": 777, "is_bot": False, "first_name": "Друг"}}, "forward_date": 1}
def arm(**extra):
    s_ = bot._get_session(CID); s_.cover_state = "want_text"
    src = bot._cover_dir(CID) + "/song.mp3"; open(src, "wb").write(b"x"); s_.cover_src = src
    for k, v in extra.items(): setattr(s_, k, v)
    bot._store.put(s_)
variants = {
  "plain lyrics": ({}, dict(text=LY)),
  "with style line": ({}, dict(text="Стиль: рок\n" + LY)),
  "one short line": ({}, dict(text="ля ля ля")),
  "forwarded lyrics": ({}, dict(text=LY, **USRFWD)),
  "after a fwd voice": ({"fwd_transcript": "старое", "fwd_transcript_done": False}, dict(text=LY)),
  "song_topic left armed": ({"reg_state": "song_topic"}, dict(text=LY)),
  "pending_instruction": ({"pending_instruction": "x"}, dict(text=LY)),
}
for name, (st, kw) in variants.items():
    arm(**st); GO.clear(); n0 = len(INVOKES); n = len(bot.sent)
    time.sleep(0.3)
    bot._dispatch(m(**kw))
    ok = wait(lambda: GO, 6)
    check(f"lyrics as {name}: the cover starts", ok and len(INVOKES) == n0,
          [t[:60] for _, t in bot.sent[n:]])
    s_ = bot._get_session(CID); s_.fwd_transcript = ""; s_.reg_state = ""; s_.pending_instruction = ""; s_.song_draft = ""; bot._store.put(s_)
# A clip at the text step is the SECOND song: the first song's voice sings its melody (one remix button).
for name, kw in {"audio": dict(audio=A), "voice": dict(voice={"file_id": "V", "duration": 30}),
                 "video": dict(video={"file_id": "VD", "duration": 30, "mime_type": "video/mp4"})}.items():
    arm(); GO.clear()
    bot._dispatch(m(**kw))
    ok = wait(lambda: GO, 6)
    check(f"a second song as {name}: the voice remix starts", ok and GO and GO[0][-1].endswith(("song2.mp3", "song2.ogg", "song2.mp4", "song2.mp3")) or (ok and GO and "song2" in GO[0][-1]), GO)
    check(f"...and the first song is the voice ({name})", ok and GO and GO[0][2].endswith(("song.mp3", ".mp3")), GO)
# A YouTube link is the song too (owner 10-03), then the words as usual.
import tg_links
_real_fetch = tg_links.fetch_video
_real_busy = bot._run_busy
bot._run_busy = lambda cid, fn, *a: fn(*a)
tg_links.fetch_video = lambda url, *a, **k: {"data": b"MP4", "seconds": 200, "title": "t"}
s_ = bot._get_session(CID); s_.cover_state = "want_audio"; s_.cover_src = ""; bot._store.put(s_)
n = len(bot.sent)
bot._dispatch(m(text="https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
check("a YouTube link is taken as the song", wait(lambda: bot._get_session(CID).cover_state == "want_text", 5)
      and bot._get_session(CID).cover_src.endswith("song.mp4"), [t[:60] for _, t in bot.sent[n:]])
GO.clear()
bot._run_busy = lambda cid, fn, *a: GO.append(a)
bot._dispatch(m(text="Новые слова про ютуб\nhttps://youtu.be/xyz в куплете"))
check("...then a lyric that quotes a link is still the lyric", wait(lambda: GO, 5), GO)
bot._run_busy = lambda cid, fn, *a: fn(*a)
tg_links.fetch_video = lambda url, *a, **k: {"too_long": True, "seconds": 3600, "title": "t"}
s_ = bot._get_session(CID); s_.cover_state = "want_audio"; bot._store.put(s_)
n = len(bot.sent)
bot._dispatch(m(text="https://youtu.be/long"))
check("an hour-long video is named as too long", wait(lambda: any("60 мин" in t for _, t in bot.sent[n:]), 5),
      [t[:60] for _, t in bot.sent[n:]])
tg_links.fetch_video = _real_fetch
bot._run_busy = _real_busy

# _cover_render with a second song: the bar-aligned mashup goes out first, then the voice remix;
# a failed mashup never costs the remix.
import mashup_auto, mashup_stems
mashup_stems.release_separator = lambda: None
AUD = []
bot._send_audio = lambda cid, path, cap="", **k: (AUD.append((path, cap)), True)[1]
mashup_auto.make = lambda ctx, a, b, *x: f"mash:{os.path.basename(a)}+{os.path.basename(b)}"
remix.remix_voice = lambda ctx, a, b: "voice.mp3"
remix.resing_available = lambda: False
bot._cover_render(CID, "ru", "/x/song.mp3", "", "/x/song2.mp3")
check("second song: the mashup (song 1 vocal over song 2) goes first, then the voice remix",
      [p for p, _ in AUD] == ["mash:song.mp3+song2.mp3", "voice.mp3"] and "Мэшап" in AUD[0][1], AUD)
AUD.clear()
def _boom(*a): raise RuntimeError("demucs died")
mashup_auto.make = _boom
bot._cover_render(CID, "ru", "/x/song.mp3", "", "/x/song2.mp3")
check("a failed mashup still sends the voice remix", [p for p, _ in AUD] == ["voice.mp3"], AUD)
# 10-07 «говновоз делать по 2 ссылкам»: with YuE2 the second song gives the WORDS and the first
# sings them on its own melody (after the quick mashup).
AUD.clear()
mashup_auto.make = lambda ctx, a, b, *x: "mash.mp3"
remix.resing_available = lambda: True
remix.lyrics_of = lambda ctx, song: "слова из " + os.path.basename(song)
SUNG = []
def _resing2(ctx, a, l, info=None):
    SUNG.append((os.path.basename(a), l))
    info.update({"rvc": "artist", "trained": True})
    return "resing2.mp3"
remix.resing = _resing2
bot._cover_render(CID, "ru", "/x/song.mp3", "", "/x/song2.mp3")
check("two songs: mashup, then song 1 re-sung with song 2's words",
      [p for p, _ in AUD] == ["mash.mp3", "resing2.mp3"] and SUNG == [("song.mp3", "слова из song2.mp3")], (AUD, SUNG))
AUD.clear(); remix.remix_words = lambda ctx, a, l: "words.mp3"; mashup_auto.make = lambda *a: "mash.mp3"
remix.resing = lambda ctx, a, l, info=None: "resing.mp3"
remix.resing_available = lambda: False
bot._cover_render(CID, "ru", "/x/song.mp3", "новые слова", "")
check("new words without YuE2: the syllable remix, no mashup", [p for p, _ in AUD] == ["words.mp3"], AUD)
# 10-07: new words are re-sung on the original's score («МОЛОДЕЦ!!!»), not squeezed into its syllables.
AUD.clear(); remix.resing_available = lambda: True
bot._cover_render(CID, "ru", "/x/song.mp3", "новые слова", "")
check("new words with YuE2: the song is re-sung", [p for p, _ in AUD] == ["resing.mp3"], AUD)
# 10-07: a new artist's first cover goes out in a zero-shot timbre, then again in their own RVC
# voice trained on the original's vocal; a known artist's comes once, already in it.
AUD.clear()
def _resing_new(ctx, a, l, info=None):
    info.update({"rvc": "artist", "trained": False})
    return "resing.mp3"
remix.resing = _resing_new
remix.resing_rvc = lambda ctx, info: "resing_rvc.mp3"
bot._cover_render(CID, "ru", "/x/song.mp3", "новые слова", "")
check("new artist: zero-shot cover first, then the same cover in the trained RVC voice",
      [p for p, _ in AUD] == ["resing.mp3", "resing_rvc.mp3"], AUD)
AUD.clear()
def _resing_known(ctx, a, l, info=None):
    info.update({"rvc": "artist", "trained": True})
    return "resing.mp3"
remix.resing = _resing_known
bot._cover_render(CID, "ru", "/x/song.mp3", "новые слова", "")
check("known artist: one cover, already in their voice", [p for p, _ in AUD] == ["resing.mp3"], AUD)

# 10-09: the voice picker under «Песню получил» -- original singer (auto) / no swap / a star;
# the choice is kept on the session and goes into the render; a picked voice is the only pass.
import rvc_voice
rvc_voice.available = lambda: True
rvc_voice.stars = lambda: [("tsoi_hq", "Виктор Цой")]
_s = bot._get_session(CID); _s.cover_voice = ""
kb = bot._cover_voice_kb(_s, "ru")["inline_keyboard"]
check("voice picker: auto (ticked), no swap, the trained stars",
      [r[0]["callback_data"] for r in kb] == ["cvv:auto", "cvv:none", "cvv:tsoi_hq"] and kb[0][0]["text"].startswith("✅"), kb)
EDITS = []
_real_edit = bot._edit_text
bot._edit_text = lambda cid, mid, text, parse_mode=None, keyboard=None: EDITS.append(keyboard)
bot._cb_cover_voice(CID, {"message_id": 5}, "cvv:tsoi_hq")
check("picking a star keeps it and moves the tick",
      bot._get_session(CID).cover_voice == "tsoi_hq" and EDITS and EDITS[-1]["inline_keyboard"][2][0]["text"].startswith("✅"), EDITS)
bot._cb_cover_voice(CID, {"message_id": 5}, "cvv:evil_model")
check("an unknown voice name is ignored", bot._get_session(CID).cover_voice == "tsoi_hq")
AUD.clear(); GOT = {}
def _resing_pick(ctx, a, l, info=None):
    GOT.update(info); info.update({"rvc": "artist", "trained": False})
    return "resing.mp3"
remix.resing = _resing_pick
bot._cover_render(CID, "ru", "/x/song.mp3", "новые слова", "")
check("the picked voice goes into the render and no second RVC pass follows",
      GOT.get("voice") == "tsoi_hq" and [p for p, _ in AUD] == ["resing.mp3"], (GOT, AUD))
# «▶️ спеть эту песню как есть»: shown only for a star while the song waits for words
_s = bot._get_session(CID); _s.cover_voice = "tsoi_hq"; _s.cover_state = "want_text"
_src = os.path.join(tempfile.mkdtemp(), "kukla.mp3"); open(_src, "wb").close(); _s.cover_src = _src
bot._store.put(_s)
kb = bot._cover_voice_kb(_s, "ru")["inline_keyboard"]
check("a star while the song waits: the «sing it as is» button", kb[-1][0]["callback_data"] == "cvv:go", kb)
STAR = []
remix.sing_original_as = lambda ctx, src, name: (STAR.append((os.path.basename(src), name)), "star.mp3")[1]
_real_busy2 = bot._run_busy
bot._run_busy = lambda cid, fn, *a: fn(*a)
AUD.clear()
bot._cb_cover_voice(CID, {"message_id": 5}, "cvv:go")
check("«sing it as is»: the song goes out in the star's voice, the flow disarmed",
      STAR == [("kukla.mp3", "tsoi_hq")] and [p for p, _ in AUD] == ["star.mp3"]
      and bot._get_session(CID).cover_state == "", (STAR, AUD))
bot._cb_cover_voice(CID, {"message_id": 5}, "cvv:go")
check("a second tap does nothing (the song was taken)", len(STAR) == 1, STAR)
_s = bot._get_session(CID); _s.cover_voice = "none"; _s.cover_state = "want_text"; bot._store.put(_s)
check("no star: no «sing it as is» button",
      all(r[0]["callback_data"] != "cvv:go" for r in bot._cover_voice_kb(_s, "ru")["inline_keyboard"]))
_s.cover_state = ""; bot._store.put(_s)
bot._run_busy = _real_busy2
bot._cb_cover_voice(CID, {"message_id": 5}, "cvv:auto")
check("auto clears the choice", bot._get_session(CID).cover_voice == "")
bot._edit_text = _real_edit

bot._running = False; thr.join(timeout=8)
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
