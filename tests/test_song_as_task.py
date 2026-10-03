"""A song is a queued task, not a bare thread.

Live, 2026-09-12: a song ran on its own thread, so ⛔ Стоп answered "Сейчас
ничего не выполняется" and the song arrived anyway 60 s later; the chat never
looked busy, there was no status line and no queue position. Now the topic
becomes a "[song:N] topic" task handled in _run_task_inner under the task's
status line and cancel event; a Stop mid-write is silent, not "the writer
fumbled".

Pure: music.py and the LLM are stubbed; no GPU.

Run: venv/Scripts/python.exe tests/test_song_as_task.py
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import tg_songs as S
import intent   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py,
# bench/intent_song_weather_live.py
_MAKE = {"song": "make"}
_PREV = {"song": "make", "song_uses_previous_text": True}
intent.STUB = {"колыбельная на английском": {"reply_language": "en", "song": "make"},
               "а теперь сочини на него песню, 30 секунд": _PREV, "а теперь сделай из него песню": _PREV,
               "положи это на музыку": _PREV, "а можешь это спеть?": _PREV,
               "сочини песню про кота": _MAKE, "сочини гимн нашего отдела": _MAKE,
               "сделай колыбельную для сына": _MAKE, "сделай джингл для кофейни": _MAKE,
               "напиши частушку про тёщу": _MAKE,
               "а теперь то же самое в стиле рок": {"song": "sound"}, "а теперь в стиле рок": {"song": "sound"},
               "а можно версию подлиннее, на 2 минуты": {"song": "words", "song_seconds": 120},
               "минуты на две": {"song_seconds": 120}, "песню на полторы минуты": {"song": "make", "song_seconds": 90},
               "замени припев на повеселее": {"song": "words"}, "сделай подлиннее": {"song": "words"}}.get
import music as M

T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_song_task_"))

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


check("payload round-trips", S.parse_song_payload(S.song_payload("про дождь", 30)) == ("про дождь", 30))
check("no length -> 0", S.parse_song_payload(S.song_payload("про дождь")) == ("про дождь", 0))
check("ordinary text is not a payload", S.parse_song_payload("сочини песню") == ("", 0))
check("a payload is not re-detected as a request", S.song_request("[song:30] сочини песню")[0] is False)


class Ctx:
    model_name = "m"; stage_callback = None
    def __init__(self): self.cancel_event = threading.Event()
    def is_cancelled(self): return self.cancel_event.is_set()
    def set_stage(self, s):
        if self.stage_callback: self.stage_callback(s)


def make_bot():
    bot = T.TelegramBot("1:TEST", lambda: Ctx(), lambda: None, lambda: {}, silent_mode=True)
    bot.sent, bot.audio = [], []
    bot._send_text = lambda cid, text, **k: bot.sent.append(text)
    bot._send_audio = lambda cid, path, caption="": (bot.audio.append(path), True)[1]
    bot._user_store.put(T._User(chat_id=5001, name="F", status="approved"))
    return bot


_engine, _cap, _gen = M.engine_available, M.build_structured_caption, M.generate_music
M.engine_available = lambda ctx=None, preset=None: (True, "")
M.build_structured_caption = lambda ctx, topic, lang, **kw: {"lyrics": "la la", "style": "pop"}
M.generate_music = lambda ctx, lyrics, style, **kw: "/tmp/song.wav"
try:
    # 1. a typed request is enqueued as a task
    bot = make_bot()
    enq = []
    bot._enqueue_item = lambda cid, item: enq.append(item)
    bot._start_song_generation(5001, "про дождь", "ru", 30)
    check("the flow enqueues a [song] task", enq and enq[0]["text"] == "[song:30] про дождь", enq)

    # 2. the generation honours the task's stage callback and delivers
    bot = make_bot()
    ctx = Ctx(); stages = []
    ctx.stage_callback = stages.append
    sent = bot._generate_song(5001, "про дождь", "ru", 30, ctx=ctx)
    check("the song is delivered", sent is True and bot.audio == ["/tmp/song.wav"], bot.audio)
    check("stages went through the task's callback", "Writing the lyrics" in stages, stages)

    # 3. Stop while the lyrics are being written: silent, nothing delivered
    bot = make_bot()
    ctx = Ctx()

    def _writer_interrupted(ctx_, topic, lang, **kw):
        ctx.cancel_event.set()
        raise M.SongwritingFailed("interrupted")
    M.build_structured_caption = _writer_interrupted
    sent = bot._generate_song(5001, "про дождь", "ru", 30, ctx=ctx)
    check("a cancelled write delivers nothing", sent is False and bot.audio == [])
    check("and does not blame the writer", not any("Не получилось" in t or "lyrics" in t for t in bot.sent), bot.sent)
    M.build_structured_caption = lambda ctx, topic, lang, **kw: {"lyrics": "la la", "style": "pop"}

    # 4. Stop between the write and the render: no render, nothing delivered
    bot = make_bot()
    ctx = Ctx(); rendered = []

    def _cap_then_stop(ctx_, topic, lang, **kw):
        ctx.cancel_event.set()
        return {"lyrics": "la", "style": "pop"}
    M.build_structured_caption = _cap_then_stop
    M.generate_music = lambda *a, **k: rendered.append(1) or "/tmp/song.wav"
    sent = bot._generate_song(5001, "про дождь", "ru", 30, ctx=ctx)
    check("Stop before the render skips it", sent is False and rendered == [] and bot.audio == [])

    # 5. the task runner recognises the payload (source contract)
    import inspect, tg_tasks
    src = inspect.getsource(tg_tasks)
    check("_run_task_inner handles the song payload", "parse_song_payload(task.user_text)" in src)
    check("under the task's cancel and stage", "ctx.stage_callback = _song_stage" in src)
finally:
    M.engine_available, M.build_structured_caption, M.generate_music = _engine, _cap, _gen

# "а теперь сочини на него песню" after the bot wrote four lines: the song
# must use THOSE lines (live, 2026-09-12: it sang about city lights instead).
check("'на него' refers to the previous text", S.refers_to_previous_text("а теперь сочини на него песню, 30 секунд"))
check("a plain topic does not", not S.refers_to_previous_text("сочини песню про кота"))
check("'из него' / 'положи это на музыку' / 'можешь это спеть' sing the previous text",
      all(S.song_request(t)[0] and S.refers_to_previous_text(t) for t in
          ("а теперь сделай из него песню", "положи это на музыку", "а можешь это спеть?")))
check("'как научиться петь' is not a song", not S.song_request("как научиться петь")[0])
check("sung forms are songs: гимн, колыбельная, джингл, частушка",
      all(S.song_request(t)[0] for t in ("сочини гимн нашего отдела", "сделай колыбельную для сына",
                                         "сделай джингл для кофейни", "напиши частушку про тёщу"))
      and not S.song_request("расскажи про гимн России")[0])
_lyr = "Золотые листья кружат в танце,\nОсень тихо входит в города.\nСерый дождь застыл,\nУлетают птицы навсегда."
check("four short lines look like lyrics", S.looks_like_lyrics(_lyr))
check("a paragraph does not", not S.looks_like_lyrics("Это длинный ответ в одну строку про всё на свете и ещё немного, чтобы точно превысить лимит символов на строку и не быть стихами."))
check("tagged lyrics keep every line", all(ln in S.tag_lyrics(_lyr) for ln in _lyr.splitlines()) and "[verse]" in S.tag_lyrics(_lyr))
# The engine check hits the live ComfyUI (node list); offline it must be stubbed
# again or the song is refused before the lyrics are even looked at.
M.engine_available = lambda ctx=None, preset=None: (True, "")
M.build_structured_caption = lambda ctx, topic, lang, **kw: {"lyrics": "WRITER", "style": "pop"}
_got = {}
M.generate_music = lambda ctx, lyrics, style, **kw: _got.update(lyrics=lyrics) or "/tmp/song.wav"
bot = make_bot()
bot._generate_song(5001, S.LYRICS_MARK + _lyr, "ru", 30, ctx=Ctx())
check("the given lyrics reach the renderer, not the writer's", "Золотые листья" in _got.get("lyrics", "") and "WRITER" not in _got.get("lyrics", ""), _got)
# The words typed under the request itself: those are sung, the request line steers the sound.
_seen = {}
M.build_structured_caption = lambda ctx, topic, lang, **kw: _seen.update(topic=topic) or {"lyrics": "WRITER", "style": "pop"}
bot._generate_song(5001, "сделай песню в стиле панк-рок на мои стихи:\n" + S.LYRICS_MARK + _lyr, "ru", 30, ctx=Ctx())
check("inline lyrics are sung verbatim", "Золотые листья" in _got.get("lyrics", "") and "WRITER" not in _got.get("lyrics", ""), _got)
check("and the request line reaches the style", "панк-рок" in _seen.get("topic", ""), _seen)
check("a style wish after a song is a redo", S.song_followup("а теперь то же самое в стиле рок"))
check("and a longer one keeps its length", S.request_seconds("а можно версию подлиннее, на 2 минуты") == 120)
check("minutes in words", (S.request_seconds("минуты на две"), S.request_seconds("песню на полторы минуты")) == (120, 90))
_langs = []
M.build_structured_caption = lambda ctx, topic, lang, **kw: _langs.append(lang) or {"lyrics": "WRITER", "style": "pop"}
bot._generate_song(5001, "колыбельная на английском", "ru", 30, ctx=Ctx())
check("'на английском' writes English lyrics in a Russian chat", _langs[-1:] == ["en"], _langs)
M.build_structured_caption = lambda ctx, topic, lang, **kw: _seen.update(topic=topic) or {"lyrics": "WRITER", "style": "pop"}
check("a picture in a style is not", not S.song_followup("нарисуй кота в стиле аниме"))
bot._generate_song(5001, "про дождь", "ru", 30, ctx=Ctx())
check("a delivered song is remembered for the next message", S.LAST_SONG.get(5001) == "WRITER", S.LAST_SONG)
import tg_resolve
_rs = inspect.getsource(tg_resolve)
check("the router redoes the last song", "_tg_songs.followup_topic(raw, _last_song)" in _rs)
check("a new sound keeps the words", S.LYRICS_MARK in S.followup_topic("а теперь в стиле рок", "la"))
check("a new chorus or length rewrites them", S.song_followup("замени припев на повеселее")
      and not any(S.LYRICS_MARK in S.followup_topic(t, "la") for t in ("замени припев на повеселее", "сделай подлиннее")))
check("the router splits typed lyrics off the request", '_head + "\\n" + _tg_songs.LYRICS_MARK + _body' in _rs)
check("the router carries the previous reply when the request refers to it", "_tg_songs.LYRICS_MARK + _prev" in _rs)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
