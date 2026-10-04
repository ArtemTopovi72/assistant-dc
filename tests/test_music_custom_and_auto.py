"""Song settings: typed length / BPM within the engine's range, «на усмотрение
бота» for the length, the bot's own picks for every Auto field before the
songwriting, and the «🎲 Бот выбрал сам» line after the song. The engine is
stubbed; the settings, capture and reporting are real."""
import sys, os, types, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
_RESETS = {"♻️ сбрось настройки песни", "Сбросить настройки песни", "сброс настроек песни", "reset song settings", "сбрось настройки песни"}
intent.STUB = lambda t: {"song": "reset"} if t.strip() in _RESETS else None
intent.CHOICE_STUB = lambda q, t: "song" if t.strip() in _RESETS else "nothing"
os.environ.setdefault("F5_TEST_RUN", "1")
_DATA = tempfile.mkdtemp(prefix="tg_music_custom_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)
import tg_music as TM
import music as M
import config as C
import tg_songs as TS
M.MUSIC_ENGINE = "music3"   # this suite covers the Music3 knobs (length is a Music3 promise)

CID = 515151
PASSED = FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond: PASSED += 1; print("PASS ", label)
    else:    FAILED += 1; print("FAIL ", label, " ", str(detail)[:250])


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent, bot.edits, bot.audio_sent = [], [], []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append(text), 1)[-1]
    bot._edit_text = lambda cid, mid, text, **kw: (bot.edits.append((text, kw.get("keyboard"))), True)[-1]
    bot._send_audio = lambda cid, path, caption="": (bot.audio_sent.append(path), True)[-1]
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


def kb_texts(kb):
    return [b["text"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


# ── engine-side contracts ────────────────────────────────────────────────────
check("custom_bpm parses bpm:N inside the range only",
      M.custom_bpm("bpm:118") == 118 and M.custom_bpm("bpm:30") == 0 and M.custom_bpm("fast") == 0)
check("a pinned BPM becomes an exact tempo requirement",
      M.prefs_from(tempo="bpm:118")["tempo"] == "exactly 118 BPM")
check("render_ceiling gives the ask headroom for the outro",
      M.render_ceiling(100) == 130 and M.render_ceiling(180) <= C.MUSIC_CEILING_SECONDS)

import tempfile as _tf, config as _cfg2
_cfg_out = _cfg2.OUTPUT_DIR; M.OUTPUT_DIR = _tf.mkdtemp(prefix="music_eta_")   # never touch the live file
check("the quality ETA is measured for a 180-second song, not the old 30-second figure",
      M.ETA_QUOTE_SECONDS == 180 and 300 <= M.eta_seconds("fast", steps=30) <= 450
      and M.eta_seconds("max", steps=30) > 1500 and M.eta_label(336, "ru") == "~6 мин" and M.eta_label(45, "ru") == "~45 с")
check("the default is 20 steps and it is cheaper than 30, and 50 dearer",
      M.MUSIC_STEPS == 20 and M.STEPS_RANGE == (20, 50)
      and M.eta_seconds("fast", steps=20) < M.eta_seconds("fast", steps=30) < M.eta_seconds("fast", steps=50))
check("max streams its DiT, so steps weigh more there",
      M.eta_seconds("max", steps=50) / M.eta_seconds("max", steps=20)
      > M.eta_seconds("fast", steps=50) / M.eta_seconds("fast", steps=20))
check("clamp_steps keeps the range and falls back to the default",
      M.clamp_steps(35) == 35 and M.clamp_steps(19) == 20 and M.clamp_steps(51) == 20 and M.clamp_steps("x") == 20)
for _ in range(3):
    M.record_render_time("fast", 180, 300, steps=30)
check("a real render feeds the median, normalised to its step count",
      300 <= M.eta_seconds("fast", steps=30) <= 336 and M.eta_seconds("fast", steps=20) < 300,
      (M.eta_seconds("fast", steps=30), M.eta_seconds("fast", steps=20)))
M.record_render_time("fast", 180, 250, steps=20)
check("a 20-step render is stored with its steps",
      any(r[2] == 20 for r in M._eta_runs()["fast"]))
check("the quality buttons quote the measured time",
      "~" in TM._value_label("quality", "max", "ru") and "мин" in TM._value_label("quality", "max", "ru"))

# ── the settings screens ─────────────────────────────────────────────────────
bot = make_bot(); sess = bot._get_session(CID)
kb = TM._field_kb(sess, "duration", "ru")
check("the length picker offers «на усмотрение бота», the presets and «своя длина»",
      "🎲 На усмотрение бота" in kb_texts(kb) and "✏️ Своя длина" in kb_texts(kb), kb_texts(kb))
check("the tempo picker offers «свой BPM»", "✏️ Свой BPM" in kb_texts(TM._field_kb(sess, "tempo", "ru")))

bot._cb_song_settings(CID, {"message_id": 7}, "music:set:duration:auto")
sess = bot._get_session(CID)
check("Auto length is stored as a real choice and resolves to 0 (the bot decides)",
      sess.music_duration == "auto" and TM.resolve_duration(sess) == 0 and TM.auto_fields(sess)["duration"])
check("the menu says so", "🎲 На усмотрение бота" in TM._current_label(sess, "duration", "ru"))

bot._cb_song_settings(CID, {"message_id": 7}, "music:custom:duration")
sess = bot._get_session(CID)
check("«своя длина» arms the capture and asks for a number in range",
      sess.reg_state == "music_custom:duration" and "от 20 до 180" in bot.sent[-1], (sess.reg_state, bot.sent[-1:]))
bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 2, "text": "999"})
check("an out-of-range number is refused and the capture stays armed",
      "от 20 до 180" in bot.sent[-1] and bot._get_session(CID).reg_state == "music_custom:duration", bot.sent[-1])
bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 3, "text": "100 секунд"})
sess = bot._get_session(CID)
check("a number in range is stored, confirmed, and the menu is redrawn",
      sess.music_duration == "100" and sess.reg_state == "" and TM.resolve_duration(sess) == 100
      and any("Установлено: 100 с" in t for t in bot.sent), (sess.music_duration, bot.sent[-2:]))
check("the custom length shows on its button with a ✅",
      any(t.startswith("✅ ✏️ Своя длина (100 с)") for t in kb_texts(TM._field_kb(sess, "duration", "ru"))),
      kb_texts(TM._field_kb(sess, "duration", "ru")))

bot._cb_song_settings(CID, {"message_id": 7}, "music:custom:tempo")
bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 4, "text": "118"})
sess = bot._get_session(CID)
check("a typed BPM is stored as bpm:N, labelled «118 BPM», and reaches the brief as a requirement",
      sess.music_tempo == "bpm:118" and TM._current_label(sess, "tempo", "ru") == "118 BPM"
      and TM.prefs_of(sess).get("tempo") == "exactly 118 BPM", (sess.music_tempo, TM.prefs_of(sess)))
check("/cancel abandons a capture", (bot._cb_song_settings(CID, {"message_id": 7}, "music:custom:tempo"),
      bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 5, "text": "/cancel"}),
      bot._get_session(CID).reg_state)[-1] == "")
# ── steps live on the Quality screen ─────────────────────────────────────────
bot = make_bot(); sess = bot._get_session(CID)
kb = kb_texts(TM._field_kb(sess, "quality", "ru"))
check("the quality screen carries the presets, an Auto/turbo choice, then 20/25/30/40/50 steps with a time each, then a typed one",
      any(t.startswith("✅ ⚡ Быстро · ~") for t in kb) and any(t.startswith("✅ ⚡ Авто (турбо, 8)") for t in kb)
      and all(any(t.startswith(f"{n} · ~") or t.startswith(f"✅ {n} · ~") for t in kb) for n in (20, 25, 30, 40, 50))
      and "✏️ Другое число шагов" in kb and any(t.startswith("🔬 Шагов: 8") for t in kb), kb)
check("the default is the turbo LoRA's 8 steps (None to generate_music), and the menu line says so",
      TM.resolve_steps(sess) == 8 and TM.resolve_steps_for_generate(sess) is None
      and "8 шагов" in TM._current_label(sess, "quality", "ru"),
      TM._current_label(sess, "quality", "ru"))
bot._cb_song_settings(CID, {"message_id": 7}, "music:set:steps:50")
sess = bot._get_session(CID)
kb50 = kb_texts(TM._field_kb(sess, "quality", "ru"))
check("50 steps is stored, ticked, and every preset's time grows",
      sess.music_steps == "50" and TM.resolve_steps(sess) == 50 and any(t.startswith("✅ 50 · ~") for t in kb50)
      and M.eta_seconds("fast", steps=50) > M.eta_seconds("fast", steps=20)
      and "50 шагов" in TM._current_label(sess, "quality", "ru"), (sess.music_steps, kb50))
fast50 = next(t for t in kb50 if "⚡" in t); fast20 = next(t for t in kb if "⚡" in t)
check("the ⚡ button quotes a longer wait at 50 than at 20", fast50 != fast20, (fast20, fast50))
bot._cb_song_settings(CID, {"message_id": 7}, "music:set:steps:99")
check("an out-of-range button value falls back to Auto/turbo, not a crash",
      bot._get_session(CID).music_steps == "" and TM.resolve_steps(bot._get_session(CID)) == 8
      and TM.resolve_steps_for_generate(bot._get_session(CID)) is None)
bot._cb_song_settings(CID, {"message_id": 7}, "music:custom:steps")
sess = bot._get_session(CID)
check("«другое число шагов» arms the capture in 20..50",
      sess.reg_state == "music_custom:steps" and "от 20 до 50" in bot.sent[-1], (sess.reg_state, bot.sent[-1:]))
bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 2, "text": "60"})
check("60 is refused", "от 20 до 50" in bot.sent[-1] and bot._get_session(CID).reg_state == "music_custom:steps")
n_before = len(bot.sent)
bot._user_gate(CID, {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 3, "text": "35"})
sess = bot._get_session(CID)
check("35 is stored, confirmed with the quality line, and the QUALITY screen comes back (not the menu)",
      sess.music_steps == "35" and sess.reg_state == "" and any("Установлено:" in t and "35 шагов" in t for t in bot.sent[n_before:])
      and "Шаги — проходы сэмплера" in bot.sent[-1], (sess.music_steps, bot.sent[n_before:]))
check("a typed count shows on its button with a ✅",
      any(t.startswith("✅ ✏️ Другое число шагов (35)") for t in kb_texts(TM._field_kb(sess, "quality", "ru"))))
bot._cb_song_settings(CID, {"message_id": 7}, "music:reset")
check("reset clears the steps too", bot._get_session(CID).music_steps == "" )

check("a typed «song for 100 seconds» keeps its number instead of snapping to a preset",
      TS.snap_duration(100) == 100 and TS.snap_duration(5) == 20 and TS.snap_duration(0) == 0)

# ── a TYPED «сбрось настройки песни» resets, instead of the model claiming it did
check("the reset phrase is recognised in RU/EN, genitive included, and only as a whole message",
      all(TM.is_reset_phrase(t) for t in ("♻️ сбрось настройки песни", "Сбросить настройки песни",
                                           "сброс настроек песни", "reset song settings"))
      and not any(TM.is_reset_phrase(t) for t in ("напиши песню про сброс настроек", "сбрось", "настройки песни")))
bot = make_bot(); sess = bot._get_session(CID)
sess.music_tempo = "bpm:118"; sess.music_steps = "40"; sess.music_genre = "rock"; sess.reg_state = ""; bot._store.put(sess)
bot._backend = T.InMemoryBackend(); bot._get_ctx = lambda: None
bot._resolve_and_push(CID, [{"type": "text", "text": "♻️ сбрось настройки песни"}])
sess = bot._get_session(CID)
check("every field is back on Auto, the ♻️ line is said and the menu redrawn; nothing reaches the agent",
      sess.music_tempo == "" and sess.music_steps == "" and sess.music_genre == ""
      and any("сброшены на «Авто»" in t for t in bot.sent) and any("Настройки песни" in t for t in bot.sent[-1:])
      and bot._backend.depth() == 0, (sess.music_tempo, sess.music_steps, bot.sent[-2:], bot._backend.depth()))

# ── the bot's own picks and the report ───────────────────────────────────────
import llm as _llm
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: (
    '{"genre": "synth", "bpm": 118, "vocal": "female", "seconds": 90, "why": "Тема про ночной город просит неон."}')
picked = M.choose_auto_params(None, "ночной город", "ru", genre=True, tempo=True, vocal=True, duration=True)
check("choose_auto_params validates every field",
      picked == {"genre": "synth", "bpm": 118, "vocal": "female", "seconds": 90,
                 "why": "Тема про ночной город просит неон."}, picked)
check("only the fields asked for come back",
      set(M.choose_auto_params(None, "x", "ru", tempo=True)) == {"bpm", "why"})
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: '{"genre": "polka", "bpm": 900, "seconds": 5}'
# A too-short length is the nearest one the engine can make («джингл на 15 секунд»).
check("junk from the model is dropped, not stored",
      M.choose_auto_params(None, "x", "ru", genre=True, tempo=True, duration=True) == {"seconds": 20, "why": ""})
check("nothing on Auto -> no call at all", M.choose_auto_params(None, "x", "ru") == {})
asked = {}
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: (asked.__setitem__("user", user), '{"bpm": 92, "why": "ok"}')[1]
M.choose_auto_params(None, "дорога домой", "ru", tempo=True, fixed={"genre": "Country", "vocal": "FEMALE vocal", "seconds": 100})
check("what the user DID choose goes into the ask as fixed conditions, not guessed blind",
      "Already fixed by the user" in asked["user"] and "genre: Country" in asked["user"]
      and "vocal: FEMALE vocal" in asked["user"] and "seconds: 100" in asked["user"], asked)
line = TM.chose_line(picked, "ru")
check("the report names each pick and the reason",
      line.startswith("🎲 <b>Бот выбрал сам:</b>") and "жанр — 🎹 Синти-поп" in line or "жанр — " in line
      and "темп — 118 BPM" in line and "длина — 90 с" in line and "неон" in line, line)

# end to end through _generate_song with stubs: everything on Auto
bot = make_bot(); sess = bot._get_session(CID)
sess.music_duration = "auto"; sess.music_tempo = ""; bot._store.put(sess)   # the typed BPM above is this chat's
seen = {}
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: (
    '{"genre": "pop", "bpm": 104, "vocal": "male", "seconds": 60, "why": "Лёгкая тема."}')
M.engine_available = lambda ctx, preset=None: (True, "")
def _cap(ctx, topic, lang, duration_s=None, prefs=None):
    seen["duration"] = duration_s; seen["prefs"] = dict(prefs or {})
    return {"lyrics": "[verse]\nla", "style": "pop"}
M.build_structured_caption = _cap
M.generate_music = lambda ctx, lyrics, style, duration_s=60, preset=None, **kw: (seen.__setitem__("render", duration_s), seen.__setitem__("steps", kw.get("steps")), "x.wav")[2]
bot._get_ctx = lambda: None
bot._generate_song(CID, "про лето", "ru")
check("the picks reach the brief as requirements and the length comes from the bot",
      seen.get("duration") == 60 and seen["prefs"].get("tempo") == "exactly 104 BPM"
      and "Pop" in seen["prefs"].get("genre", "") and "MALE" in seen["prefs"].get("vocal", ""), seen)
check("an untouched chat's render gets None, so build_workflow's turbo LoRA gate fires",
      seen.get("steps") is None, seen.get("steps"))
check("the song is delivered, then the «бот выбрал сам» line",
      bot.audio_sent == ["x.wav"] and bot.sent and bot.sent[-1].startswith("🎲 <b>Бот выбрал сам:</b>")
      and "темп — 104 BPM" in bot.sent[-1] and "длина — 60 с" in bot.sent[-1], bot.sent[-1:])

# the user's own choice wins over the bot's pick
bot = make_bot(); sess = bot._get_session(CID)
sess.music_genre = "rock"; sess.music_duration = "100"; sess.music_tempo = ""; sess.music_steps = "40"; bot._store.put(sess)
asked_e2e = {}
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: (asked_e2e.__setitem__("user", user),
    '{"bpm": 104, "vocal": "male", "why": "Лёгкая тема."}')[1]
bot._get_ctx = lambda: None
bot._generate_song(CID, "про лето", "ru")
check("a chosen step count reaches the render", seen.get("steps") == 40, seen.get("steps"))
check("a chosen genre and a typed length are kept; only tempo and vocal are the bot's",
      seen["duration"] == 100 and "rock" in seen["prefs"]["genre"].lower()
      and "genre: " in asked_e2e.get("user", "") and "seconds: 100" in asked_e2e.get("user", "")
      and "жанр" not in bot.sent[-1] and "длина" not in bot.sent[-1] and "темп — 104 BPM" in bot.sent[-1],
      (seen, bot.sent[-1:]))

# the vocal button decides the singer: a caption naming the other gender is rewritten
_m = M.enforce_vocal("Vocal Details: a soulful female singer, her voice airy.", "a MALE lead vocal")
check("a MALE choice rewrites a female caption and says so",
      "male singer" in _m and "MALE (a man" in _m and "female singer" not in _m, _m)
_f = M.enforce_vocal("Vocal Details: a gritty male baritone, he growls.", "a FEMALE lead vocal")
check("a FEMALE choice rewrites a male caption", "female alto" in _f and "FEMALE (a woman" in _f, _f)
check("a duet or no choice leaves the caption alone",
      M.enforce_vocal("x", "a male/female duet trading lines") == "x" and M.enforce_vocal("x", "") == "x")

print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
sys.exit(1 if FAILED else 0)
