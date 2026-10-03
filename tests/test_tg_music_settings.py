"""🎛 Song settings: the picker, what it persists, and what actually reaches
the songwriter.

The point of this suite is the LAST part. A settings screen that stores a
genre nobody reads is worse than no settings screen at all: the user picks
"Jazz", waits several minutes, and gets back a pop song with nothing anywhere
saying their choice was ignored. So every check below that matters follows a
value from a button tap through the session, through tg_music.prefs_of, into
the actual system prompt music.py sends.

Run: venv/Scripts/python.exe tests/test_tg_music_settings.py
"""
import sys, os, types, tempfile, threading, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import tg_music as TM
import music as M
import config as C

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_musicset_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


class _SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
    def start(self):
        self._target(*self._args, **self._kwargs)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent, bot.sent_kb, bot.edits, bot.audio_sent = [], [], [], []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append(text), bot.sent_kb.append(kw.get("keyboard")), 1)[-1]
    bot._edit_text = lambda cid, mid, text, **kw: (
        bot.edits.append((text, kw.get("keyboard"))), True)[-1]
    bot._send_audio = lambda cid, path, caption="": (
        bot.audio_sent.append((cid, path, caption)), True)[-1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    return bot


def cb(chat_id, data, msg_id=555):
    return {"callback_query": {
        "id": "cbq1", "data": data, "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}


def msg(cid, text, mid=1):
    return {"chat": {"id": cid}, "from": {"id": cid, "username": "u"},
            "message_id": mid, "text": text}


def kb_texts(kb):
    return [b["text"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


def kb_datas(kb):
    return [b["callback_data"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


bot = make_bot()
threading.Thread = _SyncThread
# The song is a queued task now (tg_songs.song_payload); drive the generation
# directly so the settings contract below is still exercised end to end.
bot._start_song_generation = lambda cid, topic, lang, duration=0: bot._generate_song(cid, topic, lang, duration)
CID = 999901
bot._user_store.put(T._User(chat_id=CID, name="U", status="approved", is_admin=False))

print("=" * 70)
print("1. The button exists, is localized, and opens the menu")
print("=" * 70)

check("the 'song_setup' key has a button label", bool(T._BTN.get("song_setup")))
check("...in Russian too (house language)",
      bool(T._BTN.get("song_setup", {}).get("ru")), T._BTN.get("song_setup"))
check("it is wired to a direct action", T._DIRECT_KB.get("song_setup") == "__song_settings__")
check("it appears on the creativity keyboard",
      any("song_setup" == T._LABEL2KEY.get(b)
          for row in T._cr_music_kb("ru")["keyboard"] for b in row),
      T._cr_music_kb("ru")["keyboard"])

bot.sent.clear(); bot.sent_kb.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("song_setup", "ru")}])
check("pressing it sends the settings menu", len(bot.sent) == 1, bot.sent)
check("...with an inline keyboard", bool((bot.sent_kb[-1] or {}).get("inline_keyboard")))
check("...carrying a row per setting",
      all(any(f"music:open:{f}" == d for d in kb_datas(bot.sent_kb[-1]))
          for f in TM.FIELDS), kb_datas(bot.sent_kb[-1]))
check("...and a reset row", "music:reset" in kb_datas(bot.sent_kb[-1]))
check("the session stays in the creativity menu (Back stays sane)",
      bot._get_session(CID).menu == "cr_music", bot._get_session(CID).menu)

print()
print("=" * 70)
print("2. Picking a value persists it and moves the ✅")
print("=" * 70)

bot.edits.clear()
bot._dispatch(cb(CID, "music:open:genre"))
check("opening a field redraws IN PLACE, not as a new message", len(bot.edits) == 1)
check("...showing that field's values", any("music:set:genre:" in d
      for d in kb_datas(bot.edits[-1][1])), kb_datas(bot.edits[-1][1]))
check("...with a back-to-menu row", "music:menu" in kb_datas(bot.edits[-1][1]))

bot.edits.clear()
bot._dispatch(cb(CID, "music:set:genre:jazz"))
check("the choice is persisted on the session",
      bot._get_session(CID).music_genre == "jazz", bot._get_session(CID).music_genre)
check("...and survives a reload from the store",
      bot._store.get(CID).music_genre == "jazz")
check("the redraw marks it with a ✅",
      any(t.startswith("✅") and "Джаз" in t for t in kb_texts(bot.edits[-1][1])),
      kb_texts(bot.edits[-1][1]))

for data, field, want in (("music:set:tempo:slow", "music_tempo", "slow"),
                          ("music:set:vocal:female", "music_vocal", "female"),
                          ("music:set:duration:120", "music_duration", "120")):
    bot._dispatch(cb(CID, data))
    check(f"{field} persisted as {want!r}",
          getattr(bot._get_session(CID), field) == want,
          getattr(bot._get_session(CID), field))

bot.edits.clear()
bot._dispatch(cb(CID, "music:menu"))
check("the menu now states every chosen value",
      all(s in " ".join(kb_texts(bot.edits[-1][1]))
          for s in ("Джаз", "Медленно", "Женский", "120")),
      kb_texts(bot.edits[-1][1]))

print()
print("=" * 70)
print("3. Auto and reset put it back to 'no opinion'")
print("=" * 70)

bot._dispatch(cb(CID, "music:set:genre:auto"))
check("choosing Auto clears the stored value (not stores 'auto')",
      bot._get_session(CID).music_genre == "", bot._get_session(CID).music_genre)
check("...so prefs_of drops the field entirely",
      "genre" not in TM.prefs_of(bot._get_session(CID)),
      TM.prefs_of(bot._get_session(CID)))

bot.sent.clear()
bot._dispatch(cb(CID, "music:reset"))
s = bot._get_session(CID)
check("reset clears every field",
      not any(getattr(s, "music_" + f) for f in TM.FIELDS),
      {f: getattr(s, "music_" + f) for f in TM.FIELDS})
check("...and says so", any("Авто" in t or "Auto" in t for t in bot.sent), bot.sent)
check("...leaving prefs_of empty", TM.prefs_of(s) == {}, TM.prefs_of(s))

print()
print("=" * 70)
print("4. A junk / stale tap cannot corrupt the setting or the ✅")
print("=" * 70)
print("""
Telegram buttons never expire, so a keyboard from an older build (or a
hand-crafted callback) can arrive at any time. The depth picker's fix applies
here for the same reason: returning silently would leave the ✅ on screen
pointing at a value we just refused to store, so the keyboard is redrawn
against what IS active.
""")

bot._dispatch(cb(CID, "music:set:genre:jazz"))
bot.edits.clear()
bot._dispatch(cb(CID, "music:set:genre:polka_not_a_genre"))
check("an unknown genre is refused", bot._get_session(CID).music_genre == "jazz",
      bot._get_session(CID).music_genre)
check("...and the keyboard is redrawn honestly rather than left lying",
      len(bot.edits) == 1 and any(t.startswith("✅") and "Джаз" in t
                                  for t in kb_texts(bot.edits[-1][1])),
      kb_texts(bot.edits[-1][1]) if bot.edits else "no redraw")

bot._dispatch(cb(CID, "music:set:duration:120"))
bot._dispatch(cb(CID, "music:set:duration:99999"))
check("an off-menu duration is refused, leaving the previous one intact",
      bot._get_session(CID).music_duration == "120",
      bot._get_session(CID).music_duration)
bot._dispatch(cb(CID, "music:set:nosuchfield:x"))
check("an unknown FIELD does not crash the dispatcher", True)
bot.edits.clear()
bot._dispatch(cb(CID, "music:"))
check("a malformed payload falls back to the menu, not silence",
      len(bot.edits) == 1, bot.edits)

# Corrupted persisted state must not take the keyboard down with it.
s = bot._get_session(CID); s.music_duration = {"not": "a number"}; bot._store.put(s)
try:
    TM._music_menu_kb(s, "ru"); ok_corrupt = True
except Exception as exc:
    ok_corrupt = False; print("   ", exc)
check("a corrupted stored value does not crash the keyboard", ok_corrupt)
s.music_duration = ""; bot._store.put(s)

print()
print("=" * 70)
print("5. The settings actually REACH the songwriter")
print("=" * 70)
print("""
The whole point. A picker whose value never reaches the prompt is a lie told
with a checkmark -- so these follow the value all the way into the system
prompt string that music.py hands to the LLM.
""")

s = bot._get_session(CID)
s.music_genre, s.music_tempo, s.music_vocal, s.music_duration = \
    "jazz", "slow", "female", "180"
bot._store.put(s)

prefs = TM.prefs_of(s)
check("prefs_of maps genre to an English phrase for the model",
      "jazz" in prefs.get("genre", "").lower(), prefs)
check("...tempo to a BPM range, not an invented exact number",
      "65-75" in prefs.get("tempo", ""), prefs)
check("...and vocal to an explicit gender",
      "FEMALE" in prefs.get("vocal", ""), prefs)
check("resolve_duration returns the chosen length", TM.resolve_duration(s) == 180,
      TM.resolve_duration(s))

captured = []
import llm as L
_real = L.call_llm_simple
L.call_llm_simple = lambda ctx, sys_p, user_p, **kw: (
    captured.append(sys_p),
    json.dumps({"lyrics": "[verse]\nx",
                "style": "Global Metadata: x Vocal Details: y Arrangement: z"}))[-1]
try:
    M.build_structured_caption(None, "a rainy evening", "ru",
                               duration_s=180, prefs=prefs)
finally:
    L.call_llm_simple = _real

sys_p = captured[0]
check("the prompt carries a HARD REQUIREMENTS block", "HARD REQUIREMENTS" in sys_p)
check("...naming the chosen genre", "jazz" in sys_p.lower(), sys_p[-500:])
check("...the chosen tempo", "65-75" in sys_p, sys_p[-500:])
check("...and the chosen vocal gender", "FEMALE" in sys_p, sys_p[-500:])
check("...and the requested duration", "180" in sys_p, sys_p[-600:])

# No settings chosen -> no requirements block at all. An unset setting must
# not silently become an instruction.
captured.clear()
L.call_llm_simple = lambda ctx, sys_p, user_p, **kw: (
    captured.append(sys_p),
    json.dumps({"lyrics": "[verse]\nx",
                "style": "Global Metadata: x Vocal Details: y Arrangement: z"}))[-1]
try:
    M.build_structured_caption(None, "a rainy evening", "ru", prefs={})
finally:
    L.call_llm_simple = _real
check("no chosen settings -> no requirements block",
      "HARD REQUIREMENTS" not in captured[0])

print()
print("=" * 70)
print("6. Instrumental is handled as its own shape, not just a label")
print("=" * 70)
print("""
MiniMax's guide says instrumental work must SAY it is instrumental and name
the instrument carrying the lead line. Leaving sung words in `lyrics` while
the caption says "no singer" is a straight contradiction, and the guide's
precedence rules make that the caption's problem to lose -- so the lyrics are
dropped to bare tags instead.
""")

s.music_vocal = "instrumental"; bot._store.put(s)
iprefs = TM.prefs_of(s)
check("prefs_of flags it distinctly, not as a 'vocal'",
      iprefs.get("instrumental") is True and "vocal" not in iprefs, iprefs)

captured.clear()
L.call_llm_simple = lambda ctx, sys_p, user_p, **kw: (
    captured.append(sys_p),
    json.dumps({"lyrics": "[verse]\n[chorus]",
                "style": "Global Metadata: x Vocal Details: y Arrangement: z"}))[-1]
try:
    M.build_structured_caption(None, "a rainy evening", "ru", prefs=iprefs)
finally:
    L.call_llm_simple = _real
check("the prompt declares it INSTRUMENTAL", "INSTRUMENTAL" in captured[0])
check("...asks for the lead instrument to be named",
      "lead melodic line" in captured[0], captured[0][-400:])
check("...and forbids words under the tags",
      "ONLY section tags" in captured[0], captured[0][-400:])

print()
print("=" * 70)
print("7. The topic prompt shows what you are about to get")
print("=" * 70)

s.music_vocal = "female"; bot._store.put(s)
bot.sent.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("songs", "ru")}])
joined = " ".join(bot.sent)
check("the topic ask names the current settings",
      all(x in joined for x in ("Джаз", "Женский", "180")), bot.sent)
check("...and still arms the topic capture",
      bot._get_session(CID).reg_state == "song_topic")

print()
print("=" * 70)
print("8. Generation uses the chosen duration and prefs end to end")
print("=" * 70)

seen = {}
_oc, _og, _oe = (M.build_structured_caption, M.generate_music, M.engine_available)
M.engine_available = lambda ctx=None, preset=None: (True, "")
M.build_structured_caption = lambda ctx, topic, lang, **kw: (
    seen.update(caption_kw=kw), {"lyrics": "[verse]\nx", "style": "pop"})[-1]
M.generate_music = lambda ctx, lyrics, style, **kw: (
    seen.update(gen_kw=kw), "/tmp/song.wav")[-1]
bot.audio_sent.clear()
try:
    bot._user_gate(CID, msg(CID, "a song about rain"))
finally:
    M.build_structured_caption, M.generate_music, M.engine_available = _oc, _og, _oe

check("a song was delivered", len(bot.audio_sent) == 1, bot.audio_sent)
check("the songwriter got the chosen duration",
      seen.get("caption_kw", {}).get("duration_s") == 180, seen.get("caption_kw"))
check("the songwriter got the chosen prefs",
      "jazz" in str(seen.get("caption_kw", {}).get("prefs", {})).lower(),
      seen.get("caption_kw"))
check("the RENDER got the same duration, not the config default",
      seen.get("gen_kw", {}).get("duration_s") == 180, seen.get("gen_kw"))

print()
print("=" * 70)
print("9. Every string this feature added is translated")
print("=" * 70)

_keys = (["ms_title", "ms_hint", "ms_field_title", "ms_reset", "ms_back",
          "ms_was_reset", "m_auto", "m_secs", "song_setup"]
         + [f"ms_{f}" for f in TM.FIELDS]
         + [f"ms_pick_{f}" for f in TM.FIELDS]
         + [f"mg_{g}" for g in TM.GENRES if g != "auto"]
         + [f"mt_{t}" for t in TM.TEMPOS if t != "auto"]
         + [f"mv_{v}" for v in TM.VOCALS if v != "auto"])
_missing = [k for k in _keys
            if not (T._MSG.get(k) or T._BTN.get(k))
            or not (T._MSG.get(k) or T._BTN.get(k)).get("ru")
            or not (T._MSG.get(k) or T._BTN.get(k)).get("en")]
check("every new key has both an EN and a RU string", not _missing, _missing)
# Every value offered by a keyboard must have a label, or a button renders
# blank -- the tables and the strings drifting apart is the failure mode.
_unlabelled = [f"{p}{v}" for p, table in (("mg_", TM.GENRES), ("mt_", TM.TEMPOS),
                                          ("mv_", TM.VOCALS))
               for v in table if v != "auto" and not T._MSG.get(f"{p}{v}")]
check("every genre/tempo/vocal value has a label", not _unlabelled, _unlabelled)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
