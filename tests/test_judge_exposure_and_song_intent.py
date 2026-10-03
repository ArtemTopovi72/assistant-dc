"""Two things the live drive of 2026-09-12 found (bench/live_tg_scenarios.py).

1. The outer image judge called a sunlit orange cat "a black silhouette" on
   three renders in a row (5/10, 4/10, 5/10), so a plain scene cost three
   renders, an agent inspect/edit/redraw cycle and an apology for a shadow
   that was not there. exposure.py measures the pixels; a darkness-only
   complaint about a normally exposed picture is overruled.

2. "Сочини короткую весёлую песню про кота" typed in chat reached the chat
   model, which wrote four lines and read them aloud. A plain-words song
   request is the song flow.

Pure: no model, no GPU.

Run: venv/Scripts/python.exe tests/test_judge_exposure_and_song_intent.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


# ── 1. exposure ──────────────────────────────────────────────────────────────
import exposure as E
from PIL import Image

d = tempfile.mkdtemp(prefix="exposure_")
bright = os.path.join(d, "bright.png"); Image.new("RGB", (64, 64), (200, 140, 60)).save(bright)
dark = os.path.join(d, "dark.png")
import random as _rnd
# dark but not a refusal card: wide spread of dark tones, many colours
_im = Image.new("RGB", (160, 160)); _im.putdata([(_rnd.randint(0, 90), _rnd.randint(0, 60), _rnd.randint(0, 70))
                                                 for _ in range(160 * 160)]); _im.save(dark)

check("a lit picture measures as normally exposed", E.normally_exposed(E.measure(bright)))
check("a dark picture does not", not E.normally_exposed(E.measure(dark)))
check("evidence is written for the lit picture", "normally exposed" in E.evidence(bright))
check("and withheld for the dark one", E.evidence(dark) == "")
check("a missing file is no evidence", E.evidence(os.path.join(d, "nope.png")) == "")

check("silhouette complaint is a darkness complaint",
      E.is_dark_complaint("The cat is extremely dark/underexposed, appearing almost as a "
                          "black silhouette. The window frame also has heavy black artifacts."))
check("'obscured by a large black bar' is one too",
      E.is_dark_complaint("The main subject is almost entirely obscured by a large black bar "
                          "at the bottom of the frame."))
check("an absent subject in another sentence keeps the complaint",
      not E.is_dark_complaint("The cat is a dark silhouette. The requested dog is absent."))
check("a second copy keeps it",
      not E.is_dark_complaint("The cat is a black silhouette; there is a second cat."))
check("a plain wrong-subject complaint is untouched",
      not E.is_dark_complaint("Asked for a cat, got a dog."))
check("empty is not a complaint", not E.is_dark_complaint(""))

# The outer judge: a darkness-only refine on a lit picture becomes an accept.
import image_generate as G
import image as I


class _Ctx:
    model_name = "test"; no_think = False; reasoning_effort = "high"
    def is_cancelled(self): return False
    def set_stage(self, *_): pass


_orig = I.analyze_image_with_llm
_calls = []


def _fake(ctx, image_path=None, user_text="", system_prompt="", **kw):
    _calls.append(user_text)
    return ('{"verdict": "refine", "score": 5, "reason": "The cat is extremely dark, '
            'a black silhouette.", "prompt_patch": "well lit cat", '
            '"negative_prompt_patch": "silhouette", "steps": 0, "cfg": 0, "width": 0, "height": 0}')


I.analyze_image_with_llm = _fake
try:
    r = G.evaluate_image(_Ctx(), bright, "a ginger cat", "a ginger cat", "", 8, 1.0, 64, 64)
    check("darkness-only refine on a lit picture is overruled", r.get("verdict") == "success", r)
    check("and scores at the accept floor", int(r.get("score", 0)) >= 8, r)
    check("the patches are dropped with it", not r.get("prompt_patch") and not r.get("negative_prompt_patch"))
    check("the judge was told the measured exposure", any("Measured exposure" in c for c in _calls))
    _calls.clear()
    r2 = G.evaluate_image(_Ctx(), dark, "a ginger cat", "a ginger cat", "", 8, 1.0, 64, 64)
    check("the same complaint about a genuinely dark picture stands", r2.get("verdict") == "refine", r2)
    check("and no exposure evidence was claimed for it", not any("Measured exposure" in c for c in _calls))
finally:
    I.analyze_image_with_llm = _orig

# draw_agent's critic shares the measurement
import draw_agent as DA
check("draw_agent drops the same complaints",
      DA._drop_unfounded(bright, ["the scene is extremely dark", "the cat is missing"]) == ["the cat is missing"])
check("but not on a dark picture",
      len(DA._drop_unfounded(dark, ["the scene is extremely dark"])) == 1)

# the layout critic: a blind verdict is discarded, taste is not a problem
import llm as _llm
_real_vision = _llm.analyze_image_with_llm
LAY = {"background": "a street", "elements": [
    {"desc": "a cafe", "text": "", "x": 0.1, "y": 0.2, "w": 0.8, "h": 0.7},
    {"desc": "a sign", "text": "У ОЛЬГИ", "x": 0.2, "y": 0.05, "w": 0.6, "h": 0.15}]}


def _critic_says(problems, ok=False, score=2):
    import json as _j
    _llm.analyze_image_with_llm = lambda ctx, **kw: _j.dumps(
        {"ok": ok, "score": score, "problems": problems, "ops": []})


try:
    _critic_says(["The image is almost entirely black and lacks most elements",
                  "The flower boxes are missing", "The street is missing"])
    v = DA.critique(_Ctx(), bright, LAY)
    check("a critic that calls a lit picture black overall gets no vote",
          v["ok"] and v["source"] == "blind", v)
    v = DA.critique(_Ctx(), dark, LAY)
    check("the same words about a dark picture stand", not v["ok"] and len(v["problems"]) == 3, v)
    _critic_says(["The text 'У ОЛЬГИ' is present but the background of the sign is messy.",
                  "The sign is not clean/solid as requested"])
    v = DA.critique(_Ctx(), bright, LAY)
    check("taste complaints are dropped", not v["problems"], v)
    check("and with nothing left the picture is ok", v["ok"], v)
    _critic_says(["The text on the sign is 'УОЛЬГИ' instead of 'У ОЛЬГИ'",
                  "the background of the sign is messy"])
    v = DA.critique(_Ctx(), bright, LAY)
    check("a real problem survives beside a taste one",
          not v["ok"] and len(v["problems"]) == 1 and "instead of" in v["problems"][0], v)
    _critic_says(["the cat appears twice"])
    check("a doubled subject is a problem", not DA.critique(_Ctx(), bright, LAY)["ok"])
    _critic_says(["the cafe is missing"])
    check("a missing element is a problem", not DA.critique(_Ctx(), bright, LAY)["ok"])
    _critic_says(["There is a third cat visible on the left side which was not in the layout"])
    check("an EXTRA subject is a problem, not taste", not DA.critique(_Ctx(), bright, LAY)["ok"])
    _critic_says(["An extra sign hangs above the door"])
    check("'extra' counts", not DA.critique(_Ctx(), bright, LAY)["ok"])
    _critic_says(["The windowsill is too large and covers more than the bottom 25% of the frame"])
    check("size taste is still taste", DA.critique(_Ctx(), bright, LAY)["ok"])
finally:
    _llm.analyze_image_with_llm = _real_vision

# ── 2. song intent ───────────────────────────────────────────────────────────
import tg_songs as S

# What is a song and how long is the model's read (intent.song /
# song_seconds); the phrases run live in bench/intent_song_weather_live.py.
import intent
READS = {"сочини песню про кота, секунд 30": {"song": "make", "song_seconds": 30},
         "сделай трек на 2 минуты": {"song": "make", "song_seconds": 120},
         "напиши текст песни про кота": {"song": "lyrics"},
         "нарисуй кота": {"wants": ["generate_image"]}}
intent.STUB = READS.get
check("song: a read 'make' is a song", S.song_request("сочини песню про кота, секунд 30")[0])
for t in ("напиши текст песни про кота", "нарисуй кота"):
    check("not a song: " + t, not S.song_request(t)[0], S.song_request(t))
check("seconds are read", S.song_request("сочини песню про кота, секунд 30")[1] == 30)
check("minutes are read", S.song_request("сделай трек на 2 минуты")[1] == 120)
# The engine takes any length inside DURATION_RANGE now (music3-length-is-lyric-
# volume): a request is clamped to the range, not snapped to a menu step.
_lo, _hi = S._music_mod.DURATION_RANGE
check("a length is clamped to what the engine offers",
      S.snap_duration(45) == 45 and S.snap_duration(100) == 100
      and S.snap_duration(_hi + 500) == _hi and S.snap_duration(1) == _lo)
check("no length asked -> 0 (the setting decides)", S.snap_duration(0) == 0)

# wired: the text branch of the resolver hands a typed song request to the flow
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_song_intent_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
started = []
bot._start_song_generation = lambda cid, topic, lang, duration=0: started.append((cid, topic, duration))
bot._send_text = lambda *a, **k: None
u = T._User(chat_id=777001, name="t", status="approved"); bot._user_store.put(u)
sess = bot._get_session(777001); sess.lang = "ru"; bot._store.put(sess)
enq = []
bot._enqueue_task = lambda *a, **k: enq.append(a)
bot._resolve_and_enqueue(777001, [{"type": "text", "text": "сочини песню про кота, секунд 30"}]) \
    if hasattr(bot, "_resolve_and_enqueue") else None
import inspect
src = inspect.getsource(sys.modules["tg_resolve"])
check("the resolver consults song_request before the agent", "song_request(raw)" in src)
check("and starts the flow with the snapped length", "snap_duration(_secs)" in src)

# ── 3. weather intent ────────────────────────────────────────────────────────
import datetime as _dt
import tg_weather as W
_today = _dt.date(2026, 9, 12)
# The place and day are the model's read (intent.weather); the phrases run
# live in bench/intent_song_weather_live.py. Here: what the read turns into.
_W = {"какая погода в Казани завтра?": {"city": "Казани", "when": "tomorrow"},
      "какая сейчас погода?": {"city": "", "when": "now"},
      "погода в субботу": {"city": "", "when": "saturday"},
      "погода на выходных": {"city": "", "when": "weekend"},
      "а послезавтра?": {"city": "", "when": "day_after_tomorrow"}}
intent.STUB = lambda t: {"weather": _W[t]} if t in _W else {"wants": []}
w = W.weather_request("какая погода в Казани завтра?", _today)
check("a typed forecast question is recognised", bool(w), w)
check("the city is lifted from the sentence", w.get("city") == "Казани", w)
check("'завтра' is tomorrow", w.get("date") == _dt.date(2026, 9, 13), w)
check("'сейчас' means current conditions", W.weather_request("какая сейчас погода?", _today).get("hours") == 0)
check("no city -> the remembered one decides", W.weather_request("какая сейчас погода?", _today).get("city") == "")
check("a weekday is the next one", W.weather_request("погода в субботу", _today).get("date") == _dt.date(2026, 9, 12))
check("the weekend is 48 hours", W.weather_request("погода на выходных", _today).get("hours") == 48)
check("'а послезавтра?' moves the date",
      W.weather_followup("а послезавтра?", _today) == {"city": "", "date": _today + _dt.timedelta(days=2), "hours": None})
check("no weather read -> not weather", W.weather_request("привет", _today) == {})
intent.STUB = None
check("model down -> not weather (the full loop answers)", W.weather_request("какая погода в Казани завтра?", _today) == {})
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(W.__file__))), "bot/tg_resolve.py"), encoding="utf-8").read()
check("the router tries the follow-up only inside the weather menu",
      'sess.menu == "weather"' in _src and "_tg_weather.weather_followup(raw)" in _src)
src = inspect.getsource(sys.modules["tg_resolve"])
check("the resolver consults weather_request before the agent", "weather_request(raw)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
