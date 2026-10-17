"""Long videos, portion by portion (tg_video).

Creativity ▸ 🎬 Video settings state their value and are redrawn in place;
a long video is OFFERED (not started) with ▶️ / ⚙️ / ✖; ▶️ runs the job in
portions with absolute timecodes, delivering each portion as it is ready,
and ⛔ Stop ends it between portions. The heavy parts (download, ASR,
vision, retelling) are stubbed; the cutting and the sheet are real ffmpeg.
"""
import os, sys, tempfile, threading, subprocess
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="tg_long_video_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
import types as _types
_T = [None]   # the text model the context hands out
TXT = _types.SimpleNamespace(text_model=lambda *a, **k: _T[0](*a, **k))

T.redirect_data_dir(_DATA)
import tg_video as V
import video_look as VL

CID = 424242
PASSED = FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond: PASSED += 1; print("PASS ", label)
    else:    FAILED += 1; print("FAIL ", label, " ", str(detail)[:200])


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: bot.sent.append((t, kw.get("keyboard")))
    bot._edit_text = lambda cid, mid, t, **kw: bot.sent.append((t, kw.get("keyboard")))
    bot._send_photo = lambda cid, path, **kw: bot.sent.append(("<photo>", None))
    bot._send_get_id = lambda cid, t, **kw: bot.sent.append((t, kw.get("keyboard"))) or 77
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


def texts(bot): return "\n".join(t for t, _ in bot.sent)


# ── settings ─────────────────────────────────────────────────────────────────
bot = make_bot()
sess = bot._get_session(CID)
check("defaults: 5-minute portions, 40 frames, brief (one retelling + questions), long from 2 min",
      (V.chunk_seconds(sess), V.frames_per_chunk(sess), V.output_mode(sess), V.long_threshold(sess))
      == (300, 40, "brief", 120))
kb = V._video_menu_kb(sess, "ru")
check("the menu states every value", all(":" in r[0]["text"] for r in kb["inline_keyboard"][:4]), kb)
bot._cb_video_settings(CID, {"message_id": 7}, "video:set:chunk:10")
check("a tap changes the setting", V.chunk_seconds(bot._get_session(CID)) == 600)
bot._cb_video_settings(CID, {"message_id": 7}, "video:set:chunk:99")
check("an unknown value is ignored", V.chunk_seconds(bot._get_session(CID)) == 600)
bot._cb_video_settings(CID, {"message_id": 7}, "video:reset")
check("reset returns to defaults", V.chunk_seconds(bot._get_session(CID)) == 300)
check("the creativity keyboard offers 🎬 Video",
      any("🎬" in b for row in T._creativity_kb("ru")["keyboard"] for b in row))

# ── the offer ────────────────────────────────────────────────────────────────
bot = make_bot(); sess = bot._get_session(CID)
check("a 90 s note is short, a 3-minute video is long", not V.is_long(sess, 90) and V.is_long(sess, 180))
bot._long_video_offer(CID, sess, "ru", {"file_id": "F1", "seconds": 1800, "caption": ""})
t, kb = bot.sent[-1]
check("the offer names the length and the number of steps", "30 мин" in t and "6 шагов" in t, t)
data = [b["callback_data"] for r in kb["inline_keyboard"] for b in r]
check("the offer has whole / by parts / ask / ⚙️ / ✖", any(d.startswith("lv:go:") for d in data)
      and any(d.startswith("lv:each:") for d in data) and any(d.startswith("lv:ask:") for d in data)
      and "video:menu" in data and any(d.startswith("lv:skip:") for d in data), data)
job_id = bot._get_session(CID).long_video["id"]
bot._cb_long_video(CID, "lv:go:stale")
check("a stale ▶️ is refused", "не в работе" in bot.sent[-1][0], bot.sent[-1][0])
bot._cb_long_video(CID, f"lv:skip:{job_id}")
check("✖ clears the pending video", bot._get_session(CID).long_video is None)

# ── the words already say what is wanted: no menu ───────────────────────────
# «перескажи кратко это интервью, кто что говорит: <ссылка>» got «Что нужно?» (2026-10-17)
import types as _types
for words, want in (("перескажи кратко это интервью, кто что говорит: https://youtu.be/x", "retell"),
                    ("о чём он говорит на 10-й минуте? https://youtu.be/x", "question")):
    bot = make_bot(); sess = bot._get_session(CID)
    picked = []
    bot.reads = _types.SimpleNamespace(choice=lambda q, t, opts, default="", _w=want: (
        picked.append((q, t)), _w)[1] if "{text}" in q else default)
    ran = []
    bot._long_video_start = lambda cid, s, l, pending, _r=ran: _r.append(pending)
    bot.sent.clear()
    bot._long_video_offer(CID, sess, "ru", {"url": "https://youtu.be/x", "seconds": 1560, "caption": words})
    check(f"«{want}» in the words starts the job without the menu",
          ran and not bot.sent and "youtu" not in picked[0][1]
          and (ran[0].get("out") == "brief" if want == "retell" else ran[0].get("question") == picked[0][1]),
          (ran, bot.sent, picked))
bot = make_bot(); sess = bot._get_session(CID)
bot.reads = _types.SimpleNamespace(choice=lambda q, t, opts, default="": "none")
bot._long_video_offer(CID, sess, "ru", {"url": "https://youtu.be/x", "seconds": 1560, "caption": "https://youtu.be/x"})
check("a bare link still gets the menu", bot.sent and "lv:go:" in str(bot.sent[-1][1]), bot.sent[-1:])

# ── the job, with stubs around the heavy parts ───────────────────────────────
tmp = Path(tempfile.mkdtemp())
clip = tmp / "long.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=s=320x240:d=150:r=10",
                "-f", "lavfi", "-i", "sine=f=440:d=150",
                "-shortest", "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=120)
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; sess.video_out = "both"; bot._store.put(sess)  # 150 s -> 2 portions
bot._dl_bytes = lambda fid: clip.read_bytes()
bot._get_ctx = lambda: TXT
asr_calls, look_calls, retell_calls = [], [], []
import audio as _audio, llm as _llm
from tests.fakes import Over
ASR = [lambda ctx, p, lang_hint="ru": asr_calls.append(p) or "речь части"]
AU = Over(_audio, transcribe_audio_file=lambda ctx, p, lang_hint="ru": ASR[0](ctx, p, lang_hint))
_orig_look = VL.look_at_video
def _fake_look(ctx, path, **kw):
    look_calls.append(kw.get("offset"))
    r = _orig_look(ctx, path, vision=lambda *a: "x — кадр", **kw)
    r["description"] = "\n".join(f"{VL._stamp(t)} — кадр" for t in r["times"])
    return r
VLO = Over(VL, look_at_video=_fake_look)
bot.modules = {"video_look": VLO, "audio": AU}
_T[0] = lambda ctx, sysm, user, **kw: retell_calls.append(user) or "пересказ части"
stop = threading.Event()
bot._lv_jobs()[CID] = stop
bot._long_video_job(CID, {"id": "j1", "file_id": "F1", "seconds": 150}, stop)
out = texts(bot)
check("two portions were cut and transcribed", len(asr_calls) == 2, asr_calls)
check("the second portion is looked at with an absolute offset", look_calls == [0, 120], look_calls)
check("part headers carry absolute ranges", "Часть 1/2" in out and "2:00–2:30" in out, out)
check("the storyboard of part 2 has timecodes past 2:00",
      any(l.startswith("2:") for l in out.splitlines() if "— кадр" in l), out)
check("the retelling gets transcript + storyboard", retell_calls and "TRANSCRIPT" in retell_calls[0]
      and "STORYBOARD" in retell_calls[0], retell_calls[:1])
check("the job reports done", "Готово: 2" in out, out)
check("the job slot is released", CID not in bot._lv_jobs())
check("a long-video portion: Основная тема + timecoded Ключевые моменты, NO «что требуется» (a tutorial asks nothing)",
      all(k in V._structure_rules("ru", True, ask=False) for k in ("Основная тема:", "Ключевые моменты:", "m:ss"))
      and "Что требуется" not in V._structure_rules("ru", True, ask=False))
check("a voice note / round video: the same shape plus «Что требуется»",
      "Что требуется:" in V._structure_rules("ru", False) and "m:ss" not in V._structure_rules("ru", False))
html = V.render_retelling("**Основная тема:** лампа.\nКлючевые моменты:\n- Свет: слабый\nЧто требуется:\n• ничего — вопросов к слушателю нет", "Пересказ")
check("render_retelling: bold title + bold labels + bullets, markdown stars gone",
      html.startswith("📋 <b>Пересказ</b>") and "<b>Основная тема:</b> лампа." in html and "\n• Свет: слабый" in html
      and "<b>Что требуется:</b>" in html and "**" not in html, html)

check("a detailed run still saves its notes, so ❓ works after it too",
      (Path(T._IMAGE_DIR) / "longvideo_j1" / "material.json").exists())

# ── brief (the default): nothing per portion, one retelling, then questions ──
# 10-09: 11 portions x storyboard + retelling + sheet buried the chat
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; sess.video_out = ""; bot._store.put(sess)
bot._dl_bytes = lambda fid: clip.read_bytes(); bot._get_ctx = lambda: TXT
bot.modules = {"video_look": VLO, "audio": AU}; ASR[0] = lambda ctx, p, lang_hint="ru": "речь части"
calls = []
_T[0] = lambda ctx, sysm, user, **kw: calls.append((sysm, user)) or (
    "Основная тема: ремонт.\nКлючевые моменты:\n• 0:10 — плата" if "WHOLE" in sysm else
    "0:10 — ответ по видео" if "QUESTION" in user else "пересказ части")
stop = threading.Event(); bot._lv_jobs()[CID] = stop
bot._long_video_job(CID, {"id": "b1", "file_id": "F1", "seconds": 150}, stop)
out = texts(bot)
check("brief: no per-part storyboards, retellings or sheets in the chat",
      "Часть 1/2" not in out and "<photo>" not in out and "пересказ части" not in out, out)
check("brief: one progress line, edited in place", out.count("Смотрю видео") == 2 and "Посмотрел: 2" in out, out)
check("brief: one retelling of the whole video, built from the parts' retellings",
      "Пересказ всего видео" in out and any("WHOLE" in s and "пересказ части" in u for s, u in calls), out)
kb = bot.sent[-1][1] or {}
data = [b["callback_data"] for r in kb.get("inline_keyboard", []) for b in r]
check("under it: ❓ ask and 📜 every part", "lv:more:b1" in data and "lv:parts:b1" in data, data)

bot.sent.clear()
bot._cb_long_video(CID, "lv:parts:b1")
check("📜 opens every part on demand", "Часть 1/2" in texts(bot) and "Часть 2/2" in texts(bot), texts(bot))
bot.sent.clear()
bot._cb_long_video(CID, "lv:more:b1")
check("❓ arms the next message as the question", bot._get_session(CID).reg_state == "lv_more:b1"
      and "Что тебя интересует" in texts(bot), texts(bot))
bot.spawn = lambda fn, *a, **k: fn(*a)
bot.sent.clear(); calls.clear()
bot._user_gate(CID, {"text": "какая частота на осциллографе?", "from": {}})
check("the question is answered from the saved notes, no re-watch",
      "ответ по видео" in texts(bot) and calls and "STORYBOARD" in calls[-1][1]
      and "какая частота" in calls[-1][1] and bot._get_session(CID).reg_state == "", texts(bot))

# ── ❓ before the job: the job answers the question instead of retelling ─────
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; bot._store.put(sess)
bot._dl_bytes = lambda fid: clip.read_bytes(); bot._get_ctx = lambda: TXT
bot._long_video_offer(CID, sess, "ru", {"file_id": "F1", "seconds": 150})
job = bot._get_session(CID).long_video["id"]
bot._cb_long_video(CID, f"lv:ask:{job}")
check("❓ on the offer asks for the question and keeps the video pending",
      bot._get_session(CID).reg_state == f"lv_question:{job}" and bot._get_session(CID).long_video)
ran = []
bot._long_video_start = lambda cid, s, l, pending: ran.append(pending)
bot._user_gate(CID, {"text": "где он паяет транзистор?", "from": {}})
check("the question starts the job with it", ran and ran[0].get("question") == "где он паяет транзистор?", ran)
check("material_text keeps every storyboard and trims transcripts to the budget",
      len(V.material_text([{"a": "0:00", "b": "5:00", "board": "0:10 — плата", "said": "слово " * 9000}], 3000)) <= 3000
      and "0:10 — плата" in V.material_text([{"a": "0:00", "b": "5:00", "board": "0:10 — плата", "said": "x" * 9000}], 3000))

# ── Stop between portions ────────────────────────────────────────────────────
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; sess.video_out = "both"; bot._store.put(sess)
bot._dl_bytes = lambda fid: clip.read_bytes(); bot._get_ctx = lambda: TXT
stop = threading.Event(); bot._lv_jobs()[CID] = stop
bot.modules = {"video_look": VLO, "audio": AU}; ASR[0] = lambda ctx, p, lang_hint="ru": (stop.set(), "речь")[1]   # Stop during part 1
bot._long_video_job(CID, {"id": "j2", "file_id": "F1", "seconds": 150}, stop)
out = texts(bot)
check("Stop ends the job between portions", "Остановлено после" in out and "Часть 2/2" not in out, out)
check("_stop_and_report reaches a running job",
      (bot._lv_jobs().__setitem__(CID, threading.Event()) or bot._long_video_stop(CID)) is True)

# ── ⛔ Стоп right under the progress message ──────────────────────────────────
bot.sent.clear()
stop = threading.Event(); bot._lv_jobs()[CID] = stop
bot.modules = {"video_look": VLO, "audio": AU}; ASR[0] = lambda ctx, p, lang_hint="ru": "речь"
bot._long_video_job(CID, {"id": "j3", "file_id": "F1", "seconds": 150}, stop)
btns = [b["callback_data"] for t, kb in bot.sent if kb for row in kb.get("inline_keyboard", []) for b in row]
check("the running job shows a ⛔ button", "lv:stop:j3" in btns, btns)
bot.sent.clear()
ev = threading.Event(); bot._lv_jobs()[CID] = ev
bot._cb_long_video(CID, "lv:stop:j3")
check("pressing ⛔ stops the running job and says so", ev.is_set() and "Останавливаю" in texts(bot), texts(bot))
bot._lv_jobs().pop(CID, None); bot.sent.clear()
bot._cb_long_video(CID, "lv:stop:j3")
check("⛔ with nothing running says so", "ничего не выполняется" in texts(bot), texts(bot))

# ── the retelling check: a figure the portion does not support is dropped ──────
import tg_video as V2
_seen = {}
class _Slides:
    @staticmethod
    def claim_support(ctx, claims, facts):
        _seen["claims"], _seen["facts"] = claims, facts
        return [0.02 if "999" in c else 0.9 for c in claims]
_ctx = _types.SimpleNamespace(modules={"slides": _Slides})
_tale = "Суть: обзор железа\n• 0:17 — Блок питания: мощность 1000 Вт\n• 1:02 — Видеокарта: 999 ГБ памяти\n• 2:00 — Сборка корпуса без цифр"
_out = V2.drop_unsupported_lines(_ctx, _tale, "Блок питания на 1000 Вт. Дальше видеокарта.", "0:17 блок питания CORSAIR")
check("a line with a figure the portion does not support is dropped", "999" not in _out and "1000" in _out, _out)
check("only lines with a figure or a quote are sent to the checker", len(_seen["claims"]) == 2, _seen)
check("the transcript is split into sentences, the storyboard lines follow",
      _seen["facts"].splitlines()[:2] == ["Блок питания на 1000 Вт.", "Дальше видеокарта."], _seen["facts"])
check("a timecode alone is not a figure", V2.drop_unsupported_lines(_ctx, "• 0:17 — Заставка", "x", "y") == "• 0:17 — Заставка")
_Slides.claim_support = staticmethod(lambda ctx, c, f: None)
check("no checker, no change", V2.drop_unsupported_lines(_ctx, _tale, "a", "b") == _tale)
check("the carry is the last lines of the previous retelling", V2._carry("a\n\nb\nc\nd\ne") == "c\nd\ne" and V2._carry("") == "")

print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
sys.exit(1 if FAILED else 0)
