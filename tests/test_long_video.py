"""Long videos, portion by portion (tg_video).

Creativity ▸ 🎬 Video settings state their value and are redrawn in place;
a long video is OFFERED (not started) with ▶️ / ⚙️ / ✖; ▶️ runs the job in
portions with absolute timecodes, delivering each portion as it is ready,
and ⛔ Stop ends it between portions. The heavy parts (download, ASR,
vision, retelling) are stubbed; the cutting and the sheet are real ffmpeg.
"""
import os, sys, tempfile, threading, time, subprocess
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="tg_long_video_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
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
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


def texts(bot): return "\n".join(t for t, _ in bot.sent)


# ── settings ─────────────────────────────────────────────────────────────────
bot = make_bot()
sess = bot._get_session(CID)
check("defaults: 5-minute portions, 12 frames, both, long from 2 min",
      (V.chunk_seconds(sess), V.frames_per_chunk(sess), V.output_mode(sess), V.long_threshold(sess))
      == (300, 12, "both", 120))
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
check("the offer names the length and the setup", "30 мин" in t and "Порция" in t, t)
data = [b["callback_data"] for r in kb["inline_keyboard"] for b in r]
check("the offer has ▶️ / ⚙️ / ✖", any(d.startswith("lv:go:") for d in data)
      and "video:menu" in data and any(d.startswith("lv:skip:") for d in data), data)
job_id = bot._get_session(CID).long_video["id"]
bot._cb_long_video(CID, "lv:go:stale")
check("a stale ▶️ is refused", "не в работе" in bot.sent[-1][0], bot.sent[-1][0])
bot._cb_long_video(CID, f"lv:skip:{job_id}")
check("✖ clears the pending video", bot._get_session(CID).long_video is None)

# ── the job, with stubs around the heavy parts ───────────────────────────────
tmp = Path(tempfile.mkdtemp())
clip = tmp / "long.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=s=320x240:d=150:r=10",
                "-f", "lavfi", "-i", "sine=f=440:d=150",
                "-shortest", "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=120)
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; bot._store.put(sess)     # 150 s -> 2 portions
bot._dl_bytes = lambda fid: clip.read_bytes()
bot._get_ctx = lambda: None
asr_calls, look_calls, retell_calls = [], [], []
import audio as _audio, llm as _llm
_audio.transcribe_audio_file = lambda ctx, p, lang_hint="ru": asr_calls.append(p) or "речь части"
_orig_look = VL.look_at_video
def _fake_look(ctx, path, **kw):
    look_calls.append(kw.get("offset"))
    r = _orig_look(ctx, path, vision=lambda *a: "x — кадр", **kw)
    r["description"] = "\n".join(f"{VL._stamp(t)} — кадр" for t in r["times"])
    return r
VL.look_at_video = _fake_look
_llm.call_llm_simple = lambda ctx, sysm, user, **kw: retell_calls.append(user) or "пересказ части"
# look_at_video's close-up pass is a LIVE vision call (48 requests, 2.5 min) --
# it must never reach the operator's LM Studio from a suite.
import vision_close as _vclose
_vclose.tile_notes = lambda *a, **k: ""
T._IMAGE_DIR = str(tmp)
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

# ── Stop between portions ────────────────────────────────────────────────────
bot = make_bot(); sess = bot._get_session(CID)
sess.video_chunk = "2"; sess.video_frames = "6"; bot._store.put(sess)
bot._dl_bytes = lambda fid: clip.read_bytes(); bot._get_ctx = lambda: None
stop = threading.Event(); bot._lv_jobs()[CID] = stop
_audio.transcribe_audio_file = lambda ctx, p, lang_hint="ru": (stop.set(), "речь")[1]   # Stop during part 1
bot._long_video_job(CID, {"id": "j2", "file_id": "F1", "seconds": 150}, stop)
out = texts(bot)
check("Stop ends the job between portions", "Остановлено после" in out and "Часть 2/2" not in out, out)
check("_stop_and_report reaches a running job",
      (bot._lv_jobs().__setitem__(CID, threading.Event()) or bot._long_video_stop(CID)) is True)

print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
sys.exit(1 if FAILED else 0)
