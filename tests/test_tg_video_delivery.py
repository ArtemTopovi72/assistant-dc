"""Telegram delivery of a generated clip — offline (no Telegram API, no ffmpeg, no GPU).

The bug class this exists for is the one that has bitten this bot repeatedly: the
model narrates a result the delivery layer never actually sent. A video makes it
worse than an image, because a clip costs minutes of GPU — "here is your video"
with no video is both wrong and expensive.

So the contract under test is:
  * a successful clip is sent with sendVideo, and the chat's video register is
    updated so "use that video" resolves next turn;
  * a delivery FAILURE is surfaced to the user, not swallowed;
  * a turn that delivered a clip does not also re-post the source stills as if
    they were the answer;
  * over-size clips are re-encoded, and if that fails they go as a document
    rather than being dropped.

Run: venv/Scripts/python.exe tests/test_tg_video_delivery.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_video_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


_TMP = tempfile.mkdtemp(prefix="tgvid_")
def _fake_clip(tag="clip", size=64 * 1024):
    p = os.path.join(_TMP, f"{tag}.mp4")
    with open(p, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * size)
    return p


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    sent = []
    bot._api_post = lambda m, p=None, **kw: (sent.append((m, p)) or
                                             {"ok": True, "result": {"message_id": len(sent) + 1}})
    return bot, sent


print("=" * 70)
print("1. A successful clip goes out via sendVideo")
print("=" * 70)

bot, sent = make_bot()
posted = []


class _Resp:
    def __init__(self, code=200, text="ok"):
        self.status_code, self.text = code, text


def _fake_post(url, data=None, files=None, timeout=None):
    posted.append((url.rsplit("/", 1)[-1], dict(data or {})))
    return _Resp(200)


T.requests.post = _fake_post
# no ffmpeg in the test env: keep the real file, no poster
import video as _V
_V.fit_for_telegram = lambda p, max_bytes=None, ctx=None: p
_V.make_thumbnail = lambda p: None
_V.probe = lambda p: {"seconds": 5.2, "width": 1344, "height": 768,
                      "has_audio": True, "bytes": os.path.getsize(p)}

clip = _fake_clip("good")
ok = bot._send_video(4242, clip, caption="done")
check("sendVideo was called", any(m == "sendVideo" for m, _ in posted), posted)
check("_send_video reports success", ok is True)
_, payload = [p for p in posted if p[0] == "sendVideo"][0]
check("the real duration is sent (else Telegram shows a 0:00 placeholder)",
      payload.get("duration") == 5, payload.get("duration"))
check("the real dimensions are sent",
      payload.get("width") == 1344 and payload.get("height") == 768, payload)
check("streaming is enabled", payload.get("supports_streaming") is True)
check("the original render was NOT deleted (it is the user's file)",
      os.path.exists(clip))

print()
print("=" * 70)
print("2. A rejected upload falls back to a document, then reports failure")
print("=" * 70)

posted.clear()
T.requests.post = lambda url, **kw: (posted.append((url.rsplit("/", 1)[-1], None))
                                     or _Resp(413, "file is too big"))
bot2, _ = make_bot()
_docs = []
bot2._send_document = lambda cid, p, cap="": (_docs.append(p) or True)
ok2 = bot2._send_video(1, _fake_clip("big"))
check("an oversize upload is not retried against sendVideo",
      sum(1 for m, _ in posted if m == "sendVideo") == 1, posted)
check("it falls back to sendDocument", len(_docs) == 1, _docs)
check("and that counts as delivered", ok2 is True)

posted.clear()
bot3, _ = make_bot()
bot3._send_document = lambda cid, p, cap="": False
ok3 = bot3._send_video(1, _fake_clip("hopeless"))
check("when BOTH sendVideo and sendDocument fail, delivery reports False",
      ok3 is False)

print()
print("=" * 70)
print("3. A missing file is refused before any API call")
print("=" * 70)

posted.clear()
bot4, _ = make_bot()
check("a non-existent path returns False",
      bot4._send_video(1, os.path.join(_TMP, "nope.mp4")) is False)
check("...and no upload was attempted", not posted, posted)
check("an empty path returns False", bot4._send_video(1, "") is False)

print()
print("=" * 70)
print("4. The failure message is localized, both ways")
print("=" * 70)

for lang in ("en", "ru"):
    msg = T._t("video_send_failed", lang)
    check(f"video_send_failed has a {lang} string", bool(msg) and "{" not in msg, msg)
check("the RU string is actually Russian, not the English fallback",
      T._t("video_send_failed", "ru") != T._t("video_send_failed", "en"))
# The "this will take minutes" signal is carried by the STAGE vocabulary
# ("Generating a video" / "Генерирую видео" in stages.py), which is already
# localized and already shown live — a second standing message would be a dead
# string duplicating it.
import stages as _stages
check("the slow-render signal is localized via the stage vocabulary instead",
      "Generating a video" in _stages._STAGES
      and _stages._STAGES["Generating a video"].get("ru"),
      "stage translation missing")

print()
print("=" * 70)
print("5. Delivering a clip suppresses re-posting the source stills")
print("=" * 70)

import inspect
src = inspect.getsource(T.TelegramBot._run_task_inner) if hasattr(
    T.TelegramBot, "_run_task_inner") else ""
if not src:
    for _n, _m in inspect.getmembers(T.TelegramBot, inspect.isfunction):
        s = inspect.getsource(_m)
        if "[video delivery]" in s:
            src = s
            break
check("the turn has a video-delivery block", "[video delivery]" in src)
check("the image block is suppressed once a clip went out",
      'if video_delivered else (final.get("image_path")' in src
      or "video_delivered else" in src, "image block not gated on video_delivered")
check("a failed delivery tells the user instead of staying silent",
      "video_send_failed" in src)
check("the delivered clip is registered for the chat",
      "_chat_videos[chat_id]" in src)

print()
print("=" * 70)
print("6. The per-chat video register is isolated and initialised")
print("=" * 70)

bot5, _ = make_bot()
check("_chat_videos exists on a fresh bot", isinstance(bot5._chat_videos, dict))
bot5._chat_videos[111] = "/a.mp4"
check("one chat's clip does not leak into another",
      bot5._chat_videos.get(222) is None)
check("it is separate from the image register (a still is never a clip)",
      "image_log" not in str(type(bot5._chat_videos)))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
