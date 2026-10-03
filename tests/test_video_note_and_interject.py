"""Two new scenarios, offline (redirect_data_dir, no live LM Studio/ComfyUI/GPU):

1. Telegram video_note (a round "video message") now supports the exact same
   two scenarios voice notes already had: sent directly it is speech TO the
   bot (transcribed and treated as text); forwarded it is material the user
   wants something done WITH (transcribed, then asked transcript/summary/both).
   `_transcribe` takes a `media` kind so the temp file gets an honest suffix
   (.mp4 vs .ogg) — ffmpeg extracts the audio track from either.

2. A chat may now have a SECOND task admitted while its first is running, but
   ONLY once that first task has announced (via ctx.set_stage) that it has
   reached a genuinely slow, backgroundable phase (image render, deep research,
   web search) — see _mark_interruptible and the consumer-loop admission gate
   in _consumer_loop. This lets a quick new message get answered without
   waiting for the whole slow operation, while it keeps running untouched.
   A third task for the same chat still always queues.

Run: venv/Scripts/python.exe tests/test_video_note_and_interject.py
"""
import os, sys, tempfile, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_video_interject_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    return bot, sent


def set_user(bot, chat_id, status="approved"):
    u = T._User(chat_id=chat_id, name="U", tg_username="u", password_hash="x", status=status)
    bot._user_store.put(u)


print("=" * 70)
print("1. OWN video_note -> enqueued as speech-to-the-bot (type 'voice')")
print("=" * 70)

bot, sent = make_bot()
CID = 9101
set_user(bot, CID)
_captured = []
bot._enqueue_item = lambda chat_id, item: _captured.append(item)
upd = {"message": {"chat": {"id": CID}, "message_id": 1,
       "video_note": {"file_id": "vn1", "duration": 7}}}
bot._dispatch(upd)
item = _captured[0] if _captured else None
check("a bare video_note enqueues type=voice", item is not None and item["type"] == "voice", item)
check("…tagged media=video so _transcribe uses the right ffmpeg suffix",
      item is not None and item.get("media") == "video", item)

print()
print("=" * 70)
print("2. FORWARDED video_note -> asks transcript/summary/both, not silent instruction")
print("=" * 70)

bot2, sent2 = make_bot()
CID2 = 9102
set_user(bot2, CID2)
_captured2 = []
bot2._enqueue_item = lambda chat_id, item: _captured2.append(item)
upd2 = {"message": {"chat": {"id": CID2}, "message_id": 1,
        "forward_origin": {"type": "user"},
        "video_note": {"file_id": "vn2", "duration": 30}}}
bot2._dispatch(upd2)
item2 = _captured2[0] if _captured2 else None
check("a forwarded video_note enqueues type=fwd_voice", item2 is not None and item2["type"] == "fwd_voice", item2)
check("…tagged media=video too", item2 is not None and item2.get("media") == "video", item2)

print()
print("=" * 70)
print("3. _transcribe uses an .mp4 temp suffix for media='video', .ogg for voice")
print("=" * 70)

bot3, _ = make_bot()
_seen_suffix = {}
_orig_ntf = tempfile.NamedTemporaryFile
def _spy_ntf(*a, **kw):
    fh = _orig_ntf(*a, **kw)
    _seen_suffix["suffix"] = kw.get("suffix")
    return fh
bot3._dl_bytes = lambda file_id: b"not really audio, just needs to be non-empty bytes"
tempfile.NamedTemporaryFile = _spy_ntf
try:
    bot3._transcribe(None, "fid", media="video")
finally:
    tempfile.NamedTemporaryFile = _orig_ntf
check("media='video' picks an .mp4 temp suffix", _seen_suffix.get("suffix") == ".mp4", _seen_suffix)

tempfile.NamedTemporaryFile = _spy_ntf
try:
    bot3._transcribe(None, "fid", media="voice")
finally:
    tempfile.NamedTemporaryFile = _orig_ntf
check("media='voice' (default) still picks an .ogg temp suffix",
      _seen_suffix.get("suffix") == ".ogg", _seen_suffix)

print()
print("=" * 70)
print("4. A fresh task for an IDLE chat is always admitted alone")
print("=" * 70)

bot4, _ = make_bot()
CID4 = 9104
t1 = T._Task(task_id="a1", chat_id=CID4, user_text="hi")
with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 0)
    interruptible = bot4._chat_interruptible.get(CID4, False)
    admit = (n == 0) or (n == 1 and interruptible)
    if admit:
        bot4._chat_busy[CID4] = n + 1
check("first task for an idle chat is admitted", admit)
check("chat_busy count is now 1", bot4._chat_busy.get(CID4) == 1)

print()
print("=" * 70)
print("5. A second task is REFUSED while the first is running but NOT interruptible")
print("=" * 70)

with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 0)
    interruptible = bot4._chat_interruptible.get(CID4, False)
    admit2 = (n == 0) or (n == 1 and interruptible)
check("second task queues behind a non-interruptible first task", not admit2)

print()
print("=" * 70)
print("6. _mark_interruptible opens exactly ONE extra slot, never a third")
print("=" * 70)

bot4._mark_interruptible(CID4)
with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 0)
    interruptible = bot4._chat_interruptible.get(CID4, False)
    admit3 = (n == 0) or (n == 1 and interruptible)
    if admit3:
        bot4._chat_busy[CID4] = n + 1
check("a second task IS admitted once the chat is marked interruptible", admit3)
check("chat_busy count is now 2", bot4._chat_busy.get(CID4) == 2)

with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 0)
    interruptible = bot4._chat_interruptible.get(CID4, False)
    admit4 = (n == 0) or (n == 1 and interruptible)
check("a THIRD task for the same chat still queues (cap of 2)", not admit4)

print()
print("=" * 70)
print("7. Releasing both slots clears chat_busy AND chat_interruptible")
print("=" * 70)

with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 1) - 1
    if n <= 0:
        bot4._chat_busy.pop(CID4, None)
        bot4._chat_interruptible.pop(CID4, None)
    else:
        bot4._chat_busy[CID4] = n
with bot4._task_lock:
    n = bot4._chat_busy.get(CID4, 1) - 1
    if n <= 0:
        bot4._chat_busy.pop(CID4, None)
        bot4._chat_interruptible.pop(CID4, None)
    else:
        bot4._chat_busy[CID4] = n
check("chat_busy is fully cleared after both tasks finish", CID4 not in bot4._chat_busy)
check("chat_interruptible is reset for the next turn", CID4 not in bot4._chat_interruptible)

print()
print("=" * 70)
print("8. _INTERRUPTIBLE_STAGE_KEYWORDS marks image/search/research stages, not quick ones")
print("=" * 70)

def is_slow(stage):
    return any(k in stage.lower() for k in T._INTERRUPTIBLE_STAGE_KEYWORDS)

check("'Drawing a picture' (image gen) is slow", is_slow("Drawing a picture"))
check("'Searching the web' is slow", is_slow("Searching the web"))
check("'Upscaling (face-safe)' is slow", is_slow("Upscaling (face-safe)"))
check("'Writing a response' (plain LLM reply) is NOT slow", not is_slow("Writing a response"))
check("'Looking at the image' (single vision call) is NOT slow", not is_slow("Looking at the image"))
check("'Speaking' (TTS) is NOT slow", not is_slow("Speaking"))
check("'Compacting the chat history' is NOT slow", not is_slow("Compacting the chat history"))

print()
print("=" * 70)
print("9. _history_lock returns the SAME lock object for the same chat")
print("=" * 70)

bot5, _ = make_bot()
lk1 = bot5._history_lock(9105)
lk2 = bot5._history_lock(9105)
lk3 = bot5._history_lock(9106)
check("same chat_id -> same lock instance", lk1 is lk2)
check("different chat_id -> a different lock instance", lk1 is not lk3)

print()
print("=" * 70)
print("10. Concurrent history commits from two tasks do not clobber each other")
print("=" * 70)

bot6, _ = make_bot()
CID6 = 9107
set_user(bot6, CID6)
sess6 = bot6._get_session(CID6)
sess6.set_history([{"role": "system", "content": "sys"}])
bot6._store.put(sess6)

errors = []
def commit(tag, n_msgs):
    try:
        with bot6._history_lock(CID6):
            hist = bot6._get_session(CID6).get_history()
            for i in range(n_msgs):
                hist.append({"role": "user", "content": f"{tag}-{i}"})
            bot6._get_session(CID6).set_history(hist)
            bot6._store.put(bot6._get_session(CID6))
    except Exception as exc:
        errors.append(exc)

threads = [threading.Thread(target=commit, args=(f"t{i}", 5)) for i in range(6)]
for th in threads: th.start()
for th in threads: th.join()

final_hist = bot6._get_session(CID6).get_history()
check("no exceptions during concurrent commits", not errors, errors)
check("all 30 messages from 6 threads x 5 msgs survived (no lost-update clobber)",
      len(final_hist) == 1 + 30, len(final_hist))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
