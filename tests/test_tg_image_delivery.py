"""Two bugs a user hit in the same screenshot, and they feed each other.

1. THE BOT SAID IT DID SOMETHING AND SENT NOTHING.
   The image-delivery block has five ways to send no picture — no path, a failed
   status, a path that does not exist, an intermediate artifact, sendPhoto AND
   sendDocument both failing — and every one of them only wrote a line to the
   activity log. The text reply goes out BEFORE that block, so the user had
   already been told "I expanded the borders of your image" and then received
   nothing at all, with no way to tell a silent failure from a slow one.
   Worst case: an image tool returns "[TOOL ERROR] … do NOT claim it succeeded",
   leaving image_path and image_status empty, and the model narrates success
   anyway — so the delivery block does not even consider itself involved.

2. THE REPLY WAS PRINTED TWICE.
   Users double-tap when a tap looks like it did nothing — i.e. because of (1).
   Two taps inside one debounce window arrive as two identical items, and the
   batch is joined with newlines, so the model was handed "outpaint the current
   image…" twice and answered it twice inside one message.

Run: venv/Scripts/python.exe tests/test_tg_image_delivery.py
"""
import sys, os, types, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_imgdel_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999951
OUTPAINT = T._CB_CMDS["change_clothes"]
UPSCALE = T._CB_CMDS["regenerate"]


class _Ctx:
    def __init__(self):
        import threading
        self.cancel_event = threading.Event()
        self.stage_callback = None
        self.session_memory = []
        self.pinned_facts = []
        self.memory_lock = threading.Lock()
        self.memory_text = lambda *a, **k: ""
        self.total_user_turns = 0
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.tts_disabled = True
    def set_stage(self, s):
        if self.stage_callback: self.stage_callback(s)


def run_turn(final_state, user_text=OUTPAINT, photo_ok=True, lang="ru"):
    """Run one delivery cycle with a graph that returns `final_state`."""
    ctx = _Ctx()
    seen = {"text": [], "photo": []}

    class _Graph:
        def invoke(self, state, cfg=None):
            return dict(final_state)

    bot = T.TelegramBot("1:T", lambda: ctx, lambda: _Graph(),
                        lambda: {"messages": []}, silent_mode=True)
    bot._send_text = lambda cid, text, **k: (seen["text"].append(text), 1)[1]
    bot._send_get_id = lambda cid, text, **k: 11
    bot._edit_text = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._send_photo = lambda cid, path, **k: (seen["photo"].append(path), photo_ok)[1]
    bot._send_voice_from_wav = lambda *a, **k: False
    bot._send_voice = lambda *a, **k: False

    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID); sess.lang = lang; sess.voice_on = False
    bot._store.put(sess)

    task = T._Task(task_id="t", chat_id=CID, user_text=user_text)
    bot._run_task_inner(task, bot._get_session(CID), bot._user_store.get(CID),
                        "U", lang, ctx, _Graph())
    return seen


NARRATION = "я расширил границы изображения, чтобы показать больше пространства."
WARN_RU = T._t("img_missing", "ru")
WARN_EN = T._t("img_missing", "en")


def warned(seen):
    return any(WARN_RU in t or WARN_EN in t for t in seen["text"])


print("=" * 70)
print("A CLAIM THE DELIVERY CANNOT BACK UP IS CORRECTED")
print("=" * 70)

good = os.path.join(_DATA_DIR, "out.png")
open(good, "wb").write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

# The happy path must be untouched — no scary correction after a real delivery.
seen = run_turn({"final_answer": NARRATION, "image_path": good, "image_status": "ok"})
check("a delivered image is sent", seen["photo"] == [good], str(seen["photo"]))
check("and nothing is retracted", not warned(seen), str(seen["text"]))

# The screenshot's case: the tool errored, so the pipeline reports no image at
# all, and the model claimed success anyway.
seen = run_turn({"final_answer": NARRATION})
check("a tool-error turn sends no photo", seen["photo"] == [])
check("and the false claim is corrected", warned(seen), str(seen["text"])[:120])

for label, state in {
    "a failed image_status": {"final_answer": NARRATION, "image_path": good,
                              "image_status": "fail"},
    "an empty image_status": {"final_answer": NARRATION, "image_path": good,
                              "image_status": ""},
    "a path that does not exist": {"final_answer": NARRATION,
                                   "image_path": os.path.join(_DATA_DIR, "gone.png"),
                                   "image_status": "ok"},
}.items():
    seen = run_turn(state)
    check(f"corrected: {label}", warned(seen), str(seen["text"])[:110])

# sendPhoto AND sendDocument both failing is the one case where the pipeline did
# everything right and the network did not.
seen = run_turn({"final_answer": NARRATION, "image_path": good,
                 "image_status": "ok"}, photo_ok=False)
check("corrected: upload failed", warned(seen), str(seen["text"])[:110])
check("and it did try to send", seen["photo"] == [good])

# An intermediate editing tile must not be delivered — and its absence is still
# a failure to deliver, so it must be corrected too.
tile = os.path.join(_DATA_DIR, "_INTERMEDIATE_firered_tile_1.png")
open(tile, "wb").write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
seen = run_turn({"final_answer": NARRATION, "image_path": tile, "image_status": "ok"})
check("an intermediate artifact is never delivered", seen["photo"] == [],
      str(seen["photo"]))
check("and that also counts as a failure to deliver", warned(seen),
      str(seen["text"])[:110])

print()
print("=" * 70)
print("AND A TURN THAT OWED NO PICTURE IS LEFT ALONE")
print("=" * 70)

seen = run_turn({"final_answer": "Париж — столица Франции."},
                user_text="какая столица франции")
check("an ordinary question is not accused of losing an image", not warned(seen),
      str(seen["text"])[:110])

# "describe and analyze this image" is an image INTENT but produces no new image,
# which is why the delivery check needs its own, narrower pattern.
seen = run_turn({"final_answer": "На фото кот."},
                user_text=T._DIRECT_KB["analyze"])
check("analysing a photo is not treated as owing a new one", not warned(seen),
      str(seen["text"])[:110])
check("the analyse prompt is still an image-quota intent",
      T._classify_task(T._DIRECT_KB["analyze"]) == T.KIND_IMAGE)
check("but it is not an image-PRODUCING request",
      not T._IMAGE_PRODUCING_RE.match(T._DIRECT_KB["analyze"]))
for payload in (OUTPAINT, UPSCALE,
                "generate an image of: a cat",
                "edit the image: remove the hat"):
    check(f"is image-producing: {payload[:34]!r}",
          bool(T._IMAGE_PRODUCING_RE.match(payload)))

# The correction has to be in the user's language, like everything else.
seen = run_turn({"final_answer": NARRATION}, lang="en")
check("the correction is English for an English user",
      any(WARN_EN in t for t in seen["text"]), str(seen["text"])[:110])
seen = run_turn({"final_answer": NARRATION}, lang="ru")
check("and Russian for a Russian user",
      any(WARN_RU in t for t in seen["text"]), str(seen["text"])[:110])


print()
print("=" * 70)
print("A DOUBLE TAP IS ONE INSTRUCTION, NOT TWO")
print("=" * 70)

def resolve(items, running=None):
    pushed = []
    bot = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    sent = []
    bot._send_text = lambda cid, text, **k: (sent.append(text), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(push=pushed.append, depth=lambda: 0,
                                         name=lambda: "s", chat_depth=lambda c: 0,
                                         drop_chat=lambda c: 0, close=lambda: None)
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    if running is not None:
        bot._running_task[CID] = [T._Task(task_id="r", chat_id=CID, user_text=running)]
    bot._resolve_and_push(CID, items)
    return pushed, sent

pushed, _ = resolve([{"type": "text", "text": OUTPAINT},
                     {"type": "text", "text": OUTPAINT}])
check("a double tap makes one task", len(pushed) == 1, str(len(pushed)))
check("carrying the instruction exactly once",
      pushed and pushed[0].user_text == OUTPAINT, repr(pushed[0].user_text))

pushed, _ = resolve([{"type": "text", "text": OUTPAINT}] * 5)
check("five taps still make one instruction",
      pushed and pushed[0].user_text == OUTPAINT, repr(pushed[0].user_text))

# Two DIFFERENT messages in a burst must still merge into one turn — that is the
# whole point of the debounce, and collapsing must not eat it.
pushed, _ = resolve([{"type": "text", "text": "what is"},
                     {"type": "text", "text": "the capital of France"}])
check("different messages still merge into one turn",
      pushed and pushed[0].user_text == "what is\nthe capital of France",
      repr(pushed[0].user_text))

pushed, _ = resolve([{"type": "text", "text": "hi"},
                     {"type": "text", "text": "hi"},
                     {"type": "text", "text": "there"}])
check("only the repeat is collapsed, the rest survives",
      pushed and pushed[0].user_text == "hi\nthere", repr(pushed[0].user_text))

# A repeat that arrives AFTER the first task started cannot be seen by the batch
# collapse — it would run the whole image job a second time on the one GPU.
pushed, sent = resolve([{"type": "text", "text": OUTPAINT}], running=OUTPAINT)
check("a repeat of the running action is dropped", pushed == [], str(pushed))
check("and the user is told why, not ignored",
      any(T._t("already_running", "en") in s or T._t("already_running", "ru") in s
          for s in sent), str(sent)[:110])

pushed, _ = resolve([{"type": "text", "text": UPSCALE}], running=OUTPAINT)
check("a DIFFERENT action while one runs is still accepted", len(pushed) == 1)

pushed, _ = resolve([{"type": "text", "text": "tell me about cats"}],
                    running="tell me about cats")
check("a repeated typed sentence is NOT dropped (people mean those)",
      len(pushed) == 1, str(pushed))

check("every inline action payload counts as machine-generated",
      set(T._CB_CMDS.values()) <= T._MACHINE_PAYLOADS)
check("and no empty string slipped into the set", "" not in T._MACHINE_PAYLOADS)

print()
print("=" * 70)
print("A REGISTERED PHOTO SURVIVES ITS OWN TASK'S CLEANUP")
print("=" * 70)
# _resolve_and_push downloads an uploaded photo's bytes and hands the SAME path
# to two places: task.image_path (read once, then cleaned up by the consumer
# loop in tg_queue.py after the task finishes) and sess.image_log via
# _log_image (meant to persist, so a later "the one I sent earlier" still
# resolves). Those two lifetimes conflict: if the saved path is an ordinary
# tempfile, the post-task cleanup deletes it out from under the register the
# moment the task completes, and every plain photo upload becomes
# unreferenceable one turn later. The fix keeps the register path under
# tg_bot._IMAGE_DIR (the same persistent directory _save_incoming_photo already
# uses for a reply-fetched photo) and has the cleanup skip anything under it.

import threading, time as _time, copy as _copy

def _make_photo_bot():
    ctx = _Ctx()
    bot = T.TelegramBot("1:T", lambda: ctx, lambda: object(),
                        lambda: {"messages": []}, silent_mode=True)
    bot._backend = T.InMemoryBackend()
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: (bot.sent.append(t), 1)[1]
    bot._send_get_id = lambda cid, t, **kw: len(bot.sent)
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"REG" * 40
    bot._running = True
    bot._consumers = [threading.Thread(target=bot._consumer_loop, daemon=True,
                                       name="c0")]
    for t in bot._consumers:
        t.start()
    return bot

def _photo_update(cid, fid):
    return {"update_id": 1, "message": {"message_id": 1,
            "chat": {"id": cid, "type": "private"}, "from": {"id": cid},
            "photo": [{"file_id": fid, "file_size": 10, "width": 10, "height": 10}],
            "caption": "", "date": int(_time.time())}}

CID_REG = 999952
bot = _make_photo_bot()
bot._user_store.put(T._User(chat_id=CID_REG, name="U", status="approved"))
sess = bot._get_session(CID_REG); sess.clear_context(); sess.lang = "en"
bot._store.put(sess)

bot._dispatch(_photo_update(CID_REG, "fid1"))
deadline = _time.monotonic() + 8
while _time.monotonic() < deadline and not bot.sent:
    _time.sleep(0.1)
_time.sleep(1.0)   # let the consumer's post-task cleanup run too

bot._running = False
for t in bot._consumers:
    t.join(timeout=8)

sess = bot._get_session(CID_REG)
check("the photo was registered", len(sess.image_log) == 1, sess.image_log)
if sess.image_log:
    reg_path = sess.image_log[0]["path"]
    check("the register path lives under the persistent image dir",
          Path(reg_path).resolve().is_relative_to(T._IMAGE_DIR.resolve()), reg_path)
    check("and the file itself still exists after its own task's cleanup ran",
          os.path.exists(reg_path), reg_path)

print()
print("=" * 70)
print("A TURN THAT RAN OUT OF TIME STILL HANDS OVER A FINISHED PICTURE")
print("=" * 70)
# Live: the render finished at 09:06:57 and scored 9/10, then two inspect_image
# calls pushed the turn past the 300s watchdog and the user got "try again".
# The deadline is not proof that nothing was made.

def salvage(render_path, render_status="success", also_uploaded=""):
    ctx = _Ctx()
    ctx.last_render_path = render_path
    ctx.last_render_status = render_status
    ctx.last_image_path = also_uploaded
    seen = {"photo": [], "log": []}

    class _G:
        def invoke(self, state, cfg=None): return {}

    bot = T.TelegramBot("1:T", lambda: ctx, lambda: _G(),
                        lambda: {"messages": []}, silent_mode=True)
    bot._api_post = lambda *a, **k: {}
    bot._send_photo = lambda cid, path, **k: (seen["photo"].append(path), True)[1]
    bot._activity.log = lambda *a, **k: seen["log"].append(a)
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID)
    task = T._Task(task_id="t", chat_id=CID, user_text="нарисуй курицу")
    sent = bot._deliver_salvaged_render(ctx, CID, sess, task, "ru", "U")
    return sent, seen, sess

sent, seen, sess = salvage(good)
check("a finished render is handed over after the timeout", sent and seen["photo"] == [good],
      str(seen["photo"]))
check("...and it is registered so the buttons under it resolve",
      len(sess.image_log) >= 1, sess.image_log)

sent, seen, _ = salvage(good, render_status="fail")
check("a FAILED render is not salvaged", not sent and seen["photo"] == [])

sent, seen, _ = salvage(os.path.join(_DATA_DIR, "vanished.png"))
check("a render whose file is gone is not salvaged", not sent and seen["photo"] == [])

# The one that would be worse than the timeout notice: posting the user's own
# photo back at them. last_image_path holds an upload just as readily as a
# render, which is exactly why the salvage never reads it.
sent, seen, _ = salvage("", also_uploaded=good)
check("the user's own upload is never posted back as a result",
      not sent and seen["photo"] == [], str(seen["photo"]))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
