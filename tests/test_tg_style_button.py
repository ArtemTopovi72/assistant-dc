"""🎭 Change style button: TG had NO path at all to transfer_image before this.

transfer_image (tool_image_handlers._handle_transfer_image) needs TWO images in
ctx.reference_images, which only the desktop GUI's Transfer tab ever populated.
Pressing 🎭 under a picture arms sess.pending_style_target; the very next photo
this chat sends is consumed as the style reference (tg_resolve.py's
_push_style_transfer_task), producing a _Task whose style_ref_path rides
alongside image_path -- tg_tasks.py's _execute_task reads it and fills
ctx.reference_images = [target, reference] before the turn runs.

Offline: bot._backend is a stub recording pushed _Task objects -- nothing pops
the queue, so no worker thread runs and no request reaches ComfyUI/LM Studio.

Run: venv/Scripts/python.exe tests/test_tg_style_button.py
"""
# Script, not pytest: check() only counts, and the module-level sys.exit lives
# under `if __name__ == "__main__":` (indented), which tests/run_all.py's
# classify() cannot see -- undeclared, this file silently fell through to
# "skip" and never ran as part of the suite at all (caught 2026-09-18: it was
# missing from run_all's own summary).
RUN_AS_SCRIPT = True
import os, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_style_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999903
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_style_imgs_")


def make_png(name):
    p = os.path.join(IMG_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
    return p


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot.pushed = []

    def _send(cid, text, **kw):
        bot.sent.append((text, kw.get("keyboard")))
        return 1
    bot._send_text = _send
    bot._send_get_id = lambda cid, text, **kw: (_send(cid, text, **kw), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda t: bot.pushed.append(t), depth=lambda: 0, name=lambda: "stub",
        chat_depth=lambda cid: 0, drop_chat=lambda cid: 0, close=lambda: None,
        tasks_ahead=lambda tid: [])
    bot._activity.log = lambda *a, **k: None
    # transfer_image is never actually called (offline); avoid a real download.
    bot._dl_bytes = lambda file_id: b"\x89PNG\r\n\x1a\n" + os.urandom(32)
    u = T._User(chat_id=CID, name="Tester", status="approved")
    bot._user_store.put(u)
    return bot


def seed_one_image(bot):
    s = bot._get_session(CID)
    s.clear_context()
    a = make_png("target.png")
    id_a = T._log_image(s, a, msg_id=101, label="target", src="bot")
    s.lang = "en"
    bot._store.put(s)
    return a, id_a


def press(bot, data, msg_id=900):
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": msg_id}}})
    deadline = time.monotonic() + 5.0
    base_sent = len(bot.sent)
    while time.monotonic() < deadline:
        if len(bot.sent) != base_sent:
            time.sleep(0.05)
            return
        time.sleep(0.03)


def send_photo(bot, file_id="ref_file_1", caption=""):
    bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                               "message_id": 901, "date": int(time.time()),
                               "photo": [{"file_id": file_id, "file_size": 1234,
                                         "width": 100, "height": 100}],
                               "caption": caption}})
    deadline = time.monotonic() + 5.0
    base = len(bot.pushed)
    while time.monotonic() < deadline:
        if len(bot.pushed) != base:
            time.sleep(0.1)
            return
        time.sleep(0.03)


print("=" * 70)
print("🎭 Change style: button -> ask for reference -> photo -> transfer task")
print("=" * 70)

bot = make_bot()
path_a, id_a = seed_one_image(bot)

press(bot, f"style:{id_a}")
sess = bot._get_session(CID)
check("pending_style_target armed on the pressed image",
      sess.pending_style_target == id_a, sess.pending_style_target)
check("bot asked for the reference photo", bool(bot.sent), bot.sent)

send_photo(bot)

check("exactly one task was pushed", len(bot.pushed) == 1, bot.pushed)
if bot.pushed:
    task = bot.pushed[0]
    check("task targets the ORIGINAL picture", task.image_path == path_a,
          (task.image_path, path_a))
    check("task carries a style_ref_path (the just-sent photo)",
          bool(task.style_ref_path) and os.path.exists(task.style_ref_path),
          task.style_ref_path)
    check("style_ref_path is NOT the target path",
          task.style_ref_path != task.image_path)
    check("instruction names transfer_image",
          "transfer_image" in task.user_text, task.user_text)

sess2 = bot._get_session(CID)
check("pending_style_target cleared after the reference photo arrived",
      sess2.pending_style_target == "", sess2.pending_style_target)

print("=" * 70)
print("A photo with NO pending style target is not hijacked")
print("=" * 70)
bot2 = make_bot()
seed_one_image(bot2)
send_photo(bot2, file_id="unrelated")
check("the photo still goes through the ORDINARY flow, not style",
      len(bot2.pushed) == 1 and not bot2.pushed[0].style_ref_path, bot2.pushed)

print("=" * 70)
print("Pressing 🎭 with no valid target image reports img_gone, arms nothing")
print("=" * 70)
bot3 = make_bot()
sess3 = bot3._get_session(CID)
sess3.clear_context()
bot3._store.put(sess3)
press(bot3, "style")
sess3b = bot3._get_session(CID)
check("nothing armed when there is no target picture",
      sess3b.pending_style_target == "", sess3b.pending_style_target)

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
