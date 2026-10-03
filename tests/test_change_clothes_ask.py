"""👗 Change outfit button: asks WHAT to change into, instead of silently
applying a random "something different" swap.

Live, 2026-09-19: the user asked for the same "what to what" free-text
capture ✏️ Edit already has ("I write, for example, the hat to a hat")
after complaining the old bare-verb button changed the outfit but never
asked. tg_callbacks._cb_change_clothes now arms pending_prefix = "change
the outfit to: " and asks describe_clothes, mirroring _cb_edit_image
exactly; the next free-text message becomes "change the outfit to: <text>"
via the normal pending_prefix consumption path (tg_bot._resolve_and_push).

Run: venv/Scripts/python.exe tests/test_change_clothes_ask.py
"""
RUN_AS_SCRIPT = True
import os, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_clothes_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999904
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_clothes_imgs_")


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


def send_text(bot, text, msg_id=902):
    bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                               "message_id": msg_id, "date": int(time.time()),
                               "text": text}})
    deadline = time.monotonic() + 5.0
    base = len(bot.pushed)
    while time.monotonic() < deadline:
        if len(bot.pushed) != base:
            time.sleep(0.1)
            return
        time.sleep(0.03)


print("=" * 70)
print("👗 Change outfit: button asks first, does not auto-apply")
print("=" * 70)

bot = make_bot()
path_a, id_a = seed_one_image(bot)

press(bot, f"change_clothes:{id_a}")
sess = bot._get_session(CID)
check("pending_prefix armed to the clothes-specific prefix",
      sess.pending_prefix == "change the outfit to: ", sess.pending_prefix)
check("target_image resolved to the pressed picture",
      sess.target_image == id_a, sess.target_image)
check("bot asked what to change into instead of enqueueing anything",
      bool(bot.sent) and not bot.pushed, (bot.sent, bot.pushed))
check("the ask text is the describe_clothes string",
      bot.sent and "outfit" in bot.sent[-1][0].lower(), bot.sent)

print("=" * 70)
print("Free text after the press becomes 'change the outfit to: <text>'")
print("=" * 70)
send_text(bot, "the hat into a cap")
check("exactly one task was pushed", len(bot.pushed) == 1, bot.pushed)
if bot.pushed:
    task = bot.pushed[0]
    # NOTE: an ordinary typed message (unlike the id-based style/animate
    # preset presses) does not pre-attach image_path on the Task itself --
    # the graph resolves "the current image" from sess.target_image/context
    # at execution time. This flow is about the PREFIX/TEXT, not the id.
    check("task text carries the clothes prefix and the user's own words",
          task.user_text == "change the outfit to: the hat into a cap",
          task.user_text)

sess2 = bot._get_session(CID)
check("pending_prefix cleared after the free text was consumed",
      sess2.pending_prefix == "", sess2.pending_prefix)

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
