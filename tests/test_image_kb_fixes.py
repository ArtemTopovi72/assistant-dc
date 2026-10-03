"""Regression tests for three confirmed bugs in the inline image-action keyboard,
proved by a real-chat audit (scratchpad/audit_03_imagekb.md, 2026-08-03):

1. Two queued presses on two DIFFERENT pictures both silently resolved to the
   LAST one — the picture was resolved at EXECUTION time from a single
   `sess.target_image` slot, which a second press already overwrote. Fixed by
   threading the pressed image id through the enqueued item and onto `_Task`
   itself (`_Task.image_id`), captured at PRESS time.

2. The duplicate-press guard compared only the machine payload TEXT, which is
   identical across all six verbs ("on the current image"). Pressing the SAME
   verb on a DIFFERENT picture while one was already running was wrongly
   refused as a dupe. Fixed by folding the target image id into the dedupe key.

3. `describe:<id>` was an accepted verb (`_IMAGE_VERBS`) with no handler in
   `_CB_CMDS` — it silently ARMED `sess.target_image` and then matched nothing,
   leaving a stale armed target. Fixed by removing "describe" from
   `_IMAGE_VERBS` entirely (it was never emitted by `_image_kb`), so the
   callback now falls through as an ordinary unmatched string instead of
   arming anything.

Offline: `bot._backend` is a stub recording pushed `_Task` objects — nothing
ever pops the queue, so no worker thread runs and no request reaches
ComfyUI/LM Studio/GPU.

Run: venv/Scripts/python.exe tests/test_image_kb_fixes.py
"""
import os, sys, tempfile, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_imgkb_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999902
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_imgkb_imgs_")


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
    # This is the "nothing renders" boundary: push() only records the _Task —
    # no pop(), no worker thread, no graph.invoke(), no ComfyUI/GPU call.
    bot._backend = types.SimpleNamespace(
        push=lambda t: bot.pushed.append(t), depth=lambda: 0, name=lambda: "stub",
        chat_depth=lambda cid: 0, drop_chat=lambda cid: 0, close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    u = T._User(chat_id=CID, name="Tester", status="approved")
    bot._user_store.put(u)
    return bot


def seed_two_images(bot):
    s = bot._get_session(CID)
    s.clear_context()
    a, b = make_png("a.png"), make_png("b.png")
    id_a = T._log_image(s, a, msg_id=101, label="a", src="bot")
    id_b = T._log_image(s, b, msg_id=202, label="b", src="bot")
    s.lang = "en"
    bot._store.put(s)
    return a, b, id_a, id_b


def press(bot, data, msg_id=900):
    """Drive a button press straight through _dispatch (real update shape),
    directly through the enqueue path — no debounce wait needed since our
    test doesn't depend on batching multiple items."""
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": msg_id}}})
    # The item is queued via _enqueue_item's debounce worker thread; give it a
    # moment to hand off to _resolve_and_push (which then calls the stub
    # push() synchronously — no real render happens).
    import time
    deadline = time.monotonic() + 5.0
    base = len(bot.pushed)
    while time.monotonic() < deadline:
        if len(bot.pushed) != base or bot.sent:
            time.sleep(0.1)
            return
        time.sleep(0.03)


print("=" * 70)
print("1. Two presses on two DIFFERENT pictures resolve to DISTINCT images")
print("=" * 70)

bot = make_bot()
path_a, path_b, id_a, id_b = seed_two_images(bot)

press(bot, f"regenerate:{id_a}")
press(bot, f"regenerate:{id_b}")

check("two tasks were pushed", len(bot.pushed) == 2, bot.pushed)
if len(bot.pushed) == 2:
    task_a, task_b = bot.pushed
    check("task A carries image_id A", task_a.image_id == id_a,
          (task_a.image_id, id_a))
    check("task B carries image_id B", task_b.image_id == id_b,
          (task_b.image_id, id_b))
    check("task A and task B target DIFFERENT images",
          task_a.image_id != task_b.image_id, (task_a.image_id, task_b.image_id))

    # The session-level slot has since moved on to B (or been cleared) — proving
    # the OLD lazy-read-at-execution-time approach would have gotten task A
    # wrong, and that resolving from task.image_id is what actually saves it.
    entry_a = T._image_by_id(bot._get_session(CID), task_a.image_id)
    entry_b = T._image_by_id(bot._get_session(CID), task_b.image_id)
    check("task A resolves to picture A's path",
          entry_a and entry_a["path"] == path_a, entry_a)
    check("task B resolves to picture B's path",
          entry_b and entry_b["path"] == path_b, entry_b)

print()
print("=" * 70)
print("2. Duplicate guard keys on (verb, image) — not verb text alone")
print("=" * 70)

bot = make_bot()
path_a, path_b, id_a, id_b = seed_two_images(bot)

press(bot, f"regenerate:{id_a}")
check("first regenerate on A was pushed", len(bot.pushed) == 1, bot.pushed)
task_a = bot.pushed[0]
# Simulate "still running": the dedupe guard compares against _running_task.
bot._running_task[CID] = [task_a]

bot.sent.clear()
press(bot, f"regenerate:{id_b}")
check("regenerate on a DIFFERENT picture while A is 'running' is NOT refused",
      len(bot.pushed) == 2, bot.pushed)
check("the second task targets B, not A",
      bot.pushed and bot.pushed[-1].image_id == id_b,
      bot.pushed[-1].image_id if bot.pushed else None)
check("no 'already doing exactly that' refusal was sent for the different picture",
      not any("already" in (t or "").lower() for t, _ in bot.sent), bot.sent)

# Same verb, same picture, while still "running" — THIS must still be refused.
bot.sent.clear()
bot._running_task[CID] = [task_a]
pushed_before = len(bot.pushed)
press(bot, f"regenerate:{id_a}")
check("same verb + SAME picture while running IS still refused as a dupe",
      len(bot.pushed) == pushed_before, bot.pushed)
check("the dupe refusal text was sent",
      any("already" in (t or "").lower() or "⏳" in (t or "") for t, _ in bot.sent),
      bot.sent)

print()
print("=" * 70)
print("3. describe:<id> is not a recognised verb — no silent arm-and-drop")
print("=" * 70)

bot = make_bot()
path_a, path_b, id_a, id_b = seed_two_images(bot)
sess = bot._get_session(CID)
sess.target_image = ""
bot._store.put(sess)

check("'describe' is not in _IMAGE_VERBS", "describe" not in T._IMAGE_VERBS)

bot.sent.clear(); bot.pushed.clear()
press(bot, f"describe:{id_a}", msg_id=901)

check("describe:<id> pushed no task", len(bot.pushed) == 0, bot.pushed)
after = bot._get_session(CID)
check("describe:<id> left sess.target_image UNARMED",
      not getattr(after, "target_image", ""), after.target_image)

print()
print("=" * 70)
summary = f"TOTAL: {OK} passed, {BAD} failed"
print(summary)
print("=" * 70)
sys.exit(1 if BAD else 0)
