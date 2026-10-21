"""A photo sent while another chat's song held the whole card waited 55 s with
no word: the ⏳ note only counted tasks queued AHEAD, and a render in another
chat is not one of them.

Run: venv/Scripts/python.exe tests/test_tg_card_busy_note.py
"""
import os, sys, tempfile, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_cardbusy_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def note(ahead, busy, rendering=()):
    bot = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    bot._send_get_id = lambda cid, text, **k: (sent.append(text), len(sent))[1]
    bot._backend = types.SimpleNamespace(tasks_ahead=lambda tid: list(ahead))
    bot._card_busy = lambda: busy
    bot._running_task = {c: [types.SimpleNamespace(delivered=False)] for c in rendering}
    bot._chat_interruptible = {c: True for c in rendering}
    bot._queue_note(5, types.SimpleNamespace(task_id="t"), "ru")
    return sent


sent = note([], "YuE2")
check("nobody ahead but the card is rendering: the user is told", sent and "Видеокарта занята" in sent[0], sent)
check("...without naming whose render it is", sent and "YuE2" not in sent[0], sent)
check("nobody ahead and the card shared: no note", note([], "") == [])
ahead = [T._Task(task_id="a", chat_id=1, user_text="x")]
sent = note(ahead, "YuE2")
check("a real queue still shows its place", len(sent) == 1 and "#2" in sent[0], sent)
sent = note([], "", rendering=(9,))
check("another chat is rendering, the card shared: #2 and why", sent and "#2" in sent[0] and "генерация" in sent[0], sent)
sent = note([], "", rendering=(5,))
check("only this chat's own render: no note", sent == [], sent)
sent = note(ahead, "", rendering=(9, 8))
check("queue plus two renders: place counts both", sent and "#4" in sent[0], sent)


def early(rendering=(9,)):
    bot = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    bot._send_get_id = lambda cid, text, **k: (sent.append(text), len(sent))[1]
    bot._backend = types.SimpleNamespace(tasks_ahead=lambda tid: [])
    bot._card_busy = lambda: ""
    bot._running_task = {c: [types.SimpleNamespace(delivered=False)] for c in rendering}
    bot._chat_interruptible = {c: True for c in rendering}
    return bot, sent


bot, sent = early()
bot._busy_note(5, "ru")
check("the plan's ▶ is answered at the click, before any task exists", sent and "#2" in sent[0], sent)
bot._queue_note(5, types.SimpleNamespace(task_id="t"), "ru")
check("the queue's own note right after does not double it", len(sent) == 1, sent)
bot, sent = early(rendering=())
bot._busy_note(5, "ru")
check("nobody busy: ▶ stays quiet", sent == [], sent)

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
