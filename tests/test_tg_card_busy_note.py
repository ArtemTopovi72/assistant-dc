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


def note(ahead, busy):
    bot = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    bot._send_text = lambda cid, text, **k: sent.append(text)
    bot._backend = types.SimpleNamespace(tasks_ahead=lambda tid: list(ahead))
    bot._card_busy = lambda: busy
    bot._queue_note(5, types.SimpleNamespace(task_id="t"), "ru")
    return sent


sent = note([], "YuE2")
check("nobody ahead but the card is rendering: the user is told", sent and "Видеокарта занята" in sent[0], sent)
check("...without naming whose render it is", sent and "YuE2" not in sent[0], sent)
check("nobody ahead and the card shared: no note", note([], "") == [])
ahead = [T._Task(task_id="a", chat_id=1, user_text="x")]
sent = note(ahead, "YuE2")
check("a real queue still shows its place", len(sent) == 1 and "#2" in sent[0], sent)

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
