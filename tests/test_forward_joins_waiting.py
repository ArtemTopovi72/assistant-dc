"""A forward that lands while the chat's previous turn still waits joins it.

Open item 2026-10-05: a forwarded post arriving after the debounce had already
queued the user's turn (a late part of the same paste, a getUpdates gap) became
a second task — two answers to one question. A forward now folds into the
newest WAITING task of the same chat: never into one a worker has started,
never one older than two minutes, never two pictures into one turn.
"""
import logging
import os
import sys
import tempfile
import time

os.environ.setdefault("F5_TEST_RUN", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T  # noqa: E402

T.redirect_data_dir(tempfile.mkdtemp(prefix="fwdjoin_"))

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond)
    BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


bot = T.TelegramBot("123:TEST", lambda: None, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend()
bot._send_text = lambda *a, **k: 1
bot._send_get_id = lambda *a, **k: 1
bot._edit_text = lambda *a, **k: None
bot._api_post = lambda *a, **k: {}
bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._try_steer = lambda *a, **k: False

CID = 9_300_777
bot._user_store.put(T._User(chat_id=CID, name="J", status="approved", is_admin=False))
sess = bot._get_session(CID)
sess.clear_context()
sess.lang = "ru"
sess.reg_state = ""
bot._store.put(sess)


def waiting():
    return [t for t in bot._backend.service_order() if t.chat_id == CID]


def queue(text, age=0.0):
    t = T._Task(task_id="prev-" + str(time.time()), chat_id=CID, user_text=text,
                enqueue_ts=time.time() - age)
    bot._backend.push(t)
    return t


fwd = [{"type": "text", "text": "пересланный пост про погоду", "forwarded": True,
        "forward_from": "Канал"}]

queue("что тут написано?")
bot._resolve_and_push(CID, list(fwd))
w = waiting()
check("a forward joins the chat's waiting turn: one task, not two", len(w) == 1, [t.user_text for t in w])
check("...the waiting turn's words come first", w and w[0].user_text.startswith("что тут написано?"),
      w and w[0].user_text)
check("...and the forwarded post is in it", w and "пересланный пост" in w[0].user_text, w and w[0].user_text)

bot._backend.drop_chat(CID)
queue("старый вопрос", age=600)
bot._resolve_and_push(CID, list(fwd))
check("a turn queued ten minutes ago is left alone", len(waiting()) == 2, [t.user_text for t in waiting()])

bot._backend.drop_chat(CID)
bot._resolve_and_push(CID, [{"type": "text", "text": "сам по себе"}])
bot._resolve_and_push(CID, [{"type": "text", "text": "ещё вопрос"}])
check("ordinary typed messages are not folded", len(waiting()) == 2, [t.user_text for t in waiting()])

print(f"\n{OK}/{OK + BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
