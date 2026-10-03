"""🔐 Admin console (tg_admin): live panel, cancel / bump / stop / say / ban, on the REAL bot.

Run: venv/Scripts/python.exe tests/test_tg_admin_console.py
"""
import os, sys, tempfile, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_admin_"))
from tg_queue_backends import _Task

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else: BAD += 1; print(f"FAIL  {name}   {extra}")

ADM, U1, U2, EVIL = 500, 501, 502, 503
bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
bot.sent, bot.edits, bot.api = [], [], []
bot._send_text = lambda cid, text, **kw: bot.sent.append((cid, text))
bot._send_get_id = lambda cid, text, **kw: (bot.sent.append((cid, text)), 77)[1]
bot._edit_text = lambda cid, mid, text, **kw: bot.edits.append((cid, mid, text, kw.get("keyboard")))
bot._api_post = lambda m, p=None, **k: bot.api.append((m, p)) or {}
for cid, name, adm in ((ADM, "Ivan", True), (U1, "Влад", False), (U2, "Stepan", False), (EVIL, "Evil", False)):
    bot._user_store.put(T._User(chat_id=cid, name=name, status="approved", is_admin=adm))

# a running task for Влад, two queued for Stepan
run = _Task(task_id="r" * 36, chat_id=U1, user_text="помоги ответить на вопрос")
ev = threading.Event()
bot._running_task[U1] = [run]; bot._task_started[run.task_id] = time.monotonic() - 65
bot._task_cancels[run.task_id] = ev
bot._active_stages[U1] = "👁 Looking at the image"
q1 = _Task(task_id="a" * 36, chat_id=U2, user_text="[song:0] про кота")
q2 = _Task(task_id="b" * 36, chat_id=U2, user_text="нарисуй кота")
bot._backend.push(q1); bot._backend.push(q2)

s = bot.admin_snapshot()
check("snapshot: running with user, stage, elapsed",
      s["running"][0]["user"] == "Влад" and "Looking" in s["running"][0]["stage"] and s["running"][0]["elapsed"] >= 60)
check("snapshot: queue in service order, song payload readable",
      [q["task_id"] for q in s["queued"]] == [q1.task_id, q2.task_id] and s["queued"][0]["text"].startswith("🎵 про кота"))

bot._send_admin_panel(ADM)
text = bot.sent[-1][1]
check("panel shows who runs what and the queue", "Влад" in text and "1:0" in text and "Stepan" in text, text)

def press(data, who=ADM):
    bot._dispatch_callback({"id": "cb", "from": {"id": who},
                            "message": {"chat": {"id": ADM}, "message_id": 77}, "data": data})

press("adm:b:" + q2.task_id)
check("⬆️ bump serves the second one first", bot._backend.service_order()[0].task_id == q2.task_id)
press("adm:c:" + q1.task_id)
check("✖ drops a queued task and tells its owner",
      all(t.task_id != q1.task_id for t in bot._backend.service_order())
      and any(c == U2 and "Администратор отменил" in t for c, t in bot.sent), bot.sent[-3:])
press("adm:c:" + run.task_id)
check("⛔ cancels the running task (its own event)", ev.is_set())
press("adm:s:%d" % U2)
check("stop chat empties that user's queue", bot._backend.chat_depth(U2) == 0)

n = len(bot._backend.service_order())
bot._backend.push(_Task(task_id="c" * 36, chat_id=U2, user_text="x"))
press("adm:c:" + "c" * 36, who=EVIL)
check("a non-admin pressing an old panel button does nothing",
      len(bot._backend.service_order()) == n + 1 and any(p and p.get("text") == "Только для админов" for m, p in bot.api))

press("adm:say:%d" % U1)
check("✉️ arms the admin's next text", bot._get_session(ADM).reg_state == "admin_say:%d" % U1)
bot._user_gate(ADM, {"text": "Влад, иди учи уроки", "chat": {"id": ADM}, "from": {"id": ADM}})
check("the text goes to the user as the administration",
      any(c == U1 and "Администрация" in t and "учи уроки" in t for c, t in bot.sent)
      and bot._get_session(ADM).reg_state == "", bot.sent[-2:])

press("adm:say:%d" % U1)
press("adm:v:users")
check("another panel press drops a pending ✉️", bot._get_session(ADM).reg_state == "")
press("adm:say:%d" % U1)
_n = len(bot.sent)
import tg_strings as _S
bot._user_gate(ADM, {"text": _S._BTN["cover_btn"]["ru"], "chat": {"id": ADM}, "from": {"id": ADM}})
check("a menu button after ✉️ is not sent to the user",
      not any(c == U1 for c, t in bot.sent[_n:]) and bot._get_session(ADM).reg_state == "")

press("adm:ban:%d" % U2)
check("🚫 bans", bot._user_store.get(U2).status == "banned")
press("adm:ok:%d" % U2)
check("✅ unbans", bot._user_store.get(U2).status == "approved")

press("adm:v:users")
check("users view lists everyone", "Stepan" in bot.edits[-1][2] and "Влад" in bot.edits[-1][2])
press("adm:v:u:%d" % U1)
check("user card has the actions", "Написать" in str(bot.edits[-1][3]))

# --- the desktop tab over the same bot ------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
import types
app = QApplication.instance() or QApplication([])
import gui_admin_tab as G
tab = G.AdminTab(types.SimpleNamespace(telegram_tab=types.SimpleNamespace(_bot=bot)))
tab._poller.stop(); tab._poller.wait(3000)
bot._backend.push(_Task(task_id="d" * 36, chat_id=U1, user_text="первый"))
bot._backend.push(_Task(task_id="e" * 36, chat_id=U1, user_text="второй"))
tab._on_snap(bot.admin_snapshot())
check("tab lists the queue and the users", tab.q_t.rowCount() >= 2 and tab.u_t.rowCount() == 4)
rows = {tab.q_t.item(r, 0).data(G.Qt.UserRole): r for r in range(tab.q_t.rowCount())}
tab.q_t.selectRow(rows["e" * 36])
done = threading.Event()
_orig_act = tab._act
tab._act = lambda label, fn, *a: (fn(bot, *a), done.set())
tab._bump()
check("⬆️ from the tab reorders the real queue",
      [t.task_id for t in bot._backend.service_order() if t.chat_id == U1][0] == "e" * 36)
tab.to.setCurrentIndex(tab.to.findData(U2)); tab.msg.setText("привет от админа"); tab._say()
check("📢 from the tab reaches the user", any(c == U2 and "привет от админа" in t for c, t in bot.sent))
tab.shutdown()

bot._running = False
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
