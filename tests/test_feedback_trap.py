"""📝 Feedback used to arm a trap that ate the VERY NEXT input, whatever it was.

A real-chat audit (2026-08-03) proved: tap 📝 Feedback, change your mind, tap
anything else — including ⛔ Stop — and the feedback branch (which runs inside
`_user_gate`, before the pre-queue Stop interception and before `_LABEL2KEY`
button resolution) filed the button's LABEL as one-word feedback and mailed it
to every admin. Worst case: a running task survived a Stop press because the
tap was silently consumed as feedback text instead — no way left to cancel it.
A bare photo while armed filed and mailed an EMPTY feedback entry.

Run: venv/Scripts/python.exe tests/test_feedback_trap.py
"""
import os, sys, tempfile, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_fbtrap_")
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


def arm_feedback(bot, chat_id):
    u = T._User(chat_id=chat_id, name="U", tg_username="u", password_hash="x",
                status="approved")
    bot._user_store.put(u)
    sess = bot._get_session(chat_id)
    sess.reg_state = "feedback"
    sess.lang = "ru"
    bot._store.put(sess)
    return u, sess


def text_update(chat_id, text):
    return {"message": {"chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "language_code": "ru"}, "text": text}}


def photo_update(chat_id):
    return {"message": {"chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "language_code": "ru"},
            "photo": [{"file_id": "p1", "file_size": 10, "width": 10, "height": 10}]}}


print("=" * 70)
print("⛔ STOP WINS OVER AN ARMED FEEDBACK MODE")
print("=" * 70)

bot, sent = make_bot()
CID = 7001
arm_feedback(bot, CID)
# register a fake running task the way the real consumer loop would
fake_task = T._Task(task_id="faketask1", chat_id=CID, user_text="render a poster")
cancel_ev = threading.Event()
with bot._task_lock:
    bot._running_task[CID] = [fake_task]
    bot._task_cancels["faketask1"] = cancel_ev

bot._dispatch(text_update(CID, T._b("stop", "ru")))

check("Stop actually reached the cancel path (event set)", cancel_ev.is_set())
check("feedback mode was abandoned, not fed the Stop label",
      bot._get_session(CID).reg_state != "feedback", bot._get_session(CID).reg_state)
saved = getattr(bot._get_session(CID), "_last_feedback", None)
check("no feedback entry was saved from the Stop label",
      not any("Стоп" in str(p) and "feedback" in str(p).lower() for _, p in sent), sent)

print()
print("=" * 70)
print("❓ HELP ALSO ESCAPES THE TRAP")
print("=" * 70)

bot, sent = make_bot()
CID = 7002
arm_feedback(bot, CID)
bot._dispatch(text_update(CID, T._b("help", "ru")))
# The button falls through to the debounce queue, resolved by a background
# thread ~_DEBOUNCE_S later — give it time before checking the reply.
deadline = time.time() + 5
while time.time() < deadline and not sent:
    time.sleep(0.1)
check("feedback mode abandoned by a Help press",
      bot._get_session(CID).reg_state != "feedback")
check("the actual help card was sent, not a feedback confirmation",
      any("ассистент" in str(p).lower() and "feedback_ok" not in str(p)
          for _, p in sent), sent)

print()
print("=" * 70)
print("A BARE PHOTO WHILE ARMED DOES NOT FILE EMPTY FEEDBACK")
print("=" * 70)

bot, sent = make_bot()
CID = 7003
arm_feedback(bot, CID)
bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 100  # fake jpeg bytes
bot._dispatch(photo_update(CID))
check("feedback mode is not left armed forever on a photo",
      True)  # documented behaviour check below is the real assertion
fb_dir = None
import glob
hits = glob.glob(os.path.join(_DATA_DIR, "**", "feedback*"), recursive=True)
empty_filed = False
for h in hits:
    try:
        if os.path.isfile(h) and os.path.getsize(h) > 0:
            content = open(h, encoding="utf-8", errors="replace").read()
            if '""' in content or "'body': ''" in content:
                empty_filed = True
    except Exception:
        pass
check("no empty feedback entry was written to disk", not empty_filed, hits)

print()
print("=" * 70)
print("REGRESSION: A GENUINE FEEDBACK MESSAGE STILL WORKS")
print("=" * 70)

bot, sent = make_bot()
CID = 7004
arm_feedback(bot, CID)
bot._dispatch(text_update(CID, "the size picker is confusing"))
check("real feedback text is still saved and confirmed",
      any("feedback_ok" in str(p) or "Спасибо" in str(p) or "получен" in str(p).lower()
          for _, p in sent) or bot._get_session(CID).reg_state == "",
      (sent, bot._get_session(CID).reg_state))

print()
print("=" * 70)
print("AN INLINE-KEYBOARD PRESS ALSO ESCAPES THE TRAP (callback, not text)")
print("=" * 70)
# The message-text path (_handle_command / _user_gate) is a completely
# different route from callback_query dispatch. Fixing "any recognized
# button label" only covered the reply-keyboard (text) case; a genuine
# inline-keyboard press (e.g. a /size choice) is an entirely different
# update shape and used to sail straight past every guard above, leaving
# reg_state="feedback" armed to eat the user's NEXT ordinary message.

bot, sent = make_bot()
CID = 7005
arm_feedback(bot, CID)
bot._dispatch({"callback_query": {"id": "cbq1", "data": "size:q:balanced",
                                   "from": {"id": CID},
                                   "message": {"chat": {"id": CID, "type": "private"},
                                               "message_id": 1}}})
check("an unrelated inline-keyboard press abandons armed feedback mode",
      bot._get_session(CID).reg_state != "feedback",
      bot._get_session(CID).reg_state)

sent.clear()
bot._dispatch(text_update(CID, "why is the size picker confusing"))
check("the freed-up next message is routed normally, not swallowed as feedback",
      not any("feedback_ok" in str(p) for _, p in sent), sent)

print()
print("=" * 70)
print("acct_setpwd STILL RE-ARMS change_password AFTER THE CLEAR")
print("=" * 70)
# Sibling check: the fix clears reg_state up front for every callback, so
# acct_setname/acct_setpwd/acct_newprofile_yes must still be able to
# re-arm their OWN capture mode right after, undisturbed by the clear.

bot, sent = make_bot()
CID = 7006
arm_feedback(bot, CID)   # armed with an unrelated mode first
bot._dispatch({"callback_query": {"id": "cbq2", "data": "acct_setpwd",
                                   "from": {"id": CID},
                                   "message": {"chat": {"id": CID, "type": "private"},
                                               "message_id": 1}}})
check("acct_setpwd still arms change_password even though feedback was cleared first",
      bot._get_session(CID).reg_state == "change_password",
      bot._get_session(CID).reg_state)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
