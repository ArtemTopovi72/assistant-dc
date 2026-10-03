"""UX sweep: EVERY keyboard label, in every language, pressed through the real
dispatcher of a real TelegramBot (network stubbed). A label must reach its
handler byte-identical and never be queued for the model as if it were a
question.

Live 2026-09-27: «🎵 Songs» was rewritten to «Ыщтпы» before any handler saw it
and redrew a photo, while every unit suite was green. The transcript auditor
(scripts/ux_audit.py) must also flag each known failure shape — checked on a
synthetic chat at the end.

Run: venv/Scripts/python.exe tests/test_ux_every_button.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import tg_strings

T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_uxbtn_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999871


def make_bot(lang):
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent, bot.pushed = [], []
    bot._send_text = lambda cid, text, **k: (bot.sent.append(text), 1)[1]
    bot._send_get_id = lambda cid, text, **k: (bot.sent.append(text), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend.push = lambda task: bot.pushed.append(task)
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID)
    try: sess.lang = lang
    except Exception: pass
    return bot


def upd(text):
    return {"update_id": 1, "message": {"message_id": 1, "chat": {"id": CID},
                                        "from": {"id": CID}, "text": text}}


labels = [(lang, v) for forms in tg_strings._BTN.values() for lang, v in forms.items()]

# 1. the label reaches the dispatcher unchanged (layout fixer, any pre-processing)
for lang, label in labels:
    bot = make_bot(lang)
    seen = []
    bot._dispatch = lambda u: seen.append(u["message"]["text"])
    bot._dispatch_logged(upd(label))
    check("reaches handler unchanged", seen == [label], (lang, label, seen, bot.sent[:1]))

# 2. pressing it never queues the bare label as a question for the model
for lang, label in labels:
    bot = make_bot(lang)
    try:
        bot._dispatch_logged(upd(label))
        bot._flush_debounce(CID) if hasattr(bot, "_flush_debounce") else None
    except Exception as exc:
        check("does not crash", False, (lang, label, repr(exc)[:160]))
        continue
    asked = [t.user_text for t in bot.pushed if (t.user_text or "").strip() == label]
    check("not sent to the model as text", not asked, (lang, label))

# 3. the transcript auditor flags each failure shape
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ux_audit
labs = {v for _l, v in labels}
row = lambda ts, kind, data: {"ts": "2026-09-27 11:" + ts, "kind": kind, "data": data}
chat = [
    row("34:33", "out", {"text": "⌨️ Looks like the wrong keyboard layout. You meant: «<i>🎵 Ыщтпы</i>»",
                         "method": "sendMessage"}),
    row("37:36", "out", {"text": "⚙️ <b>Starting…</b>", "method": "sendMessage"}),
    row("37:39", "in", {"button": "cancel:abc"}),
    row("37:41", "out", {"text": "🎨 Got it, redrawing — a couple of minutes.", "method": "sendMessage"}),
    row("37:55", "out", {"text": "👌 Noted: “Stop”", "method": "sendMessage"}),
    row("38:00", "in", {"text": "🎨 Creativity"}),
    row("38:10", "in", {"text": "🎨 Creativity"}),
    row("38:20", "in", {"text": "🎨 Creativity"}),
]
rules = {f[2] for f in ux_audit.audit_chat(chat, labs)}
for r in ("layout-rewrite", "cancel-ignored", "stop-as-steer", "repeat"):
    check("auditor flags " + r, r in rules, rules)
clean = [row("40:00", "in", {"button": "cancel:x"}),
         row("40:01", "out", {"text": "⛔ Request cancelled.", "method": "sendMessage"}),
         row("40:05", "out", {"text": "🎨 Got it, redrawing", "method": "sendMessage"})]
check("a Cancel that was honoured is not flagged",
      "cancel-ignored" not in {f[2] for f in ux_audit.audit_chat(clean, labs)})

print(f"\n{OK} passed, {BAD} failed  ({len(labels)} labels x 2 checks + auditor)")
sys.exit(1 if BAD else 0)
