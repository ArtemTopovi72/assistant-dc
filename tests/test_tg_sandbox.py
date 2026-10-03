"""The sandbox as it is reached from Telegram.

Two things are being checked, and both are about the boundary rather than the
sandbox itself (which has its own suite):

  1. the grant is honoured on the way IN — a user without it cannot list, reset
     or learn anything about a working folder, and every one of the three
     commands says the same thing, so nobody maps the feature by noticing which
     one answers differently;

  2. an uploaded file lands in the RIGHT user's folder, under a name that came
     from an attacker-controlled field. Telegram will carry a document called
     "../../evil.txt" without comment.

Run: venv/Scripts/python.exe tests/test_tg_sandbox.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

# Before any bot exists — otherwise fixture users land in the LIVE store.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_sandbox_")
T.redirect_data_dir(_DATA_DIR)

import sandbox_access as A
import code_sandbox as CS
from pathlib import Path

# Sandboxes go somewhere disposable too: the real base is under runtime/ and a
# test must not leave chat folders in it.
_SBX_BASE = Path(tempfile.mkdtemp(prefix="tgtest_sbxroot_"))
CS.SANDBOX_BASE = _SBX_BASE

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload or {}))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    return bot, sent


def add_user(bot, chat_id, level=None, name="tester"):
    u = T._User(chat_id=chat_id, name=name, status="approved")
    if level:
        u.prefs = {"sandbox": level}
    bot._user_store.put(u)
    return u


def last_text(sent):
    for method, payload in reversed(sent):
        if method == "sendMessage":
            return payload.get("text", "")
    return ""


# ── the grant is honoured ────────────────────────────────────────────────────
bot, sent = make_bot()
add_user(bot, 100)                       # no grant at all
sess = bot._get_session(100)

for cmd in ("/files", "/sandbox", "/reset_sandbox"):
    sent.clear()
    bot._handle_sandbox_command(100, sess, bot._user_store.get(100), "ru", cmd)
    txt = last_text(sent)
    check(f"{cmd} refused without a grant", "не включена" in txt, txt[:80])

# All three must say the SAME thing: a different answer per command is a map of
# the feature for someone who is not supposed to have it.
sent.clear()
answers = []
for cmd in ("/files", "/sandbox", "/reset_sandbox"):
    sent.clear()
    bot._handle_sandbox_command(100, sess, bot._user_store.get(100), "ru", cmd)
    answers.append(last_text(sent))
check("the refusal is identical for all three", len(set(answers)) == 1, answers)

# A granted user gets a real answer.
add_user(bot, 101, level=A.FILES)
sess101 = bot._get_session(101)
sent.clear()
bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "ru", "/files")
check("granted user gets the folder", "пуста" in last_text(sent), last_text(sent)[:80])

CS.sandbox_for(101).write_text("notes.txt", "hi")
sent.clear()
bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "ru", "/files")
def last_kb(sent):
    import json as _json
    for method, payload in reversed(sent):
        if method in ("sendMessage", "editMessageText") and payload.get("reply_markup"):
            return _json.loads(payload["reply_markup"])
    return {}
_kb = last_kb(sent)
_labels = [b["text"] for row in _kb.get("inline_keyboard", []) for b in row]
check("the listing shows the file as a button", any("notes.txt" in t for t in _labels), _labels)

sent.clear()
bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "ru", "/sandbox")
_status = last_text(sent)
check("/sandbox reports the access level", "Доступ" in _status, _status[:80])
check("/sandbox reports whether code can run",
      "Выполнение кода" in _status, _status[:120])

sent.clear()
bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "ru", "/reset_sandbox")
check("reset empties it", not CS.sandbox_for(101).list_dir("."), "still has files")
check("reset says what it did", "очищена" in last_text(sent), last_text(sent)[:80])


# ── revocation takes effect immediately ──────────────────────────────────────
A.set_level(bot._user_store, 101, A.OFF)
sent.clear()
bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "ru", "/files")
check("a revoked user loses access on the next command",
      "не включена" in last_text(sent), last_text(sent)[:80])


# ── uploads land in the right folder, under a safe name ──────────────────────
A.set_level(bot._user_store, 101, A.FILES)
add_user(bot, 202, level=A.FILES)

box101 = CS.sandbox_for(101)
box202 = CS.sandbox_for(202)
box101.write_text("mine.txt", "A")
box202.write_text("mine.txt", "B")
check("two chats do not share a folder",
      box101.read_text("mine.txt") == "A" and box202.read_text("mine.txt") == "B")

# The filename is attacker-controlled: Telegram carries whatever was sent.
escaped = False
for hostile in ("../../evil.txt", "..\\..\\evil.txt", "C:/Windows/evil.txt",
                "/etc/evil.txt"):
    safe = Path(hostile).name or "upload.bin"
    try:
        target = box101.resolve(safe)
        target.write_bytes(b"x")
        if box101.root not in target.parents and target != box101.root:
            escaped = True
    except Exception:
        pass
check("a hostile filename cannot leave the folder", not escaped)
check("nothing was written outside the sandbox base",
      not (_SBX_BASE.parent / "evil.txt").exists())


# ── an English user gets English ─────────────────────────────────────────────
# The whole point of this surface is being understood, and it shipped as Russian
# string literals: an English user was told what to do in Russian.
from tg_strings import _t
A.set_level(bot._user_store, 101, A.CODE)
for cmd in ("/files", "/sandbox", "/reset_sandbox"):
    sent.clear()
    bot._handle_sandbox_command(101, sess101, bot._user_store.get(101), "en", cmd)
    txt = last_text(sent)
    check(f"{cmd} answers an English user in English",
          "How to use it" in txt and "Как пользоваться" not in txt, txt[:90])

sent.clear()
bot._handle_sandbox_command(100, bot._get_session(100),
                            bot._user_store.get(100), "en", "/files")
check("the refusal is translated too",
      "not enabled" in last_text(sent), last_text(sent)[:80])

# Nothing may be left as a literal: the detector rule is that a new string
# reaches the user through _t(), not that it happens to be Russian today.
import inspect, re, tg_commands
_src = inspect.getsource(tg_commands.CommandsMixin._handle_sandbox_command)
_ru = [l for l in re.findall(r'"([^"]{12,})"', _src) if re.search(r"[А-Яа-я]", l)]
check("no Russian literals left in the handler", not _ru, _ru[:3])


# ── the commands are advertised ──────────────────────────────────────────────
_cmds = {c for c, _en, _ru in T.TelegramBot._COMMANDS}
for name in ("files", "sandbox", "reset_sandbox"):
    check(f"/{name} is registered with Telegram", name in _cmds, sorted(_cmds))


print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(0 if BAD == 0 else 1)
