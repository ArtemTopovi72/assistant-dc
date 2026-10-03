"""The Sandbox button in the Creativity menu.

It shares a row with Mashup by request. Two things are asserted, and neither is
a grep of the source: that the button is REACHABLE (label -> sentinel ->
dispatch, in every language), and that pressing it lands on the same handler
/files uses, so the grant check cannot drift apart from the command's.

Run: venv/Scripts/python.exe tests/test_tg_sandbox_button.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_sbxbtn_"))

import code_sandbox as CS
from pathlib import Path
CS.SANDBOX_BASE = Path(tempfile.mkdtemp(prefix="tgtest_sbxbtn_root_"))

import sandbox_access as A
import tg_keyboards as K
from tg_strings import _b, _BTN

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


# -- the row ----------------------------------------------------------------
for lang in ("ru", "en"):
    rows = K._creativity_kb(lang)["keyboard"]
    row = [r for r in K._cr_music_kb(lang)["keyboard"] if _b("mashup", lang) in r]
    check(f"[{lang}] mashup row exists", bool(row), rows)
    if row:
        # Mashup sits with Songs in the Music tier; the sandbox stays on top.
        check(f"[{lang}] songs sit beside mashup", _b("songs", lang) in row[0], row[0])
    check(f"[{lang}] sandbox is in the creativity menu",
          any(_b("sandbox_btn", lang) in r for r in rows), rows)

check("the label is translated, not a raw key",
      _BTN["sandbox_btn"]["ru"] != _BTN["sandbox_btn"]["en"])


# -- the label routes -------------------------------------------------------
# _DIRECT_KB is keyed by string KEY; the dispatcher matches on the sentinel.
check("the button has a routing sentinel",
      T._DIRECT_KB.get("sandbox_btn") == "__sandbox__",
      T._DIRECT_KB.get("sandbox_btn"))


# -- pressing it reaches the sandbox handler --------------------------------
bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                    silent_mode=True)
sent = []
bot._api_post = lambda method, payload=None, **kw: (
    sent.append((method, payload or {})),
    {"ok": True, "result": {"message_id": len(sent)}})[1]

u = T._User(chat_id=77, name="t", status="approved")
u.prefs = {"sandbox": A.FILES}
bot._user_store.put(u)
CS.sandbox_for(77).write_text("notes.txt", "hi")

seen = {}
_orig = bot._handle_sandbox_command
def _rec(chat_id, sess, user, lang, cmd):
    seen["cmd"] = cmd
    return _orig(chat_id, sess, user, lang, cmd)
bot._handle_sandbox_command = _rec

for lang in ("ru", "en"):
    seen.clear(); sent.clear()
    sess = bot._get_session(77)
    sess.lang = lang
    # _resolve_and_push is where the button text becomes an action. Going in
    # through _dispatch would only park the message on the debounce queue and
    # hand it to a worker thread, which a synchronous test cannot observe.
    bot._resolve_and_push(77, [{"type": "text", "text": _b("sandbox_btn", lang)}])
    check(f"[{lang}] the press reaches the sandbox handler",
          seen.get("cmd") == "/files", seen)
    # The entries are inline buttons now (a tap downloads / opens), so the
    # file name lives in the keyboard, not the message text.
    txt = " ".join(p.get("text", "") + " " + str(p.get("reply_markup", ""))
                   for m, p in sent if m == "sendMessage")
    check(f"[{lang}] and the folder listing comes back",
          "notes.txt" in txt, txt[:200])

# The listing carries INLINE buttons, so the Creativity reply keyboard stays on
# screen -- and the session must say so. A keyboard and a menu that disagree is
# how a hidden pending_prefix survives and eats the next message.
check("the keyboard on screen and the session menu agree",
      bot._get_session(77).menu == "creativity" and
      not bot._get_session(77).pending_prefix,
      (bot._get_session(77).menu, bot._get_session(77).pending_prefix))

# A user without the grant gets the SAME refusal as /files, not a hint that a
# button exists for people who have it.
u2 = T._User(chat_id=88, name="n", status="approved")
bot._user_store.put(u2)
sent.clear()
bot._resolve_and_push(88, [{"type": "text", "text": _b("sandbox_btn", "ru")}])
btn_txt = " ".join(p.get("text", "") for m, p in sent if m == "sendMessage")
sent.clear()
sess88 = bot._get_session(88)
bot._handle_sandbox_command(88, sess88, bot._user_store.get(88), "ru", "/files")
cmd_txt = " ".join(p.get("text", "") for m, p in sent if m == "sendMessage")
check("an ungranted press refuses exactly like /files does",
      btn_txt.strip() == cmd_txt.strip(), (btn_txt[:80], cmd_txt[:80]))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(0 if BAD == 0 else 1)
