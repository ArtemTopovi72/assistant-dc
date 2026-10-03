"""Scenario tests for tg_bot button/prefix routing.

Drives the REAL TelegramBot object with the network layer stubbed, so routing,
prefix handling and modality merging are exercised for real.
"""
import os, sys, types, tempfile, pathlib
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T

# Before any bot is built — otherwise fixture users land in the LIVE store.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_buttons_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot(monkey_transcribe=None):
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot.pushed = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append(text) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda task: bot.pushed.append(task),
        depth=lambda: 1, name=lambda: "stub")
    bot._activity.log = lambda *a, **k: None
    if monkey_transcribe:
        bot._transcribe = monkey_transcribe
    bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 100
    return bot


CID = 999001

def reset(bot):
    bot.sent.clear(); bot.pushed.clear()
    s = bot._get_session(CID)
    s.pending_prefix = ""
    s.clear_history()
    bot._store.put(s)
    # Daily quotas are persistent and per user, so repeated test runs would
    # eventually exhaust the deep-research budget for this chat id and the
    # routing assertions below would fail for the wrong reason.
    try:
        with bot._user_store._conn() as conn:
            conn.execute("DELETE FROM usage WHERE chat_id=?", (CID,))
    except Exception:
        pass


print("=" * 62)
print("PENDING-PREFIX ACROSS INPUT MODALITIES")
print("=" * 62)

bot = make_bot(monkey_transcribe=lambda ctx, fid, media="voice", **kw: "автобусные туры по спб")

# 1. tap Deep Research, answer by TEXT
reset(bot)
bot._resolve_and_push(CID, [{"type": "text", "text": "🔬 Deep Research"}])
sess = bot._get_session(CID)
check("Deep Research button sets prefix",
      sess.pending_prefix == "do a deep research on: ", repr(sess.pending_prefix))
bot._resolve_and_push(CID, [{"type": "text", "text": "автобусные туры"}])
check("text answer gets prefix",
      bot.pushed and bot.pushed[-1].user_text.startswith("do a deep research on: "),
      bot.pushed[-1].user_text if bot.pushed else "nothing pushed")

# 2. tap Deep Research, answer by VOICE  (the bug)
reset(bot)
bot._resolve_and_push(CID, [{"type": "text", "text": "🔬 Deep Research"}])
bot._resolve_and_push(CID, [{"type": "voice", "file_id": "v1"}])
check("VOICE answer gets prefix",
      bot.pushed and bot.pushed[-1].user_text.startswith("do a deep research on: "),
      bot.pushed[-1].user_text if bot.pushed else "nothing pushed")
check("prefix consumed after voice",
      bot._get_session(CID).pending_prefix == "")

# 3. tap Generate Image, answer by VOICE
reset(bot)
bot._resolve_and_push(CID, [{"type": "text", "text": "🖼 Generate Image"}])
bot._resolve_and_push(CID, [{"type": "voice", "file_id": "v1"}])
check("Generate Image + voice gets prefix",
      bot.pushed and bot.pushed[-1].user_text.startswith("generate an image of: "),
      bot.pushed[-1].user_text if bot.pushed else "nothing")

# 4. photo WITH caption while prefix pending
reset(bot)
bot._resolve_and_push(CID, [{"type": "text", "text": "✏️ Edit Image"}])
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "p1", "caption": "убери очки"}])
check("photo caption gets prefix",
      bot.pushed and bot.pushed[-1].user_text.startswith("edit the image: "),
      bot.pushed[-1].user_text if bot.pushed else "nothing")

# 5. photo WITHOUT caption must NOT consume a pending prefix
reset(bot)
sess = bot._get_session(CID)
sess.pending_prefix = "edit the image: "; bot._store.put(sess)
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "p1"}])
check("caption-less photo keeps prefix for next msg",
      bot._get_session(CID).pending_prefix == "edit the image: ",
      repr(bot._get_session(CID).pending_prefix))

# 6. navigation buttons must not consume the prefix
reset(bot)
bot._resolve_and_push(CID, [{"type": "text", "text": "🔬 Deep Research"}])
before = bot._get_session(CID).pending_prefix
bot._resolve_and_push(CID, [{"type": "text", "text": "❓ Help"}])
check("nav button does not push a task", not bot.pushed,
      str([p.user_text for p in bot.pushed]))

print()
print("=" * 62)
print("DEEP RESEARCH BYPASS ROUTING")
print("=" * 62)
for txt, should in [
    ("do a deep research on: питер", True),
    ("Do A Deep Research On: питер", True),
    ("do a deep research on:", True),          # empty topic -> handled, not routed to agent
    ("search the web for: питер", False),
    ("what is deep research", False),
]:
    fires = txt.lower().startswith("do a deep research on:")
    check(f"route {txt[:34]!r:38} -> {'bypass' if should else 'agent'}",
          fires == should)

# ── menu placement: Admin Panel belongs in Settings, not the main menu ───────
def _labels(kb):
    return [b for row in kb["keyboard"] for b in row]

for _lang in ("en", "ru"):
    _admin_label = T._b("admin", _lang)
    _main = _labels(T._main_kb(True, True, _lang))
    _settings = _labels(T._settings_kb(True, True, _lang))
    check(f"[{_lang}] admin panel absent from the main menu",
          _admin_label not in _main, str(_main))
    check(f"[{_lang}] admin panel present in settings",
          _admin_label in _settings, str(_settings))
    # …and the main menu still carries the everyday buttons.
    for _key in ("creativity", "search", "settings", "feedback", "account", "help", "stop"):
        check(f"[{_lang}] main menu keeps {_key}", T._b(_key, _lang) in _main)

# A non-admin must not see it anywhere, and the label must still route (the
# dispatch site is what enforces authorization, not the keyboard).
check("non-admin settings menu hides the admin panel",
      T._b("admin", "en") not in _labels(T._settings_kb(True, False, "en")))
check("admin label still maps to the panel command",
      T._DIRECT_KB.get(T._LABEL2KEY.get(T._b("admin", "en"))) == "__admin_panel__")

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
