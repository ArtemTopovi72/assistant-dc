"""Regression tests for the 🎨 Draw submenu audit (audit_02_draw.md, 2026-08-03).

Five bugs fixed in tg_bot.py's `_resolve_and_push` / `_state_kb`:

1. A button pressed in the same debounce batch as another message used to
   `return` out of the loop and drop every sibling item (photo/typed prompt).
   Fixed: button branches `continue` instead of `return`.
2. 🔄 Regenerate with nothing to regenerate ran a full agent turn (image
   quota burnt). Fixed: same honest-refusal precondition as 📷 Analyze.
3. 📷 Analyze accepted a `last_image_path` that no longer exists on disk.
   Fixed: `os.path.exists` gate.
4. 📷 with no image dropped the user out of the Draw submenu because
   `_state_kb` only looked at `pending_prefix`, not `sess.menu`. Fixed:
   fall back to `sess.menu`.
5. An armed `pending_photo` never expired on an unrelated turn. Fixed:
   a genuine free-text turn disarms it.

Run: venv/Scripts/python.exe tests/test_draw_fixes.py
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_drawfix_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


class _FakeBackend:
    def __init__(self):
        self.pushed = []
    def push(self, task):
        self.pushed.append(task)
    def chat_depth(self, chat_id):
        return 0
    def depth(self):
        return len(self.pushed)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    bot._backend = _FakeBackend()
    return bot, sent


def approve(bot, chat_id, lang="ru"):
    u = T._User(chat_id=chat_id, name="U", tg_username="u", password_hash="x",
                status="approved")
    bot._user_store.put(u)
    sess = bot._get_session(chat_id)
    sess.lang = lang
    bot._store.put(sess)
    return u, sess


def enter_draw(bot, chat_id, lang="ru"):
    sess = bot._get_session(chat_id)
    sess.menu = "draw"
    sess.pending_prefix = "generate an image of: "
    bot._store.put(sess)
    return sess


def text_update(chat_id, text):
    return {"message": {"chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "language_code": "ru"}, "text": text}}


def photo_update(chat_id):
    return {"message": {"chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "language_code": "ru"},
            "photo": [{"file_id": "p1", "file_size": 10, "width": 10, "height": 10}]}}


def wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond(): return True
        time.sleep(0.1)
    return cond()


print("=" * 70)
print("1. BUTTON + SIBLING MESSAGE IN ONE DEBOUNCE BATCH — NOTHING IS DROPPED")
print("=" * 70)

bot, sent = make_bot()
CID = 8001
approve(bot, CID)
enter_draw(bot, CID)
lang = "ru"
batch = [
    {"type": "text", "text": T._b("gen_image", lang)},
    {"type": "text", "text": "счастливая собака"},
]
bot._resolve_and_push(CID, batch)
check("the typed prompt was NOT dropped — a task was pushed",
      len(bot._backend.pushed) == 1, bot._backend.pushed)
if bot._backend.pushed:
    check("the pushed task carries the generate prefix + the typed text",
          "generate an image of:" in bot._backend.pushed[0].user_text
          and "собака" in bot._backend.pushed[0].user_text,
          bot._backend.pushed[0].user_text)

print()
print("=" * 70)
print("1b. 📷 ANALYZE PRESSED, THEN THE PHOTO ARRIVES IN THE SAME BATCH")
print("=" * 70)

bot, sent = make_bot()
CID = 8002
approve(bot, CID)
enter_draw(bot, CID)
bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 200  # fake jpeg bytes
batch = [
    {"type": "text", "text": T._b("analyze", lang)},
    {"type": "photo", "file_id": "p1"},
]
bot._resolve_and_push(CID, batch)
check("the photo was NOT dropped — a task was pushed with the analyze prompt",
      len(bot._backend.pushed) == 1
      and "analyz" in bot._backend.pushed[0].user_text.lower(),
      bot._backend.pushed)
if bot._backend.pushed:
    check("the task carries the actual image bytes (image_path set)",
          bool(bot._backend.pushed[0].image_path), bot._backend.pushed[0])

print()
print("=" * 70)
print("2. 🔄 REGENERATE WITH NOTHING TO REGENERATE DOES NOT BURN A TASK")
print("=" * 70)

bot, sent = make_bot()
CID = 8003
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
sess.last_image_prompt = ""
sess.last_image_path = ""
bot._store.put(sess)
batch = [{"type": "text", "text": T._b("regenerate", lang)}]
bot._resolve_and_push(CID, batch)
check("no task was enqueued for an empty regenerate",
      len(bot._backend.pushed) == 0, bot._backend.pushed)
check("an honest refusal was sent instead",
      any("nothing" in str(p).lower() or "нечего" in str(p).lower()
          or "no_regenerate" in str(p).lower() for _, p in sent)
      or any(T._t("no_regenerate", lang) in str(p) for _, p in sent),
      sent)

print()
print("=" * 70)
print("2b. REGENERATE STILL WORKS WHEN THERE IS SOMETHING TO REGENERATE")
print("=" * 70)

bot, sent = make_bot()
CID = 8004
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
sess.last_image_prompt = "a red bicycle"
bot._store.put(sess)
batch = [{"type": "text", "text": T._b("regenerate", lang)}]
bot._resolve_and_push(CID, batch)
check("a real regenerate task was enqueued",
      len(bot._backend.pushed) == 1, bot._backend.pushed)

print()
print("=" * 70)
print("3. 📷 ANALYZE WITH A STALE (DELETED) last_image_path FALLS THROUGH")
print("=" * 70)

bot, sent = make_bot()
CID = 8005
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
_gone = os.path.join(tempfile.gettempdir(), "tgtest_gone_image_never_existed.jpg")
if os.path.exists(_gone): os.unlink(_gone)
sess.last_image_path = _gone
bot._store.put(sess)
batch = [{"type": "text", "text": T._b("analyze", lang)}]
bot._resolve_and_push(CID, batch)
check("no task fired against a vanished file",
      len(bot._backend.pushed) == 0, bot._backend.pushed)
check("pending_photo was armed instead (await-photo branch taken)",
      bool(bot._get_session(CID).pending_photo), bot._get_session(CID).pending_photo)

print()
print("=" * 70)
print("3b. 📷 ANALYZE WITH A LIVE REGISTER IMAGE STILL PROCEEDS (no regression)")
print("=" * 70)

bot, sent = make_bot()
CID = 8006
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
_real = os.path.join(tempfile.gettempdir(), "tgtest_real_image.jpg")
with open(_real, "wb") as f: f.write(b"\xff\xd8\xff" + b"0" * 50)
sess.last_image_path = _real
bot._store.put(sess)
batch = [{"type": "text", "text": T._b("analyze", lang)}]
bot._resolve_and_push(CID, batch)
check("a live last_image_path still triggers the analyze task",
      len(bot._backend.pushed) == 1, bot._backend.pushed)
try: os.unlink(_real)
except Exception: pass

print()
print("=" * 70)
print("4. 📷 WITH NO IMAGE DOES NOT DROP THE USER OUT OF THE DRAW SUBMENU")
print("=" * 70)

bot, sent = make_bot()
CID = 8007
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
sess.last_image_path = ""
bot._store.put(sess)
batch = [{"type": "text", "text": T._b("analyze", lang)}]
bot._resolve_and_push(CID, batch)
sess = bot._get_session(CID)
check("sess.menu is still 'draw' after the no-image analyze branch",
      sess.menu == "draw", sess.menu)
kb = bot._state_kb(sess, lang)
draw_kb = T._draw_kb(lang)
check("_state_kb returns the DRAW keyboard, not the main menu",
      kb == draw_kb, kb)

print()
print("=" * 70)
print("5. AN ARMED 📷 EXPIRES ON AN UNRELATED TURN")
print("=" * 70)

bot, sent = make_bot()
CID = 8008
approve(bot, CID)
enter_draw(bot, CID)
sess = bot._get_session(CID)
sess.last_image_path = ""
bot._store.put(sess)
# Arm it.
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("analyze", lang)}])
check("pending_photo is armed", bool(bot._get_session(CID).pending_photo))
# An unrelated free-text turn should disarm it.
bot._resolve_and_push(CID, [{"type": "text", "text": "какая погода в Москве"}])
check("pending_photo was disarmed by the unrelated turn",
      not bot._get_session(CID).pending_photo, bot._get_session(CID).pending_photo)
# A later caption-less photo must NOT be force-analyzed.
bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 200
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "p2"}])
check("the later photo was pushed as a plain 'image' turn, not force-analyzed",
      len(bot._backend.pushed) >= 1
      and "analyz" not in bot._backend.pushed[-1].user_text.lower(),
      bot._backend.pushed)

print()
print("=" * 70)
print("END-TO-END: REAL DEBOUNCE WINDOW, BUTTON THEN FAST TYPED PROMPT")
print("=" * 70)

bot, sent = make_bot()
CID = 8009
approve(bot, CID)
enter_draw(bot, CID)
bot._dispatch(text_update(CID, T._b("gen_image", lang)))
bot._dispatch(text_update(CID, "маленький робот"))
ok = wait_for(lambda: len(bot._backend.pushed) >= 1, timeout=6.0)
check("real dispatch + debounce still delivers the fast-typed prompt",
      ok and len(bot._backend.pushed) == 1
      and "маленький робот" in bot._backend.pushed[0].user_text,
      bot._backend.pushed)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
