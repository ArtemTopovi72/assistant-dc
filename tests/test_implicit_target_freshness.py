"""H2b: an implicit follow-up binds to the NEWEST delivered picture.

«проверь картинку: попугай на плече» after the wolf was delivered answered
about a deer from much earlier (live 2026-11-01, 4/4 in the bug hunt): queue
races leave sess.turn_image stale, and the implicit binding inherited it.
Now a follow-up with no explicit pointer (no button press, no ordinal, no
reply) is rebound to the newest bot-delivered picture. Explicit pointers stay
frozen; a user-sent photo stays theirs.

Run: venv/Scripts/python.exe tests/test_implicit_target_freshness.py
"""
RUN_AS_SCRIPT = True
import os, sys, tempfile, time, types, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_fresh_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {str(extra)[:200]}")

CID = 999_917
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_fresh_imgs_")

def make_png(name):
    p = os.path.join(IMG_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
    return p

class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event(); self.last_image_path = ""
        self.last_image_prompt = ""; self.stage_callback = None
        self.total_user_turns = 0; self.tts_disabled = True
        self.memory_lock = threading.Lock(); self.image_undo = []
    def memory_text(self): return ""
    def set_stage(self, s): pass

INVOKES = []
class StubGraph:
    def __init__(self, ctx): self.ctx = ctx
    def invoke(self, base):
        INVOKES.append(base.get("user_input", ""))
        return {"final_answer": "echo", "messages": base.get("messages", [])}

SHARED = FakeCtx()
bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: object(),
                    lambda: {"messages": []}, silent_mode=True)
bot.graph_builder = lambda ctx: StubGraph(ctx)
bot._backend = T.InMemoryBackend(); bot.sent = []; bot.pushed = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
bot._send_photo = lambda cid, p, **kw: True
bot._edit_text = lambda *a, **kw: None; bot._delete = lambda *a, **kw: None
bot._api_post = lambda *a, **k: {}; bot._api_get = lambda *a, **k: {}
ACT = []
bot._activity.log = lambda cid, kind, text="", *a, **k: ACT.append((kind, text))
_orig_push = bot._backend.push
def _push(t):
    bot.pushed.append(t)
    return _orig_push(t)
bot._backend.push = _push
bot._running = True
threading.Thread(target=bot._consumer_loop, daemon=True).start()

u = T._User(chat_id=CID, name="Tester", status="approved")
bot._user_store.put(u)
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"
bot._store.put(sess)

def target_line():
    return next((t for k, t in reversed(ACT) if k == "system" and t.startswith("[target]")), "")

def send_text(text, msg_id=902):
    ACT.clear()
    bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                               "message_id": msg_id, "date": int(time.time()),
                               "text": text}})
    deadline = time.monotonic() + 15
    base = len(INVOKES)
    while time.monotonic() < deadline:
        if len(INVOKES) != base:
            time.sleep(0.4)
            return
        time.sleep(0.05)

print("=" * 70)
print("the stale-deer scenario: implicit follow-up -> newest delivered")
print("=" * 70)
deer = make_png("deer.png")
wolf = make_png("wolf.png")
id_deer = T._log_image(sess, deer, msg_id=11, label="олень", src="bot")
id_wolf = T._log_image(sess, wolf, msg_id=12, label="волк", src="bot")
bot._store.put(sess)

# the queue race: turn_image still points at the OLD delivery
sess = bot._get_session(CID); sess.turn_image = id_deer; bot._store.put(sess)
send_text("проверь картинку: попугай на плече")
line = target_line()
check("rebound to the newest delivered picture (wolf)",
      "wolf.png" in line, line)
check("not the stale deer", "deer.png" not in line, line)

print("=" * 70)
print("explicit pointers stay frozen")
print("=" * 70)
# a button-press style task: image_id frozen at press time
class _T:
    pass
from tg_tasks import _target_id_of
fake_task = types.SimpleNamespace(image_id=id_deer, image_selection="context")
check("an explicit image_id is never rebound",
      _target_id_of(fake_task, bot._get_session(CID)) == id_deer)

print("=" * 70)
print("a user-sent photo stays the target (no bot rebound)")
print("=" * 70)
mine = make_png("mine.png")
id_mine = T._log_image(sess, mine, msg_id=13, label="моя", src="user")
sess = bot._get_session(CID); sess.turn_image = id_mine; bot._store.put(sess)
send_text("что на этой картинке?", msg_id=903)
line = target_line()
# with three live pictures and ambiguous words the bot ASKS which one (the
# standing convention) — the important part is it does not silently grab a
# bot picture over the user's current photo.
check("no silent grab of a bot picture when the user's photo is current",
      "wolf.png" not in line and "deer.png" not in line, line)

print("=" * 70)
print("already-fresh binding is untouched")
print("=" * 70)
sess = bot._get_session(CID); sess.turn_image = id_wolf; bot._store.put(sess)
send_text("проверь: волк в комикс-стиле?", msg_id=904)
line = target_line()
check("the newest picture stays the target when turn_image is fresh",
      "wolf.png" in line, line)

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
