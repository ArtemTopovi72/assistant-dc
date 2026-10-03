"""Three defects found by driving the real bot through a long live chat.

1. THE SCOPED CONTEXT NEVER REACHED THE GRAPH. `_execute_task` builds a
   per-task Context (own facts, own working image, own cancel event) and then
   handed the task the app's PREBUILT graph — which `build_graph` had closed
   over the SHARED context. So every node and every tool ran against the shared
   one. Live symptom: the bot answered "я запомнил, что вас зовут Артём",
   `remember_fact` really was called, and 🧠 My facts was still empty one turn
   later — the fact had been written into the desktop app's context instead of
   the session's. The same hole applies to the working image across chats.

2. ⛔ STOP WAS IGNORED FOR THE TASK IN FLIGHT. The post-invoke guard consulted
   only the inline ⛔ button's cancelled-set, so a render that finished before
   the cooperative cancel was noticed sailed through to delivery. Live: Stop
   answered "Останавливаю…" and the picture arrived two minutes later.

3. 📷 ANALYZE ASKED FOR A PHOTO WITH THREE IN THE CHAT. It gated on the single
   `last_image_path` slot instead of the per-chat image register the buttons
   actually resolve against.

Run: venv/Scripts/python.exe tests/test_task_scoped_graph.py
"""
import os, sys, tempfile, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_scoped_")
import tg_bot as T
T.redirect_data_dir(_DATA)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999851


class FakeCtx:
    """Enough of a Context for _execute_task; records which copy tools saw."""
    def __init__(self):
        self.session_memory = []
        self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0
        self.tts_disabled = True
        self.memory_lock = threading.Lock()

    def memory_text(self): return ""
    def set_stage(self, s): pass


def make_bot(built, graph_obj):
    bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: graph_obj,
                        lambda: {"messages": []}, silent_mode=True)
    bot._api_post = lambda *a, **k: {"ok": True, "result": {"message_id": 5}}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: bot.sent.append(t)
    bot._send_get_id = lambda cid, t, **kw: (bot.sent.append(t), 5)[1]
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


# ── 1. the graph the task runs must be bound to the task's own context ───────
print("=" * 66)
print("THE TASK'S GRAPH IS BOUND TO THE TASK'S CONTEXT")
print("=" * 66)

SHARED = FakeCtx()
built_with = []
seen_ctx = []


class Graph:
    def __init__(self, ctx=None): self.ctx = ctx
    def invoke(self, base):
        seen_ctx.append(self.ctx)
        # what a tool does: write a fact into the context it was given
        if self.ctx is not None:
            self.ctx.pinned_facts.append({"ts": time.time(), "text": "имя: Артём"})
        return {"messages": [], "final_answer": "ок"}


PREBUILT = Graph(SHARED)          # what the desktop app hands over


def fake_build_graph(ctx):
    built_with.append(ctx)
    return Graph(ctx)


import graph as graph_mod
_real_build = graph_mod.build_graph
graph_mod.build_graph = fake_build_graph
try:
    bot = make_bot(built_with, PREBUILT)
    task = T._Task(task_id="t1", chat_id=CID, user_text="запомни: меня зовут Артём")
    bot._execute_task(task)
finally:
    graph_mod.build_graph = _real_build

check("a graph was rebuilt for the task", len(built_with) == 1, built_with)
# remember_fact persists via ctx.save_memory(ctx.active_memory_dir); inheriting
# the desktop app's directory wrote a Telegram user's facts into the owner's
# profile — and, with the scoped facts list holding only this chat's, OVERWROTE
# it with them.
_dir = str(getattr(built_with[0], "active_memory_dir", "")) if built_with else ""
check("the task's context points at a PER-CHAT memory directory",
      f"chat_{CID}" in _dir, _dir)
check("it was built on the task's OWN context, not the shared one",
      built_with and built_with[0] is not SHARED)
check("the prebuilt shared graph was NOT the one invoked",
      seen_ctx and seen_ctx[0] is not SHARED, seen_ctx)
check("the fact written by the tool did not land in the shared context",
      SHARED.pinned_facts == [], SHARED.pinned_facts)
sess = bot._get_session(CID)
check("the fact IS persisted on the session (🧠 My facts can see it)",
      any("Артём" in str(f.get("text", "")) for f in sess.get_tg_facts()),
      sess.get_tg_facts())

# a broken build must not take the turn down
print()
graph_mod.build_graph = lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    bot2 = make_bot([], PREBUILT)
    bot2._execute_task(T._Task(task_id="t2", chat_id=CID, user_text="привет"))
    check("a failed rebuild falls back instead of failing the turn", True)
except Exception as exc:
    check("a failed rebuild falls back instead of failing the turn", False, exc)
finally:
    graph_mod.build_graph = _real_build

# ── 2. Stop must suppress delivery of the task that was running ─────────────
print("=" * 66)
print("STOP SUPPRESSES THE RESULT OF THE TASK IN FLIGHT")
print("=" * 66)

import inspect
src = inspect.getsource(T.TelegramBot._run_task_inner)
check("the post-invoke guard consults BOTH cancel paths",
      "self._is_cancelled(task.task_id) or self._stop_requested_after(task)" in src)

bot3 = make_bot([], Graph(FakeCtx()))
t3 = T._Task(task_id="t3", chat_id=CID, user_text="нарисуй корабль")
time.sleep(0.01)
bot3._request_stop(CID)           # Stop arrives AFTER the task was enqueued
check("a Stop after enqueue marks the running task stale",
      bot3._stop_requested_after(t3))

# ── 3. 📷 Analyze counts the image register, not just the last-image slot ────
print("=" * 66)
print("ANALYZE SEES THE PICTURES ALREADY IN THE CHAT")
print("=" * 66)

src2 = inspect.getsource(T.TelegramBot._resolve_and_push)
check("the analyze gate consults the image register",
      "_live_images(sess)" in src2 and "_have_image" in src2)

bot4 = make_bot([], Graph(FakeCtx()))
sess4 = bot4._get_session(CID)
sess4.last_image_path = ""
png = os.path.join(_DATA, "pic.png")
open(png, "wb").write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
T._log_image(sess4, png, label="трактор", src="bot")
bot4._store.put(sess4)
bot4.sent.clear()
bot4._resolve_and_push(CID, [{"type": "text", "text": T._b("analyze", "ru")}])
check("with a picture in the register it does NOT ask for a photo",
      not any("Пришли фото" in s for s in bot4.sent), bot4.sent)

# with genuinely nothing, it must still arm and ask
bot5 = make_bot([], Graph(FakeCtx()))
s5 = bot5._get_session(CID)
s5.last_image_path = ""
s5.image_log = []
bot5._store.put(s5)
bot5.sent.clear()
bot5._resolve_and_push(CID, [{"type": "text", "text": T._b("analyze", "ru")}])
check("with no picture anywhere it still asks for one",
      any("Пришли фото" in s for s in bot5.sent), bot5.sent)

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
