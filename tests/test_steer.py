"""What the user says WHILE the bot works changes the work (steer.py).

Live 2026-09-18: «прошу график, вижу, что он пишет код, вкидываю "фон пусть
будет красный" -- и он на ближайшем свободном моменте: ага, учту». A text
that arrives during a running task goes into that task's inbox, the user gets
«👌 Учту: …» at once, the agent loop folds the note into its next round, and
a note the task never read is re-queued as an ordinary request.
"""
import os, sys, types, threading
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.STUB = lambda t: {"is_question": t.rstrip().endswith("?")}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer as S

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# --- who steers ------------------------------------------------------------------
check("a plain text during a task steers", S.eligible("а фон пусть будет красный", False, False))
check("a photo does not (it is its own request)", not S.eligible("вот", True, False))
check("a command does not", not S.eligible("/cancel", False, False))
check("a question during a slow phase is answered on the side", not S.eligible("сколько времени?", False, True))
check("...but the same question during a fast phase steers", S.eligible("сколько времени?", False, False))
check("an essay does not", not S.eligible("x" * 500, False, False))
check("the RU ack quotes the words", S.ack("фон красный", "ru") == "👌 Учту: «фон красный»")
check("the EN ack", S.ack("red background", "en").startswith("👌 Noted:"))
check("a long note is shortened in the ack", len(S.ack("слово " * 40, "ru")) < 100)

# --- the inbox and the fold -------------------------------------------------------
ib = S.Inbox(); ib.put("фон красный"); ib.put("и подпиши оси")
check("unread before the drain", ib.unread() == ["фон красный", "и подпиши оси"])
msgs = [{"role": "user", "content": "нарисуй график"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "[done] script written"}]
ctx = types.SimpleNamespace(steer_inbox=ib)
n = S.drain_into(ctx, msgs)
check("both notes are folded into the last tool result (template-safe)",
      n == 2 and msgs[-1]["role"] == "tool" and "фон красный" in msgs[-1]["content"]
      and "подпиши оси" in msgs[-1]["content"] and "UPDATE FROM THE USER" in msgs[-1]["content"], msgs[-1])
check("drained notes are consumed, not unread", ib.unread() == [] and ib.consumed == ["фон красный", "и подпиши оси"])
check("nothing to drain -> 0 and untouched", S.drain_into(ctx, msgs) == 0 and msgs[-1]["content"].count("UPDATE") == 1)
m2 = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plan"}]
S.fold_into(m2, ["red"])
check("after a bare assistant message the note is a user turn", m2[-1]["role"] == "user" and "red" in m2[-1]["content"])
check("no inbox (desktop) -> 0", S.drain_into(types.SimpleNamespace(), msgs) == 0)

# --- through the real agent loop ------------------------------------------------------
import test_agent_loop_hardening as H
import tools as tools_mod
import graph as graph_mod
spec = tools_mod._BY_NAME["generate_image"]; orig = spec.handler
ctx = H._ctx(); ctx.steer_inbox = S.Inbox()
seen_prompts = []
def _handler(c, s, a):
    seen_prompts.append(str(a))
    if len(seen_prompts) == 1:
        c.steer_inbox.put("фон пусть будет красный")   # the user types while the tool runs
    return "[done] drawn"
object.__setattr__(spec, "handler", _handler)
round_inputs = []
_orig_send = H._send
def _send(c, messages, tools=None, **k):
    round_inputs.append([m.get("content") or "" for m in messages if m.get("role") == "tool"])
    return _orig_send(c, messages, tools=tools, **k)
graph_mod.send_to_lm_studio = _send
try:
    def fn(i):
        if i == 1:
            return H._msg("", tcs=[H._tc("generate_image", {"prompt": "a chart"}, "g1")])
        if i == 2:
            # a model that read the update redraws with the change
            return H._msg("", tcs=[H._tc("generate_image", {"prompt": "a chart, red background"}, "g2")])
        return H._msg("Готово: график с красным фоном.")
    H._script["fn"] = fn; H._script["n"] = 0
    g = graph_mod.build_graph(ctx)
    st = H._base_state(); st["user_input"] = "draw a chart"
    final = g.invoke(st)
finally:
    object.__setattr__(spec, "handler", orig)
    graph_mod.send_to_lm_studio = _orig_send
check("the next round's tool result carries the user's update",
      len(round_inputs) >= 2 and any("красный" in t for t in round_inputs[1]), round_inputs[1:2])
check("the work changed: the second draw has the red background", len(seen_prompts) >= 2 and "red" in seen_prompts[1], seen_prompts)
check("the turn ends with an answer", bool((final.get("final_answer") or "").strip()))

# --- a wish that lands while the model is planning: the stale call is not run ------------
ctx = H._ctx(); ctx.steer_inbox = S.Inbox()
seen_prompts = []
object.__setattr__(spec, "handler", lambda c, s, a: (seen_prompts.append(str(a)), "[done] drawn")[1])
def _send2(c, messages, tools=None, **k):
    out = _orig_send(c, messages, tools=tools, **k)
    if H._script["n"] == 1:
        c.steer_inbox.put("сделай в стиле аниме")        # typed while round 1 was thinking
    return out
graph_mod.send_to_lm_studio = _send2
try:
    H._script["fn"] = fn; H._script["n"] = 0
    g = graph_mod.build_graph(ctx)
    st = H._base_state(); st["user_input"] = "draw a chart"
    g.invoke(st)
finally:
    object.__setattr__(spec, "handler", orig)
    graph_mod.send_to_lm_studio = _orig_send
check("the call planned before the wish is skipped; only the re-issued one runs",
      len(seen_prompts) == 1 and "red" in seen_prompts[0], seen_prompts)

# --- the bot side: _try_steer and the re-queue --------------------------------------------
import tg_tasks
class _Bot:
    def __init__(self):
        self._task_lock = threading.Lock(); self._running_task = {}; self._steer_inboxes = {}
        self._chat_interruptible = {}; self.sent = []; self.queued = []; self.logged = []
        self._store = types.SimpleNamespace(put=lambda s: None)
        self._activity = types.SimpleNamespace(log=lambda *a: self.logged.append(a))
        self._user_store = types.SimpleNamespace(get=lambda cid: None)
    def _get_session(self, cid): return types.SimpleNamespace(lang="ru")
    def _lang(self, sess): return "ru"
    def _send_text(self, cid, text, **k): self.sent.append(text)
    def _enqueue_item(self, cid, item): self.queued.append(item)
    _try_steer = tg_tasks._TaskMixin._try_steer if hasattr(tg_tasks, "_TaskMixin") else None
b = _Bot()
_try = getattr(tg_tasks, "_TaskMixin", None)
if _try is None:
    # the methods live on whatever class tg_tasks defines; bind them by name
    _cls = next(v for v in vars(tg_tasks).values() if isinstance(v, type) and hasattr(v, "_try_steer"))
else:
    _cls = _try
task_running = types.SimpleNamespace(task_id="run1", chat_id=7, delivered=False)
b._running_task[7] = [task_running]; b._steer_inboxes["run1"] = S.Inbox()
arrival = types.SimpleNamespace(task_id="new1", user_text="а фон красный", image_path="", image_id="")
check("a text during a running task is taken into its inbox and acked",
      _cls._try_steer(b, 7, arrival) is True and b._steer_inboxes["run1"].unread() == ["а фон красный"]
      and b.sent == ["👌 Учту: «а фон красный»"], (b.sent, b.logged))
check("nothing running -> not taken (queues as usual)",
      _cls._try_steer(b, 8, arrival) is False)
b._chat_interruptible[7] = True
q = types.SimpleNamespace(task_id="new2", user_text="сколько времени?", image_path="", image_id="")
check("a question during a slow phase is not taken (side answer)", _cls._try_steer(b, 7, q) is False)
_cls._requeue_unread_steer(b, task_running)
check("an unread note is re-queued as an ordinary text when the task ends",
      b.queued == [{"type": "text", "text": "а фон красный"}] and "run1" not in b._steer_inboxes, b.queued)

# --- the wiring ------------------------------------------------------------------------------
src = open(tg_tasks.__file__, encoding="utf-8").read()
check("every task gets an inbox", "ctx.steer_inbox = _steer.Inbox()" in src)
check("the re-queue runs in the task's finally (failures too)",
      src.index("self._requeue_unread_steer(task)") < src.index("lst = self._running_task.get(chat_id)"))
check("the push site tries to steer first",
      "if self._try_steer(chat_id, task):" in open("bot/tg_resolve.py", encoding="utf-8").read())
check("the loop drains before every model call",
      "_steer.drain_into(ctx, messages)" in open("agent/graph_personality.py", encoding="utf-8").read())

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
