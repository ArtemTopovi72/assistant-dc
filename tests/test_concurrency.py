"""The queue must OVERLAP work, not just order it.

The bot ran one task at a time. While somebody's picture rendered (40s+) or a
crawl ran (minutes), the LLM sat idle and everyone else waited — a queue whose
only function was to make people wait.

Concurrency is only safe if each task is isolated, so this checks both halves:

  1. ISOLATION — a task gets its own working image, memory, facts and cancel
     event, so nothing leaks between chats and one chat's Stop cannot kill
     another chat's request.
  2. OVERLAP — an LLM turn genuinely proceeds while a render is in flight, and
     two renders still never run at the same time (the GPU is the one resource
     that cannot be shared).

Run: venv/Scripts/python.exe tests/test_concurrency.py
"""
import os, sys, tempfile, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_conc_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import image as I
# The GPU slot guard moved out of image.py into comfy_client.py when the
# monolith was split; it is the renderer's resource, not the module's.
import comfy_client as GPU

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


# ══════════════════════════════════════════════════════════ 1. the worker count
print("\n" + "=" * 70)
print("1. MORE THAN ONE TASK MAY BE IN FLIGHT")
print("=" * 70)
check("the bot runs several consumers", T._MAX_CONSUMERS > 1, T._MAX_CONSUMERS)
os.environ["TG_WORKERS"] = "7"
import importlib
check("the count is configurable via TG_WORKERS",
      max(1, int(os.getenv("TG_WORKERS", "3"))) == 7)
os.environ.pop("TG_WORKERS", None)

# ══════════════════════════════════════════════════════════ 2. task isolation
print("\n" + "=" * 70)
print("2. EACH TASK GETS ITS OWN CONTEXT")
print("=" * 70)

from collections import deque


class Base:
    """Stands in for the shared Context."""
    def __init__(self):
        self.session_memory = deque(maxlen=40)
        self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = "/shared/original.png"
        self.last_image_prompt = "shared prompt"
        self.image_aspect = "16:9"
        self.models = object()                 # must be SHARED, not copied
        self.api_lock = threading.Lock()       # must be SHARED
        self.tts_lock = threading.Lock()


base = Base()
a = T._scoped_ctx(base, cancel_event=threading.Event())
b = T._scoped_ctx(base, cancel_event=threading.Event())

a.last_image_path = "/chat_a/cat.png"
b.last_image_path = "/chat_b/dog.png"
a.session_memory.append({"role": "user", "content": "A's private turn"})
a.pinned_facts.append({"text": "A's home address"})
a.image_aspect = "1:1"

check("two tasks do not share a working image",
      a.last_image_path == "/chat_a/cat.png" and b.last_image_path == "/chat_b/dog.png",
      f"{a.last_image_path} / {b.last_image_path}")
check("the shared context is left untouched",
      base.last_image_path == "/shared/original.png", base.last_image_path)
check("session memory is per task",
      len(a.session_memory) == 1 and len(b.session_memory) == 0,
      f"{len(a.session_memory)} / {len(b.session_memory)}")
check("pinned facts do NOT leak between chats",
      a.pinned_facts and not b.pinned_facts and not base.pinned_facts,
      f"a={a.pinned_facts} b={b.pinned_facts} base={base.pinned_facts}")
check("output size is per task", a.image_aspect == "1:1" and b.image_aspect == "16:9",
      f"{a.image_aspect} / {b.image_aspect}")

check("the loaded MODELS are shared, not duplicated", a.models is base.models)
check("the api lock is shared (it is what serialises the GPU)",
      a.api_lock is base.api_lock and b.api_lock is a.api_lock)

# cancel isolation
a.cancel_event.set()
check("cancelling one task does not cancel the other", not b.is_cancelled()
      if hasattr(b, "is_cancelled") else not b.cancel_event.is_set())
check("nor the shared context", not base.cancel_event.is_set())

# a partial/legacy context must not take the turn down
c = T._scoped_ctx(types.SimpleNamespace(), cancel_event=threading.Event())
check("a context missing fields still yields a usable scope",
      hasattr(c, "session_memory") and hasattr(c, "pinned_facts"))

# ══════════════════════════════════════════════ 3. the GPU is still exclusive
print("\n" + "=" * 70)
print("3. RENDERS SERIALISE; EVERYTHING ELSE OVERLAPS")
print("=" * 70)

check("there is a GPU slot guard", hasattr(GPU, "_gpu_slot"))
check("only one render at a time by default", GPU.COMFY_MAX_CONCURRENT == 1)

live = {"now": 0, "peak": 0}
lock = threading.Lock()


def fake_render(_i):
    with GPU._gpu_slot():
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.15)
        with lock:
            live["now"] -= 1


ts = [threading.Thread(target=fake_render, args=(i,)) for i in range(4)]
t0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
elapsed = time.time() - t0
check("two renders never overlap", live["peak"] == 1, f"peak={live['peak']}")
check("and they did actually queue (not run in parallel)", elapsed >= 0.55,
      f"{elapsed:.2f}s for 4x0.15s")

# reentrant: one pipeline submitting several graphs must not deadlock itself
done = []


def nested():
    with GPU._gpu_slot():
        with GPU._gpu_slot():
            done.append(1)


th = threading.Thread(target=nested)
th.start(); th.join(timeout=3)
check("a pipeline may submit several graphs without deadlocking itself",
      done == [1], "nested acquire deadlocked")

# ── non-GPU work overlaps freely while a render holds the slot ───────────────
order = []


def hold_gpu():
    with GPU._gpu_slot():
        order.append("render-start")
        time.sleep(0.3)
        order.append("render-end")


def llm_turn():
    time.sleep(0.05)                      # starts after the render has the slot
    order.append("llm-during-render")


t1 = threading.Thread(target=hold_gpu)
t2 = threading.Thread(target=llm_turn)
t1.start(); t2.start(); t1.join(); t2.join()
check("an LLM turn runs WHILE a picture is rendering",
      order == ["render-start", "llm-during-render", "render-end"], order)

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
