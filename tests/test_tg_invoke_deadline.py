"""The per-turn deadline must not kill a task that is merely WAITING for the GPU.

Live bug, reported twice by the user with screenshots: a picture request came
back "⏱ Запрос выполнялся слишком долго и был отменён. Попробуй ещё раз."
about five minutes after it was sent.

Nothing was stuck. `TG_INVOKE_TIMEOUT_S` was a flat 300s cap on the whole agent
turn, and the GPU was busy with a MiniMax Music3 render -- ~70 minutes of
autoregressive sampling (observed: "AR sampling: 42%|4 | 2491/6001
[30:28<42:57]"). The picture was queued behind it, spent its entire 300s
waiting for a slot it never got, and was abandoned before running a single
step. The advice to retry was actively wrong: retrying rejoined the same queue.

A deadline exists to free a consumer thread from a HUNG call. Waiting your turn
is not hanging. So the deadline now extends for as long as the render server is
genuinely working, capped by TG_TASK_MAX_S.

Offline: the "server" is a stub, no ComfyUI, no GPU, no network.

Run: venv/Scripts/python.exe tests/test_tg_invoke_deadline.py
"""
import sys, os, time, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import comfy_client

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else: BAD += 1; print(f"FAIL  {name}   {extra}")


def wait_like_tg_tasks(thread, invoke_timeout, hard_ceiling, busy_fn):
    """The waiting policy from tg_tasks._run_task_inner, isolated.

    Kept in lock-step with the source by test_deadline_policy_matches_source
    below, which reads the real file.
    """
    STEP = min(15.0, max(0.05, invoke_timeout / 4.0))
    waited = 0.0
    while True:
        thread.join(timeout=min(STEP, max(0.01, hard_ceiling - waited)))
        if not thread.is_alive():
            return waited, "finished"
        waited += STEP
        if waited >= hard_ceiling:
            return waited, "ceiling"
        if waited >= invoke_timeout:
            try:
                if not busy_fn():
                    return waited, "deadline"
            except Exception:
                return waited, "deadline"


def worker(seconds):
    t = threading.Thread(target=lambda: time.sleep(seconds), daemon=True)
    t.start()
    return t


print("=" * 70)
print("A BUSY RENDER SERVER EXTENDS THE DEADLINE")
print("=" * 70)

t = worker(1.2)
waited, why = wait_like_tg_tasks(t, invoke_timeout=0.4, hard_ceiling=10.0,
                                 busy_fn=lambda: True)
check("a task outliving the base deadline is NOT abandoned while the GPU works",
      why == "finished", f"stopped because {why} after {waited:.2f}s")
check("...and it was actually held past the base deadline", waited >= 0.4, waited)

print()
print("=" * 70)
print("AN IDLE SERVER STILL LETS THE DEADLINE FIRE")
print("=" * 70)

t = worker(30)
t0 = time.time()
waited, why = wait_like_tg_tasks(t, invoke_timeout=0.4, hard_ceiling=10.0,
                                 busy_fn=lambda: False)
elapsed = time.time() - t0
check("a genuinely stuck call is still abandoned", why == "deadline", why)
check("...promptly, near the base deadline, not at the ceiling",
      elapsed < 3.0, f"{elapsed:.2f}s")

print()
print("=" * 70)
print("THE HARD CEILING STILL BOUNDS AN ENDLESSLY 'BUSY' SERVER")
print("=" * 70)
print("""
A server that reports busy forever must not hold a consumer thread forever --
otherwise one wedged render silently costs a worker permanently.
""")

t = worker(30)
t0 = time.time()
waited, why = wait_like_tg_tasks(t, invoke_timeout=0.2, hard_ceiling=1.0,
                                 busy_fn=lambda: True)
elapsed = time.time() - t0
check("an always-busy server is cut off at the ceiling", why == "ceiling", why)
check("...at the ceiling, not beyond it", elapsed < 3.0, f"{elapsed:.2f}s")

print()
print("=" * 70)
print("AN UNREACHABLE SERVER MUST NOT LOOK 'BUSY'")
print("=" * 70)
print("""
server_busy() is conservative on purpose: if ComfyUI cannot be reached at all,
reporting it busy would extend every task to the ceiling during an outage.
""")

_real_get = comfy_client.requests.get
class _Down:
    def __call__(self, *a, **k):
        raise comfy_client.requests.exceptions.ConnectionError("refused")
comfy_client.requests.get = _Down()
try:
    check("an unreachable ComfyUI reports NOT busy", comfy_client.server_busy() is False)
finally:
    comfy_client.requests.get = _real_get

class _Resp:
    def __init__(self, d): self._d = d
    def json(self): return self._d
comfy_client.requests.get = lambda *a, **k: _Resp({"queue_running": [], "queue_pending": []})
try:
    check("an idle queue reports NOT busy", comfy_client.server_busy() is False)
finally:
    comfy_client.requests.get = _real_get
comfy_client.requests.get = lambda *a, **k: _Resp({"queue_running": [["x"]], "queue_pending": []})
try:
    check("a running job reports busy", comfy_client.server_busy() is True)
finally:
    comfy_client.requests.get = _real_get
comfy_client.requests.get = lambda *a, **k: _Resp({"queue_running": [], "queue_pending": [["y"]]})
try:
    check("a QUEUED job also counts as busy (our turn has not come)",
          comfy_client.server_busy() is True)
finally:
    comfy_client.requests.get = _real_get

print()
print("=" * 70)
print("A TIMEOUT IS THE OPPOSITE SIGNAL FROM AN UNREACHABLE SERVER")
print("=" * 70)
print("""
Live, 2026-09-20: a 41-minute H3 video render (near-100% GPU, heavy VRAM
offload) had this /queue poll time out ONCE, deep into an otherwise-healthy
render that finished 40 seconds later. Before this fix, a Timeout was caught
by the same bare `except Exception` as a connection refusal and read as "not
busy" -- the caller's extend-while-busy loop abandoned the task 9 seconds
before the real video was ready, and the user got "cancelled, try again" for
a render that had already succeeded. A server too saturated to answer a
lightweight status check in time IS busy; only a genuine connection failure
means it might be down.
""")


class _Timeout:
    def __call__(self, *a, **k):
        raise comfy_client.requests.exceptions.Timeout("read timed out")
comfy_client.requests.get = _Timeout()
try:
    check("a TIMEOUT (server saturated, not down) reports BUSY",
          comfy_client.server_busy() is True)
finally:
    comfy_client.requests.get = _real_get

print()
print("=" * 70)
print("A CHAIN HOLDING THE CARD IS WORKING, EVEN WITH AN EMPTY QUEUE")
print("=" * 70)
print("""
Live, 2026-10-07: a four-part Herrgott video chain was abandoned at 30 min --
between two parts ComfyUI's queue is empty for a moment, and the deadline read
that as idle. A render pipeline holding the card is not a hung call.
""")
comfy_client.requests.get = lambda *a, **k: _Resp({"queue_running": [], "queue_pending": []})
try:
    check("no card held: card_in_use is False", comfy_client.card_in_use() is False)
    with comfy_client.card_session("video chain"):
        check("inside a card_session the card is in use (queue empty)",
              comfy_client.card_in_use() is True)
    check("...and free again after it", comfy_client.card_in_use() is False)
finally:
    comfy_client.requests.get = _real_get

print()
print("=" * 70)
print("THE SHIPPED POLICY MATCHES THE ONE TESTED ABOVE")
print("=" * 70)

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "bot/tg_tasks.py"), encoding="utf-8").read()
check("tg_tasks consults server_busy before abandoning", "server_busy()" in src)
check("tg_tasks also counts a held card as working", "card_in_use()" in src)
check("tg_tasks bounds the extension by TG_TASK_MAX_S", "TG_TASK_MAX_S" in src)
check("the base deadline is no longer a flat 300s",
      'TG_INVOKE_TIMEOUT_S", 300)' not in src)

import config
check("config default is generous enough for a real render",
      config.TG_INVOKE_TIMEOUT_S >= 900, config.TG_INVOKE_TIMEOUT_S)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
