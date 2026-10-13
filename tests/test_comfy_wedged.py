"""A WEDGED ComfyUI must fail fast, not burn the user's whole request.

ComfyUI can reach a state (seen after three days up) where it still accepts
the TCP connection but never answers. Every comfy call therefore paid
its FULL read timeout (the /prompt POST alone is (connect=10, read=120)), and a
draw makes several calls in a row, so the request ran ~5 minutes and then
surfaced to the user as "the request took too long and was cancelled" -- which
blamed slowness for what was actually a dead engine.

Two defects made that possible:

  1. No liveness probe. Every entry point dived straight into the 120s-read
     submit, and took the GPU slot before finding out.
  2. The poll loop's 60s outage grace caught only ConnectionError. requests'
     ReadTimeout does NOT subclass ConnectionError (it is Timeout ->
     RequestException -> OSError), so a wedged server -- as opposed to an
     absent one -- skipped the grace entirely and was retried until the full
     wall-clock timeout (up to 31 min).

Offline by construction: requests is stubbed, nothing touches the network.

Run: venv/Scripts/python.exe tests/test_comfy_wedged.py
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import requests
import comfy_client as C

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


def reset_health():
    C._health["ok"], C._health["at"] = None, 0.0


class WedgedServer:
    """Accepts the socket, never answers -> ReadTimeout on every call."""
    def __init__(self):
        self.calls = 0
    def __call__(self, *a, **kw):
        self.calls += 1
        raise requests.exceptions.ReadTimeout("read timed out")


print("=" * 66)
print("THE EXCEPTION HIERARCHY THAT CAUSED THIS")
print("=" * 66)

check("ReadTimeout is NOT a ConnectionError (the whole trap)",
      not issubclass(requests.exceptions.ReadTimeout,
                     requests.exceptions.ConnectionError))
check("...but it IS a Timeout, so catching Timeout covers it",
      issubclass(requests.exceptions.ReadTimeout, requests.exceptions.Timeout))
check("a genuinely absent server still raises a ConnectionError subclass",
      issubclass(requests.exceptions.ConnectTimeout,
                 requests.exceptions.ConnectionError))

print()
print("=" * 66)
print("A WEDGED SERVER IS DETECTED, AND FAST")
print("=" * 66)

import types
wedged = WedgedServer()
WEDGED = types.SimpleNamespace(get=wedged, post=wedged)   # the server, as the client is handed it
reset_health()
try:
    t0 = time.time()
    healthy = C.server_healthy(http=WEDGED)
    elapsed = time.time() - t0
finally:
    pass

check("server_healthy() reports a wedged server as unhealthy", healthy is False)
check("...and does so promptly, not after the submit read timeout",
      elapsed < 5.0, f"{elapsed:.2f}s")

print()
print("=" * 66)
print("THE PROBE IS CACHED (a job makes many calls)")
print("=" * 66)

reset_health()
try:
    C.server_healthy(http=WEDGED)
    after_first = wedged.calls
    for _ in range(5):
        C.server_healthy(http=WEDGED)
    after_more = wedged.calls
    forced = C.server_healthy(force=True, http=WEDGED)
    after_forced = wedged.calls
finally:
    pass

check("five further probes inside the TTL cost zero extra requests",
      after_more == after_first, f"{after_first} -> {after_more}")
check("force=True bypasses the cache", after_forced == after_more + 1)
check("the forced probe still reports unhealthy", forced is False)

print()
print("=" * 66)
print("SUBMIT ABANDONS EARLY -- WITHOUT TAKING THE GPU SLOT")
print("=" * 66)

slot_taken = {"n": 0}
class _CountingSlot:
    def __enter__(self): slot_taken["n"] += 1; return self
    def __exit__(self, *a): return False

_W = types.SimpleNamespace(deps=types.SimpleNamespace(requests=WEDGED, _gpu_slot=lambda *a, **kw: _CountingSlot()))
reset_health()
try:
    t0 = time.time()
    result = C._submit_and_poll(_W, {"1": {}}, timeout=1900, label="draw")
    elapsed = time.time() - t0
    reset_health()
    result2 = C._submit_and_collect(_W, {"1": {}}, timeout=1900, label="draw")
finally:
    pass

check("_submit_and_poll returns None against a wedged server", result is None)
check("_submit_and_collect returns None too", result2 is None)
check("...in seconds, not the 120s submit read timeout",
      elapsed < 10.0, f"{elapsed:.2f}s")
check("the GPU slot was never taken for a job that cannot run",
      slot_taken["n"] == 0, f"taken {slot_taken['n']}x")

print()
print("=" * 66)
print("THE POLL LOOP'S OUTAGE GRACE COVERS A WEDGED SERVER")
print("=" * 66)
print("""
_poll_history tolerates a brief outage (CONN_FAIL_GRACE = 60s) so a ComfyUI
restart mid-job does not lose the job. That grace has to fire for BOTH shapes of
death. Before the fix a ReadTimeout fell through to the generic handler, which
just slept and retried -- for up to 31 minutes.
""")

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "imaging/comfy_client.py"), encoding="utf-8").read()
grace_clause = "except (requests.exceptions.ConnectionError,\n                requests.exceptions.Timeout) as exc:"
check("the grace branch catches Timeout alongside ConnectionError",
      grace_clause in src)
check("...and it is the poll loop's branch (conn_down_since lives there)",
      "conn_down_since" in src.split(grace_clause)[-1][:400] if grace_clause in src else False)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
