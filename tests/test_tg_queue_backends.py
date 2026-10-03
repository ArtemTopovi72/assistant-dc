"""The queue backends: selection precedence (offline) and, when the real
services are up, a live round trip through Redis and Kafka.

Background: RedisBackend and KafkaBackend existed but had NO test coverage and
had never been exercised against a real server -- KAFKA_BROKERS/REDIS_URL were
unset, so _make_backend() silently fell through to InMemoryBackend on every
start. Two things that only show up against a real server:

  * RedisBackend's pop Lua uses LPOS, which requires Redis >= 6.0.6. The common
    Windows port is stuck at 5.0.14, where the script fails at runtime -- not at
    connect time, so a naive "did it construct?" check would call it healthy.
  * KafkaBackend is FIFO ONLY. Its own docstring says so: Kafka partitions are
    consumed in order, so the per-chat round-robin fairness that InMemory and
    Redis provide does NOT survive. One chat can starve the others.

The selection section runs everywhere and never touches the network. The live
section SKIPS (it does not fail) when a service is not answering, so this suite
stays usable on a machine without them.

Run: venv/Scripts/python.exe tests/test_tg_queue_backends.py
"""
import sys, os, re, time, socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_queue_backends as B

OK = BAD = SKIP = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else: BAD += 1; print(f"FAIL  {name}   {extra}")
def skip(name, why):
    global SKIP
    SKIP += 1; print(f"SKIP  {name}   ({why})")


def _port_open(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class _EnvGuard:
    """_make_backend reads os.getenv directly; restore whatever was there."""
    KEYS = ("KAFKA_BROKERS", "REDIS_URL")
    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS: os.environ.pop(k, None)
        return self
    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        return False


print("=" * 66)
print("BACKEND SELECTION -- NO NETWORK, ALWAYS RUNS")
print("=" * 66)

_realK, _realR = B.KafkaBackend, B.RedisBackend

class _FakeOK:
    def __init__(self, *a, **k): self.args = a
    def name(self): return type(self).__name__
    def close(self): pass
class _FakeKafkaOK(_FakeOK): pass
class _FakeRedisOK(_FakeOK): pass
class _Boom:
    def __init__(self, *a, **k): raise RuntimeError("service down")

try:
    with _EnvGuard():
        B.KafkaBackend, B.RedisBackend = _FakeKafkaOK, _FakeRedisOK
        b = B._make_backend()
        check("no env set -> InMemoryBackend", isinstance(b, B.InMemoryBackend), type(b).__name__)
        b.close()

        os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
        b = B._make_backend()
        check("REDIS_URL alone -> Redis", isinstance(b, _FakeRedisOK), type(b).__name__)
        b.close()

        os.environ["KAFKA_BROKERS"] = "127.0.0.1:9092"
        b = B._make_backend()
        check("both set -> Kafka wins (documented precedence)",
              isinstance(b, _FakeKafkaOK), type(b).__name__)
        b.close()

        B.KafkaBackend = _Boom
        b = B._make_backend()
        check("Kafka unreachable -> falls through to Redis",
              isinstance(b, _FakeRedisOK), type(b).__name__)
        b.close()

        B.RedisBackend = _Boom
        b = B._make_backend()
        check("both unreachable -> InMemory, never an exception",
              isinstance(b, B.InMemoryBackend), type(b).__name__)
        b.close()

        os.environ.pop("KAFKA_BROKERS")
        b = B._make_backend()
        check("Redis set but down -> InMemory", isinstance(b, B.InMemoryBackend), type(b).__name__)
        b.close()
finally:
    B.KafkaBackend, B.RedisBackend = _realK, _realR

print()
print("=" * 66)
print("LIVE REDIS ROUND TRIP")
print("=" * 66)

# NEVER the operator's own ring. This suite DELETES the ring and every
# per-chat queue before it measures, and the user's live bot is connected to
# the same server: pointing it at REDIS_URL wiped real queued work, pushed
# fixtures the running bot would have executed, and made the fairness
# measurement race with real traffic -- which is how it failed inside a full
# run while passing on its own. Same host, dedicated database.
def _test_db_url(url: str, db: int = 15) -> str:
    return re.sub(r"/\d+$", "", url.rstrip("/")) + "/%d" % db


REDIS_URL = _test_db_url(os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0")
if not _port_open("127.0.0.1", 6379):
    skip("live redis round trip", "nothing listening on 6379")
else:
    try:
        r = B.RedisBackend(REDIS_URL)
        r._r.delete(B.RedisBackend._RING)
        for k in r._r.keys(B.RedisBackend._PFX + "*"): r._r.delete(k)

        r.push(B._Task(task_id="t1", chat_id=111, user_text="one"))
        r.push(B._Task(task_id="t2", chat_id=222, user_text="two"))
        r.push(B._Task(task_id="t3", chat_id=111, user_text="three"))
        check("depth counts every queued task", r.depth() == 3, r.depth())

        ids = [(r.pop(timeout=3) or B._Task(task_id="<none>", chat_id=0, user_text="")).task_id
               for _ in range(3)]
        # The whole point of the ring: chat 111 queued twice in a row does not get
        # both slots before chat 222 is served once.
        check("round-robin fairness across chats (t1, t2, t3)", ids == ["t1", "t2", "t3"], ids)
        check("queue drains to empty", r.depth() == 0, r.depth())
        check("pop on an empty queue returns None", r.pop(timeout=1) is None)

        # LPOS guard: a Redis older than 6.0.6 fails INSIDE the Lua, not at connect.
        info = r._r.info("server")
        ver = info.get("redis_version", "0.0.0")
        parts = [int(x) for x in str(ver).split(".")[:3] if str(x).isdigit()]
        while len(parts) < 3: parts.append(0)
        check(f"server is new enough for LPOS (>=6.0.6, found {ver})",
              tuple(parts) >= (6, 0, 6), ver)
        r.close()
    except Exception as exc:
        check("live redis round trip", False, f"{type(exc).__name__}: {exc}")

print()
print("=" * 66)
print("LIVE KAFKA ROUND TRIP")
print("=" * 66)

BROKERS = os.environ.get("KAFKA_BROKERS") or "127.0.0.1:9092"
if not _port_open(BROKERS.split(",")[0].split(":")[0], int(BROKERS.split(",")[0].split(":")[1])):
    skip("live kafka round trip", f"nothing listening on {BROKERS}")
else:
    try:
        k = B.KafkaBackend(BROKERS)
        stamp = str(time.time())
        k.push(B._Task(task_id="k1", chat_id=333, user_text="one " + stamp))
        k.push(B._Task(task_id="k2", chat_id=333, user_text="two " + stamp))
        check("depth tracks produced-but-unconsumed lag", k.depth() == 2, k.depth())

        seen, deadline = [], time.time() + 45
        while len(seen) < 2 and time.time() < deadline:
            t = k.pop(timeout=5)
            if t and stamp in t.user_text: seen.append(t.task_id)
        check("both messages come back", len(seen) == 2, seen)
        check("FIFO order preserved", seen == ["k1", "k2"], seen)
        check("name reports the topic", "tgbot_tasks" in k.name(), k.name())
        k.close()
    except Exception as exc:
        check("live kafka round trip", False, f"{type(exc).__name__}: {exc}")

print()
print("=" * 66)
print("FAIRNESS UNDER MULTI-USER LOAD -- THE PRODUCTION-CRITICAL PROPERTY")
print("=" * 66)
print("""
The reported problem was users blocked behind each other. Measured against the
real services: with chat 100 holding 10 queued jobs, Redis served two later
light users at positions 2 and 3, while Kafka served them at 11 and 12 -- Kafka
is FIFO by construction and cannot express per-chat round robin.

That is why REDIS_URL is set and KAFKA_BROKERS is not. These checks pin the
property so the request path cannot quietly regress to FIFO.
""")

if not _port_open("127.0.0.1", 6379):
    skip("fairness under load", "nothing listening on 6379")
else:
    try:
        r = B.RedisBackend(REDIS_URL)
        r._r.delete(B.RedisBackend._RING)
        for k in r._r.keys(B.RedisBackend._PFX + "*"): r._r.delete(k)

        for i in range(10):
            r.push(B._Task(task_id=f"heavy{i}", chat_id=100, user_text="h"))
        r.push(B._Task(task_id="light200", chat_id=200, user_text="l"))
        r.push(B._Task(task_id="light300", chat_id=300, user_text="l"))

        order = []
        for _ in range(12):
            t = r.pop(timeout=3)
            if t: order.append(t.task_id)
        check("every queued task is eventually served", len(order) == 12, len(order))
        p200 = order.index("light200") + 1 if "light200" in order else 99
        p300 = order.index("light300") + 1 if "light300" in order else 99
        # FIFO would put them at 11 and 12. Round robin puts them right behind
        # the one heavy job already in flight.
        check("a light user is NOT stuck behind the whole flood (<=4)",
              p200 <= 4 and p300 <= 4, f"positions {p200}, {p300}")

        # starvation: light user arriving mid-flood
        r._r.delete(B.RedisBackend._RING)
        for k in r._r.keys(B.RedisBackend._PFX + "*"): r._r.delete(k)
        for i in range(6): r.push(B._Task(task_id=f"A{i}", chat_id=100, user_text="a"))
        r.push(B._Task(task_id="B0", chat_id=200, user_text="b"))
        for i in range(6, 12): r.push(B._Task(task_id=f"A{i}", chat_id=100, user_text="a"))
        order2 = []
        for _ in range(13):
            t = r.pop(timeout=3)
            if t: order2.append(t.task_id)
        posB = order2.index("B0") + 1 if "B0" in order2 else 99
        check("a user arriving mid-flood is not starved (<=4)", posB <= 4, f"position {posB}")
        r.close()
    except Exception as exc:
        check("fairness under load", False, f"{type(exc).__name__}: {exc}")

print()
print("=" * 66)
print("PRODUCTION SELECTION IS THE FAIR BACKEND")
print("=" * 66)

_sel_k = os.environ.get("KAFKA_BROKERS", "")
if _sel_k.strip():
    check("KAFKA_BROKERS is unset in production (Kafka is FIFO -> unfair)",
          False, f"KAFKA_BROKERS={_sel_k!r} would put FIFO in the request path")
else:
    check("KAFKA_BROKERS is unset, so the fair backend wins selection", True)

print()
print(f"{OK} passed, {BAD} failed, {SKIP} skipped")
sys.exit(1 if BAD else 0)
