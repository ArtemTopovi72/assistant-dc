"""Order and thread-safety of the durable queue backends, without live services.

RedisBackend.requeue used to inherit the default push(), which appends to the
TAIL of the chat's list: a task the consumer had to defer (its chat was already
busy) went behind the messages the user sent after it, so a busy chat's turns
were answered out of order. KafkaBackend.pop was called from every consumer
thread (TG_WORKERS, default 3) on one KafkaConsumer, which is not thread-safe.
"""
import threading
import time

import pytest

import tg_queue_backends as B


@pytest.fixture
def redis_backend(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")          # the backend's push/pop are Lua scripts
    import redis
    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis, "from_url",
                        lambda url, **k: fakeredis.FakeRedis(server=server, **k))
    return B.RedisBackend("redis://fake")


def _t(i, chat=1):
    return B._Task(task_id=f"t{i}", chat_id=chat, user_text=f"m{i}")


def test_redis_requeue_goes_to_head_of_its_chat(redis_backend):
    for i in range(3):
        redis_backend.push(_t(i))
    first = redis_backend.pop(0.1)
    assert first.task_id == "t0"
    redis_backend.requeue(first)
    assert [redis_backend.pop(0.1).task_id for _ in range(3)] == ["t0", "t1", "t2"]
    assert redis_backend.pop(0.05) is None


def test_redis_requeue_matches_inmemory_rotation(redis_backend):
    mem = B.InMemoryBackend()
    for b in (mem, redis_backend):
        b.push(_t(0, chat=1)); b.push(_t(1, chat=1)); b.push(_t(2, chat=2))
        b.requeue(b.pop(0.1))
    got = {name: [b.pop(0.1).task_id for _ in range(3)]
           for name, b in (("mem", mem), ("redis", redis_backend))}
    assert got["mem"] == got["redis"], got


def test_redis_requeue_of_emptied_chat_rejoins_ring(redis_backend):
    redis_backend.push(_t(0))
    t = redis_backend.pop(0.1)
    assert redis_backend.depth() == 0
    redis_backend.requeue(t)
    assert redis_backend.depth() == 1
    assert redis_backend.pop(0.1).task_id == "t0"


class _NotThreadSafeConsumer:
    """Raises if two threads are ever inside poll() at once, like kafka-python
    does ("KafkaConsumer is not thread-safe")."""
    def __init__(self):
        self._inside = 0
        self.overlaps = 0
        self._lock = threading.Lock()

    def poll(self, timeout_ms=0, max_records=1):
        with self._lock:
            self._inside += 1
            if self._inside > 1:
                self.overlaps += 1
        time.sleep(0.01)
        with self._lock:
            self._inside -= 1
        return {}

    def close(self):
        pass


def test_kafka_pop_serializes_access_to_the_consumer():
    k = B.KafkaBackend.__new__(B.KafkaBackend)
    k._consumer = _NotThreadSafeConsumer()
    k._lag = 0
    k._lag_lock = threading.Lock()
    k._poll_lock = threading.Lock()
    threads = [threading.Thread(target=k.pop, args=(0.15,)) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert k._consumer.overlaps == 0
