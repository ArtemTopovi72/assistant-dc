"""A restart never loses a message, and two instances never fight over the poll.

Live 2026-10-21 («бот меня заигнорил»): a new instance started while the old one was tearing down;
both polled (409), the message went to the old one, was enqueued at 00:21:37 and the process killed
at 00:21:44 -- popped from Redis, held in RAM, never started; the restart skipped it as «still
queued»."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_bot  # noqa: E402
import tg_dispatch  # noqa: E402
import tg_queue  # noqa: E402


class _FakeRedis:
    def __init__(self):
        self.kv = {}

    def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return False
        self.kv[k] = v
        return True

    def get(self, k):
        return self.kv.get(k)

    def expire(self, k, ttl):
        return k in self.kv

    def delete(self, k):
        self.kv.pop(k, None)


def _bot(r):
    b = types.SimpleNamespace(_backend=types.SimpleNamespace(_r=r))
    b._poll_lease = types.MethodType(tg_dispatch.DispatchMixin._poll_lease, b)
    b._release_poll_lease = types.MethodType(tg_dispatch.DispatchMixin._release_poll_lease, b)
    return b


def test_one_poller_per_token():
    r = _FakeRedis()
    old, new = _bot(r), _bot(r)
    assert old._poll_lease() and old._poll_lease()          # renews its own
    assert not new._poll_lease()                             # waits instead of a 409
    old._release_poll_lease()
    assert new._poll_lease()


def test_no_redis_no_lease_needed():
    assert _bot(None)._poll_lease()


def test_a_task_popped_but_never_started_is_run_after_the_restart():
    task = tg_bot._Task(task_id="e4782f4f00", chat_id=386455236, user_text="нарисуй кота")
    requeued, notified = [], []

    class RedisLike:                                     # a durable backend: not the in-memory one
        def __init__(self, queued):
            self.queued = queued

        def position_of(self, task_id):
            return 1 if task_id in self.queued else 0    # 1-based; 0 = a consumer had it

        def requeue(self, t):
            requeued.append(t.task_id)
    bot = types.SimpleNamespace(_backend=RedisLike(set()), _send_text=lambda *a, **k: notified.append(a))
    tg_queue.QueueMixin._recover_inflight(bot, [dict(task.to_dict(), _state="pending")])
    assert requeued == ["e4782f4f00"] and not notified
    # still in Redis: the fresh consumers run it, nothing to do
    requeued.clear()
    bot._backend = RedisLike({"e4782f4f00"})
    tg_queue.QueueMixin._recover_inflight(bot, [dict(task.to_dict(), _state="pending")])
    assert not requeued and not notified
