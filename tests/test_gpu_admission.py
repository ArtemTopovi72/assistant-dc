import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import redis
import gpu_admission as queue


@pytest.fixture
def client(monkeypatch):
    url = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')
    r = redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
    try:
        r.ping()
    except redis.RedisError:
        pytest.skip('Redis unavailable')
    prefix = 'assistant:test:gpu:' + uuid.uuid4().hex + ':'
    monkeypatch.setattr(queue, '_QUEUE', prefix + 'queue')
    monkeypatch.setattr(queue, '_PRESENCE', prefix + 'waiter:')
    yield r
    keys = list(r.scan_iter(prefix + '*'))
    if keys:
        r.delete(*keys)


def test_fifo_and_cancel_cleanup(client):
    order = []
    cancelled = threading.Event()
    def waiter(name, ctx=None):
        try:
            with queue.admit(ctx=ctx, timeout=5, client=client):
                order.append(name)
        except RuntimeError:
            order.append('cancelled')
    with queue.admit(timeout=5, client=client):
        one = threading.Thread(target=waiter, args=('one',))
        one.start()
        deadline = time.monotonic() + 2
        while client.llen(queue._QUEUE) < 2 and time.monotonic() < deadline: time.sleep(.01)
        assert client.llen(queue._QUEUE) == 2
        two = threading.Thread(target=waiter, args=('two', SimpleNamespace(is_cancelled=cancelled.is_set)))
        two.start()
        deadline = time.monotonic() + 2
        while client.llen(queue._QUEUE) < 3 and time.monotonic() < deadline: time.sleep(.01)
        assert client.llen(queue._QUEUE) == 3
        cancelled.set()
        two.join(3)
    one.join(3)
    assert order == ['cancelled', 'one']
    assert client.llen(queue._QUEUE) == 0


def test_stale_head_removed(client):
    client.rpush(queue._QUEUE, 'dead-process')
    with queue.admit(timeout=2, client=client):
        assert client.llen(queue._QUEUE) == 1
    assert client.llen(queue._QUEUE) == 0


def test_waiter_reports_queue_status(client):
    messages = []
    ctx = SimpleNamespace(set_progress=lambda done, total, what: messages.append(what))
    def waiter():
        with queue.admit(ctx=ctx, timeout=3, client=client):
            pass
    with queue.admit(timeout=3, client=client):
        thread = threading.Thread(target=waiter)
        thread.start()
        deadline = time.monotonic() + 2
        while not messages and time.monotonic() < deadline: time.sleep(.01)
        assert messages == ['In the GPU queue (1 ahead)']
    thread.join(3)
    assert not thread.is_alive()


def test_position_counts_all_live_tickets_ahead(client):
    messages = []
    cancelled = threading.Event()
    def wait(ctx):
        try:
            with queue.admit(ctx=ctx, timeout=3, client=client):
                pass
        except RuntimeError:
            pass
    first = SimpleNamespace(is_cancelled=cancelled.is_set)
    second = SimpleNamespace(is_cancelled=cancelled.is_set, set_progress=lambda _a, _b, what: messages.append(what))
    with queue.admit(timeout=3, client=client):
        one = threading.Thread(target=wait, args=(first,)); one.start()
        deadline = time.monotonic() + 2
        while client.llen(queue._QUEUE) < 2 and time.monotonic() < deadline: time.sleep(.01)
        assert client.llen(queue._QUEUE) == 2
        two = threading.Thread(target=wait, args=(second,)); two.start()
        deadline = time.monotonic() + 2
        while not messages and time.monotonic() < deadline: time.sleep(.01)
        assert messages[0] == 'In the GPU queue (2 ahead)'
        cancelled.set()
        one.join(2); two.join(2)


def test_cleanup_failure_does_not_mask_success(client, monkeypatch):
    original = client.eval
    def evaluate(script, *args):
        if script == queue._REMOVE:
            raise redis.ConnectionError('disconnected')
        return original(script, *args)
    monkeypatch.setattr(client, 'eval', evaluate)
    with queue.admit(timeout=2, client=client):
        pass


def test_nested_video_claim_does_not_wait_behind_teacher(client, monkeypatch):
    import comfy_client
    original = queue.admit
    monkeypatch.setattr(queue, 'admit', lambda ctx=None, timeout=600: original(ctx=ctx, timeout=timeout, client=client))
    ctx = SimpleNamespace(deps=SimpleNamespace(COMFY_EVICT_LLM=False))
    entered = threading.Event()
    def teacher():
        with comfy_client.speech_session(timeout=3):
            entered.set()
    with comfy_client.card_session('chain', ctx=ctx):
        thread = threading.Thread(target=teacher)
        thread.start()
        deadline = time.monotonic() + 2
        while client.llen(queue._QUEUE) == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        assert client.llen(queue._QUEUE) == 1
        with comfy_client._gpu_slot(exclusive=True, ctx=ctx):
            assert not entered.is_set()
    thread.join(3)
    assert entered.is_set()
