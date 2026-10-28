"""FIFO admission for GPU work, separate from the Telegram message queue."""
import contextlib
import logging
import os
import threading
import time
import uuid

_QUEUE = 'assistant:gpu:admission'
_PRESENCE = 'assistant:gpu:waiter:'
_TTL = 30
log = logging.getLogger('assistant.gpu_queue')
_HEAD = """
local ahead = 0
local tickets = redis.call('LRANGE', KEYS[1], 0, -1)
for _, ticket in ipairs(tickets) do
  if redis.call('EXISTS', ARGV[1] .. ticket) == 0 then
    redis.call('LREM', KEYS[1], 0, ticket)
  elseif ticket == ARGV[2] then
    return ahead + 1
  else
    ahead = ahead + 1
  end
end
return 0
"""
_REMOVE = """
redis.call('LREM', KEYS[1], 0, ARGV[1])
redis.call('DEL', ARGV[2] .. ARGV[1])
return 1
"""


@contextlib.contextmanager
def admit(ctx=None, timeout=600, client=None):
    if client is None:
        url = os.getenv('REDIS_URL', '').strip()
        if not url or os.getenv('F5_TEST_RUN'):
            yield
            return
        import redis
        client = redis.from_url(url, socket_timeout=3, socket_connect_timeout=3)
    token = uuid.uuid4().hex
    presence = _PRESENCE + token
    pipe = client.pipeline(transaction=True)
    pipe.set(presence, '1', ex=_TTL)
    pipe.rpush(_QUEUE, token)
    pipe.execute()
    stopped = threading.Event()
    failure = threading.Event()

    def heartbeat():
        while not stopped.wait(_TTL / 3):
            try:
                if not client.expire(presence, _TTL):
                    failure.set()
                    return
            except Exception:
                failure.set()
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    last_ahead = None
    try:
        while True:
            if failure.is_set():
                raise RuntimeError('GPU queue connection lost')
            if ctx is not None and getattr(ctx, 'is_cancelled', lambda: False)():
                raise RuntimeError('Cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('The GPU queue is busy')
            position = int(client.eval(_HEAD, 1, _QUEUE, _PRESENCE, token))
            if position == 0:
                raise RuntimeError('GPU queue ticket lost')
            if position == 1:
                break
            import comfy_client
            ahead = position - 1 + int(not comfy_client._SPEECH_FREE.is_set())
            if ahead != last_ahead and ctx is not None:
                import progress
                progress.report(ctx, 0, 0, f'In the GPU queue ({ahead} ahead)')
                last_ahead = ahead
            time.sleep(.2)
        yield
    finally:
        stopped.set()
        thread.join(timeout=4)
        try:
            client.eval(_REMOVE, 1, _QUEUE, token, _PRESENCE)
        except Exception:
            log.warning('GPU queue cleanup failed; the ticket will expire', exc_info=True)
