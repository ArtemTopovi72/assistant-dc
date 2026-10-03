"""The task queue backends: the queued item and the three transports.

Lifted out of tg_bot.py whole. Nothing here knows about Telegram, sessions or
the agent — a backend only moves _Task records around and answers questions
about who is waiting and in what order. tg_bot re-exports every name, because
the suites and the sibling tg_* modules address them as tg_bot._Task,
tg_bot.InMemoryBackend and so on.
"""
import abc
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("assistant.tg_bot")


@dataclass
class _Task:
    """A fully-resolved agent task, ready for graph.invoke()."""
    task_id:     str
    chat_id:     int
    user_text:   str
    image_path:  str = ""
    # The picture this task acts on, captured at PRESS time (when the
    # id-carrying callback like "upscale:ab12cd" was parsed) — not read lazily
    # from sess.target_image at execution time. A second button press on a
    # DIFFERENT picture, queued while the first is still pending, used to
    # overwrite the single sess.target_image slot before the first task ever
    # ran, so both tasks silently resolved to whichever picture was pressed
    # LAST. Threading the id through the task closes that race.
    image_id:    str = ""
    # True only for a picture the user actually SENT this turn (a photo or the
    # first frame of an album) -- never an internally-derived image such as a
    # video's frame sheet, which rides image_path/image_id identically but
    # must not be offered picture-editing actions as if it were the user's own
    # upload (see tg_tasks.py's use of it, and _photo_upload in tg_resolve.py).
    image_is_photo: bool = False
    # Set only by the 🎭 Style flow: a second image (the style reference) that
    # rides alongside image_path (the target). _execute_task populates
    # ctx.reference_images from the pair so transfer_image has both to work
    # with -- ordinary tasks never touch this field.
    style_ref_path: str = ""
    enqueue_ts:  float = field(default_factory=time.time)
    # Set once the reply (text / picture / file) has landed in the chat. What
    # runs after that — history compaction — is housekeeping the user never
    # sees, so a Stop pressed then must not claim to be "stopping" anything.
    delivered:   bool = False
    reply_to:    int = 0          # the user's message this task answers (threads the reply)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "_Task":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _Backend(abc.ABC):
    @abc.abstractmethod
    def push(self, task: _Task) -> None: ...
    @abc.abstractmethod
    def pop(self, timeout: float = 5.0) -> Optional[_Task]: ...
    @abc.abstractmethod
    def depth(self) -> int: ...
    @abc.abstractmethod
    def position_of(self, task_id: str) -> int: ...
    def name(self) -> str: return type(self).__name__
    def close(self) -> None: pass

    def requeue(self, task: _Task) -> None:
        """Put a task BACK at the head of its chat's queue.

        Used when a worker picks up a task for a chat that another worker is
        already running: the task must wait, but it must not lose its place to
        the messages the user sent after it.
        """
        self.push(task)

    # ── fairness API (default: FIFO semantics) ────────────────────────────────
    def service_order(self) -> list:
        """Waiting tasks in the order they will actually be served."""
        return []

    def chat_depth(self, chat_id: int) -> int:
        return sum(1 for t in self.service_order() if t.chat_id == chat_id)

    def drop_chat(self, chat_id: int) -> int:
        """Remove every waiting task belonging to one chat. Returns how many."""
        return 0

    def drop_task(self, task_id: str) -> bool:
        """Remove one waiting task by id. Returns True if it was still queued."""
        return False

    def bump(self, task_id: str) -> bool:
        """Admin: serve this waiting task next (head of its chat, chat first
        in the rotation). True if it was still queued."""
        return False

    def tasks_ahead(self, task_id: str) -> list:
        order = self.service_order()
        for i, t in enumerate(order):
            if t.task_id == task_id:
                return order[:i]
        return []


class InMemoryBackend(_Backend):
    """Round-robin queue: one FIFO per chat, served in rotation.

    A single global FIFO meant one user's 40-minute deep research blocked every
    other user behind it — with one GPU pipeline that is the difference between a
    bot that works for a group and one that works for whoever queued first. Tasks
    are now taken from a different chat each turn, so a burst from one user costs
    everyone else at most one task of latency.
    """

    def __init__(self) -> None:
        self._chats: dict[int, list] = {}     # chat_id -> [_Task, ...] (FIFO)
        self._ring:  list[int] = []           # rotation order of chat ids
        self._cv     = threading.Condition()
        self._count  = 0

    def push(self, task: _Task) -> None:
        with self._cv:
            q = self._chats.setdefault(task.chat_id, [])
            q.append(task)
            if task.chat_id not in self._ring:
                self._ring.append(task.chat_id)
            self._count += 1
            self._cv.notify()

    def requeue(self, task: _Task) -> None:
        with self._cv:
            q = self._chats.setdefault(task.chat_id, [])
            q.insert(0, task)              # HEAD: preserve the user's order
            if task.chat_id not in self._ring:
                self._ring.append(task.chat_id)
            self._count += 1
            self._cv.notify()

    def pop(self, timeout: float = 5.0) -> Optional[_Task]:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                task = self._take_locked()
                if task is not None:
                    return task
                left = deadline - time.monotonic()
                if left <= 0 or not self._cv.wait(left):
                    return self._take_locked()

    def _take_locked(self) -> Optional[_Task]:
        for _ in range(len(self._ring)):
            chat_id = self._ring.pop(0)
            q = self._chats.get(chat_id) or []
            if not q:
                self._chats.pop(chat_id, None)
                continue
            task = q.pop(0)
            self._count -= 1
            if q:
                self._ring.append(chat_id)      # back of the rotation
            else:
                self._chats.pop(chat_id, None)
            return task
        return None

    def depth(self) -> int:
        with self._cv:
            return self._count

    def service_order(self) -> list:
        """Simulate the rotation so a queue position reflects reality — under
        round-robin the Nth task pushed is NOT the Nth task served."""
        with self._cv:
            pending = {cid: list(q) for cid, q in self._chats.items()}
            ring = list(self._ring)
        out: list = []
        while ring:
            cid = ring.pop(0)
            q = pending.get(cid) or []
            if not q:
                continue
            out.append(q.pop(0))
            if q:
                ring.append(cid)
        return out

    def chat_depth(self, chat_id: int) -> int:
        with self._cv:
            return len(self._chats.get(chat_id) or [])

    def drop_chat(self, chat_id: int) -> int:
        with self._cv:
            q = self._chats.pop(chat_id, None) or []
            if chat_id in self._ring:
                self._ring.remove(chat_id)
            self._count -= len(q)
            return len(q)

    def drop_task(self, task_id: str) -> bool:
        with self._cv:
            for cid, q in list(self._chats.items()):
                for i, t in enumerate(q):
                    if t.task_id == task_id:
                        del q[i]
                        self._count -= 1
                        if not q:
                            self._chats.pop(cid, None)
                            if cid in self._ring:
                                self._ring.remove(cid)
                        return True
        return False

    def position_of(self, task_id: str) -> int:
        for i, t in enumerate(self.service_order(), 1):
            if t.task_id == task_id:
                return i
        return 0

    def bump(self, task_id: str) -> bool:
        with self._cv:
            for cid, q in self._chats.items():
                for i, t in enumerate(q):
                    if t.task_id == task_id:
                        q.insert(0, q.pop(i))
                        if cid in self._ring:
                            self._ring.remove(cid)
                        self._ring.insert(0, cid)
                        return True
        return False


class RedisBackend(_Backend):
    """Same round-robin fairness as InMemoryBackend, one Redis list per chat.

    push/pop run as Lua scripts so the ring and the per-chat lists can never drift
    apart under concurrent consumers — a plain multi-command sequence could pop a
    chat off the ring and then crash before pushing it back, stranding that chat's
    queue forever.
    """
    _RING = "tgbot:ring"
    _PFX  = "tgbot:q:"

    _PUSH_LUA = """
    local key = KEYS[2] .. ARGV[1]
    redis.call('RPUSH', key, ARGV[2])
    if redis.call('LPOS', KEYS[1], ARGV[1]) == false then
        redis.call('RPUSH', KEYS[1], ARGV[1])
    end
    return 1
    """
    # Same as push, but to the HEAD of the chat's list (see _Backend.requeue).
    # Inheriting the default push() put a deferred task at the TAIL, behind the
    # messages the user sent after it, so with several consumers a busy chat's
    # turns were answered out of order.
    _REQUEUE_LUA = """
    local key = KEYS[2] .. ARGV[1]
    redis.call('LPUSH', key, ARGV[2])
    if redis.call('LPOS', KEYS[1], ARGV[1]) == false then
        redis.call('RPUSH', KEYS[1], ARGV[1])
    end
    return 1
    """
    _POP_LUA = """
    local n = redis.call('LLEN', KEYS[1])
    for i = 1, n do
        local chat = redis.call('LPOP', KEYS[1])
        if not chat then return nil end
        local key = KEYS[2] .. chat
        local task = redis.call('LPOP', key)
        if task then
            if redis.call('LLEN', key) > 0 then
                redis.call('RPUSH', KEYS[1], chat)
            end
            return task
        end
    end
    return nil
    """

    def __init__(self, url: str) -> None:
        import redis as _redis
        self._r = _redis.from_url(url, decode_responses=False)
        self._r.ping()
        self._push_sha = self._r.register_script(self._PUSH_LUA)
        self._pop_sha  = self._r.register_script(self._POP_LUA)
        self._requeue_sha = self._r.register_script(self._REQUEUE_LUA)
        logger.info("RedisBackend connected: %s", url)

    def _chats(self) -> list[str]:
        try:
            return [c.decode() if isinstance(c, bytes) else str(c)
                    for c in (self._r.lrange(self._RING, 0, -1) or [])]
        except Exception:
            return []

    def push(self, task: _Task) -> None:
        self._push_sha(keys=[self._RING, self._PFX],
                       args=[str(task.chat_id),
                             json.dumps(task.to_dict()).encode()])

    def requeue(self, task: _Task) -> None:
        self._requeue_sha(keys=[self._RING, self._PFX],
                          args=[str(task.chat_id),
                                json.dumps(task.to_dict()).encode()])

    def pop(self, timeout: float = 5.0) -> Optional[_Task]:
        # No blocking primitive covers "rotate then pop", so poll. The interval is
        # short enough that a freshly pushed task starts within a quarter second.
        deadline = time.monotonic() + timeout
        while True:
            try:
                raw = self._pop_sha(keys=[self._RING, self._PFX])
            except Exception as exc:
                logger.error("Redis pop error: %s", exc)
                raw = None
            if raw:
                try:
                    return _Task.from_dict(json.loads(raw))
                except Exception as exc:
                    logger.error("Redis pop parse error: %s", exc)
                    return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)

    def depth(self) -> int:
        try:
            return sum(int(self._r.llen(self._PFX + c) or 0) for c in self._chats())
        except Exception:
            return 0

    def _pending(self) -> dict:
        out: dict = {}
        for c in self._chats():
            tasks = []
            for raw in (self._r.lrange(self._PFX + c, 0, -1) or []):
                try: tasks.append(_Task.from_dict(json.loads(raw)))
                except Exception: pass
            out[c] = tasks
        return out

    def service_order(self) -> list:
        pending = self._pending()
        ring = self._chats()
        out: list = []
        while ring:
            c = ring.pop(0)
            q = pending.get(c) or []
            if not q:
                continue
            out.append(q.pop(0))
            if q:
                ring.append(c)
        return out

    def chat_depth(self, chat_id: int) -> int:
        try:
            return int(self._r.llen(self._PFX + str(chat_id)) or 0)
        except Exception:
            return 0

    def drop_chat(self, chat_id: int) -> int:
        n = self.chat_depth(chat_id)
        try:
            self._r.delete(self._PFX + str(chat_id))
            self._r.lrem(self._RING, 0, str(chat_id))
        except Exception as exc:
            logger.warning("Redis drop_chat(%s): %s", chat_id, exc)
            return 0
        return n

    def drop_task(self, task_id: str) -> bool:
        for c in self._chats():
            key = self._PFX + c
            for raw in (self._r.lrange(key, 0, -1) or []):
                try:
                    if json.loads(raw).get("task_id") != task_id:
                        continue
                except Exception:
                    continue
                try:
                    return bool(self._r.lrem(key, 1, raw))
                except Exception:
                    return False
        return False

    def position_of(self, task_id: str) -> int:
        for i, t in enumerate(self.service_order(), 1):
            if t.task_id == task_id:
                return i
        return 0

    _BUMP_LUA = """
    local key = KEYS[2] .. ARGV[1]
    if redis.call('LREM', key, 1, ARGV[2]) == 0 then return 0 end
    redis.call('LPUSH', key, ARGV[2])
    redis.call('LREM', KEYS[1], 0, ARGV[1])
    redis.call('LPUSH', KEYS[1], ARGV[1])
    return 1
    """

    def bump(self, task_id: str) -> bool:
        for c in self._chats():
            for raw in (self._r.lrange(self._PFX + c, 0, -1) or []):
                try:
                    if json.loads(raw).get("task_id") != task_id:
                        continue
                except Exception:
                    continue
                try:   # one script: the list and the ring move together
                    return bool(self._r.eval(self._BUMP_LUA, 2, self._RING, self._PFX, c, raw))
                except Exception as exc:
                    logger.warning("Redis bump(%s): %s", task_id, exc)
                    return False
        return False

    def name(self) -> str: return f"Redis({self._RING})"


class KafkaBackend(_Backend):
    """FIFO only — Kafka partitions are consumed in order and offsets cannot be
    reordered client-side, so the round-robin fairness of the other two backends
    does not apply here. Stop still works (it drops stale tasks by timestamp)."""
    _TOPIC = "tgbot_tasks"

    def __init__(self, brokers: str) -> None:
        from kafka import KafkaProducer, KafkaConsumer
        self._producer = KafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: json.dumps(v).encode())
        self._consumer = KafkaConsumer(
            self._TOPIC, bootstrap_servers=brokers,
            value_deserializer=lambda v: json.loads(v),
            auto_offset_reset="earliest", enable_auto_commit=True,
            group_id="tgbot_workers", consumer_timeout_ms=1000)
        self._lag = 0; self._lag_lock = threading.Lock()
        # KafkaConsumer is NOT thread-safe, and the bot runs TG_WORKERS (3)
        # consumer threads that all call pop() on this one instance.
        self._poll_lock = threading.Lock()
        logger.info("KafkaBackend connected: brokers=%s topic=%s", brokers, self._TOPIC)

    def push(self, task: _Task) -> None:
        self._producer.send(self._TOPIC, value=task.to_dict())
        self._producer.flush()
        with self._lag_lock: self._lag += 1

    def pop(self, timeout: float = 5.0) -> Optional[_Task]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._poll_lock:
                batch = self._consumer.poll(timeout_ms=500, max_records=1)
            for _, msgs in batch.items():
                for msg in msgs:
                    with self._lag_lock: self._lag = max(0, self._lag - 1)
                    try: return _Task.from_dict(msg.value)
                    except Exception as exc:
                        logger.error("Kafka pop parse error: %s", exc)
        return None

    def depth(self) -> int:
        with self._lag_lock: return self._lag

    def position_of(self, task_id: str) -> int:
        with self._lag_lock: return max(0, self._lag)

    def name(self) -> str: return f"Kafka({self._TOPIC})"

    def close(self) -> None:
        try: self._producer.close(timeout=5)
        except Exception: pass
        try:
            with self._poll_lock: self._consumer.close()
        except Exception: pass


# Set by tg_bot.redirect_data_dir, which every test that builds a bot already
# calls. Without it a suite inherits the operator's REDIS_URL and pushes its
# fixtures into the LIVE ring: the running bot would pick them up and execute
# them, and leftovers from one suite made the next one fail at random ("three
# messages became one task" passing, the same call with a photo pushing
# nothing). The queue is persistent shared state exactly like tg_users.db, so
# it is redirected in the same place and for the same reason.
_FORCE_INMEMORY = False


def force_inmemory() -> None:
    global _FORCE_INMEMORY
    _FORCE_INMEMORY = True


def _make_backend() -> _Backend:
    if _FORCE_INMEMORY:
        b = InMemoryBackend()
        logger.info("Queue backend: InMemory (redirected for a test)")
        return b
    kafka_url = os.getenv("KAFKA_BROKERS", "").strip()
    if kafka_url:
        try:
            b = KafkaBackend(kafka_url)
            logger.info("Queue backend: %s", b.name()); return b
        except Exception as exc:
            logger.warning("Kafka unavailable (%s), trying Redis…", exc)
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            b = RedisBackend(redis_url)
            logger.info("Queue backend: %s", b.name()); return b
        except Exception as exc:
            logger.warning("Redis unavailable (%s), falling back to in-memory.", exc)
    b = InMemoryBackend()
    logger.info("Queue backend: InMemory"); return b
