"""A Cancel button belongs to one chat; startup must not replay work already claimed."""
import threading
import types

import tg_bot as T


class DurableQueue:
    """The external queue contract without a Redis service."""
    def __init__(self):
        self.queue = T.InMemoryBackend()

    def __getattr__(self, name):
        return getattr(self.queue, name)


def queue_bot(backend=None):
    bot = T.TelegramBot.__new__(T.TelegramBot)
    bot._backend = backend or T.InMemoryBackend()
    bot._task_lock = threading.Lock()
    bot._cancelled = T._RecentIds(512)
    bot._running_task = {}
    bot._task_cancels = {}
    bot._pending_journal = {}
    bot._incoming = types.SimpleNamespace(ack=lambda ids: None, pending=lambda: [], claim=lambda bot: True)
    bot._write_inflight = lambda: None
    bot._activity = types.SimpleNamespace(log=lambda *a: None)
    bot._queued_work_allowed = lambda cid: True
    return bot


def test_cancel_cannot_drop_another_chats_queued_task():
    bot = queue_bot()
    task = T._Task(task_id="other-task", chat_id=202, user_text="hello")
    bot._backend.push(task)
    bot._pending_journal[task.task_id] = task
    assert bot._cancel_task(101, task.task_id) == ""
    assert bot._backend.pop(timeout=0).task_id == "other-task"
    assert not bot._is_cancelled(task.task_id)


def test_cancel_cannot_poison_another_chats_running_task():
    bot = queue_bot()
    task = T._Task(task_id="running-other", chat_id=202, user_text="hello")
    bot._running_task[202] = [task]
    event = bot._task_cancels[task.task_id] = threading.Event()
    assert bot._cancel_task(101, task.task_id) == ""
    assert not bot._is_cancelled(task.task_id)
    assert not event.is_set()


def test_owner_can_cancel_a_restored_queue_without_a_local_journal():
    bot = queue_bot(DurableQueue())
    task = T._Task(task_id="restored", chat_id=202, user_text="hello")
    bot._backend.push(task)
    assert bot._cancel_task(202, task.task_id) == "dropped"
    assert bot._backend.depth() == 0


def test_owner_can_cancel_a_task_popped_but_not_yet_started():
    bot = queue_bot()
    task = T._Task(task_id="claimed", chat_id=202, user_text="hello")
    bot._pending_journal[task.task_id] = task
    bot._cancel_task(202, task.task_id)
    assert bot._is_cancelled(task.task_id)


def test_restore_pending_skips_bad_entries_but_keeps_valid_work():
    bot = queue_bot(DurableQueue())
    task = T._Task(task_id="valid-after-bad", chat_id=303, user_text="hello")
    data = [None, 42, "bad", dict(task.to_dict(), _state="pending")]
    bot._restore_pending(data)
    assert bot._backend.pop(timeout=0).task_id == "valid-after-bad"
    bot._recover_inflight(data)
    assert bot._backend.depth() == 0


def test_start_does_not_requeue_a_task_the_new_consumer_already_claimed():
    bot = queue_bot(DurableQueue())
    task = T._Task(task_id="pending-at-shutdown", chat_id=303, user_text="hello")
    bot._backend.push(task)
    bot._running = False
    bot._identify = lambda: ({"username": "AuditTestBot"}, "")
    bot._register_commands = lambda: None
    bot._take_inflight = lambda: [dict(task.to_dict(), _state="pending")]
    bot._on_status = lambda *a: None
    bot._consumers = []
    bot._silent_mode = True
    executed = []

    def consume():
        claimed = bot._backend.pop(timeout=0)
        if claimed:
            executed.append(claimed.task_id)
            bot._cancel_task(claimed.chat_id, claimed.task_id)
            assert bot._is_cancelled(claimed.task_id), "restored claimed task lost its ownership record"

    bot._consumer_loop = consume

    class Thread:
        def __init__(self, target, args=(), **kwargs):
            self.target, self.args = target, args

        def start(self):
            if self.target in (consume, bot._recover_inflight):
                self.target(*self.args)

    bot.modules = {
        "tg_local_api": types.SimpleNamespace(enabled=lambda: False),
        "reminders": types.SimpleNamespace(register=lambda *a, **k: None),
        "threading": types.SimpleNamespace(Thread=Thread),
    }
    assert bot.start()
    assert executed == ["pending-at-shutdown"]
    assert bot._backend.pop(timeout=0) is None, "startup queued a second copy of claimed work"
