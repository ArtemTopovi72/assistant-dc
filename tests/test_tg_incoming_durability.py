"""Incoming work survives the debounce window without logging account secrets."""
import json
import contextvars
import threading
import types
import pytest

import tg_bot as T
from tg_dispatch import DispatchMixin


def make_bot(tmp_path):
    T.redirect_data_dir(tmp_path)
    bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot._api_post = lambda *a, **k: {"ok": True, "result": {"message_id": 1}}
    bot._send_text = lambda *a, **k: None
    bot._activity.log = lambda *a: None
    for cid in (101, 202):
        bot._user_store.put(T._User(chat_id=cid, name="User", status="approved"))
    return bot


def hold_worker(bot, cid):
    bot._workers[cid] = types.SimpleNamespace(is_alive=lambda: True)


def test_poll_crash_does_not_confirm_an_undispatched_batch():
    class Crash(BaseException):
        pass

    class Poll(DispatchMixin):
        _running = True
        _offset = 0
        _start_time = 0
        saved = None
        _poll_lease = lambda self: True
        _fetch_updates = lambda self: [{"update_id": 100}, {"update_id": 101}]

        def _save_offset(self, offset):
            self.saved = offset

        def _dispatch_logged(self, update):
            raise Crash()

    bot = Poll()
    try:
        bot._poll_loop()
    except Crash:
        pass
    assert bot.saved is None


def test_a_buffered_item_is_durable_before_a_worker_can_take_it(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    bot._enqueue_item(101, {"type": "text", "text": "pending before debounce"})
    pending = bot._incoming.pending()
    assert len(pending) == 1
    assert pending[0]["item"]["text"] == "pending before debounce"
    restored = make_bot(tmp_path)
    hold_worker(restored, 101)
    restored._restore_incoming([])
    assert restored._queues[101].get_nowait()["text"] == "pending before debounce"


def test_task_handoff_is_persisted_before_push_and_deduplicates_recovery(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    item = {"type": "text", "text": "do this"}
    bot._enqueue_item(101, item)
    task = T._Task(task_id="accepted", chat_id=101, user_text="do this")
    bot._persist_task(task, [item["_inbox_id"]])
    saved = json.loads(T._INFLIGHT_FILE.read_text("utf-8"))
    assert saved[0]["incoming_ids"] == [item["_inbox_id"]]
    assert bot._incoming.pending() == []
    assert bot._backend.pop(timeout=0).task_id == "accepted"
    # A crash between writing the task and retiring its input must not replay both.
    bot._incoming.put(101, item, "crash-gap")
    task.incoming_ids = ["crash-gap"]
    restarted = make_bot(tmp_path)
    hold_worker(restarted, 101)
    restarted._restore_incoming([dict(task.to_dict(), _state="pending")])
    assert restarted._incoming.pending() == []
    assert 101 not in restarted._queues


def test_secret_answers_are_not_written_to_the_incoming_journal(tmp_path):
    bot = make_bot(tmp_path)
    bot._user_store.delete(101)
    sess = bot._get_session(101)
    sess.reg_state = "awaiting_password"
    sess.reg_name = "Test"
    bot._store.put(sess)
    bot._dispatch_logged({"update_id": 123, "message": {
        "message_id": 45, "date": 1, "chat": {"id": 101, "type": "private"},
        "from": {"id": 101}, "text": "a-secret-never-written"}})
    assert "a-secret-never-written" not in (tmp_path / "tg_incoming.json").read_text("utf-8")


def test_album_parts_are_durable_before_the_collect_timer(tmp_path):
    bot = make_bot(tmp_path)
    bot.album_collect_s = 60
    try:
        bot._buffer_album(101, "group", "photo-one", "caption")
        bot._buffer_album(101, "group", "photo-two", "")
        saved = bot._incoming.pending()
        assert [p["file_id"] for p in saved[0]["item"]["photos"]] == ["photo-one", "photo-two"]
    finally:
        for album in bot._albums.values():
            album["timer"].cancel()


def test_stop_discards_durable_items_including_a_buffered_album(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    bot._enqueue_item(101, {"type": "text", "text": "cancel me"})
    bot.album_collect_s = 60
    bot._buffer_album(101, "group", "photo-one", "caption")
    bot._request_stop(101)
    assert bot._incoming.pending() == []
    assert not bot._albums


def test_journal_write_failure_is_not_reported_as_acceptance(tmp_path):
    from tg_incoming import IncomingStore
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    store = IncomingStore(blocked / "inbox.json")
    try:
        store.put(101, {"type": "text", "text": "retry"}, "u1")
    except OSError:
        pass
    else:
        raise AssertionError("failed persistence was accepted")
    assert store.pending() == []


def test_stop_while_an_item_is_resolving_cannot_resurrect_its_task(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    item = {"type": "text", "text": "cancel me"}
    bot._enqueue_item(101, item)
    task = T._Task(task_id="late-resolution", chat_id=101, user_text="cancel me",
                   enqueue_ts=item["_arrival_ts"])
    bot._request_stop(101)
    assert bot._persist_task(task, [item["_inbox_id"]]) is False
    assert bot._backend.depth() == 0
    assert "late-resolution" not in bot._pending_journal


def test_a_background_followup_does_not_reuse_its_completed_update_id(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    contexts = []
    bot._dispatch_traced = lambda update: contexts.append(contextvars.copy_context())
    bot._dispatch_logged({"update_id": 456})
    thread = threading.Thread(target=contexts[0].run,
        args=(bot._enqueue_item, 101, {"type": "text", "text": "background continuation"}))
    thread.start()
    thread.join(3)
    assert len(bot._incoming.pending()) == 1


def test_stop_remains_effective_after_a_late_queue_push_and_restart(tmp_path):
    bot = make_bot(tmp_path)
    task = T._Task(task_id="raced-stop", chat_id=101, user_text="cancelled")
    bot._request_stop(101)
    bot._backend.push(task)  # a resolver that was already past its initial Stop check
    restarted = make_bot(tmp_path)
    assert restarted._stop_requested_after(task)


def test_old_store_cannot_overwrite_inputs_accepted_by_a_new_store(tmp_path):
    from tg_incoming import IncomingStore
    old = IncomingStore(tmp_path / "inbox.json")
    old.put(101, {"text": "old"}, "old")
    new = IncomingStore(tmp_path / "inbox.json")
    new.put(202, {"text": "new"}, "new")
    old.ack(["old"])
    assert [entry["id"] for entry in IncomingStore(tmp_path / "inbox.json").pending()] == ["new"]


def test_replacement_waits_until_the_previous_resolver_is_finished(tmp_path):
    old = make_bot(tmp_path)
    alive = [True]
    old._workers[101] = types.SimpleNamespace(is_alive=lambda: alive[0])
    assert old._incoming.claim(old)
    old._stopping = True
    new = make_bot(tmp_path)
    assert not new._incoming.claim(new)
    alive[0] = False
    assert new._incoming.claim(new)


def test_failed_recovery_leaves_start_stopped_and_the_saved_task_intact(tmp_path):
    bot = make_bot(tmp_path)
    task = T._Task(task_id="retry-start", chat_id=101, user_text="still saved")
    T._INFLIGHT_FILE.write_text(json.dumps([dict(task.to_dict(), _state="pending")]), "utf-8")
    bot._identify = lambda: ({"username": "Test"}, "")
    bot._register_commands = lambda: None
    bot.modules = {"tg_local_api": types.SimpleNamespace(enabled=lambda: False),
                   "reminders": types.SimpleNamespace(register=lambda *a, **k: None)}
    class BrokenBackend:
        def position_of(self, task_id):
            return 0
        def requeue(self, task):
            raise OSError("temporary queue outage")
    bot._backend = BrokenBackend()
    assert bot.start() is False
    assert not bot._running
    assert json.loads(T._INFLIGHT_FILE.read_text("utf-8"))[0]["task_id"] == "retry-start"


def test_ban_discards_only_that_users_buffered_work(tmp_path):
    bot = make_bot(tmp_path)
    bot._user_store.put(T._User(chat_id=101, name="User", status="approved"))
    for cid in (101, 202):
        hold_worker(bot, cid)
        bot._enqueue_item(cid, {"type": "text", "text": "queued"})
    bot.ban_user(101)
    assert [entry["chat_id"] for entry in bot._incoming.pending()] == [202]


def test_joining_a_queued_forward_keeps_the_old_task_until_replacement_is_durable(tmp_path):
    bot = make_bot(tmp_path)
    previous = T._Task(task_id="old-forward", chat_id=101, user_text="forwarded material")
    bot._persist_task(previous)
    combined = T._Task(task_id="combined", chat_id=101, user_text="what to do with it")
    assert bot._join_waiting(101, combined)
    bot._write_inflight()  # another writer, or a crash, before the new task is accepted
    assert json.loads(T._INFLIGHT_FILE.read_text("utf-8"))[0]["task_id"] == "old-forward"
    bot._persist_task(combined)
    saved = json.loads(T._INFLIGHT_FILE.read_text("utf-8"))
    assert len(saved) == 1 and saved[0]["task_id"] == "combined"
    assert "forwarded material" in saved[0]["user_text"]


def test_bot_stop_during_callback_does_not_acknowledge_unaccepted_work(tmp_path):
    bot = make_bot(tmp_path)
    def callback(update):
        bot._stopping = True
        bot._persist_task(T._Task(task_id="not-accepted", chat_id=101, user_text="animate"))
    bot._dispatch_traced = callback
    try:
        bot._dispatch_logged({"update_id": 789})
    except RuntimeError:
        pass
    assert not bot._incoming.seen("update:789")
    assert bot._backend.depth() == 0


def test_registered_reminder_sender_reports_a_rejected_delivery(tmp_path):
    bot = make_bot(tmp_path)
    bot._identify = lambda: ({"username": "Test"}, "")
    bot._register_commands = lambda: None
    bot._send_get_id = lambda *a, **k: None
    captured = []
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    bot.modules = {"tg_local_api": types.SimpleNamespace(enabled=lambda: False),
        "reminders": types.SimpleNamespace(register=lambda sender, *a, **k: captured.append(sender)),
        "threading": types.SimpleNamespace(Thread=Thread)}
    assert bot.start()
    try:
        captured[0](101, "due reminder")
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed reminder delivery was silently accepted")


def test_recovery_lookup_failure_preserves_both_durable_copies(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    item = {"type": "text", "text": "keep this"}
    bot._enqueue_item(101, item)
    task = T._Task(task_id="lookup-outage", chat_id=101, user_text="keep this",
                   incoming_ids=[item["_inbox_id"]])
    saved = [dict(task.to_dict(), _state="pending")]
    T._INFLIGHT_FILE.write_text(json.dumps(saved), "utf-8")
    class OfflineQueue:
        def position_of(self, task_id):
            raise OSError("queue unavailable")
    bot._backend = OfflineQueue()
    try:
        bot._restore_pending(saved)
        bot._restore_incoming(saved)
    except OSError:
        pass
    else:
        raise AssertionError("failed startup recovery was accepted")
    assert json.loads(T._INFLIGHT_FILE.read_text("utf-8")) == saved
    assert bot._incoming.pending()[0]["id"] == item["_inbox_id"]


@pytest.mark.parametrize("status", ["banned", "rejected", "pending", "missing", "logged_out"])
def test_recovery_discards_inputs_and_tasks_of_revoked_users(tmp_path, status):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    item = {"type": "text", "text": "accepted before ban"}
    bot._enqueue_item(101, item)
    task = T._Task(task_id="before-ban", chat_id=101, user_text="accepted before ban")
    bot._persist_task(task)
    # Simulate a crash after the status write but before _request_stop.
    user = bot._user_store.get(101)
    if status == "missing":
        bot._user_store.delete(101)
    elif status == "logged_out":
        sess = bot._get_session(101)
        sess.reg_state = "awaiting_login"
        bot._store.put(sess)
    else:
        user.status = status
        bot._user_store.put(user)
    restored = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    restored._backend = bot._backend
    hold_worker(restored, 101)
    restored._restore_pending([dict(task.to_dict(), _state="pending")])
    restored._restore_incoming([])
    assert restored._incoming.pending() == []
    assert restored._backend.depth() == 0
    assert not restored._pending_journal
    assert 101 not in restored._queues


def steer_note(bot):
    from steer import Inbox
    hold_worker(bot, 101)
    item = {"type": "text", "text": "make background red"}
    bot._enqueue_item(101, item)
    target = T._Task(task_id="being-made", chat_id=101, user_text="make a chart")
    bot._running_task[101] = [target]
    inbox = bot._steer_inboxes[target.task_id] = Inbox()
    note = T._Task(task_id="correction", chat_id=101, user_text=item["text"])
    bot._resolve_items = lambda cid, batch: bot._try_steer(cid, note)
    bot._resolve_and_push(101, [item])
    assert inbox.unread() == [item["text"]]
    return item, target, inbox


def test_steered_input_stays_durable_until_the_running_task_finishes(tmp_path):
    bot = make_bot(tmp_path)
    item, target, inbox = steer_note(bot)
    assert bot._incoming.pending()[0]["id"] == item["_inbox_id"]
    restored = make_bot(tmp_path)
    hold_worker(restored, 101)
    restored._restore_incoming([dict(target.to_dict(), _state="running")])
    assert restored._queues[101].get_nowait()["text"] == item["text"]
    inbox.drain()
    bot._requeue_unread_steer(target)
    assert bot._incoming.pending() == []


@pytest.mark.parametrize("queue_fails", [False, True])
def test_unread_steering_is_durable_during_its_requeue(tmp_path, queue_fails):
    bot = make_bot(tmp_path)
    item, target, inbox = steer_note(bot)
    if queue_fails:
        def unavailable(*a):
            raise OSError("temporarily unavailable")
        bot._enqueue_item = unavailable
    bot._requeue_unread_steer(target)
    pending = bot._incoming.pending()
    assert len(pending) == 1
    assert pending[0]["item"]["text"] == item["text"]
    assert (pending[0]["id"] == item["_inbox_id"]) is queue_fails


@pytest.mark.parametrize("stop_bot", [False, True])
def test_stopping_does_not_resurrect_or_lose_steered_notes(tmp_path, stop_bot):
    bot = make_bot(tmp_path)
    item, target, inbox = steer_note(bot)
    if stop_bot:
        bot._stopping = True
    else:
        bot._request_stop(101)
    bot._requeue_unread_steer(target)
    pending = bot._incoming.pending()
    assert [entry["id"] for entry in pending] == ([item["_inbox_id"]] if stop_bot else [])


def test_failed_steer_retirement_does_not_abort_task_cleanup(tmp_path):
    bot = make_bot(tmp_path)
    item, target, inbox = steer_note(bot)
    inbox.drain()
    def unavailable(keys):
        raise OSError("disk unavailable")
    bot._incoming.ack = unavailable
    bot._requeue_unread_steer(target)
    assert bot._incoming.pending()[0]["id"] == item["_inbox_id"]
    assert item["_inbox_id"] in bot._steer_incoming[target.task_id]


def test_fresh_input_after_stop_does_not_steer_into_the_stopped_task(tmp_path):
    bot = make_bot(tmp_path)
    _, old, inbox = steer_note(bot)
    bot._request_stop(101)
    fresh = {"type": "text", "text": "now make a different chart"}
    bot._enqueue_item(101, fresh)
    task = T._Task(task_id="fresh-after-stop", chat_id=101, user_text=fresh["text"])
    def resolve(cid, batch):
        if not bot._try_steer(cid, task):
            bot._persist_task(task)
    bot._resolve_items = resolve
    bot._resolve_and_push(101, [fresh])
    bot._requeue_unread_steer(old)
    assert bot._backend.pop(timeout=0).task_id == task.task_id


def test_multiple_consumers_cannot_admit_a_later_message_first(tmp_path):
    bot = make_bot(tmp_path)
    first_popped = threading.Event()
    later_started = threading.Event()
    pop = bot._backend.pop
    def delayed_pop(timeout):
        task = pop(timeout=0.05)
        if task and task.task_id == "first":
            first_popped.set()
            later_started.wait(1)
        return task
    bot._backend.pop = delayed_pop
    order = []
    def execute(task):
        order.append(task.task_id)
        if task.task_id == "second":
            later_started.set()
        if len(order) == 2:
            bot._running = False
    bot._execute_task = execute
    for name in ("first", "second"):
        bot._backend.push(T._Task(task_id=name, chat_id=101, user_text=name))
    bot._running = True
    threads = [threading.Thread(target=bot._consumer_loop, daemon=True) for _ in range(2)]
    try:
        threads[0].start()
        assert first_popped.wait(2)
        threads[1].start()
        for thread in threads:
            thread.join(3)
        assert order == ["first", "second"]
    finally:
        bot._running = False
        later_started.set()


@pytest.mark.parametrize("during_pop", [False, True])
def test_stopped_consumer_keeps_the_candidate_without_executing(tmp_path, during_pop):
    bot = make_bot(tmp_path)
    reached = threading.Event()
    release = threading.Event()
    if during_pop:
        pop = bot._backend.pop
        def blocked_pop(timeout):
            task = pop(timeout=0)
            reached.set()
            assert release.wait(2)
            return task
        bot._backend.pop = blocked_pop
    else:
        lock = bot._queue_claim_lock
        class Gate:
            def __enter__(self):
                reached.set()
                lock.acquire()
            def __exit__(self, *args):
                lock.release()
        lock.acquire()
        bot._queue_claim_lock = Gate()
    executed = []
    bot._execute_task = lambda task: executed.append(task.task_id)
    bot._backend.push(T._Task(task_id="keep-on-stop", chat_id=101, user_text="saved"))
    bot._running = True
    thread = threading.Thread(target=bot._consumer_loop, daemon=True)
    try:
        thread.start()
        assert reached.wait(2)
        bot._running = False
        bot._stopping = True
    finally:
        if during_pop:
            release.set()
        else:
            lock.release()
        thread.join(3)
    assert not thread.is_alive()
    assert executed == []
    assert bot._backend.depth() == 1
