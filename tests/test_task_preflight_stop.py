"""Stop during a semantic preflight must not start the agent afterwards."""
import threading
from types import SimpleNamespace

import tg_bot as T
from fakes import reads
from test_tg_incoming_durability import make_bot


def test_stop_during_language_read_skips_the_agent_and_cleans_inflight(tmp_path):
    bot = make_bot(tmp_path)
    entered, release = threading.Event(), threading.Event()
    invoked, sent, errors = [], [], []

    def semantic(text):
        entered.set()
        assert release.wait(3)
        return {"language_mode": ""}

    shared = SimpleNamespace(**reads(read=semantic), load_memory=lambda path: None)
    bot.reads = shared
    bot._get_ctx = lambda: shared
    bot._get_graph = lambda: object()
    bot.graph_builder = lambda ctx: SimpleNamespace(invoke=lambda state: invoked.append("graph"))
    bot._run_task_inner = lambda *args: invoked.append("agent")
    bot._send_text = lambda cid, text, **kw: sent.append(text)
    task = T._Task(task_id="stopped-preflight", chat_id=101, user_text="Расскажи про фотографию")
    bot._persist_task(task)
    assert bot._backend.pop(timeout=0) is task

    def run():
        try:
            bot._execute_task(task)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        assert not bot._running_task.get(101)
        bot._request_stop(101)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert invoked == []
    assert sent.count(T._t("cancel_done", "ru")) == 1
    assert not bot._running_task.get(101)
    assert task.task_id not in bot._task_started
    assert task.task_id not in bot._task_cancels
    assert task.task_id not in bot._pending_journal


def test_clear_during_language_read_preserves_the_new_generation_language(tmp_path):
    bot = make_bot(tmp_path)
    entered, release = threading.Event(), threading.Event()
    invoked, errors = [], []

    def semantic(text):
        entered.set()
        assert release.wait(3)
        return {"language_mode": "en"}

    shared = SimpleNamespace(**reads(read=semantic, pick=lambda *a: "english"),
                             load_memory=lambda path: None)
    bot.reads = shared
    bot._get_ctx = lambda: shared
    bot._get_graph = lambda: object()
    bot.graph_builder = lambda ctx: invoked.append("build") or object()
    bot._run_task_inner = lambda *args: invoked.append("agent")
    sess = bot._get_session(101)
    task = T._Task(task_id="cleared-preflight", chat_id=101, user_text="Answer in English forever")
    bot._persist_task(task)
    assert bot._backend.pop(timeout=0) is task

    def run():
        try:
            bot._execute_task(task)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        sess.clear_context()
        generation = sess.context_generation
        sess.turn_lang, sess.lang_pin, sess.lang_pin_ts = "ru", "", 123.0
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert sess.context_generation == generation
    assert (sess.turn_lang, sess.lang_pin, sess.lang_pin_ts) == ("ru", "", 123.0)
    assert invoked == []
    assert task.task_id not in bot._pending_journal
