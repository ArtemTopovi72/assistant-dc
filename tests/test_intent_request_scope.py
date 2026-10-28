"""A request owns its read even when another chat sends identical words."""
import contextvars
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import intent


def test_identical_words_do_not_claim_another_threads_context():
    ready = threading.Event()
    release = threading.Event()
    seen = {}

    def chat_a():
        intent.bind_turn(["again"], "private A", "a photo", entry=True)
        ready.set()
        release.wait(5)
        seen["a"] = intent._bound("again", "private A", "a photo")

    worker = threading.Thread(target=chat_a)
    worker.start()
    assert ready.wait(5)
    try:
        with intent.turn_scope():
            intent.bind_turn(["again"], "private B", "a video", entry=True)
            # A separate graph thread must not discover either chat by words.
            with ThreadPoolExecutor(1) as pool:
                seen["graph"] = pool.submit(
                    lambda: (intent.bind_turn(["again"], "private A", "a photo"),
                             intent._bound("again", "private A", "a photo"))[1]).result()
            assert seen["graph"] == ("again", "private A", "a photo")
    finally:
        release.set()
        worker.join(5)
    assert seen["a"] == ("again", "private A", "a photo")


def test_snapshot_aliases_contextual_reads_and_scope_reset():
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(user)
        return json.dumps({"needs_tool": False, "ask": {"detailed": True}})
    ctx = SimpleNamespace(text_model=model, deps=SimpleNamespace(live=True))
    before = intent.current_turn()
    with intent.turn_scope():
        intent.bind_turn(["original"], "private A", "a photo", entry=True)
        intent.read(ctx, "original")
        snapshot = json.loads(json.dumps(intent.snapshot_turn()))
    assert intent.current_turn() is before
    with intent.turn_scope(snapshot):
        intent.bind_turn(["original"], "other view", "a photo")
        intent.turn_alias("translation")
        with ThreadPoolExecutor(1) as pool:
            copied = contextvars.copy_context()
            pool.submit(copied.run, intent.read, ctx, "original translation").result()
        assert len(calls) == 1
        intent.read(ctx, "original", "", "several pictures earlier in the chat")
        assert len(calls) == 2
    with intent.turn_scope():
        intent.bind_turn(["original"], "private B", "a video")
        intent.read(ctx, "original")
    assert len(calls) == 3
    assert "private B" in calls[-1] and "private A" not in calls[-1]


def test_ask_read_honors_intent_seam_without_cross_request_cache():
    import ask_read
    for detailed in (True, False):
        ctx = SimpleNamespace(intent_read=lambda *a, **kw: {"ask": {"detailed": detailed}},
                              deps=SimpleNamespace(live=True))
        with intent.turn_scope():
            assert ask_read.read("explain", ctx)["detailed"] is detailed


def test_graph_nodes_share_request_but_reused_gui_state_starts_fresh():
    import graph
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(user)
        return json.dumps({"needs_tool": False})
    def personality(ctx, state):
        intent.read(ctx, "translation")
        intent.read(ctx, "original translation")
        return state
    ctx = SimpleNamespace(text_model=model, deps=SimpleNamespace(
        live=True,
        _translate_to_english=lambda *a, **kw: "translation",
        personality_node=personality, tts_node=lambda ctx, state: state))
    compiled = graph.build_graph(ctx)
    def gui():
        state = {"user_input": "original", "messages": []}
        state = compiled.invoke(state)
        state["user_input"] = "original"
        compiled.invoke(state)
        assert intent.current_turn() is None
    with ThreadPoolExecutor(1) as pool:
        pool.submit(gui).result()
    assert len(calls) == 2
    with intent.turn_scope():
        intent.bind_turn(["original"], "private previous", entry=True)
        intent.read(ctx, "original")
        snapshot = intent.snapshot_turn()
    with intent.turn_scope(snapshot):
        compiled.invoke({"user_input": "original", "messages": []})
    assert len(calls) == 3
    assert "private previous" in calls[-1]


def test_concurrent_readers_share_one_model_call():
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(user)
        entered.set()
        assert release.wait(5)
        return json.dumps({"needs_tool": False})
    ctx = SimpleNamespace(text_model=model, deps=SimpleNamespace(live=True))
    with intent.turn_scope():
        intent.bind_turn(["same"])
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(contextvars.copy_context().run, intent.read, ctx, "same")
            assert entered.wait(5)
            second = pool.submit(contextvars.copy_context().run, intent.read, ctx, "same")
            release.set()
            assert first.result() == second.result()
    assert len(calls) == 1


def test_failed_read_is_shared_and_exception_restores_outer_scope():
    calls = []
    def broken(ctx, system, user, **kw):
        calls.append(user)
        raise RuntimeError("offline model failure")
    ctx = SimpleNamespace(text_model=broken, deps=SimpleNamespace(live=True))
    with intent.turn_scope() as outer:
        try:
            with intent.turn_scope():
                intent.bind_turn(["same"])
                assert intent.read(ctx, "same")["ok"] is False
                assert intent.read(ctx, "same")["ok"] is False
                raise ValueError("leave request")
        except ValueError:
            pass
        assert intent.current_turn() is outer
    assert len(calls) == 1
