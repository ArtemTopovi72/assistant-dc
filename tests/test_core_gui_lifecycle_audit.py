"""Filesystem/state regressions; no GUI, model or live service is started."""
import json
import threading
import types

import pytest

from app_runtime import _migrate_memory_to_profiles
from gui import AssistantWindow
from gui_workers import LibraryBuildWorker
from gui_database_tab import DatabaseTabMixin
from gui_busy_state import BusyStateMixin
from models import _atomic_write_json


def test_api_model_select_rejects_busy_and_switches_when_idle(monkeypatch):
    import llm
    import comfy_client
    monkeypatch.setattr(llm, 'inflight', lambda: 0)
    monkeypatch.setattr(comfy_client, 'card_in_use', lambda *a, **kw: False)
    gate = threading.Event()
    gate.set()
    bot = types.SimpleNamespace(_chat_busy={1: 1}, _task_lock=threading.Lock(),
                                _queue_claim_lock=threading.Lock(),
                                _backend=types.SimpleNamespace(depth=lambda: 0))
    calls = []
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(model_name='old', no_think=True),
        _active_model_id='old', model_switch_worker=None, worker=None, _running_task=None,
        _task_queue=[], telegram_tab=types.SimpleNamespace(_bot=bot),
        _model_switch_ready=gate, _api_old_no_think=None,
        _start_model_switch=lambda model, **kw: calls.append((model, kw)))
    assert 'error' in AssistantWindow._api_model_select(host, 'new', True)
    assert gate.is_set() and host.ctx.model_name == 'old'
    bot._chat_busy.clear()
    bot._queue_claim_lock.acquire()
    try:
        assert 'error' in AssistantWindow._api_model_select(host, 'new', True)
        assert gate.is_set() and not calls
    finally:
        bot._queue_claim_lock.release()
    assert AssistantWindow._api_model_select(host, 'new', True)['ok']
    assert not gate.is_set() and host.ctx.model_name == 'new' and not host.ctx.no_think
    assert calls == [('new', {'restore_on_failure': True})]


def test_telegram_context_waits_until_api_model_switch_finishes():
    import time
    from gui_telegram_tab import TelegramTab
    gate = threading.Event()
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(model_name='new'), _model_switch_ready=gate)
    tab = types.SimpleNamespace(host=host, _bot_ctx_view=None, _bot_cancel=threading.Event())
    results = []
    worker = threading.Thread(target=lambda: results.append(TelegramTab._bot_ctx(tab)))
    worker.start()
    time.sleep(.05)
    assert worker.is_alive() and not results
    gate.set()
    worker.join(2)
    assert results and results[0].model_name == 'new'


def test_stop_cancels_database_build_worker():
    worker = LibraryBuildWorker([])
    host = types.SimpleNamespace(
        ctx=types.SimpleNamespace(cancel_event=threading.Event()),
        _busy=lambda: True, _lib_build_worker=worker, _scan_worker=None,
        _add_system=lambda text: None, _set_status=lambda text: None,
        stage=types.SimpleNamespace(set_stage=lambda text: None),
    )
    AssistantWindow._cancel_current(host)
    assert worker._cancel is True


def test_database_build_stop_finish_and_build_again():
    class Control:
        def __init__(self):
            self.enabled = True

        def setEnabled(self, enabled):
            self.enabled = enabled

        def setText(self, text):
            pass

        def setRange(self, *args):
            pass

        def show(self):
            pass

        def hide(self):
            pass

    class Worker(LibraryBuildWorker):
        def start(self):
            pass

    class Host(DatabaseTabMixin, BusyStateMixin):
        _cancel_current = AssistantWindow._cancel_current

        def _db_listed_paths(self):
            return ["fixture.txt"]

        def _maybe_drain_queue(self):
            pass

        def _db_update_stats(self):
            pass

        def _add_system(self, text):
            pass

        def _set_status(self, text):
            pass

    host = Host()
    host.ctx = types.SimpleNamespace(cancel_event=threading.Event(), mic_disabled=False)
    host.parts = {"LibraryBuildWorker": Worker, "GUI_SERIALIZE_GPU_JOBS": False}
    host.graph = object()
    for name in ("_library", "_lib_build_worker", "_scan_worker", "worker", "model_switch_worker",
                 "redraw_worker", "research_worker", "transcribe_worker", "compact_worker", "cap"):
        setattr(host, name, None)
    for name in ("db_build_btn", "db_progress", "db_status", "input", "send_btn", "cam_btn",
                 "redraw_btn", "style_preset_btn", "fixhands_btn", "fixartifact_btn", "whole_frame_btn",
                 "retryhands_btn", "talk_btn", "capture_btn", "stop_btn"):
        setattr(host, name, Control())
    host.stop_btn.enabled = False
    host.stage = types.SimpleNamespace(set_stage=lambda text: None)
    host.worker = object()
    host.ctx.cancel_event.set()
    host._db_build()
    assert host._lib_build_worker is None
    assert host.ctx.cancel_event.is_set()
    host.worker = None
    for _ in range(2):
        host._db_build()
        assert host.stop_btn.enabled is True
        assert not host.ctx.cancel_event.is_set()
        worker = host._lib_build_worker
        host._cancel_current()
        assert worker._cancel is True
        host._on_db_finished()
        assert host._lib_build_worker is None
        assert host.stop_btn.enabled is False


def test_memory_migration_resumes_without_overwriting_existing_profile(tmp_path):
    default = tmp_path / "default"
    default.mkdir()
    session = tmp_path / "session_memory.json"
    session.write_text('[{"kind":"fact","text":"legacy fact"}]', encoding="utf-8")
    legacy_summary = tmp_path / "summary.json"
    legacy_summary.write_text('{"text":"legacy summary"}', encoding="utf-8")
    current_summary = default / "summary.json"
    current_summary.write_text('{"text":"current summary"}', encoding="utf-8")

    _migrate_memory_to_profiles(tmp_path)

    assert json.loads((default / session.name).read_text(encoding="utf-8")) == [
        {"kind": "fact", "text": "legacy fact"}]
    assert not session.exists()
    assert json.loads(current_summary.read_text(encoding="utf-8")) == {"text": "current summary"}
    assert legacy_summary.exists()


def test_atomic_json_serialization_failure_cleans_private_temp(tmp_path):
    target = tmp_path / "facts.json"
    target.write_text('[{"text":"previous facts"}]', encoding="utf-8")
    with pytest.raises(TypeError):
        _atomic_write_json(target, {"private": "sensitive memory", "bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == [{"text": "previous facts"}]
    assert list(tmp_path.glob("*.tmp")) == []
