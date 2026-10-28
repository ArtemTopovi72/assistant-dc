import threading
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from gui_translator_tab import TranslatorTab
from gui_busy_state import BusyStateMixin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_text_translation_is_threaded_and_copy_and_clear_work(app):
    calls = []
    ctx = SimpleNamespace(text_model=lambda c, s, u, **kw: calls.append((s,u)) or '{"translation":"Bonjour"}')
    tab = TranslatorTab(SimpleNamespace(ctx=ctx))
    tab.target.setCurrentIndex(tab.target.findData('fr'))
    tab.text_in.setPlainText('Hello')
    tab.translate_btn.click()
    for _ in range(200):
        QTest.qWait(10)
        if tab.translate_btn.isEnabled(): break
    assert tab.result.toPlainText() == 'Bonjour' and 'French' in calls[0][0]
    tab.copy_btn.click(); assert QApplication.clipboard().text() == 'Bonjour'
    tab.clear_btn.click(); assert not tab.text_in.toPlainText() and not tab.result.toPlainText()
    tab.close()


def test_translator_worker_is_cancelled_at_window_shutdown(app):
    entered, cancel = threading.Event(), threading.Event()
    def model(ctx, *a, **kw):
        entered.set(); ctx.cancel_event.wait(3)
        return 'Late result'
    class Host(BusyStateMixin): pass
    host = Host(); host.ctx = SimpleNamespace(text_model=model, cancel_event=cancel)
    host.translator_tab = tab = TranslatorTab(host)
    tab.text_in.setPlainText('Hello'); tab.translate_btn.click()
    try:
        assert entered.wait(1)
        assert tab.worker in host._live_threads()
        assert host._join_live_threads(1000) == []
        app.processEvents()
        assert not cancel.is_set() and not tab.worker.isRunning()
        assert tab.result.toPlainText() != 'Late result'
    finally:
        tab.worker.cancel(); tab.worker.wait(3000); tab.close()


def test_clear_and_cancel_stop_voice_playback(app):
    stopped = []
    tab = TranslatorTab()
    tab.player = SimpleNamespace(stop=lambda: stopped.append(True))
    tab._clear(); tab._cancel()
    assert len(stopped) == 2
    tab.close()
