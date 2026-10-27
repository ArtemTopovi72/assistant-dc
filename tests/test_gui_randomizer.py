import pytest
import threading
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from gui_randomizer_tab import RandomizerTab
from gui_busy_state import BusyStateMixin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class Cherries:
    def choice(self, values):
        return '🍒'


def test_spin_finishes_and_reenables_the_button(app):
    tab = RandomizerTab()
    tab.rng = Cherries()
    tab.spin_button.click()
    assert not tab.spin_button.isEnabled()
    for _ in range(8):
        tab._tick()
    assert tab.spin_button.isEnabled() and not tab.timer.isActive()
    assert tab.reels.text().count('🍒') == 3
    assert 'Jackpot' in tab.spin_result.text()
    tab.close()


def test_range_errors_do_not_crash_and_a_single_value_is_valid(app):
    tab = RandomizerTab()
    tab.low.setText('5'); tab.high.setText('1')
    tab.number_button.click()
    assert 'Lower bound' in tab.number_result.text()
    tab.low.setText('7'); tab.high.setText('7')
    tab.number_button.click()
    assert tab.number_result.text() == '7'
    tab.low.setText('no')
    tab.number_button.click()
    assert 'integers' in tab.number_result.text()
    tab.low.setText('1 2'); tab.high.clear()
    tab.number_button.click()
    assert 'integers' in tab.number_result.text()
    tab.close()


def test_prediction_runs_off_the_ui_thread_and_uses_the_period(app):
    calls = []
    ctx = SimpleNamespace(text_model=lambda c, s, u, **kw: calls.append(u) or 'A calm week ahead.')
    tab = RandomizerTab(SimpleNamespace(ctx=ctx))
    tab.period.setCurrentIndex(1)
    tab.predict_button.click()
    for _ in range(200):
        QTest.qWait(10)
        if tab.predict_button.isEnabled():
            break
    assert tab.predict_button.isEnabled() and tab.period.isEnabled()
    assert 'A calm week ahead.' in tab.prediction_result.text()
    assert 'week' in calls[0]
    tab.close()


def test_window_shutdown_cancels_and_joins_prediction_without_cancelling_chat(app):
    entered, chat_cancel = threading.Event(), threading.Event()
    def model(ctx, *a, **kw):
        entered.set()
        ctx.cancel_event.wait(2)
        return 'Must not be delivered after cancellation'
    class Window(BusyStateMixin):
        pass
    window = Window()
    window.ctx = SimpleNamespace(text_model=model, cancel_event=chat_cancel)
    window.randomizer_tab = tab = RandomizerTab(window)
    tab.predict_button.click()
    try:
        assert entered.wait(1)
        assert tab.worker in window._live_threads()
        assert window._join_live_threads(1000) == []
        assert not tab.worker.isRunning() and not chat_cancel.is_set()
        app.processEvents()
        assert 'Must not be delivered' not in tab.prediction_result.text()
    finally:
        chat_cancel.set()
        if hasattr(tab.worker, 'cancel'):
            tab.worker.cancel()
        tab.worker.wait(2500)
        tab.close()
