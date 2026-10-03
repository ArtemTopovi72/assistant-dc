import time
import tg_weather as W


def test_slow_advice_falls_back():
    slow = W._deadline(lambda *a: (time.sleep(2), {"k": "x"})[1], 0.2)
    t = time.time()
    assert slow([], "ru") is None and time.time() - t < 1


def test_fast_advice_passes_through():
    assert W._deadline(lambda *a: {"k": "x"}, 1)([], "ru") == {"k": "x"}


def test_chat_busy_until_forecast_sent():
    import threading
    gate, done = threading.Event(), threading.Event()

    import tg_queue

    class B(tg_queue.QueueMixin):
        _task_lock = threading.Lock()
        _chat_busy = {}
        def _send_text(self, *a): pass
        def _send_weather(self, *a):
            gate.wait(2); done.set()

    b = B()
    W.WeatherMixin._start_weather_lookup(b, 7, "Сочи", "ru")
    assert b._chat_busy.get(7) == 1
    gate.set(); done.wait(2); time.sleep(0.1)
    assert 7 not in b._chat_busy
