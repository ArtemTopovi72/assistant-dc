import threading
from types import SimpleNamespace

import ozon_client as ozon
import pytest


class Worker:
    def __init__(self, warm=False):
        self.pending = 0
        self._thread = SimpleNamespace(is_alive=lambda: warm)
        self.calls = []

    def fetch(self, path):
        self.calls.append(path)
        return path


def test_sequential_requests_reuse_warm_browser(monkeypatch):
    cold, warm = Worker(), Worker(warm=True)
    monkeypatch.setattr(ozon, '_BROWSERS', [cold, warm])
    assert ozon.fetch_page('search') == 'search'
    assert ozon.fetch_page('reviews') == 'reviews'
    assert warm.calls == ['search', 'reviews']
    assert cold.calls == []


def test_parallel_requests_use_another_worker(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    first, second = Worker(warm=True), Worker()
    def slow(path):
        entered.set()
        assert release.wait(5)
        return path
    first.fetch = slow
    monkeypatch.setattr(ozon, '_BROWSERS', [first, second])
    thread = threading.Thread(target=ozon.fetch_page, args=('first',))
    thread.start()
    try:
        assert entered.wait(5)
        assert ozon.fetch_page('second') == 'second'
        assert second.calls == ['second']
    finally:
        release.set()
        thread.join(5)
    assert first.pending == second.pending == 0


def test_failed_request_releases_the_worker(monkeypatch):
    worker = Worker(warm=True)
    def fail(path):
        raise ozon.OzonError('failed')
    worker.fetch = fail
    monkeypatch.setattr(ozon, '_BROWSERS', [worker])
    with pytest.raises(ozon.OzonError):
        ozon.fetch_page('fail')
    assert worker.pending == 0
