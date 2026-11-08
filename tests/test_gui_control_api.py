import threading
import time
import pytest

from PyQt5.QtWidgets import QApplication, QPushButton
from gui_control import GuiControl
import debug_user


def test_gui_control_runs_allowlisted_button_on_qt_thread():
    app = QApplication.instance() or QApplication([])
    button = QPushButton('Voice')
    hits = []
    button.clicked.connect(lambda: hits.append(threading.current_thread().name))
    control = GuiControl({'voice': button})
    result = []
    worker = threading.Thread(target=lambda: result.append(control.press('voice')))
    worker.start()
    end = time.monotonic() + 3
    while worker.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    worker.join(1)
    assert result == [{'ok': True}]
    assert hits == [threading.current_thread().name]
    assert control.press('restart') == {'error': 'unknown action'}


def test_registration_rotates_local_token(monkeypatch, tmp_path):
    monkeypatch.setattr(debug_user, 'ROOT', str(tmp_path))
    monkeypatch.setattr(debug_user, '_GUI_CONTROL', None)
    monkeypatch.setattr(debug_user, '_GUI_TOKEN', '')
    debug_user.register_gui_control(object())
    first = (tmp_path / 'gui-control-token').read_text(encoding='ascii')
    debug_user.register_gui_control(object())
    second = (tmp_path / 'gui-control-token').read_text(encoding='ascii')
    assert len(first) >= 32 and first != second


def test_gui_control_does_not_activate_disabled_button():
    app = QApplication.instance() or QApplication([])
    button = QPushButton('Stop')
    button.setEnabled(False)
    control = GuiControl({'stop': button})
    result = []
    worker = threading.Thread(target=lambda: result.append(control.press('stop')))
    worker.start()
    end = time.monotonic() + 3
    while worker.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    worker.join(1)
    assert result == [{'error': 'action disabled'}]


def test_gui_endpoint_requires_token_and_rejects_browser_origin(monkeypatch):
    import json
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(debug_user, '_GUI_TOKEN', 'secret')
    monkeypatch.setattr(debug_user, '_GUI_CONTROL', SimpleNamespace(
        buttons={'voice': None}, press=lambda action: calls.append(action) or {'ok': True}))
    monkeypatch.setattr(debug_user, 'enabled', lambda: True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), debug_user._api_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/gui/press'

    def post(headers):
        req = urllib.request.Request(url, data=json.dumps({'action': 'voice'}).encode(),
                                     headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    try:
        assert post({}) == 403
        assert post({'X-Assistant-Control': 'secret', 'Origin': 'https://example.com'}) == 403
        assert post({'X-Assistant-Control': 'secret'}) == 200
        assert calls == ['voice']
        req = urllib.request.Request(url.replace('/press', '/actions'),
                                     headers={'X-Assistant-Control': 'secret'})
        with urllib.request.urlopen(req, timeout=3) as response:
            assert json.load(response) == {'actions': ['voice']}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('action', ['voice', 'restart'])
def test_http_press_activates_qt_button_while_window_is_hidden(monkeypatch, action):
    import json
    import urllib.request
    from http.server import ThreadingHTTPServer

    app = QApplication.instance() or QApplication([])
    button = QPushButton('Voice')
    hits = []
    button.clicked.connect(lambda: hits.append(True))
    control = GuiControl({action: button})
    monkeypatch.setattr(debug_user, '_GUI_CONTROL', control)
    monkeypatch.setattr(debug_user, '_GUI_TOKEN', 'secret')
    monkeypatch.setattr(debug_user, 'enabled', lambda: True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), debug_user._api_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    response = []

    def send():
        request = urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}/gui/press',
            data=json.dumps({'action': action}).encode(),
            headers={'X-Assistant-Control': 'secret'}, method='POST')
        with urllib.request.urlopen(request, timeout=3) as result:
            response.append(json.load(result))

    worker = threading.Thread(target=send)
    try:
        worker.start()
        end = time.monotonic() + 3
        while worker.is_alive() and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        worker.join(1)
        assert not button.isVisible()
        assert response == [{'ok': True}]
        assert hits == [True]
    finally:
        server.shutdown()
        server.server_close()


def test_restart_acknowledges_before_slow_shutdown():
    app = QApplication.instance() or QApplication([])
    button = QPushButton('Restart')
    done = threading.Event()
    acknowledged_while_running = []
    button.clicked.connect(lambda: acknowledged_while_running.append(done.wait(1)))
    control = GuiControl({'restart': button})
    result = []
    worker = threading.Thread(target=lambda: (result.append(control.press('restart')), done.set()))
    worker.start()
    end = time.monotonic() + 3
    while worker.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.01)
    worker.join(2)
    assert result == [{'ok': True}]
    assert acknowledged_while_running == [True]


def test_model_control_runs_on_gui_thread():
    app = QApplication.instance() or QApplication([])
    calls = []
    control = GuiControl({}, model_state=lambda: {'model': 'old', 'switching': False},
                         model_select=lambda model, thinking, lock: calls.append(
                             (threading.current_thread().name, model, thinking)) or {'ok': True})
    result = []
    worker = threading.Thread(target=lambda: result.append(control.model_select('new', True)))
    worker.start()
    end = time.monotonic() + 3
    while worker.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.01)
    worker.join(1)
    assert result == [{'ok': True}]
    assert calls == [(threading.current_thread().name, 'new', True)]
    assert control.model_state() == {'model': 'old', 'switching': False}


def test_model_http_api_requires_token_and_validates_installed_model(monkeypatch):
    import json
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(debug_user, '_GUI_TOKEN', 'secret')
    monkeypatch.setattr(debug_user, 'enabled', lambda: True)
    monkeypatch.setattr(debug_user, '_available_gui_models', lambda: ['model-a'])
    monkeypatch.setattr(debug_user, '_GUI_CONTROL', SimpleNamespace(
        model_state=lambda: {'model': 'old', 'switching': False},
        model_select=lambda model, thinking: calls.append((model, thinking)) or {'ok': True}))
    server = ThreadingHTTPServer(('127.0.0.1', 0), debug_user._api_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/gui/model'

    def request(body=None, headers=None):
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers=headers or {}, method='GET' if body is None else 'POST')
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)
    try:
        assert request()[0] == 403
        assert request(headers={'X-Assistant-Control': 'secret', 'Origin': 'https://example.com'})[0] == 403
        assert request(headers={'X-Assistant-Control': 'secret'})[1]['model'] == 'old'
        assert request({'model': 'missing', 'thinking': True}, {'X-Assistant-Control': 'secret'})[0] == 400
        assert request({'model': 'model-a', 'thinking': 'yes'}, {'X-Assistant-Control': 'secret'})[0] == 400
        assert request({'model': 'model-a', 'thinking': True}, {'X-Assistant-Control': 'secret'})[0] == 200
        assert calls == [('model-a', True)]
    finally:
        server.shutdown()
        server.server_close()


def test_model_control_waits_for_queue_lock_without_blocking_gui():
    app = QApplication.instance() or QApplication([])
    lock = threading.Lock()
    lock.acquire()
    seen, result = [], []
    control = GuiControl({}, model_lock=lambda: lock,
                         model_select=lambda model, thinking, held: seen.append(held is lock) or {'ok': True})
    worker = threading.Thread(target=lambda: result.append(control.model_select('new', False)))
    worker.start()
    app.processEvents()
    assert worker.is_alive() and not seen
    lock.release()
    end = time.monotonic() + 3
    while worker.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.01)
    worker.join(1)
    assert result == [{'ok': True}] and seen == [True]


def test_idle_queue_claim_does_not_hold_lock_while_waiting():
    from types import SimpleNamespace
    from tg_queue import QueueMixin

    timeouts = []
    bot = SimpleNamespace(_queue_claim_lock=threading.Lock(), _running=True,
                          _stopping=False,
                          _backend=SimpleNamespace(pop=lambda timeout: timeouts.append(timeout)))
    assert QueueMixin._claim_task(bot) is None
    assert timeouts == [0]
    assert bot._queue_claim_lock.acquire(blocking=False)
    bot._queue_claim_lock.release()
