import os
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

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


def test_http_press_activates_qt_button_while_window_is_hidden(monkeypatch):
    import json
    import urllib.request
    from http.server import ThreadingHTTPServer

    app = QApplication.instance() or QApplication([])
    button = QPushButton('Voice')
    hits = []
    button.clicked.connect(lambda: hits.append(True))
    control = GuiControl({'voice': button})
    monkeypatch.setattr(debug_user, '_GUI_CONTROL', control)
    monkeypatch.setattr(debug_user, '_GUI_TOKEN', 'secret')
    monkeypatch.setattr(debug_user, 'enabled', lambda: True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), debug_user._api_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    response = []

    def send():
        request = urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}/gui/press',
            data=json.dumps({'action': 'voice'}).encode(),
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
