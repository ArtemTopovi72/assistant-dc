"""Keep the pytest half of the suite off the machine's real services.

`tests/offline_guard.py` already stubs `llm.send_to_lm_studio`, but it only
protects the four suites that remember to import it, and it only covers that
one entry point. Everything else -- a `requests.post` to ComfyUI, a raw
`urlopen` to LM Studio -- went straight out to the GPU. Four suites were
driving real renders (they now live in `bench/live_image*.py`), and the cost
was not just slowness: a run competed for the very card it was pretending not
to touch, and passed or failed on whether a model happened to be loaded.

So the socket layer itself refuses, at the two ports that matter. The error
names the offender, because a test that reaches for a live service is a defect
in the test, not a missing server.

Set ALLOW_LIVE=1 to lift it -- for a deliberate live run under pytest.
"""
import os
import socket

BLOCKED_PORTS = {1234, 8000}

_real_create = socket.create_connection
_real_connect = socket.socket.connect


class LiveServiceBlocked(RuntimeError):
    pass


def _port_of(address):
    try:
        return int(address[1])
    except Exception:
        return None


def _refuse(port):
    raise LiveServiceBlocked(
        "this suite tried to open a connection to port %d (LM Studio / ComfyUI). "
        "Tests under tests/ must be offline and deterministic -- stub the call, or "
        "move the suite to bench/. Set ALLOW_LIVE=1 to override." % port)


def pytest_configure(config):
    if os.environ.get("ALLOW_LIVE") == "1":
        return

    def create_connection(address, *a, **kw):
        p = _port_of(address)
        if p in BLOCKED_PORTS:
            _refuse(p)
        return _real_create(address, *a, **kw)

    def connect(self, address, *a, **kw):
        p = _port_of(address)
        if p in BLOCKED_PORTS:
            _refuse(p)
        return _real_connect(self, address, *a, **kw)

    socket.create_connection = create_connection
    socket.socket.connect = connect


def pytest_sessionfinish(session, exitstatus):
    """Release Qt widgets while the QApplication is still alive.

    37 suites here build widgets; two released them. The rest left the C++
    objects to be destroyed at interpreter shutdown, in an order CPython does
    not define against the QApplication -- which is a segfault inside Qt, not
    an exception. test_gui_supplement10 exited 0xC0000005 with every check
    printed PASS, and a sweep can only read that as a failure with no failing
    test.

    Doing it here covers every pytest suite at once, instead of waiting for
    each to rediscover the cure. A suite with no QApplication gets a no-op.
    """
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    from qt_teardown import release_widgets
    release_widgets(app)
