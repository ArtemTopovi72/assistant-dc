"""Release Qt widgets while the QApplication is still alive.

At interpreter shutdown CPython drops the QApplication and the widgets in an
order it does not define, and a QWidget outliving its QApplication is a
segfault inside Qt -- not an exception.  The suite prints every PASS, exits
139 (0xC0000005 on Windows), and reads in a sweep as a failure with no failing
test, which is the worst shape a flake can take: it accuses the code under
test instead of the harness.

Measured on test_gui_supplement10 before this helper reached it: 6 crashes in
15 runs with all 22 checks passing every time.

test_gui_supplement9 discovered the recipe and kept a private copy; it lives
here now so the next Qt suite inherits it instead of rediscovering it at 40%.
Call it once, immediately before sys.exit.
"""


def release_widgets(app) -> None:
    """Close, unparent and delete every top-level widget, then drain the queue.

    deleteLater() only schedules; without the processEvents() that follows,
    the deletion is still pending when the interpreter starts tearing down.
    """
    try:
        for w in list(app.topLevelWidgets()):
            w.close()
            w.setParent(None)
            w.deleteLater()
        app.processEvents()
    except Exception:
        pass          # teardown must never turn a green run red


def release(widget) -> None:
    """Release ONE widget the same way, for a suite that builds several.

    close() alone leaves the C++ object alive and owned by Python, so it is
    destroyed later -- possibly during an unrelated test, which is how this
    shows up under pytest as a crash in the middle of a run rather than at
    exit.
    """
    try:
        widget.close()
        widget.setParent(None)
        widget.deleteLater()
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def install(app) -> None:
    """Release at interpreter exit, however the suite was started.

    A script calls release_widgets() itself before sys.exit, but pytest never
    runs that block -- and run_all hands most Qt suites to pytest. atexit fires
    in both, while the QApplication is still alive.
    """
    import atexit
    atexit.register(release_widgets, app)
