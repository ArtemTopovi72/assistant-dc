"""Route Python logging records onto the Qt event loop.

A logging handler can be called from ANY thread — the model loader, a worker,
the Telegram poller — but a Qt widget may only be touched from the GUI thread.
_LogBridge is a QObject whose signal does that hop: emit() from any thread,
the connected slot runs on the GUI thread and appends to the log pane.

That indirection is the whole reason this exists. Appending to the widget
directly from a worker is the classic way to make PyQt die without a traceback.
"""
import logging
import re

from PyQt5.QtCore import QObject, pyqtSignal


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BRAILLE_RE = re.compile(r"[⠀-⣿]")


def clean_terminal_text(text: str) -> str:
    """Printable text only for the log pane.

    Live 2026-09-13: a "model revive ok" record carried the raw `lms load`
    TUI -- ESC[?25l cursor codes and a braille spinner -- and
    QTextEdit.append died with an access violation inside DirectWrite while
    laying out those glyphs (the whole app went down with it). ANSI escapes
    and control characters are stripped, a CR-driven progress line keeps
    its last frame, spinner glyphs are dropped.
    """
    text = _ANSI_RE.sub("", str(text))
    text = _BRAILLE_RE.sub("", text)
    if "\r" in text:
        text = "\n".join(seg.split("\r")[-1] for seg in text.split("\n"))
    return _CTRL_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Live log handler -> mini terminal
# --------------------------------------------------------------------------- #
class _LogBridge(QObject):
    line = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Pure-Python logging.Handler that forwards lines through a QObject signal.

    Composition (not QObject subclassing) keeps the handler's lifetime decoupled
    from Qt, so logging.shutdown never touches a deleted C++ object.
    """

    def __init__(self):
        super().__init__()
        self.bridge = _LogBridge()
        self.line = self.bridge.line
        self.setFormatter(logging.Formatter("%(asctime)s  %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.bridge.line.emit(clean_terminal_text(self.format(record)))
        except Exception:
            pass
