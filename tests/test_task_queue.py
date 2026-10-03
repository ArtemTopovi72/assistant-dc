"""Deterministic regression test for the GUI task queue's execution-state machine
(AssistantWindow). Runs headless under offscreen Qt with real QWidgets but a stubbed
dispatcher, so it exercises the REAL queue methods without LM Studio / ComfyUI / a worker
thread. QTimer.singleShot is made synchronous for determinism.

Run:  venv/Scripts/python.exe tests/test_task_queue.py   (exit 0 = all pass)
"""
import os
import sys
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PyQt5.QtWidgets import QApplication, QListWidget, QPushButton, QLineEdit
from PyQt5.QtCore import QTimer

app = QApplication.instance() or QApplication([])
import gui


def build_shim():
    class Shim:
        pass
    s = Shim()
    s._task_queue = []
    s._queue_paused = False
    s._running_task = None
    s._turn_had_error = False
    s._turn_cancelled = False
    s._running_dispatched = False
    s._completed_count = 0
    s._failed_count = 0
    s._cancelled_count = 0
    s.queue_list = QListWidget()
    s.input = QLineEdit()
    s.q_up = QPushButton(); s.q_down = QPushButton(); s.q_del = QPushButton()
    s.q_clear = QPushButton(); s.q_pause = QPushButton(); s.q_pause.setCheckable(True)
    s._queue_bar_widgets = (s.q_up, s.q_down, s.q_del, s.q_clear, s.q_pause)
    s._busy_flag = False
    s.dispatched = []
    s._busy = lambda: s._busy_flag

    def _disp(t):
        s.dispatched.append(t)
        s._busy_flag = True   # dispatching starts a worker -> busy
    s._dispatch_user_text = _disp
    s._add_system = lambda t: None
    for name in ["_Q_ICON", "_refresh_queue_ui", "_enqueue", "_pending_indices",
                 "_queue_move", "_queue_remove", "_queue_clear", "_queue_set_paused",
                 "_maybe_drain_queue", "_dispatch_queued"]:
        attr = getattr(gui.AssistantWindow, name)
        setattr(s, name, attr if isinstance(attr, dict) else types.MethodType(attr, s))
    return s


def main():
    QTimer.singleShot = staticmethod(lambda ms, fn: fn())  # synchronous for the test
    s = build_shim()

    def finish(error=False):
        s._turn_had_error = error
        s._busy_flag = False        # worker done -> _set_busy(False) -> drain
        s._maybe_drain_queue()

    ok = []
    def chk(c, m):
        ok.append(bool(c)); print(("PASS " if c else "FAIL ") + m)

    s._busy_flag = True
    s._enqueue("A"); s._enqueue("B"); s._enqueue("C")
    chk(len(s._task_queue) == 3 and s.dispatched == [], "3 pending while busy")
    chk(all(it["status"] == "pending" for it in s._task_queue), "all pending status")

    s._busy_flag = False; s._maybe_drain_queue()
    chk(s.dispatched == ["A"] and s._running_task["text"] == "A"
        and s._running_task["status"] == "running", "A dispatched + marked running")
    chk([it["status"] for it in s._task_queue] == ["running", "pending", "pending"],
        "states running/pending/pending")

    s.queue_list.setCurrentRow(0); s._queue_remove()
    chk(len(s._task_queue) == 3, "running task not removable")

    s.queue_list.setCurrentRow(2); s._queue_move(-1)
    chk([it["text"] for it in s._task_queue] == ["A", "C", "B"], "pending reorder (running fixed)")

    finish()
    chk(s._completed_count == 1 and s.dispatched == ["A", "C"]
        and s._running_task["text"] == "C", "A done (count=1), C running")

    finish(error=True)
    chk(s._failed_count == 1 and s.dispatched == ["A", "C", "B"]
        and s._running_task["text"] == "B", "C failed (count=1), B running")

    s._queue_set_paused(True); finish()
    chk(s._completed_count == 2 and s._task_queue == [] and s._running_task is None,
        "B done, queue empty")

    s._enqueue("D")
    chk([it["status"] for it in s._task_queue] == ["pending"] and s.dispatched[-1] == "B",
        "paused: D not dispatched")
    s._queue_set_paused(False)
    chk(s.dispatched[-1] == "D" and s._running_task["text"] == "D", "resume dispatches D")
    s._queue_clear()
    chk(len(s._task_queue) == 1 and s._task_queue[0]["text"] == "D", "clear keeps running task")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
