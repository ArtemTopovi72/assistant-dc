"""The in-flight journal never goes back to an older snapshot.

Bug: _write_inflight took its snapshot under the task lock but wrote it after
releasing it. A writer that stalled in between could land last, putting a
task that had already finished back into tg_inflight.json; after a crash its
user was told the task was "interrupted".
"""
import json
import os
import sys
import tempfile
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")


def test_a_stale_snapshot_does_not_overwrite_a_newer_one(monkeypatch):
    import tg_bot as T
    import tg_queue as Q
    T.redirect_data_dir(tempfile.mkdtemp())
    bot = T.TelegramBot("1:T", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)

    # Thread A stalls between its snapshot and its write: the moment it reads
    # the journal path. Everyone else goes straight through.
    in_gap, go_on = threading.Event(), threading.Event()
    real = T

    class Stalling(types.ModuleType):
        def __getattr__(self, name):
            if name == "_INFLIGHT_FILE" and threading.current_thread().name == "A":
                in_gap.set()
                go_on.wait(10)
            return getattr(real, name)
    monkeypatch.setattr(Q, "tg_bot", Stalling("tg_bot"))

    bot._running_task[1] = [T._Task(task_id="done", chat_id=1, user_text="old")]
    a = threading.Thread(target=bot._write_inflight, name="A")
    a.start()
    assert in_gap.wait(10)
    bot._running_task.clear()                   # the task finished...
    bot._write_inflight()                       # ...and the journal says so
    go_on.set()
    a.join(10)
    data = json.loads(T._INFLIGHT_FILE.read_text(encoding="utf-8"))
    assert data == [], data
