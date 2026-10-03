"""A reminder far in the future must not arm one giant Timer.

On Windows a lock wait is limited to threading.TIMEOUT_MAX (~49.7 days); a
Timer for "через 2 года" raised OverflowError on its own thread and the
reminder never fired. Timers are now capped and re-arm until the due time.
"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reminders


def test_far_reminder_waits_in_capped_steps_and_still_fires(monkeypatch):
    waits = []
    real_timer = threading.Timer

    def spy_timer(interval, fn, args=()):
        waits.append(interval)
        return real_timer(interval, fn, args)

    monkeypatch.setattr(reminders.threading, "Timer", spy_timer)
    monkeypatch.setattr(reminders, "_MAX_TIMER_S", 0.2)
    got = []
    d = tempfile.mkdtemp()
    reminders.register(lambda cid, t: got.append(t), os.path.join(d, "r.json"))
    try:
        reminders.add(5, 1.5, "far away")
        time.sleep(1.4)
        assert got and "far away" in got[0], got
        assert len(waits) >= 3, waits          # woke early and re-armed
        assert max(waits) <= 0.2 + 1e-9, waits
    finally:
        reminders.stop()


def test_two_year_reminder_arms_within_timeout_max():
    d = tempfile.mkdtemp()
    reminders.register(lambda cid, t: None, os.path.join(d, "r.json"))
    try:
        rid = reminders.add(5, 2 * 365 * 86400, "passport")
        t = reminders._TIMERS[rid]
        assert t.interval <= reminders._MAX_TIMER_S < 4294967.0
        assert t.is_alive()
        assert [i["text"] for i in reminders.pending(5)] == ["passport"]
    finally:
        reminders.stop()
