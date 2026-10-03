"""Cancelling one reminder must never wipe the others.

cancel() dropped words of two letters or fewer before matching, and an empty
word list meant "cancel everything": asking to cancel the "ТО" (car service)
reminder, or the one "в 9", deleted every reminder the user had.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reminders


@pytest.fixture
def three():
    d = tempfile.mkdtemp()
    reminders.register(lambda cid, t: None, os.path.join(d, "r.json"))
    for text in ("ТО машины", "позвонить маме в 9", "купить хлеб"):
        reminders.add(7, 3600, text)
    yield
    reminders.cancel(7, "all")
    reminders.stop()


def _left():
    return sorted(i["text"] for i in reminders.pending(7))


def test_short_word_cancels_only_its_reminder(three):
    assert reminders.cancel(7, "ТО") == ["ТО машины"]
    assert _left() == ["купить хлеб", "позвонить маме в 9"]


def test_short_word_that_matches_nothing_cancels_nothing(three):
    assert reminders.cancel(7, "ДР") == []
    assert len(_left()) == 3


def test_long_word_still_matches_by_stem(three):
    assert reminders.cancel(7, "напоминание про хлебом") == ["купить хлеб"]


@pytest.mark.parametrize("phrase", ["all", "все", "все напоминания", "all my reminders", ""])
def test_all_phrases_cancel_everything(three, phrase):
    assert len(reminders.cancel(7, phrase)) == 3
    assert _left() == []


def test_unreadable_file_is_kept_aside_not_overwritten(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"abc": {"owner": "7", "due": 1', encoding="utf-8")   # torn write
    reminders.register(lambda cid, t: None, path)
    try:
        reminders.add(7, 3600, "новое")
        kept = list(tmp_path.glob("r.json.unreadable-*"))
        assert len(kept) == 1 and kept[0].read_text(encoding="utf-8").startswith('{"abc"')
    finally:
        reminders.cancel(7, "all")
        reminders.stop()


def test_a_malformed_entry_does_not_stop_the_others(tmp_path):
    import json
    import time
    path = tmp_path / "r.json"
    later = time.time() + 3600
    path.write_text(json.dumps({
        "bad1": {"owner": "7", "text": "no due"},
        "bad2": {"owner": "7", "due": "tomorrow", "text": "string due"},
        "bad3": {"owner": "7", "due": later, "text": "spam", "every": 0.5},
        "good": {"owner": "7", "due": later, "text": "купить хлеб"},
    }), encoding="utf-8")
    try:
        assert reminders.register(lambda cid, t: None, path) == 1
        assert [i["text"] for i in reminders.pending(7)] == ["купить хлеб"]
        assert reminders.cancel(7, "all") == ["купить хлеб"]
    finally:
        reminders.stop()
