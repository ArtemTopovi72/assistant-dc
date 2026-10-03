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
