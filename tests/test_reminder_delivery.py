"""Accepted delivery stages survive failures; unsuccessful delivery stays due."""
import json
import time

import pytest

import reminders
from test_reminder_persistence import Timer


@pytest.fixture
def path(tmp_path):
    reminders.stop()
    yield tmp_path / "reminders.json"
    reminders.stop()


@pytest.mark.parametrize("repeat", [0, 60])
@pytest.mark.parametrize("restart", [False, True])
def test_failed_sender_keeps_due_reminder_until_retry(path, repeat, restart):
    delivered = []
    calls = []

    def sender(owner, text):
        calls.append(text)
        if len(calls) == 1:
            raise OSError("delivery unavailable")
        delivered.append(text)

    reminders.register(sender, path, timer=Timer)
    rid = reminders.add(7, 1, "original", every_s=repeat)
    due = time.time() - 5
    reminders.update(7, rid, due=due)
    reminders._fire(rid)
    assert reminders.get(7, rid)["due"] == due
    assert json.loads(path.read_text())[rid]["due"] == due
    assert reminders._TIMERS[rid].interval >= 1
    if restart:
        reminders.stop()
        reminders.register(sender, path, timer=Timer)
    reminders._fire(rid)
    assert len(delivered) == 1
    if repeat:
        assert reminders.get(7, rid)["due"] > time.time()
    else:
        assert reminders.get(7, rid) is None


def test_task_retry_does_not_repeat_accepted_notification(path):
    sent, attempts = [], []

    def runner(owner, text):
        attempts.append(text)
        if len(attempts) == 1:
            raise OSError("queue unavailable")

    reminders.register(lambda owner, text: sent.append(text), path, timer=Timer, runner=runner)
    rid = reminders.add(7, 1, "scheduled task", kind="task")
    reminders.update(7, rid, due=time.time() - 5)
    reminders._fire(rid)
    assert reminders.get(7, rid) is not None
    reminders.stop()
    reminders.register(lambda owner, text: sent.append(text), path, timer=Timer, runner=runner)
    reminders._fire(rid)
    assert len(sent) == 1
    assert attempts == ["scheduled task", "scheduled task"]
    assert not reminders.pending(7)


def test_price_retry_sends_only_the_unsent_part(path):
    sent, checked = [], []
    fail = [True]

    def sender(owner, text):
        if text.startswith("Price watch ended") and fail[0]:
            fail[0] = False
            raise OSError("delivery unavailable")
        sent.append(text)

    def price_check(item):
        checked.append(item["price"])
        return "price dropped", {"price": 50}

    reminders.register(sender, path, timer=Timer, price_check=price_check)
    rid = reminders.add(7, 1, "watched item", every_s=60, kind="price", lang="en",
                        extra={"product": "https://example.com/item", "price": 100,
                               "until": time.time() - 1})
    reminders.update(7, rid, due=time.time() - 5)
    reminders._fire(rid)
    assert reminders.get(7, rid) is not None
    reminders._fire(rid)
    assert checked == [100]
    assert len(sent) == 2 and sent[0] == "price dropped"
    assert not reminders.pending(7)


def test_successful_send_is_not_repeated_when_its_checkpoint_save_fails(path):
    sent = []

    def sender(owner, text):
        sent.append(text)
        path.with_suffix(".tmp").mkdir()

    reminders.register(sender, path, timer=Timer)
    rid = reminders.add(7, 1, "original")
    reminders.update(7, rid, due=time.time() - 5)
    reminders._fire(rid)
    assert reminders.get(7, rid) is not None
    path.with_suffix(".tmp").rmdir()
    reminders._fire(rid)
    assert len(sent) == 1
    assert not reminders.pending(7)


def test_editing_a_failed_delivery_replaces_its_cached_text(path):
    sent, attempts = [], []

    def sender(owner, text):
        attempts.append(text)
        if len(attempts) == 1:
            raise OSError("delivery unavailable")
        sent.append(text)

    reminders.register(sender, path, timer=Timer)
    rid = reminders.add(7, 1, "old text")
    reminders.update(7, rid, due=time.time() - 5)
    reminders._fire(rid)
    reminders.update(7, rid, text="new text")
    reminders._fire(rid)
    assert len(sent) == 1 and "new text" in sent[0]


@pytest.mark.parametrize("phase", ["price", "sender"])
def test_edit_during_delivery_rearms_the_replacement(path, phase):
    sent, checks = [], []

    def sender(owner, text):
        sent.append(text)
        if phase == "sender" and len(sent) == 1:
            reminders.update(7, rid, text="edited during send")

    def price_check(item):
        checks.append(item["text"])
        if len(checks) == 1:
            reminders.update(7, rid, text="edited during check")
        return "price dropped", {"price": 50}

    reminders.register(sender, path, timer=Timer, price_check=price_check)
    kwargs = {"kind": "price", "every_s": 60,
              "extra": {"product": "https://example.com/item", "price": 100}} if phase == "price" else {}
    rid = reminders.add(7, 1, "original", **kwargs)
    reminders.update(7, rid, due=time.time() - 5)
    fired_timer = reminders._TIMERS[rid]
    reminders._fire(rid)
    assert reminders._TIMERS[rid] is not fired_timer
    assert "edited during" in reminders.get(7, rid)["text"]
    reminders._fire(rid)
    if phase == "price":
        assert checks == ["original", "edited during check"]
        assert reminders.get(7, rid)["due"] > time.time()
        assert sent == ["price dropped"]
    else:
        assert len(sent) == 2 and "edited during send" in sent[-1]
        assert not reminders.pending(7)
