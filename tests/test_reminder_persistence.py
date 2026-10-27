"""A failed atomic save must not acknowledge or apply a reminder mutation."""
import pytest

import reminders


class Timer:
    def __init__(self, interval, callback, args):
        self.interval = interval
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def running(tmp_path):
    reminders.stop()
    sent = []
    path = tmp_path / "reminders.json"
    reminders.register(lambda owner, text: sent.append(text), path, timer=Timer)
    yield path, sent
    reminders.stop()


@pytest.mark.parametrize("operation", ["add", "update", "remove", "cancel"])
def test_failed_save_leaves_items_timers_and_disk_unchanged(running, operation):
    path, _ = running
    rid = reminders.add(7, 3600, "original")
    before = reminders.pending(7)
    disk = path.read_bytes()
    timer = reminders._TIMERS[rid]
    path.with_suffix(".tmp").mkdir()
    actions = {
        "add": lambda: reminders.add(7, 3600, "new"),
        "update": lambda: reminders.update(7, rid, text="changed", due=before[0]["due"] + 3600),
        "remove": lambda: reminders.remove(7, rid),
        "cancel": lambda: reminders.cancel(7, "all"),
    }
    with pytest.raises(ValueError, match="save"):
        actions[operation]()
    assert reminders.pending(7) == before
    assert reminders._TIMERS == {rid: timer}
    assert not timer.cancelled
    assert path.read_bytes() == disk


@pytest.mark.parametrize("repeat", [0, 60])
def test_failed_save_when_firing_preserves_item_and_retries(running, repeat):
    import time
    path, sent = running
    rid = reminders.add(7, 0.01, "original", every_s=repeat)
    reminders.update(7, rid, due=time.time() - 2)
    before = reminders.pending(7)
    disk = path.read_bytes()
    path.with_suffix(".tmp").mkdir()
    reminders._fire(rid)
    assert not sent
    assert reminders.pending(7) == before
    assert reminders._TIMERS[rid].interval >= 1
    assert path.read_bytes() == disk
    path.with_suffix(".tmp").rmdir()
    reminders._fire(rid)
    assert len(sent) == 1
    if repeat:
        assert reminders.get(7, rid)["due"] > time.time()
    else:
        assert not reminders.pending(7)


def test_tool_reports_failed_save_as_an_error(running):
    from types import SimpleNamespace
    import tools
    path, _ = running
    path.with_suffix(".tmp").mkdir()
    result = tools._handle_set_reminder(SimpleNamespace(reminder_owner=7), {},
                                        {"text": "new reminder", "minutes": 1})
    assert result.startswith("[TOOL ERROR]") and "save" in result.lower()
    assert not reminders.pending(7)


def test_failed_price_save_does_not_publish_the_new_price(running):
    import json
    import time
    path, sent = running

    def price_check(item):
        path.with_suffix(".tmp").mkdir()
        return "price dropped", {"price": 50}

    reminders.register(lambda owner, text: sent.append(text), path, timer=Timer,
                       price_check=price_check)
    rid = reminders.add(7, 1, "watch", every_s=60, kind="price",
                        extra={"product": "https://example.com/item", "price": 100})
    reminders.update(7, rid, due=time.time() - 2)
    reminders._fire(rid)
    assert not sent
    assert reminders.get(7, rid)["price"] == 100
    assert json.loads(path.read_text())[rid]["price"] == 100
