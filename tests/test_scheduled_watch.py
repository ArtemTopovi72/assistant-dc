"""Scheduled work: a reminder that DOES something at its time, and an Ozon price watch.

«каждое утро присылай погоду» used to deliver the words «присылай погоду» -- the user
asked for the weather, not a note about it. A price watch must stay silent until the
price really drops: a «не подешевело» every six hours is spam (2026-10-26).
"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

import reminders
import price_watch as P


class _Held:
    """A timer the test fires by hand: the suite does not sleep for hours."""
    def __init__(self, wait, fn, args):
        self.daemon = True

    def start(self):
        pass

    def cancel(self):
        pass


def _slot(rid):
    reminders._ITEMS[rid]["due"] = time.time()
    reminders._fire(rid)


def _start(**hooks):
    sent, ran = [], []
    reminders.stop()
    reminders.register(lambda cid, t: sent.append((cid, t)), os.path.join(tempfile.mkdtemp(), "r.json"),
                       runner=lambda cid, t: ran.append((cid, t)), **hooks)
    return sent, ran


def test_a_task_is_done_at_its_time_not_quoted():
    sent, ran = _start()
    reminders.add(42, 0.05, "пришли погоду в Москве", kind="task")
    time.sleep(0.4)
    assert ran == [(42, "пришли погоду в Москве")], ran
    assert len(sent) == 1 and "пришли погоду" in sent[0][1], sent      # the user sees why the bot spoke
    reminders.stop()


def test_check_speaks_only_below_the_lowest_price_seen():
    item = {"product": "https://www.ozon.ru/product/x-1/", "name": "Carlo Gattini Рюкзак", "price": 15475}
    same = P.check(item, details=lambda it: {"price": 15475, "available": True})
    up = P.check(item, details=lambda it: {"price": 16900, "available": True})
    down = P.check(item, details=lambda it: {"price": 13990, "available": True})
    assert same == (None, {}) and up == (None, {}), (same, up)
    msg, upd = down
    assert "15 475" in msg and "13 990" in msg and "ozon.ru/product/x-1" in msg, msg
    assert upd == {"price": 13990}


def test_a_target_price_waits_for_the_target():
    item = {"product": "u", "name": "n", "price": 15475, "target": 12000}
    assert P.check(item, details=lambda it: {"price": 13990, "available": True}) == (None, {})
    assert P.check(item, details=lambda it: {"price": 11990, "available": True})[0]


def test_a_failed_or_sold_out_check_is_silent():
    item = {"product": "u", "name": "n", "price": 15475}

    def broken(it):
        raise OSError("ozon down")
    assert P.check(item, details=broken) == (None, {})
    assert P.check(item, details=lambda it: {"price": 9000, "available": False}) == (None, {})


def test_the_watch_repeats_keeps_the_new_low_and_ends_on_time():
    prices = iter([15475, 13990, 13990])
    sent, ran = _start(price_check=lambda it: P.check(it, details=lambda _: {"price": next(prices),
                                                                              "available": True}),
                       timer=_Held)
    rid = reminders.add(7, 3600, "следить за ценой: рюкзак", every_s=3600, kind="price",
                        extra={"product": "u", "name": "рюкзак", "price": 15475,
                               "until": time.time() + 3 * 3600})
    _slot(rid)                                                     # 15 475: silent
    _slot(rid)                                                     # 13 990: a drop
    assert reminders._ITEMS[rid]["price"] == 13990                 # the new low is kept
    reminders._ITEMS[rid]["until"] = time.time()
    _slot(rid)                                                     # 13 990 again, and the end
    drops = [t for _, t in sent if "13 990" in t]
    assert len(drops) == 1, sent                                   # one drop, said once
    assert any("закончил" in t for _, t in sent), sent             # the end is announced
    assert all(i["id"] != rid for i in reminders.pending(7)), "an ended watch must be gone"
    reminders.stop()


def test_a_damaged_price_watch_in_the_file_is_dropped():
    assert not reminders._valid({"owner": "1", "due": 1.0, "text": "t", "kind": "price"})
    assert reminders._valid({"owner": "1", "due": 1.0, "text": "t", "kind": "price",
                             "product": "u", "price": 1})


def test_the_tools_set_a_task_and_a_price_watch():
    import tools
    import ozon_client as O
    from seams import Over
    _start(timer=_Held)
    ctx = SimpleNamespace(reminder_owner=7, ozon_owner="7", user_tz="", reply_lang="ru",
                          set_stage=lambda *a, **k: None,
                          deps=SimpleNamespace(users_dir=tempfile.mkdtemp(), ozon=Over(O, details=lambda p, **k: {
                              "name": "Carlo Gattini Рюкзак", "price": 15475, "available": True,
                              "url": "https://www.ozon.ru/product/carlo-gattini-ryukzak-366298835/"})))
    out = tools.execute_tool(ctx, {}, "set_reminder", {"text": "пришли погоду в Москве", "at": "08:00",
                                                       "repeat": "daily", "do": True})
    assert "carry it out" in out, out
    out = tools.execute_tool(ctx, {}, "ozon_watch_price", {"product": "https://ozon.ru/t/LCabAMR",
                                                            "every_hours": 6, "target_price": 13000})
    assert "15 475" in out and "13 000" in out, out
    kinds = {i["kind"]: i for i in reminders.pending(7) if i.get("kind")}
    assert kinds["task"]["every"] == 86400
    w = kinds["price"]
    assert w["price"] == 15475 and w["target"] == 13000 and w["every"] == 6 * 3600
    assert w["product"].endswith("366298835/") and w["key"] == "7"
    listed = tools.execute_tool(ctx, {}, "set_reminder", {"action": "list"})
    assert "Carlo Gattini" in listed and "15 475" in listed, listed
    reminders.stop()


def test_a_set_price_watch_is_not_called_an_unset_reminder():
    # live 10-26: «я начал следить за ценой… напишу, как только подешевеет» got the honesty
    # line «напоминание не поставлено» although ozon_watch_price had just set it
    import turn_audit
    ctx = SimpleNamespace(reply_read=lambda c, a, u="": {"did": ["reminder"]})
    assert turn_audit.unmet_claims("Слежу за ценой", {"ozon_watch_price"}, False, ctx) == []
    assert turn_audit.unmet_claims("Напомню", set(), False, ctx) == ["reminder"]
