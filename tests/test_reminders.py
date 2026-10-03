"""reminders: a scheduled reminder fires through the registered sender;
the tool refuses without an owner chat."""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reminders

got = []
d = tempfile.mkdtemp()
reminders.register(lambda cid, t: got.append((cid, t)), os.path.join(d, "r.json"))
reminders.add(42, 0.2, "выпить воды")
time.sleep(0.6)
assert got == [(42, "⏰ Напоминание: выпить воды")], got
reminders.stop()

import tools
from types import SimpleNamespace

out = tools._handle_set_reminder(SimpleNamespace(), None, {"text": "x", "minutes": 1})
assert "error" in str(out).lower() or "не" in str(out), out
print("ok")

reminders.register(lambda cid, t: got.append((cid, t)), os.path.join(d, "r2.json"))
reminders.add(7, 100, "позвонить маме"); reminders.add(7, 100, "выпить воды")
ctx = SimpleNamespace(reminder_owner=7)
assert "маме" in tools._handle_set_reminder(ctx, None, {"action": "list"})
assert "маме" in tools._handle_set_reminder(ctx, None, {"action": "cancel", "text": "про маму"})
assert [i["text"] for i in reminders.pending(7)] == ["выпить воды"]
reminders.stop()
print("ok2")

# exchange_rate reads the CBR payload (cache pre-seeded: no network in tests)
import time as _t
tools._CBR_CACHE.update(t=_t.time(), data={"Date": "2026-09-26T11:30", "Valute": {
    "USD": {"Value": 84.3414, "Nominal": 1}, "JPY": {"Value": 57.0, "Nominal": 100}}})
r = tools._handle_exchange_rate(None, None, {"codes": "usd, JPY,XXX"})
assert "1 USD = 84.3414 RUB" in r and "1 JPY = 0.5700 RUB" in r and "XXX: no CBR rate" in r, r
print("ok3")

# local_time: the zone comes from the geocoder, the clock from zoneinfo
import weather as _W
_orig = _W.resolve_city
_W.resolve_city = lambda *a, **k: {"name": "Нью-Йорк", "country": "США", "timezone": "America/New_York"}
try:
    r = tools._handle_local_time(None, None, {"city": "Нью-Йорк"})
    assert "zone America/New_York" in r and ("UTC-0400" in r or "UTC-0500" in r), r
finally:
    _W.resolve_city = _orig
print("ok4")

# a "last/current" search without a year gets the current year
import datetime as _dt
_seen = []
_orig_ws = tools.run_web_search
tools.run_web_search = lambda c, q: (_seen.append(q), "Some complete sentence about it.")[1]
try:
    _c = SimpleNamespace(set_stage=lambda *a: None, remember=lambda *a, **k: None, web_search_enabled=True)
    tools._handle_search(_c, {}, {"query": "winner of the last FIFA World Cup"})
    tools._handle_search(_c, {}, {"query": "World Cup 2022 winner"})
    assert _seen == [f"winner of the last FIFA World Cup {_dt.date.today().year}", "World Cup 2022 winner"], _seen
finally:
    tools.run_web_search = _orig_ws
print("ok5")

_seen.clear()
tools.run_web_search = lambda c, q: (_seen.append(q), "Some complete sentence about it.")[1]
try:
    tools._handle_search(_c, {"user_input": "who won the last world cup?"}, {"query": "who won the last FIFA World Cup 2022"})
    tools._handle_search(_c, {"user_input": "latest news from 2019"}, {"query": "latest news 2019"})
    _y = str(_dt.date.today().year)
    assert _seen == [f"who won the last FIFA World Cup {_y}", "latest news 2019"], _seen
finally:
    tools.run_web_search = _orig_ws
print("ok6")

_seen.clear()
tools.run_web_search = lambda c, q: (_seen.append(q), "Some complete sentence about it.")[1]
try:
    tools._handle_search(_c, {"user_input_original": "кто выиграл последний чемпионат мира?"},
                         {"query": "who won the 2022 FIFA World Cup"})
    assert _seen == [f"who won the {_y} FIFA World Cup"], _seen
finally:
    tools.run_web_search = _orig_ws
print("ok7")

# A daily reminder fires and re-arms for the next day instead of vanishing.
got.clear(); reminders.register(lambda cid, t: got.append((cid, t)), os.path.join(d, "r8.json"))
rid = reminders.add(555, 0.2, "таблетки", every_s=86400)
import time as _t
_t.sleep(0.6)
assert any("таблетки" in t for _, t in got), got
left = [i for i in reminders.pending(555) if i["text"] == "таблетки"]
assert left and left[0]["due"] > _t.time() + 86000, left
reminders.cancel(555, "all")
print("ok8")

# An English turn's reminder fires with an English prefix.
got.clear()
reminders.add(556, 0.2, "check the oven", lang="en")
_t.sleep(0.6)
assert ("556" in str(got)) and any(t.startswith("⏰ Reminder: check the oven") for _, t in got), got
print("ok9")

import graph as _G, graph_fastpath as _F
assert _F._FAST_PATH_ACTION_CLAIM_RE.search("Поставил таймер на 10 минут для пасты.")
assert _F._FAST_PATH_ACTION_CLAIM_RE.search("Хорошо, я разбужу тебя завтра в 7 утра.")
print("ok10")
print("ok11")
