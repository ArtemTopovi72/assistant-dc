"""Currency and calendar arithmetic is done in code, not by the model.

A receipt's 13 € went to a 46 s web search for «EUR to RUB» and a multiplication in the model's
head; «сколько дней до Нового года» was counted from memory (2026-10-24/25). The network is a
fake here: each source answers from a dict, so the routing between sources is what is tested.
"""
import datetime as dt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import currency as C
import dates as D

CBR = {"Date": "2026-10-24T11:30:00+03:00", "Valute": {
    "EUR": {"Value": 90.2429, "Nominal": 1}, "RSD": {"Value": 77.0, "Nominal": 100}}}
ER_EUR = {"result": "success", "time_last_update_unix": 1792800000, "rates": {"EUR": 1, "RSD": 117.1, "USD": 1.16}}
FAWAZ_EUR = {"date": "2026-10-01", "eur": {"rsd": 117.0, "btc": 0.000011}}


def net(pages, seen=None):
    def get(url):
        (seen if seen is not None else []).append(url)
        for key, data in pages.items():
            if key in url:
                if isinstance(data, Exception):
                    raise data
                return data
        raise OSError("404 " + url)
    return get


def fresh():
    C._cache.clear(); D._years.clear()


# rubles: the Central Bank, computed here
fresh()
r = C.convert(13, "eur", "rub", get=net({"daily_json": CBR}))
assert r["source"] == "Central Bank of Russia" and abs(r["result"] - 1173.1577) < 1e-3 and r["date"] == "2026-10-24", r
r = C.convert(1000, "RSD", "RUB", get=net({"daily_json": CBR}))
assert abs(r["result"] - 770.0) < 1e-9, r                              # nominal 100 is honoured
# a dated rate on a Sunday steps back to the last published day
fresh(); seen = []
r = C.convert(1, "EUR", "RUB", dt.date(2026, 10, 25), get=net({"2026/10/24/": CBR}, seen))
assert r["date"] == "2026-10-24" and any("2026/10/25" in u for u in seen), (r, seen)
# other pairs: ExchangeRate-API, then the currency-api mirrors
fresh()
r = C.convert(100, "EUR", "RSD", get=net({"open.er-api": ER_EUR}))
assert r["source"] == "ExchangeRate-API" and abs(r["result"] - 11710) < 1e-6, r
fresh()
r = C.convert(100, "EUR", "RSD", get=net({"open.er-api": OSError("down"), "pages.dev": FAWAZ_EUR}))
assert r["source"].startswith("fawazahmed0") and abs(r["result"] - 11700) < 1e-6, r
fresh()
r = C.convert(1, "EUR", "BTC", dt.date(2026, 10, 1), get=net({"currency-api@2026-10-01": FAWAZ_EUR}))
assert r["date"] == "2026-10-01" and r["rate"] == 0.000011, r
fresh()
try:
    C.convert(1, "EUR", "XYZ", get=net({"open.er-api": ER_EUR}))
    raise AssertionError("an unknown code must fail loudly, not guess")
except C.RateError:
    pass
print("ok currency: CBR for rubles, market sources otherwise, dated rates, loud failure")

# calendar
today = dt.date(2026, 10, 25)                                          # a Sunday
year = "".join("1" if dt.date(2026, 1, 1) + dt.timedelta(i) in {dt.date(2026, 12, 31)}
               or (dt.date(2026, 1, 1) + dt.timedelta(i)).weekday() >= 5 else "0" for i in range(365))
cal = net({"isdayoff": year})
assert D.parse("15 марта", today) == dt.date(2027, 3, 15)
assert D.parse("2026-12-31", today) == dt.date(2026, 12, 31)
assert D.holiday("сколько до Нового года", today) == ("новый год", dt.date(2027, 1, 1))
assert D.holiday("пасха", today)[1] == dt.date(2027, 5, 2)            # Orthodox Easter 2027
a = D.answer(today, until="Новый год", get=cal)
assert "68 дней" in a and "1 января 2027 (пятница)" in a and "воскресенье" in a, a
a = D.answer(today, until="2026-12-31", get=cal)
assert "67 дней" in a and "рабочих 48 по производственному календарю РФ" in a and "day off" in a, a
a = D.answer(today, date="2026-10-23", add_days=5, workdays=True, get=cal)
assert "30 октября 2026 (пятница)" in a, a                            # Fri + 5 working days
a = D.answer(today, date="2026-01-31", add_months_n=1, get=cal)
assert "28 февраля 2026" in a, a                                       # month end clamps
D._years.clear()
a = D.answer(today, until="2026-12-31", get=net({"isdayoff": OSError("down")}))
assert "календарь недоступен" in a, a                                  # says when it fell back
assert "Today is 25 октября 2026 (воскресенье)" in D.answer(today, get=cal)
a = D.answer(today, date="1990-05-12", until="2026-10-25", get=cal)
assert "36 лет 5 мес. 13 дн." in a, a                                  # an age, exactly
print("ok calendar: days, working days by the RU calendar, holidays, shifts, honest fallback")
