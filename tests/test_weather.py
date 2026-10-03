"""weather.py: geocoding, forecast, clothing-advice bands and error paths.
Stubs requests entirely so this runs deterministically offline.

Run: venv/Scripts/python.exe tests/test_weather.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import weather as W

RESULTS = []
def table_of(out):
    """The monospace forecast table only, without the advice lines below it.

    Period labels legitimately appear twice in a reply now -- once as a table
    row, once in the grouped advice under it -- so counting them across the
    whole string measures the wrong thing.
    """
    if "<pre>" not in out:
        return out
    return out.split("<pre>", 1)[1].split("</pre>", 1)[0]


import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


class FakeResp:
    def __init__(self, data): self._data = data
    def raise_for_status(self): pass
    def json(self): return self._data


class _Patches:
    def __init__(self, **kw): self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(W, k); setattr(W, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items(): setattr(W, k, v)


def test_geocode_city():
    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp(
                {"results": [{"latitude": 55.75, "longitude": 37.6,
                             "name": "Moscow", "country": "Russia"}]}))):
        loc = W.geocode_city("Moscow")
        check("geocode_city returns lat/lon/name", loc == {
            "lat": 55.75, "lon": 37.6, "name": "Moscow", "country": "Russia", "timezone": ""})

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp({"results": []}))):
        check("geocode_city returns None for no matches",
              W.geocode_city("Nonexistentxyz") is None)

    check("geocode_city returns None for empty input",
          W.geocode_city("") is None)
    check("geocode_city returns None for whitespace-only input",
          W.geocode_city("   ") is None)

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: (_ for _ in ()).throw(RuntimeError("net down")))):
        check("geocode_city swallows network errors, returns None",
              W.geocode_city("Moscow") is None)


def test_fetch_current():
    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp(
                {"current": {"temperature_2m": 12.3, "wind_speed_10m": 5.0,
                            "weather_code": 3}}))):
        cur = W.fetch_current(55.75, 37.6)
        check("fetch_current returns temp/wind/code",
              cur["temp_c"] == 12.3 and cur["wind_kmh"] == 5.0 and cur["code"] == 3)
        check("...and humidity, None when upstream omitted it",
              cur.get("humidity") is None, cur)

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp({"current": {}}))):
        check("fetch_current returns None when temperature is missing",
              W.fetch_current(0, 0) is None)

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("fetch_current swallows network errors, returns None",
              W.fetch_current(0, 0) is None)


def test_code_desc():
    check("known code resolves in English", W._code_desc(0, "en") == "clear sky")
    check("known code resolves in Russian", W._code_desc(0, "ru") == "ясно")
    check("unknown code has a safe fallback",
          W._code_desc(9999, "en") == "unknown conditions")


def test_clothing_advice_bands():
    # Temperature bands, not a per-city table -- same thresholds anywhere.
    check("deep freeze -> heaviest layer (en)",
          "down coat" in W._clothing_advice(-15, 0, 5, "en"))
    check("deep freeze -> heaviest layer (ru)",
          "пуховик" in W._clothing_advice(-15, 0, 5, "ru"))
    check("hot -> light clothing (en)",
          "light, loose" in W._clothing_advice(30, 0, 5, "en"))
    check("mild -> light jacket band (en)",
          "light jacket" in W._clothing_advice(15, 0, 5, "en"))
    check("rain code adds an umbrella mention (en)",
          "umbrella" in W._clothing_advice(15, 61, 5, "en"))
    check("snow code adds a waterproof mention, not umbrella (en)",
          "waterproof" in W._clothing_advice(-2, 71, 5, "en")
          and "umbrella" not in W._clothing_advice(-2, 71, 5, "en"))
    check("high wind adds a windbreaker mention (en)",
          "windbreaker" in W._clothing_advice(15, 0, 40, "en"))
    check("low wind does not mention a windbreaker (en)",
          "windbreaker" not in W._clothing_advice(15, 0, 5, "en"))

    # Ice is the one forecast fact with a safety consequence, and it reads as
    # ordinary rain by code -- freezing rain is 66/67, freezing drizzle 56/57,
    # rime fog 48. Advising an umbrella for a glazed pavement was the bug.
    for code in (48, 56, 57, 66, 67):
        adv_en = W._clothing_advice(0, code, 5, "en")
        adv_ru = W._clothing_advice(0, code, 5, "ru")
        check("code %d warns about ice, not an umbrella (en)" % code,
              "non-slip" in adv_en and "black ice" in adv_en
              and "umbrella" not in adv_en, adv_en)
        check("code %d warns about ice, not an umbrella (ru)" % code,
              "нескользящая" in adv_ru and "гололёд" in adv_ru
              and "зонт" not in adv_ru, adv_ru)
    # Wet snow has no code of its own: snowfall near freezing is slush, and
    # that's inferred from the temperature.
    check("snow near freezing is called out as wet snow (en)",
          "wet snow" in W._clothing_advice(0, 73, 5, "en"))
    check("snow near freezing is called out as wet snow (ru)",
          "мокрый снег" in W._clothing_advice(0, 73, 5, "ru"))
    check("snow well below freezing is NOT called wet snow (en)",
          "wet snow" not in W._clothing_advice(-14, 75, 5, "en"))

    # The Russian phrases are NOMINATIVE noun phrases -- they are listed under
    # a "что надеть" heading, not governed by "надень" (which is what produced
    # "надень футболка и ветровка" on every live reply, since the LLM advice
    # path returns a nominative phrase too).
    check("russian advice is nominative, not accusative",
          "куртка" in W._clothing_advice(15, 0, 5, "ru")
          and "куртку" not in W._clothing_advice(15, 0, 5, "ru"),
          W._clothing_advice(15, 0, 5, "ru"))


def test_what_to_wear_end_to_end():
    with _Patches(geocode_city=lambda city, lang="en": {"lat": 1, "lon": 2, "name": "Moscow", "country": "RU"},
                  fetch_current=lambda lat, lon: {"temp_c": 12.0, "wind_kmh": 5, "code": 2}):
        out_en = W.what_to_wear("Moscow", "en")
        check("english reply has the temperature", "+12" in out_en)
        check("english reply has the place name", "Moscow" in out_en)
        # The advice is labelled 👕 rather than prefixed with a verb -- see
        # _clothing_advice on why "Надень <nominative>" had to go.
        check("english reply has clothing advice under the 👕 label",
              "👕 " in out_en and out_en.rsplit("👕 ", 1)[1].strip(), out_en)

        out_ru = W.what_to_wear("Moscow", "ru")
        check("russian reply is actually in Russian",
              "куртка" in out_ru or "пальто" in out_ru, out_ru)
        check("russian reply no longer prefixes the advice with 'надень'",
              "надень" not in out_ru.lower(), out_ru)

    with _Patches(geocode_city=lambda city, lang="en": None):
        check("unknown city -> translated not-found message (en)",
              "Couldn't find" in W.what_to_wear("Nowhereville", "en"))
        check("unknown city -> translated not-found message (ru)",
              "Не нашёл" in W.what_to_wear("Nowhereville", "ru"))

    with _Patches(geocode_city=lambda city, lang="en": {"lat": 1, "lon": 2, "name": "X", "country": ""},
                  fetch_current=lambda lat, lon: None):
        check("forecast failure -> translated service-down message (en)",
              "unavailable" in W.what_to_wear("X", "en"))
        check("forecast failure -> translated service-down message (ru)",
              "недоступен" in W.what_to_wear("X", "ru"))


def test_resolve_city_correction():
    # No correct_fn given -- behaves exactly like a bare geocode_city call.
    with _Patches(geocode_city=lambda city, lang="en": (
            {"lat": 1, "lon": 2, "name": "Moscow", "country": "RU"} if city == "Moscow" else None)):
        check("resolves directly when the raw name geocodes",
              W.resolve_city("Moscow") == {"lat": 1, "lon": 2, "name": "Moscow", "country": "RU"})
        check("no correct_fn and a failed geocode -> None, no crash",
              W.resolve_city("Спб") is None)

    # correct_fn only gets a turn when the raw name fails.
    with _Patches(geocode_city=lambda city, lang="en": (
            {"lat": 1, "lon": 2, "name": "Saint Petersburg", "country": "RU"}
            if city == "Saint Petersburg" else None)):
        calls = []
        def correct(city):
            calls.append(city)
            return "Saint Petersburg"
        loc = W.resolve_city("Спб", correct_fn=correct)
        check("a corrected name is retried and resolves",
              loc == {"lat": 1, "lon": 2, "name": "Saint Petersburg", "country": "RU"})
        check("correct_fn was given the original (failed) input",
              calls == ["Спб"])

    with _Patches(geocode_city=lambda city, lang="en": (
            {"lat": 1, "lon": 2, "name": "Moscow", "country": "RU"}
            if city == "Moscow" else None)):
        calls2 = []
        def not_called(city):
            calls2.append(city); return "Moscow"
        loc2 = W.resolve_city("Moscow", correct_fn=not_called)
        check("correct_fn is NOT consulted when the raw name already resolves",
              calls2 == [])
        check("...and the direct result is returned", loc2["name"] == "Moscow")

    # correct_fn returning nothing useful (empty, unchanged, or itself wrong).
    with _Patches(geocode_city=lambda city, lang="en": None):
        check("empty correction -> still None, no crash",
              W.resolve_city("Спб", correct_fn=lambda c: "") is None)
        check("correction identical to the input -> not retried, still None",
              W.resolve_city("Спб", correct_fn=lambda c: "Спб") is None)
        check("a confidently-wrong correction that still doesn't geocode -> None",
              W.resolve_city("Спб", correct_fn=lambda c: "Nonexistentville") is None)

        def broken(city):
            raise RuntimeError("LM Studio unreachable")
        check("a raising correct_fn is swallowed, not propagated",
              W.resolve_city("Спб", correct_fn=broken) is None)


def test_what_to_wear_correct_fn():
    # End-to-end through what_to_wear: an unresolvable raw city plus a
    # correct_fn that fixes it should still produce a normal weather reply,
    # not the not-found message.
    seen = {"geocode_calls": []}
    def geocode(city, lang="en"):
        seen["geocode_calls"].append(city)
        return {"lat": 1, "lon": 2, "name": "Saint Petersburg", "country": "RU"} \
            if city == "Saint Petersburg" else None
    with _Patches(geocode_city=geocode,
                  fetch_current=lambda lat, lon: {"temp_c": 10.0, "wind_kmh": 5, "code": 0}):
        out = W.what_to_wear("Спб", "ru", correct_fn=lambda c: "Saint Petersburg")
        check("a correctable city produces a real weather reply, not a not-found message",
              "Не нашёл" not in out and "Saint Petersburg" in out)
        check("geocode was tried on both the raw and corrected names",
              seen["geocode_calls"] == ["Спб", "Saint Petersburg"])

        out2 = W.what_to_wear("Спб", "ru", correct_fn=lambda c: "Also Nonexistent")
        check("an incorrigible city still falls back to the not-found message",
              "Не нашёл" in out2)


def test_period_for_hour_and_label():
    check("hour 6 is morning", W._period_for_hour(6) == "morning")
    check("hour 11 is still morning", W._period_for_hour(11) == "morning")
    check("hour 12 is afternoon", W._period_for_hour(12) == "afternoon")
    check("hour 17 is still afternoon", W._period_for_hour(17) == "afternoon")
    check("hour 18 is evening", W._period_for_hour(18) == "evening")
    check("hour 23 is still evening", W._period_for_hour(23) == "evening")
    check("hour 0 (night) has no period", W._period_for_hour(0) is None)
    check("hour 5 (night) has no period", W._period_for_hour(5) is None)
    check("morning label (en)", W._period_label("morning", "en") == "Morning")
    check("morning label (ru)", W._period_label("morning", "ru") == "Утро")


def test_bucket_periods():
    import datetime as _dt
    def pt(y, m, d, h, temp, code=0, wind=5):
        return {"time": _dt.datetime(y, m, d, h), "temp_c": temp, "code": code, "wind_kmh": wind}

    # Two full days of hourly points, temperatures rising through the day.
    points = []
    for day in (1, 2):
        for h in range(24):
            points.append(pt(2026, 8, day, h, temp=h, code=(3 if h == 14 else 1)))

    start = _dt.datetime(2026, 8, 1, 0, 0)
    end = _dt.datetime(2026, 8, 3, 0, 0)
    buckets = W.bucket_periods(points, start, end)
    check("night hours (0-6, 24-30 etc) produce NO buckets",
          all(b["period"] != "night" for b in buckets))
    check("exactly 6 buckets for 2 full days x 3 periods",
          len(buckets) == 6, len(buckets))
    check("buckets come out in chronological order",
          [(b["date"], b["period"]) for b in buckets] ==
          [(_dt.date(2026, 8, 1), "morning"), (_dt.date(2026, 8, 1), "afternoon"),
           (_dt.date(2026, 8, 1), "evening"), (_dt.date(2026, 8, 2), "morning"),
           (_dt.date(2026, 8, 2), "afternoon"), (_dt.date(2026, 8, 2), "evening")])
    morning = buckets[0]
    check("morning bucket averages hours 6-11", morning["temp_c"] == sum(range(6, 12)) / 6)
    afternoon = buckets[1]
    check("the MOST FREQUENT code wins even if one hour (14:00) is an outlier",
          afternoon["code"] == 1, afternoon["code"])

    # A window that only partly overlaps a period (e.g. "starting right now"
    # mid-afternoon) must only average the REMAINING hours of that period,
    # not hours already in the past.
    partial = W.bucket_periods(points, _dt.datetime(2026, 8, 1, 14, 0),
                               _dt.datetime(2026, 8, 1, 18, 0))
    check("a window starting mid-period only covers the remaining hours",
          len(partial) == 1 and partial[0]["period"] == "afternoon")
    check("...averaging only hours 14-17, not the whole afternoon",
          partial[0]["temp_c"] == sum(range(14, 18)) / 4)

    check("an empty window produces no buckets",
          W.bucket_periods(points, _dt.datetime(2026, 1, 1), _dt.datetime(2026, 1, 1)) == [])


def test_parse_date_input():
    import datetime as _dt
    today = _dt.date(2026, 8, 15)
    check("'today' resolves to today", W.parse_date_input("today", today) == today)
    check("'сегодня' resolves to today", W.parse_date_input("сегодня", today) == today)
    check("'tomorrow' resolves to today+1",
          W.parse_date_input("tomorrow", today) == today + _dt.timedelta(days=1))
    check("'завтра' resolves to today+1",
          W.parse_date_input("завтра", today) == today + _dt.timedelta(days=1))
    check("DD.MM with no year, still ahead this year, uses this year",
          W.parse_date_input("20.08", today) == _dt.date(2026, 8, 20))
    check("DD.MM with no year, already past this year, rolls to next year",
          W.parse_date_input("01.01", today) == _dt.date(2027, 1, 1))
    check("DD.MM.YYYY with an explicit year is taken literally (even in the past)",
          W.parse_date_input("01.01.2026", today) == _dt.date(2026, 1, 1))
    check("DD-MM and DD/MM separators both work",
          W.parse_date_input("20-08", today) == _dt.date(2026, 8, 20))
    check("ISO YYYY-MM-DD works", W.parse_date_input("2026-09-01", today) == _dt.date(2026, 9, 1))
    check("2-digit year expands to 20XX",
          W.parse_date_input("20.08.26", today) == _dt.date(2026, 8, 20))
    check("garbage text -> None", W.parse_date_input("banana", today) is None)
    check("empty text -> None", W.parse_date_input("", today) is None)
    check("an impossible date (Feb 30) -> None", W.parse_date_input("30.02", today) is None)


def test_fetch_hourly():
    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp({
                "utc_offset_seconds": 10800,
                "hourly": {
                    "time": ["2026-08-15T00:00", "2026-08-15T01:00"],
                    "temperature_2m": [10.0, 9.5],
                    "weather_code": [0, 1],
                    "wind_speed_10m": [5.0, 6.0],
                }}))):
        result = W.fetch_hourly(1, 2, days=1)
        check("fetch_hourly returns the utc offset", result["utc_offset_seconds"] == 10800)
        check("fetch_hourly returns one point per hour", len(result["points"]) == 2)
        check("each point has a parsed datetime and the right fields",
              result["points"][0]["temp_c"] == 10.0 and result["points"][1]["code"] == 1)

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: FakeResp({"hourly": {}}))):
        check("missing hourly data -> None, not a crash", W.fetch_hourly(1, 2) is None)

    with _Patches(requests=types.SimpleNamespace(
            get=lambda url, params=None, timeout=8: (_ for _ in ()).throw(RuntimeError("down")))):
        check("network failure -> None, not a crash", W.fetch_hourly(1, 2) is None)


def test_what_to_wear_periods_for_loc():
    import datetime as _dt
    loc = {"lat": 1, "lon": 2, "name": "Testville", "country": ""}

    def fake_hourly(lat, lon, days=3):
        points = []
        for day_off in range(days):
            day = _dt.date(2026, 8, 15) + _dt.timedelta(days=day_off)
            for h in range(24):
                points.append({"time": _dt.datetime.combine(day, _dt.time(h, 0)),
                              "temp_c": 15.0, "code": 0, "wind_kmh": 5})
        return {"utc_offset_seconds": 0, "points": points}

    class _FrozenDatetime(_dt.datetime):
        @classmethod
        def utcnow(cls):
            return _dt.datetime(2026, 8, 15, 9, 0)

    # date.today() has to be frozen as well, not just utcnow(). The fixture
    # generates hours from a hardcoded 2026-08-15, but
    # what_to_wear_periods_for_loc sizes its fetch with
    # `(date - date.today()).days + 2` -- against the REAL today. Once the
    # calendar moved past 2026-08-15 that window stopped covering the date the
    # date-based checks below ask for, and they began failing for reasons that
    # had nothing to do with the code under test.
    class _FrozenDate(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date(2026, 8, 15)

    with _Patches(fetch_hourly=fake_hourly):
        _orig_dt, _orig_date = W._dt.datetime, W._dt.date
        W._dt.datetime, W._dt.date = _FrozenDatetime, _FrozenDate
        try:
            out = W.what_to_wear_periods_for_loc(loc, "en", hours=24)
            out_date = W.what_to_wear_periods_for_loc(
                loc, "en", date=_dt.date(2026, 8, 20))
        finally:
            W._dt.datetime, W._dt.date = _orig_dt, _orig_date
        check("header names the place and the window",
              "Testville" in out and "24h" in out)
        check("all three periods appear for a full 24h window starting at 09:00",
              "Afternoon" in out and "Evening" in out and "Morning" in out)
        # A 24h window from 09:00 spans the tail of today's morning (9-12)
        # AND the start of tomorrow's morning (6-9) -- both are real, partial
        # buckets, so "Morning" legitimately appears twice, not once.
        check("both the tail of today's morning and the start of tomorrow's appear",
              table_of(out).count("Morning") == 2, table_of(out))

        check("a date-based forecast headers with that date, not an hour count",
              "20.08.2026" in out_date)
        _t_date = table_of(out_date)
        check("a date-based forecast covers all three periods of that single day",
              _t_date.count("Morning") == 1 and _t_date.count("Afternoon") == 1
              and _t_date.count("Evening") == 1, _t_date)

    with _Patches(fetch_hourly=lambda lat, lon, days=3: None):
        check("forecast service down -> translated error, not a crash",
              "unavailable" in W.what_to_wear_periods_for_loc(loc, "en", hours=24))

    # A date so far out no hourly data covers it.
    def sparse_hourly(lat, lon, days=3):
        return {"utc_offset_seconds": 0, "points": [
            {"time": _dt.datetime(2026, 8, 15, 8, 0), "temp_c": 10, "code": 0, "wind_kmh": 5}]}
    with _Patches(fetch_hourly=sparse_hourly):
        out = W.what_to_wear_periods_for_loc(loc, "en", date=_dt.date(2030, 1, 1))
        check("a date far beyond available data -> a clear no-data message",
              "next" in out or "couple of weeks" in out)

    # advise_fn covering multiple buckets in one call.
    with _Patches(fetch_hourly=fake_hourly):
        _orig_dt = W._dt.datetime
        W._dt.datetime = _FrozenDatetime
        seen_buckets = []
        def advisor(buckets, lang):
            seen_buckets.append(len(buckets))
            return {f'{b["date"].isoformat()}_{b["period"]}': "a raincoat" for b in buckets}
        try:
            out = W.what_to_wear_periods_for_loc(loc, "en", hours=24, advise_fn=advisor)
        finally:
            W._dt.datetime = _orig_dt
        check("advise_fn is called exactly ONCE covering every bucket, not once per bucket",
              seen_buckets == [seen_buckets[0]] and len(seen_buckets) == 1)
        check("advise_fn's answer is used verbatim", "a raincoat" in out, out)
        check("...and identical advice is grouped into ONE line, not repeated "
              "once per bucket", out.count("a raincoat") == 1, out)


def test_format_forecast_layout():
    """The table has to survive a phone screen, and the tags have to be parsed.

    The first version of this shipped broken in two ways at once: the send site
    passed no parse_mode, so Telegram printed "<pre>" literally and drew the
    table in a proportional font; and the conditions description sat in the
    row, so "сильная гроза с градом" pushed a line to 44 characters and wrapped.
    """
    import datetime as _dt
    d0, d1 = _dt.date(2026, 8, 17), _dt.date(2026, 8, 18)

    def bk(date, period, t, h, w, code):
        return {"date": date, "period": period, "temp_c": t, "humidity": h,
                "wind_kmh": w, "code": code}

    # The widest cases at once: longest description, 3-digit humidity, a
    # missing humidity reading, a negative and a positive temperature.
    buckets = [bk(d0, "afternoon", -11, 100, 42, 99),
               bk(d0, "evening", -3, 95, 31, 57),
               bk(d1, "morning", 5, 88, 12, 61),
               bk(d1, "afternoon", 9, None, 4, 0)]
    for lang in ("ru", "en"):
        out = W.format_forecast(buckets, lang)
        rows = [r for r in table_of(out).splitlines() if r.strip()]
        widest = max(len(r) for r in rows)
        # 34 is about what a narrow phone fits in Telegram's monospace font;
        # the real rows land around 25-29.
        check("[%s] no table row is wide enough to wrap on a phone (%d chars)"
              % (lang, widest), widest <= 34, "\n".join(rows))

        # Columns line up only if every data row uses identical field widths.
        # Measured on everything BEFORE the trailing icon, since the icon is
        # the one variable-width cell (and last precisely so it cannot shift
        # anything). Date group lines start with a digit; data rows don't.
        data = [r for r in rows[1:] if not r[0].isdigit()]
        check("[%s] every data row is padded to the same column layout" % lang,
              len({len(r.rsplit("  ", 1)[0]) for r in data}) == 1, "\n".join(data))
        check("[%s] the temperature sign column is fixed" % lang,
              len({r.rsplit("  ", 1)[0].find("+") for r in data
                   if "+" in r.rsplit("  ", 1)[0]}) == 1, "\n".join(data))

        check("[%s] the description is NOT in the row (that is what wrapped)"
              % lang, W._code_desc(99, lang) not in table_of(out))
        check("[%s] ...but the words are still shown, in the legend" % lang,
              W._code_desc(99, lang) in out and W._code_desc(0, lang) in out, out)
        # Four buckets, four DISTINCT conditions here -> four legend lines. The
        # user's real 24h reply had "пасмурно" on all four rows; repeating it
        # four times is what the legend exists to avoid.
        check("[%s] the legend names each distinct condition exactly once" % lang,
              out.count(W._code_desc(99, lang)) == 1, out)

        check("[%s] the date is a group line, not a suffix on every row" % lang,
              table_of(out).count("18.08") == 1, table_of(out))
        # Three of the four buckets carry a humidity reading; the fourth has
        # none and must show a dash rather than being reported as 0% (which
        # would read as a real, and very wrong, measurement).
        # Counted over the DATA rows only -- the header row carries a "%" of
        # its own as the column label.
        check("[%s] a missing humidity reading is a dash, not a fake 0%%" % lang,
              sum(r.count("%") for r in data) == 3
              and any("—" in r for r in data), "\n".join(data))
        check("[%s] the advice heading replaces the per-line verb" % lang,
              "👕" in out and "надень" not in out.lower()
              and "wear " not in out.lower(), out)

    # A single repeated condition collapses to ONE legend line.
    same = [bk(d0, "afternoon", 21, 53, 16, 3), bk(d0, "evening", 17, 76, 19, 3),
            bk(d1, "morning", 13, 90, 6, 3), bk(d1, "afternoon", 17, 67, 6, 3)]
    out = W.format_forecast(same, "ru")
    check("one condition across every bucket -> one legend line",
          out.count("пасмурно") == 1, out)

    # Both surfaces send this with parse_mode="HTML"; <pre> is the only tag,
    # and everything interpolated into it must be escaped or Telegram rejects
    # the whole message (and _post_message falls back to stripped plain text).
    hostile = [bk(d0, "afternoon", 20, 50, 5, 3)]
    out = W.format_forecast(hostile, "en", advice_map={
        "2026-08-17_afternoon": "a <b>bold</b> coat & scarf"})
    check("advice markup is escaped, not passed through as live tags",
          "&lt;b&gt;" in out and "<b>" not in out and "&amp;" in out, out)
    check("the only real tag in the reply is <pre>",
          out.count("<pre>") == 1 and out.count("</pre>") == 1
          and out.replace("<pre>", "").replace("</pre>", "").count("<") == 0, out)


def test_forecast_is_sent_with_html_parse_mode():
    """The <pre> table is worthless without parse_mode="HTML" at the send site.

    This is a regression guard on _send_weather specifically: it was the ONE
    send site in the weather flow that omitted parse_mode, which is why the
    user's first real forecast showed the literal tags.

    The flow has moved once already (tg_accounts.py -> tg_weather.py), so this
    globs the tg_* family and locates the method with ast instead of naming a
    file and matching up to the next "\n    def " -- a terminator that also
    silently missed a _send_weather that happened to be the LAST method in its
    class, which it now is.
    """
    import ast, glob, re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = ""
    for path in sorted(glob.glob(os.path.join(root, "bot", "tg_*.py"))):
        text = open(path, encoding="utf-8").read()
        if "def _send_weather" not in text:
            continue
        lines = text.splitlines(keepends=True)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name == "_send_weather":
                body = "".join(lines[node.lineno - 1:node.end_lineno])
    check("_send_weather still exists to check", bool(body))
    if body:
        sends = re.findall(r"_send_text\([^)]*\)", body, re.DOTALL)
        check("_send_weather sends at least one message", bool(sends), body)
        for s in sends:
            check("the forecast is sent with parse_mode=HTML so <pre> is parsed",
                  'parse_mode="HTML"' in s, s)


def test_what_to_wear_periods_resolves_city():
    with _Patches(geocode_city=lambda city, lang="en": {"lat": 1, "lon": 2, "name": "Rome", "country": "IT"},
                  fetch_hourly=lambda lat, lon, days=3: None):
        out = W.what_to_wear_periods("Rome", "en", hours=24)
        check("what_to_wear_periods resolves the city before fetching a forecast",
              "unavailable" in out)
    with _Patches(geocode_city=lambda city, lang="en": None):
        check("an unresolvable city -> the same not-found message as the snapshot API",
              "Couldn't find" in W.what_to_wear_periods("Nowhereville", "en"))


def test_what_to_wear_advise_fn():
    # advise_fn is how the LLM plugs in -- the user wants the model to decide
    # exactly what to wear, not a fixed temperature-band table.
    with _Patches(geocode_city=lambda city, lang="en": {"lat": 1, "lon": 2, "name": "Paris", "country": "FR"},
                  fetch_current=lambda lat, lon: {"temp_c": 5.0, "wind_kmh": 10, "code": 61}):
        calls = []
        def advisor(temp_c, code, wind_kmh, desc, lang):
            calls.append((temp_c, code, wind_kmh, desc, lang))
            return "a waxed cotton jacket and waterproof boots"
        out = W.what_to_wear("Paris", "en", advise_fn=advisor)
        check("advise_fn is called with the live conditions",
              calls == [(5.0, 61, 10, "light rain", "en")])
        check("advise_fn's answer is used verbatim in the reply",
              "waxed cotton jacket" in out)

        # advise_fn raising must not break the button -- falls back to the
        # fixed-band table instead of propagating.
        def broken_advisor(*a, **k):
            raise RuntimeError("LM Studio unreachable")
        out2 = W.what_to_wear("Paris", "en", advise_fn=broken_advisor)
        check("a raising advise_fn falls back to the fixed-band table",
              "waterproof" in out2 or "jacket" in out2 or "coat" in out2)

        # advise_fn returning nothing (empty string / None) is also a fallback
        # trigger, not a bare "👕" with nothing after it.
        out3 = W.what_to_wear("Paris", "en", advise_fn=lambda *a, **k: "")
        check("an empty advise_fn answer falls back instead of an empty phrase",
              "👕 " in out3 and len(out3.rsplit("👕 ", 1)[1].strip()) > 5, out3)

        # Apostrophes must survive as apostrophes: html.escape's default turns
        # them into &#x27;, and Telegram's HTML parser only promises the named
        # entities, so a numeric reference reaches the reader as literal text.
        out4 = W.what_to_wear("Paris", "en",
                              advise_fn=lambda *a, **k: "a shepherd's coat")
        check("an apostrophe in the advice is not mangled into &#x27;",
              "shepherd's coat" in out4 and "&#x27;" not in out4, out4)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
