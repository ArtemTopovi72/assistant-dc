"""Date arithmetic for the agent: what day it is, the weekday of a date, how many days (and
Russian working days) until a date or a holiday, the date N days/weeks/months/working days on.

Counted here, not by the model: it knew today's date from its instructions and still did the
sums in its head. Working days follow the Russian production calendar (isdayoff.ru: official
transfers of days off), with plain weekends when that service is down -- and the answer says so.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import threading
from typing import Callable, Optional

logger = logging.getLogger("assistant.dates")

ISDAYOFF = "https://isdayoff.ru/api/getdata?year={year}"
_WEEKDAY_RU = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
_MONTH_RU = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
             "октября", "ноября", "декабря")
_years: dict = {}
_lock = threading.Lock()


class DateError(Exception):
    pass


# (words any of which names it, month, day); Easter is computed
_HOLIDAYS = (
    (("новый год", "нового года", "новогод", "new year"), 1, 1),
    (("православное рождество", "рождество христово", "рождества", "рождество"), 1, 7),
    (("валентин",), 2, 14),
    (("23 февраля", "защитника отечества"), 2, 23),
    (("8 марта", "женский день", "женского дня"), 3, 8),
    (("1 мая", "первомай", "праздник весны и труда"), 5, 1),
    (("9 мая", "день победы", "дня победы"), 5, 9),
    (("12 июня", "день россии", "дня россии"), 6, 12),
    (("1 сентября", "день знаний", "дня знаний"), 9, 1),
    (("хэллоуин", "хеллоуин", "halloween"), 10, 31),
    (("4 ноября", "народного единства"), 11, 4),
    (("католическое рождество", "christmas"), 12, 25),
)


def holiday(text: str, today: _dt.date) -> Optional[tuple[str, _dt.date]]:
    """(name, its next date on or after `today`) for a holiday named in `text`, else None."""
    t = " ".join((text or "").lower().replace("ё", "е").split())
    if "пасх" in t or "easter" in t:
        from dateutil.easter import easter, EASTER_ORTHODOX
        d = easter(today.year, EASTER_ORTHODOX)
        return "Пасха", d if d >= today else easter(today.year + 1, EASTER_ORTHODOX)
    for words, m, d in _HOLIDAYS:
        if any(w in t for w in words):
            when = _dt.date(today.year, m, d)
            return words[0], when if when >= today else _dt.date(today.year + 1, m, d)
    return None


def parse(text: str, today: _dt.date, future: bool = True) -> _dt.date:
    """A date from ISO, a holiday name, or words («15 марта», «через две недели»)."""
    t = (text or "").strip()
    if not t or t.lower() in ("today", "сегодня", "now", "сейчас"):
        return today
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return _dt.date(*map(int, m.groups()))
    h = holiday(t, today)
    if h:
        return h[1]
    import dateparser
    got = dateparser.parse(t, languages=["ru", "en"], settings={
        "RELATIVE_BASE": _dt.datetime.combine(today, _dt.time(12)),
        "PREFER_DATES_FROM": "future" if future else "past", "PREFER_DAY_OF_MONTH": "first"})
    if not got:
        raise DateError(f"cannot read a date from {t!r}: pass it as YYYY-MM-DD")
    return got.date()


def _offdays(year: int, get: Optional[Callable]) -> Optional[str]:
    """isdayoff.ru: one char per day of the year, '1' a day off. None when unreachable."""
    with _lock:
        if year in _years:
            return _years[year]
    try:
        if get is None:
            import requests
            r = requests.get(ISDAYOFF.format(year=year), timeout=10)
            r.raise_for_status()
            s = r.text.strip()
        else:
            s = get(ISDAYOFF.format(year=year))
        if not re.fullmatch(r"[01]{365,366}", s):
            raise ValueError(s[:40])
    except Exception as exc:
        logger.warning("dates: production calendar for %d unavailable (%s)", year, exc)
        return None
    with _lock:
        _years[year] = s
    return s


def is_day_off(day: _dt.date, get: Optional[Callable] = None) -> tuple[bool, bool]:
    """(day off, by the official calendar -- False means plain Saturday/Sunday)."""
    s = _offdays(day.year, get)
    if s is None or day.timetuple().tm_yday > len(s):
        return day.weekday() >= 5, False
    return s[day.timetuple().tm_yday - 1] == "1", True


def workdays_between(a: _dt.date, b: _dt.date, get: Optional[Callable] = None) -> tuple[int, bool]:
    """Working days after `a` up to and including `b` (a<b), and whether the official calendar
    counted them."""
    n, official = 0, True
    d = a
    while d < b:
        d += _dt.timedelta(days=1)
        off, by_cal = is_day_off(d, get)
        official &= by_cal
        n += not off
    return n, official


def add_months(d: _dt.date, months: int) -> _dt.date:
    import calendar
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    return _dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def ru(d: _dt.date) -> str:
    return f"{d.day} {_MONTH_RU[d.month - 1]} {d.year} ({_WEEKDAY_RU[d.weekday()]})"


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def answer(today: _dt.date, date: str = "", until: str = "", add_days: int = 0, add_weeks: int = 0,
           add_months_n: int = 0, workdays: bool = False, get: Optional[Callable] = None) -> str:
    """The tool's reply: plain facts the model quotes."""
    start = parse(date, today, future=not until)
    out = [f"Today is {ru(today)}, {today.isoformat()}."]
    if add_days or add_weeks or add_months_n:
        if workdays and add_days:
            d, left, official = start, abs(add_days), True
            step = 1 if add_days > 0 else -1
            while left:
                d += _dt.timedelta(days=step)
                off, by_cal = is_day_off(d, get)
                official &= by_cal
                left -= not off
            end = add_months(d + _dt.timedelta(weeks=add_weeks), add_months_n)
            how = f"{add_days} working days" + ("" if official else " (weekends only: the official calendar was unreachable)")
        else:
            end = add_months(start + _dt.timedelta(days=add_days, weeks=add_weeks), add_months_n)
            how = ", ".join(p for p in (f"{add_months_n} months" if add_months_n else "",
                                        f"{add_weeks} weeks" if add_weeks else "",
                                        f"{add_days} days" if add_days else "") if p)
        out.append(f"{ru(start)} + {how} = {ru(end)}, {end.isoformat()}.")
        start = end
    if until:
        h = holiday(until, start)
        target = h[1] if h else parse(until, start)
        days = (target - start).days
        line = (f"From {ru(start)} to {ru(target)}" + (f" ({h[0]})" if h else "")
                + f": {days} {_plural(days, 'день', 'дня', 'дней')}")
        if abs(days) >= 365:
            from dateutil.relativedelta import relativedelta
            r = relativedelta(max(start, target), min(start, target))
            line += f" ({r.years} {_plural(r.years, 'год', 'года', 'лет')} {r.months} мес. {r.days} дн.)"
        elif abs(days) >= 7:
            line += f" ({abs(days) // 7} нед. {abs(days) % 7} дн.)"
        if 0 < days <= 731:                 # an age in working days means nothing and costs a call a year
            wd, official = workdays_between(start, target, get)
            line += (f", рабочих {wd} " + ("по производственному календарю РФ" if official
                                            else "(только без суббот и воскресений: календарь недоступен)"))
        out.append(line + ".")
        start = target
    if not until and not (add_days or add_weeks or add_months_n) and start != today:
        days = (start - today).days
        n = f"{abs(days)} {_plural(days, 'день', 'дня', 'дней')}"
        out.append(f"{ru(start)}, {start.isoformat()}: " + (f"через {n}." if days > 0 else f"{n} назад."))
    off, official = is_day_off(start, get)
    out.append(f"{ru(start)} is a {'day off' if off else 'working day'} in Russia"
               + ("" if official else " (weekend rule only)") + ".")
    return " ".join(out)
