"""Currency conversion at a current (or dated) rate, computed here, never by the model.

Rubles go by the Central Bank of Russia (www.cbr-xml-daily.ru mirrors its daily XML as JSON,
with an archive by date) -- the rate a Russian user means by «по курсу». Every other pair goes
by ExchangeRate-API's open endpoint (open.er-api.com, daily, 160+ currencies), then by
fawazahmed0/currency-api (two mirrors, 200+ currencies and crypto, history by date).

The bot used to search the web for «EUR to RUB» and do the multiplication itself: 46 s and a
rate from whichever page loaded (2026-10-24).
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("assistant.currency")

CBR_LATEST = "https://www.cbr-xml-daily.ru/daily_json.js"
CBR_DAY = "https://www.cbr-xml-daily.ru/archive/{y:04d}/{m:02d}/{d:02d}/daily_json.js"
ER_LATEST = "https://open.er-api.com/v6/latest/{base}"
FAWAZ = ("https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{ver}/v1/currencies/{base}.json",
         "https://{ver}.currency-api.pages.dev/v1/currencies/{base}.json")
TTL = 3600
_cache: dict = {}
_lock = threading.Lock()


class RateError(Exception):
    pass


def _get_json(url: str):
    import requests
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def _cached(key, fetch: Callable):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
    data = fetch()
    with _lock:
        _cache[key] = (time.time(), data)
    return data


def _cbr(get, day: Optional[_dt.date]):
    """(rubles per unit by code, the rate's date). A weekend or holiday has no archive file:
    step back to the last published day."""
    if day is None:
        data = _cached(("cbr", None), lambda: get(CBR_LATEST))
    else:
        data = None
        for back in range(15):          # the New Year break publishes nothing for ~10 days
            d = day - _dt.timedelta(days=back)
            try:
                data = _cached(("cbr", d), lambda d=d: get(CBR_DAY.format(y=d.year, m=d.month, d=d.day)))
                break
            except Exception:
                continue
        if data is None:
            raise RateError(f"no Central Bank rate published on or before {day}")
    rates = {"RUB": 1.0}
    for code, v in (data.get("Valute") or {}).items():
        rates[code] = float(v["Value"]) / float(v["Nominal"])
    return rates, str(data.get("Date", ""))[:10]


def _market(get, base: str, day: Optional[_dt.date]):
    """(units of each code per 1 `base`, date, source)."""
    errors = []
    if day is None:
        try:
            data = _cached(("er", base), lambda: get(ER_LATEST.format(base=base)))
            if data.get("result") == "success":
                when = _dt.datetime.fromtimestamp(int(data.get("time_last_update_unix") or 0), _dt.timezone.utc).date()
                return {k.upper(): float(v) for k, v in data["rates"].items()}, when.isoformat(), "ExchangeRate-API"
            errors.append(str(data.get("error-type")))
        except Exception as exc:
            errors.append(f"open.er-api: {exc}")
    ver = day.isoformat() if day else "latest"
    for tpl in FAWAZ:
        try:
            data = _cached(("fawaz", base, ver), lambda tpl=tpl: get(tpl.format(ver=ver, base=base.lower())))
            return ({k.upper(): float(v) for k, v in data[base.lower()].items()},
                    str(data.get("date", ver)), "fawazahmed0 currency-api")
        except Exception as exc:
            errors.append(f"currency-api: {exc}")
    raise RateError("; ".join(errors) or "no source answered")


def convert(amount: float, frm: str, to: str, day: Optional[_dt.date] = None,
            get: Optional[Callable] = None) -> dict:
    """{"amount", "from", "to", "result", "rate", "date", "source"}; RateError when no source
    knows the pair."""
    get = get or _get_json
    frm, to = frm.strip().upper(), to.strip().upper()
    if frm == to:
        return {"amount": amount, "from": frm, "to": to, "result": amount, "rate": 1.0,
                "date": (day or _dt.date.today()).isoformat(), "source": "same currency"}
    if "RUB" in (frm, to):
        try:
            rub, when = _cbr(get, day)
            if frm in rub and to in rub:
                rate = rub[frm] / rub[to]
                return {"amount": amount, "from": frm, "to": to, "result": amount * rate, "rate": rate,
                        "date": when, "source": "Central Bank of Russia"}
        except Exception as exc:
            logger.warning("currency: CBR failed (%s), trying the market sources", exc)
    rates, when, src = _market(get, frm, day)
    if to not in rates:
        raise RateError(f"{src} has no rate for {frm}->{to}")
    rate = rates[to]
    return {"amount": amount, "from": frm, "to": to, "result": amount * rate, "rate": rate,
            "date": when, "source": src}
