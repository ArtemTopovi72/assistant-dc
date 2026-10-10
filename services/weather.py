"""Weather lookup for the "What to wear" button.

Uses Open-Meteo (open-meteo.com) for both geocoding and forecast — no API
key, no signup, generous free tier. Two calls: resolve the city name to
coordinates, then read current conditions at those coordinates.

Kept deliberately free of any Telegram/GUI import so it can be unit tested
and reused from either surface.
"""
import datetime as _dt
import html as _html_mod
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger("assistant.weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 8

# WMO weather code -> (English, Russian) short description. Grouped ranges
# per the Open-Meteo docs (https://open-meteo.com/en/docs -> WMO codes).
_CODE_DESC: dict[int, tuple] = {
    0: ("clear sky", "ясно"),
    1: ("mostly clear", "малооблачно"),
    2: ("partly cloudy", "переменная облачность"),
    3: ("overcast", "пасмурно"),
    45: ("fog", "туман"),
    48: ("icy fog", "изморозь"),
    51: ("light drizzle", "легкая морось"),
    53: ("drizzle", "морось"),
    55: ("heavy drizzle", "сильная морось"),
    56: ("freezing drizzle", "ледяная морось"),
    57: ("heavy freezing drizzle", "сильная ледяная морось"),
    61: ("light rain", "небольшой дождь"),
    63: ("rain", "дождь"),
    65: ("heavy rain", "сильный дождь"),
    66: ("freezing rain", "ледяной дождь"),
    67: ("heavy freezing rain", "сильный ледяной дождь"),
    71: ("light snow", "небольшой снег"),
    73: ("snow", "снег"),
    75: ("heavy snow", "сильный снег"),
    77: ("snow grains", "снежная крупа"),
    80: ("light showers", "небольшой ливень"),
    81: ("showers", "ливень"),
    82: ("heavy showers", "сильный ливень"),
    85: ("light snow showers", "небольшой снегопад"),
    86: ("heavy snow showers", "сильный снегопад"),
    95: ("thunderstorm", "гроза"),
    96: ("thunderstorm with hail", "гроза с градом"),
    99: ("severe thunderstorm with hail", "сильная гроза с градом"),
}


def _code_desc(code: int, lang: str) -> str:
    en, ru = _CODE_DESC.get(int(code), ("unknown conditions", "неизвестная погода"))
    return ru if lang == "ru" else en


# The forecast TABLE carries an icon rather than the description, because the
# description is what used to blow the row width: "сильная гроза с градом" is
# 22 characters, and a row that wide wraps on a phone -- the continuation
# starts back at column 0 and the columns stop lining up. The words are still
# shown, once each, in a legend under the table (see format_forecast).
_CODE_ICON: dict[int, str] = {
    0: "☀️", 1: "🌤", 2: "⛅", 3: "☁️",
    45: "🌫", 48: "🧊",
    51: "🌦", 53: "🌦", 55: "🌦", 56: "🧊", 57: "🧊",
    61: "🌧", 63: "🌧", 65: "🌧", 66: "🧊", 67: "🧊",
    71: "🌨", 73: "🌨", 75: "🌨", 77: "🌨", 85: "🌨", 86: "🌨",
    80: "🌦", 81: "🌧", 82: "🌧",
    95: "⛈", 96: "⛈", 99: "⛈",
}

# Codes that put ice underfoot, which is the one forecast fact with a safety
# consequence, so they get their own 🧊 icon rather than sharing 🌧 with plain
# rain: freezing drizzle, freezing rain, and rime fog (48 deposits ice on
# contact). WMO has no separate "sleet" code -- sleet arrives as one of these.
_ICY_CODES = (48, 56, 57, 66, 67)


def _code_icon(code: int) -> str:
    return _CODE_ICON.get(int(code), "•")


def _esc(s: str) -> str:
    """HTML-escape for a Telegram text node -- quote=False deliberately.

    html.escape's default also rewrites ' as &#x27;, and Telegram's HTML parser
    only promises the named entities &amp; &lt; &gt; &quot;. A numeric
    reference is passed through as literal text, so an apostrophe anywhere in
    the reply ("it&#x27;s bitterly cold", and any LLM advice with a
    contraction) showed up as that escape sequence. Quotes only need escaping
    inside attribute values, and this module emits none -- <pre> is the only
    tag it produces.
    """
    return _html_mod.escape(s or "", quote=False)


def _nominative(city: str) -> str:
    """«Твери» -> «Тверь», «Нижнем Новгороде» -> «Нижний Новгород».

    The geocoder matches name PREFIXES: the inflected «Твери» found Tiberias
    (Тверия, Israel, +31°) for a user asking about Tver (live 2026-09-27).
    Place names and the adjectives in them go to their dictionary form; any
    other word is kept as typed. "" when nothing changes or no morphology."""
    try:
        import pymorphy3
        m = _nominative.m = getattr(_nominative, "m", None) or pymorphy3.MorphAnalyzer()
    except Exception:
        return ""
    out = []
    for w in city.split():
        ps = m.parse(w)
        geo = [p for p in ps if "Geox" in p.tag]
        adj = [p for p in ps if "ADJF" in p.tag]
        pick = geo[0] if geo else (adj[0] if adj and len(city.split()) > 1 else None)
        out.append(pick.normal_form.capitalize() if pick else w)
    res = " ".join(out)
    return "" if res.lower() == city.lower() else res


# Colloquial names, keyed by the stem (case endings stripped). "Питер" geocoded
# to a village and gave St Petersburg +1 and snow (live 2026-09-28).
_ALIASES = {"питер": "Санкт-Петербург", "спб": "Санкт-Петербург", "мск": "Москва",
            "екб": "Екатеринбург", "ёбург": "Екатеринбург", "екатеринбург": "Екатеринбург",
            "нск": "Новосибирск", "новосиб": "Новосибирск", "нижний": "Нижний Новгород",
            "нн": "Нижний Новгород", "краснодар": "Краснодар", "владик": "Владивосток",
            "калининград": "Калининград", "челяб": "Челябинск", "ростов": "Ростов-на-Дону"}


def geocode_city(city: str, lang: str = "en") -> Optional[dict]:
    """Return {"lat", "lon", "name", "country"} for the best match, or None.

    `lang` picks the language of the returned NAME: the geocoder used to be
    asked in English only, so a Russian user got a card headed "Sochi" and
    "Kazan’" (live, 2026-09-12, journey 7). Open-Meteo answers in the
    language asked for and falls back to English when it has no local name.
    """
    city = (city or "").strip()
    if not city:
        return None
    city = _ALIASES.get(city.lower().rstrip("еуаомы"), city)
    if re.search("[а-яё]", city, re.I):
        nom = _nominative(city)
        if nom:
            found = geocode_city(nom, lang)
            if found:
                return found
    # The language also steers MATCHING: "New York" asked in Russian found
    # York, Nebraska (live 2026-09-28, local_time said 10:09 at 11:09).
    # A Latin-script name is matched in English.
    if not re.search("[а-яё]", city, re.I):
        lang = "en"
    try:
        r = requests.get(_GEOCODE_URL,
                         params={"name": city, "count": 1,
                                 "language": (lang or "en")[:2].lower()},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        top = results[0]
        return {"lat": top["latitude"], "lon": top["longitude"],
                "name": top.get("name", city), "country": top.get("country", ""),
                "timezone": top.get("timezone", "")}
    except Exception as exc:
        logger.warning("geocode_city(%r) failed: %s", city, exc)
        return None


def resolve_city(city: str, correct_fn=None, lang: str = "en") -> Optional[dict]:
    """geocode_city(city), retrying once with an LLM-corrected spelling if the
    raw input doesn't geocode. Typed city names are often abbreviations,
    informal nicknames, or plain typos ("Спб", "Питнр") that Open-Meteo's
    geocoder won't match -- `correct_fn(city) -> str`, normally an LLM call
    (see tg_accounts._weather_correct_fn), is given a chance to normalize the
    name before we give up.
    """
    loc = geocode_city(city, lang)
    if loc or not correct_fn:
        return loc
    try:
        corrected = (correct_fn(city) or "").strip()
    except Exception as exc:
        logger.warning("correct_fn(%r) failed: %s", city, exc)
        return None
    if not corrected or corrected.strip().lower() == (city or "").strip().lower():
        return None
    return geocode_city(corrected, lang)


def not_found_message(city: str, lang: str) -> str:
    # `city` is raw user input echoed back into an HTML-parsed reply, so it is
    # escaped here: a name containing "<" would otherwise get the whole message
    # rejected as malformed markup.
    city = _esc(city)
    return (f"Couldn't find a city called \"{city}\" — try a different spelling."
            if lang != "ru" else
            f"Не нашёл город «{city}» — попробуй написать иначе.")


def fetch_current(lat: float, lon: float) -> Optional[dict]:
    """Return {"temp_c", "wind_kmh", "code"} for the given coordinates, or None."""
    try:
        r = requests.get(_FORECAST_URL,
                         params={"latitude": lat, "longitude": lon,
                                "current": "temperature_2m,wind_speed_10m,weather_code,relative_humidity_2m"},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        cur = r.json().get("current") or {}
        if "temperature_2m" not in cur:
            return None
        return {"temp_c": cur["temperature_2m"],
                "wind_kmh": cur.get("wind_speed_10m", 0),
                "humidity": cur.get("relative_humidity_2m"),
                "code": cur.get("weather_code", 0)}
    except Exception as exc:
        logger.warning("fetch_current(%s,%s) failed: %s", lat, lon, exc)
        return None


# Fallback only: used when no LLM advisor is supplied (e.g. LM Studio is
# unreachable) so the button still answers something useful. The primary path
# is an LLM call (see tg_accounts._weather_advise), which reasons over the
# actual conditions instead of picking from a canned table.
_WET_CODES = frozenset((51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99))


def umbrella_line(buckets: list, lang: str = "en") -> str:
    """A yes/no for the umbrella: "погода завтра, брать зонт?" got a table and
    clothing advice but never the answer (live 2026-09-28)."""
    ru = (lang or "").startswith("ru")
    wet = [b for b in buckets if b.get("wet_hours")]
    if not wet:
        return "🌂 Зонт не нужен — дождя не ожидается" if ru else "🌂 No umbrella needed — no rain expected"
    whens = ", ".join(dict.fromkeys(_period_label(b["period"], lang).lower() for b in wet))
    return f"☔ Возьми зонт — дождь: {whens}" if ru else f"☔ Take an umbrella — rain: {whens}"


def _clothing_advice(temp_c: float, code: int, wind_kmh: float, lang: str) -> str:
    wet = code in _WET_CODES
    snowy = code in (71, 73, 75, 77, 85, 86)
    # Ice takes precedence over "wet" and "snowy": freezing rain reads as rain
    # by code, but an umbrella is not the thing that matters when the pavement
    # is glazed. Wet snow has no code of its own -- snowfall around freezing is
    # slush rather than powder, so it's inferred from the temperature, which is
    # also why it can't be a plain code lookup.
    icy = code in _ICY_CODES
    wet_snow = snowy and -2 <= temp_c <= 2
    if lang == "ru":
        # NOMINATIVE, deliberately: these are noun phrases listed under a
        # "что надеть" heading ("Утро — лёгкая куртка"), not objects of a verb.
        # They were briefly accusative, to read as "надень лёгкую куртку" --
        # but the primary advice path is the LLM (see the advise_fn contract
        # above), which returns a plain nominative noun phrase, so the verb
        # prefix produced "надень футболка и ветровка" on every real reply.
        # Dropping the verb makes both paths grammatical instead of asking a
        # model to reliably inflect Russian case.
        if temp_c <= -10:
            base = "пуховик, тёплая шапка и перчатки"
        elif temp_c <= 0:
            base = "зимняя куртка, шапка и шарф"
        elif temp_c <= 10:
            base = "тёплая куртка или пальто"
        elif temp_c <= 18:
            base = "лёгкая куртка или худи"
        elif temp_c <= 25:
            base = "футболка и лёгкие брюки/джинсы"
        else:
            base = "лёгкая, свободная одежда"
        # Garments joined with "+", and no "— сильный мороз"/"— возможен дождь"
        # explanations: three em-dash clauses in one line ran to 130 characters,
        # and the conditions they restated are already in the table and legend
        # directly above.
        if icy: base += " + нескользящая обувь, гололёд"
        elif wet_snow: base += " + непромокаемая обувь, мокрый снег"
        elif snowy: base += " + непромокаемая обувь"
        elif wet: base += " + зонт"
        if wind_kmh >= 30: base += " + ветрозащита"
        return base
    if temp_c <= -10:
        base = "a heavy down coat, warm hat and gloves"
    elif temp_c <= 0:
        base = "a winter coat, hat and scarf"
    elif temp_c <= 10:
        base = "a warm jacket or coat"
    elif temp_c <= 18:
        base = "a light jacket or hoodie"
    elif temp_c <= 25:
        base = "a t-shirt and light trousers/jeans"
    else:
        base = "light, loose clothing"
    if icy: base += " + non-slip boots, black ice"
    elif wet_snow: base += " + waterproof boots, wet snow"
    elif snowy: base += " + waterproof boots"
    elif wet: base += " + an umbrella"
    if wind_kmh >= 30: base += " + a windbreaker"
    return base


def what_to_wear_for_loc(loc: dict, lang: str = "en", advise_fn=None) -> str:
    """Same reply as what_to_wear, for a city already resolved to coordinates
    (avoids a second geocode+correction round trip when the caller already
    resolved the location itself, e.g. to decide what to remember as the
    user's default)."""
    cur = fetch_current(loc["lat"], loc["lon"])
    if not cur:
        return ("Weather service is unavailable right now — try again in a bit."
                if lang != "ru" else
                "Сервис погоды сейчас недоступен — попробуй чуть позже.")
    temp = cur["temp_c"]
    desc = _code_desc(cur["code"], lang)
    advice = ""
    if advise_fn is not None:
        try:
            advice = (advise_fn(temp, cur["code"], cur["wind_kmh"], desc, lang) or "").strip()
        except Exception as exc:
            logger.warning("advise_fn failed, falling back to fixed bands: %s", exc)
    if not advice:
        advice = _clothing_advice(temp, cur["code"], cur["wind_kmh"], lang)
    # Escaped because this reply is sent with parse_mode=HTML, and the place
    # name traces back to whatever the user typed as a city.
    place = _esc(loc["name"])
    ru = lang == "ru"
    # Humidity and wind are shown here too, not just in the forecast table:
    # "+21°C, overcast" alone leaves out the two numbers that decide whether
    # +21 feels pleasant or muggy.
    bits = [f"{temp:+.0f}°C", desc]
    hum = cur.get("humidity")
    if hum is not None:
        bits.append(f"💧 {hum:.0f}%")
    bits.append(f"💨 {cur['wind_kmh']:.0f} " + ("км/ч" if ru else "km/h"))
    line = f"📍 {place}: {_code_icon(cur['code'])} " + ", ".join(bits)
    # A 👕 label instead of "Надень ..." -- see _clothing_advice on why the
    # verb had to go.
    return f"{line}\n👕 {_esc(advice)}"


def what_to_wear(city: str, lang: str = "en", advise_fn=None, correct_fn=None) -> str:
    """Full one-line reply: "+12°C, cloudy — wear a light jacket...".

    `advise_fn(temp_c, code, wind_kmh, desc, lang) -> str`, if given, decides
    the clothing (normally an LLM call — see tg_accounts._weather_advise_fn).
    Falls back to the fixed temperature-band table if omitted, or if
    `advise_fn` itself raises or returns nothing (e.g. LM Studio down) — the
    button still answers something useful rather than erroring out.

    `correct_fn(city) -> str`, if given, is tried once when the raw city
    fails to geocode (see resolve_city) — normally an LLM call that maps a
    typo/abbreviation/nickname to a name the geocoder recognizes.

    Never raises -- network/lookup failures come back as a translated,
    user-facing error string instead of propagating.
    """
    loc = resolve_city(city, correct_fn=correct_fn, lang=lang)
    if not loc:
        return not_found_message(city, lang)
    return what_to_wear_for_loc(loc, lang, advise_fn)


# ── multi-period forecast (morning / afternoon / evening) ──────────────────
# Night (0-6) is deliberately not a bucket -- the feature request was
# specifically morning/afternoon/evening, and nobody needs clothing advice
# for while they're asleep.
_PERIOD_RANGES = (("morning", 6, 12), ("afternoon", 12, 18), ("evening", 18, 24))
_PERIOD_LABELS: dict[str, tuple] = {
    "morning":   ("Morning",   "Утро"),
    "afternoon": ("Afternoon", "День"),
    "evening":   ("Evening",   "Вечер"),
}


def _period_for_hour(hour: int) -> Optional[str]:
    for name, start, end in _PERIOD_RANGES:
        if start <= hour < end:
            return name
    return None


def _period_label(period: str, lang: str) -> str:
    en, ru = _PERIOD_LABELS[period]
    return ru if lang == "ru" else en


def fetch_hourly(lat: float, lon: float, days: int = 3) -> Optional[dict]:
    """Return {"utc_offset_seconds", "points": [{"time" (naive local
    datetime), "temp_c", "code", "wind_kmh"}, ...]} for the next `days` days
    at the given coordinates, or None on failure. `timezone=auto` makes
    Open-Meteo return hour timestamps already in the location's own local
    time, and utc_offset_seconds alongside them -- that's what lets a caller
    work out "the next N hours from right now" in the FORECAST location's
    time, not the server's.
    """
    days = max(1, min(int(days), 16))
    try:
        r = requests.get(_FORECAST_URL,
                         params={"latitude": lat, "longitude": lon,
                                "hourly": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                                "forecast_days": days, "timezone": "auto"},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        codes = hourly.get("weather_code") or []
        winds = hourly.get("wind_speed_10m") or []
        # Humidity is treated as OPTIONAL rather than required: an upstream
        # response that omits it must degrade to a forecast without the
        # humidity column, not to no forecast at all.
        hums = hourly.get("relative_humidity_2m") or []
        if not times or not (len(times) == len(temps) == len(codes) == len(winds)):
            return None
        points = []
        for i, t in enumerate(times):
            try:
                points.append({"time": _dt.datetime.fromisoformat(t),
                               "temp_c": temps[i], "code": codes[i],
                               "wind_kmh": winds[i],
                               "humidity": (hums[i] if i < len(hums) else None)})
            except (ValueError, TypeError):
                continue
        return {"utc_offset_seconds": data.get("utc_offset_seconds", 0),
                "points": points}
    except Exception as exc:
        logger.warning("fetch_hourly(%s,%s) failed: %s", lat, lon, exc)
        return None


def bucket_periods(points: list, start_dt, end_dt) -> list:
    """Group hourly points within [start_dt, end_dt) into one entry per
    (date, morning/afternoon/evening) that has data, in chronological order.
    Each entry averages temperature, takes the worst-case wind, and picks
    the MOST FREQUENT weather code in that window (ties broken toward the
    numerically higher/more severe code) rather than e.g. the first or last
    hour's code, so a bucket isn't misdescribed by one atypical hour."""
    buckets: dict = {}
    order: list = []
    for pt in points:
        t = pt["time"]
        if not (start_dt <= t < end_dt):
            continue
        period = _period_for_hour(t.hour)
        if period is None:
            continue
        key = (t.date(), period)
        if key not in buckets:
            buckets[key] = {"temps": [], "codes": [], "winds": [], "hums": []}
            order.append(key)
        buckets[key]["temps"].append(pt["temp_c"])
        buckets[key]["codes"].append(pt["code"])
        buckets[key]["winds"].append(pt["wind_kmh"])
        if pt.get("humidity") is not None:
            buckets[key]["hums"].append(pt["humidity"])
    result = []
    for key in order:
        b = buckets[key]
        avg_temp = sum(b["temps"]) / len(b["temps"])
        counts: dict = {}
        for c in b["codes"]:
            counts[c] = counts.get(c, 0) + 1
        top = max(counts.values())
        code = max(c for c, n in counts.items() if n == top)
        hums = b.get("hums") or []
        result.append({"date": key[0], "period": key[1], "temp_c": avg_temp,
                       "code": code, "wind_kmh": max(b["winds"]),
                       # hours with rain/drizzle: the dominant code above hides
                       # a one-hour shower, and "брать зонт?" is about exactly that
                       "wet_hours": sum(c in _WET_CODES for c in b["codes"]),
                       # None (not 0) when upstream gave us nothing: 0% is a
                       # real reading and must not be faked from missing data.
                       "humidity": (sum(hums) / len(hums)) if hums else None})
    return result


def parse_date_input(text: str, today: Optional[_dt.date] = None) -> Optional[_dt.date]:
    """Parse a user-typed date for the forecast: 'today'/'сегодня',
    'tomorrow'/'завтра', DD.MM[.YYYY] (also accepting '-'/'/' separators),
    or ISO YYYY-MM-DD. A bare DD.MM that has already passed this year is
    assumed to mean next year -- a forecast for a date in the past makes no
    sense, so "20.01" asked for in December almost certainly means next
    January, not eleven months ago."""
    text = (text or "").strip().lower()
    if not text:
        return None
    today = today or _dt.date.today()
    if text in ("today", "сегодня"):
        return today
    if text in ("tomorrow", "завтра"):
        return today + _dt.timedelta(days=1)
    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?$", text)
    if not m:
        try:
            return _dt.date.fromisoformat(text)
        except ValueError:
            return None
    day_s, month_s, year_s = m.groups()
    try:
        day, month = int(day_s), int(month_s)
        if year_s:
            year = int(year_s)
            if year < 100:
                year += 2000
            return _dt.date(year, month, day)
        d = _dt.date(today.year, month, day)
    except ValueError:
        return None
    if d < today:
        try:
            d = d.replace(year=d.year + 1)
        except ValueError:
            return None
    return d


def no_forecast_message(lang: str) -> str:
    return ("I don't have forecast data that far out yet — try a date within "
            "the next couple of weeks."
            if lang != "ru" else
            "У меня пока нет прогноза на такую дату — попробуй дату в "
            "пределах ближайших двух недель.")


def _forecast_header(place: str, lang: str, hours: Optional[int], date) -> str:
    # Escaped: `place` comes from the geocoder, seeded by user-typed input, and
    # the reply is sent with parse_mode=HTML.
    place = _esc(place)
    if date is not None:
        d = date.strftime("%d.%m.%Y")
        if hours and hours > 24:
            d += "–" + (date + _dt.timedelta(hours=hours - 1)).strftime("%d.%m.%Y")
        return f"📍 {place} — {d}"
    h = hours or 24
    if lang == "ru":
        return f"📍 {place} — ближайшие {h} ч"
    return f"📍 {place} — next {h}h"


def forecast_data(loc: dict, lang: str = "en", hours: Optional[int] = None,
                  date=None, advise_fn=None) -> tuple:
    """(buckets, advice_map, error) — the DATA behind a periods forecast.

    Split out of what_to_wear_periods_for_loc so a surface that draws its own
    table can have the buckets instead of a pre-rendered string: the desktop
    Weather tab uses a real QTableWidget, and re-parsing the monospace block
    meant for Telegram would be absurd. `error` is a translated, user-facing
    string when there is nothing to show (service down, or a date beyond
    available data), and empty otherwise; buckets is empty in that case.

    Never raises -- see what_to_wear_periods_for_loc for the window and
    advise_fn contracts, which are unchanged.
    """
    days_needed = 3
    if date is not None:
        days_needed = max(1, (date - _dt.date.today()).days + 1 + max(1, -(-int(hours or 24) // 24)))
        if days_needed > 16:
            # Open-Meteo rejects forecast_days > 16, which read as "service is
            # unavailable" instead of "no forecast that far out".
            return [], {}, no_forecast_message(lang)
    elif hours:
        days_needed = max(2, -(-int(hours) // 24) + 1)
    hourly = fetch_hourly(loc["lat"], loc["lon"], days=days_needed)
    if not hourly:
        return [], {}, ("Weather service is unavailable right now — try again in a bit."
                        if lang != "ru" else
                        "Сервис погоды сейчас недоступен — попробуй чуть позже.")
    local_now = _dt.datetime.utcnow() + _dt.timedelta(seconds=hourly["utc_offset_seconds"])
    if date is not None:
        start_dt = _dt.datetime.combine(date, _dt.time(0, 0))
        # date + hours = a window of that many hours from the date ("на выходных")
        end_dt = start_dt + _dt.timedelta(hours=hours or 24)
        # "Today" asked at 18:00 described the morning that is already over.
        start_dt = max(start_dt, local_now.replace(minute=0, second=0, microsecond=0))
    else:
        start_dt = local_now
        end_dt = start_dt + _dt.timedelta(hours=hours or 24)
    buckets = bucket_periods(hourly["points"], start_dt, end_dt)
    if not buckets and date is not None:
        # Late at night nothing of "today" is left: show the whole day.
        buckets = bucket_periods(hourly["points"],
                                 _dt.datetime.combine(date, _dt.time(0, 0)), end_dt)
    if not buckets:
        return [], {}, no_forecast_message(lang)
    advice_map: dict = {}
    if advise_fn is not None:
        try:
            advice_map = advise_fn(buckets, lang) or {}
        except Exception as exc:
            logger.warning("periods advise_fn failed, falling back to fixed bands: %s", exc)
    return buckets, advice_map, ""


def bucket_advice(b: dict, advice_map: dict, lang: str) -> str:
    """The advice for ONE bucket: the advisor's answer, else the fixed bands.

    Extracted so both renderers resolve advice identically -- the fallback has
    to apply per-bucket, since a live advisor can answer for some periods and
    miss others (a truncated reply, a dropped key), and a bucket with no advice
    must still say something.
    """
    key = f'{b["date"].isoformat()}_{b["period"]}'
    return ((advice_map or {}).get(key) or "").strip() or _clothing_advice(
        b["temp_c"], b["code"], b["wind_kmh"], lang)


# ── the LLM hooks both surfaces plug in ────────────────────────────────────
# These live here, not on the Telegram bot where they started, because the
# desktop Weather tab needs the SAME advisor: two copies of a prompt is two
# places for the two surfaces to start giving different advice for identical
# weather. `llm` is imported lazily inside the closures -- importing it at
# module scope would drag the model stack into weather.py's unit tests, which
# run offline.
def periods_advise_fn(ctx):
    """One LLM call covering EVERY period bucket in a forecast (not one call
    per bucket) -- the house model is a slow reasoner with no fast-path, so
    asking it separately for up to 6 buckets (48h = 2 days x 3 periods) would
    make the button take minutes instead of seconds."""
    def advisor(buckets, lang):
        import llm as _llm
        lines_in = []
        for b in buckets:
            # "_" not ":" -- the key itself must not contain the same
            # character the KEY/advice split below uses, or the split lands
            # inside the key instead of between key and advice (reproduced
            # live: "2026-08-15:afternoon: <advice>" split on the FIRST colon
            # returned "2026-08-15" as the key and "afternoon: <advice>" as
            # the "advice", so nothing ever matched a real bucket and every
            # reply silently fell back to the fixed temperature-band table).
            key = f'{b["date"].isoformat()}_{b["period"]}'
            lines_in.append(f'{key} | {b["temp_c"]:.0f}C | '
                            f'wind {b["wind_kmh"]:.0f}km/h | '
                            f'{_code_desc(b["code"], lang)}')
        sys_p = ("You are a concise clothing advisor. You will be given "
                 "several time periods, one per line, formatted "
                 "'KEY | temperature | wind | conditions'. For EACH line, "
                 "answer with exactly one output line 'KEY: <short noun "
                 "phrase naming what to wear>' -- no leading 'wear'/"
                 "'надень', no trailing period, no extra commentary. Keep "
                 "the same KEYs and the same order as the input, one "
                 "output line per input line, and answer in the "
                 "requested language.")
        user_p = ("Language: " + ("Russian" if lang == "ru" else "English")
                  + "\n\n" + "\n".join(lines_in))
        raw = _llm.call_llm_simple(ctx, sys_p, user_p,
                                   temperature=0.4, max_tokens=500) or ""
        result = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, _, advice = line.partition(":")
            key = key.strip()
            if key:
                result[key] = advice.strip()
        return result
    return advisor


def city_correct_fn(ctx):
    """Builds the LLM hook that fixes a city name resolve_city() couldn't
    geocode -- typos, abbreviations ("Спб"), and informal nicknames ("Питер")
    that Open-Meteo's own geocoder won't match. Letting the LLM do this
    (rather than a hand-built alias table) means it isn't limited to a fixed
    list of cities or languages."""
    def corrector(city):
        import llm as _llm
        sys_p = ("You correct city names for a weather geocoding API. "
                 "Given a possibly misspelled, abbreviated, or informal "
                 "city name in any language, answer with ONLY the "
                 "standard city name in English that a geocoder would "
                 "recognize -- no commentary, no punctuation, no "
                 "explanation. If you cannot confidently identify a real "
                 "city, answer with the input unchanged.")
        # max_tokens has to cover the model's own reasoning, not just the
        # one-line answer -- the house model (Gemma, reasoning-only, no off
        # switch) has been observed spending its whole budget "thinking" and
        # getting cut off before writing the answer.
        return _llm.call_llm_simple(ctx, sys_p, f"City: {city}",
                                    temperature=0.2, max_tokens=200)
    return corrector


def group_advice(buckets: list, advice_map: dict, lang: str = "en") -> list:
    """[(advice, [when, ...]), ...] with consecutive identical advice merged.

    For a 48h forecast the same phrase otherwise appears up to six times, which
    is most of what made the old reply a wall of text. Shared by both surfaces
    so the desktop table and the chat message group identically -- `when` is a
    period label, carrying its date only for days after the first (the first
    day is the one the reader is already looking at).
    """
    first_date = buckets[0]["date"] if buckets else None
    groups: list = []
    for b in buckets:
        advice = bucket_advice(b, advice_map, lang)
        label = _period_label(b["period"], lang)
        when = label if b["date"] == first_date else f'{label} {b["date"].strftime("%d.%m")}'
        if groups and groups[-1][0] == advice:
            groups[-1][1].append(when)
        else:
            groups.append([advice, [when]])
    return groups


def what_to_wear_periods_for_loc(loc: dict, lang: str = "en",
                                 hours: Optional[int] = None, date=None,
                                 advise_fn=None) -> str:
    """Forecast already resolved to a location, broken into morning/
    afternoon/evening buckets, rendered for a chat message.

    Exactly one of `hours`/`date` drives the window:
    - `date` (a datetime.date): the full morning/afternoon/evening of that
      calendar day, in the FORECAST location's own local time.
    - `hours` (default 24 if both are omitted): a ROLLING window starting
      right now (in the forecast location's local time) and running for
      that many hours -- so a lookup made in the evening naturally covers
      less of "today" and more of "tomorrow", rather than always describing
      a fixed calendar day.

    `advise_fn(buckets, lang) -> {"<date-iso>_<period>": advice, ...}`, if
    given, is one LLM call covering every bucket at once (not one call per
    bucket) — the house model is a slow reasoner with no fast-path, so
    asking it up to 6 separate times for a 48h forecast would make the
    button take minutes instead of seconds. Falls back to the fixed
    temperature-band table per-bucket if omitted, raising, or missing a key.
    """
    buckets, advice_map, error = forecast_data(loc, lang, hours=hours,
                                               date=date, advise_fn=advise_fn)
    if error:
        return error
    header = _forecast_header(loc["name"], lang, hours, date)
    return header + "\n\n" + format_forecast(buckets, lang, advice_map)


def format_forecast(buckets: list, lang: str = "en", advice_map: dict = None) -> str:
    """Render buckets as a monospace table, a conditions legend, and advice.

    Was one prose line per bucket, each ending in its own clothing sentence.
    For a 48h forecast that repeated the SAME advice up to six times ("надень
    лёгкую куртку или худи" five times over), which is what made the reply
    look like a wall of text — the numbers a reader actually wants to compare
    were buried inside repeated sentences.

    A <pre> block is used because Telegram renders it in a monospace font,
    which is the only way columns line up in a chat message. NOTE that this
    makes parse_mode="HTML" mandatory at the send site: without it Telegram
    shows the tags literally AND draws the table in a proportional font, so
    every column is ragged (which is exactly what shipped first).

    Three things are de-duplicated rather than repeated per row, because for a
    48h forecast each of them was otherwise printed up to six times:
      * the date — a group line, not a suffix on every row;
      * the conditions wording — an icon in the table, the words once in a
        legend (also what keeps rows narrow enough not to wrap on a phone);
      * the advice — identical advice for consecutive periods is stated ONCE
        with the periods it covers.

    Column widths are computed from the labels actually present, since the
    English period names ("Afternoon") are nearly twice the Russian ones.
    """
    advice_map = advice_map or {}
    ru = (lang or "").startswith("ru")

    when_hdr = "Когда" if ru else "When"
    lw = max([len(when_hdr)]
             + [len(_period_label(b["period"], lang)) for b in buckets])
    unit = "км/ч" if ru else "km/h"
    rows = [f"{when_hdr:<{lw}} {'°C':>4} {'%':>4} {unit:>5}"]

    legend: list = []                # (icon, description), first-seen order
    shown_date = None                # the date the rows below currently belong to
    for b in buckets:
        label = _period_label(b["period"], lang)
        # Keyed on the last date actually PRINTED, not on the previous row:
        # checking only the previous line re-emitted the separator before
        # every row of the same day.
        if b["date"] != shown_date:
            wd = (("пн", "вт", "ср", "чт", "пт", "сб", "вс") if ru else
                  ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))[b["date"].weekday()]
            rows.append(f'{b["date"].strftime("%d.%m")} {wd}')
            shown_date = b["date"]
        hum = b.get("humidity")
        hum_s = f"{hum:.0f}%" if hum is not None else "—"
        icon, desc = _code_icon(b["code"]), _code_desc(b["code"], lang)
        rows.append(f'{label:<{lw}} {b["temp_c"]:>+4.0f} {hum_s:>4} '
                    f'{b["wind_kmh"]:>5.0f}  {icon}')
        if (icon, desc) not in legend:
            legend.append((icon, desc))

    advice_groups = group_advice(buckets, advice_map, lang)
    out = ["<pre>" + _esc("\n".join(rows)) + "</pre>"]
    out.append("")
    out += [_esc(f"{icon} {desc}") for icon, desc in legend]
    out.append(_esc(umbrella_line(buckets, lang)))
    out.append("")
    out.append("👕 " + ("Что надеть" if ru else "What to wear"))
    for advice, whens in advice_groups:
        # One label needs no list; several are joined so the advice is stated
        # once rather than repeated per period. Only the first keeps its
        # capital, so a span reads as one phrase ("День, вечер — ...") rather
        # than as a list of headings.
        span = ", ".join([whens[0]] + [w[:1].lower() + w[1:] for w in whens[1:]])
        out.append(_esc(f"{span} — {advice}"))
    return "\n".join(out)


def what_to_wear_periods(city: str, lang: str = "en", hours: Optional[int] = None,
                         date=None, advise_fn=None, correct_fn=None) -> str:
    """Same as what_to_wear_periods_for_loc, resolving `city` first."""
    loc = resolve_city(city, correct_fn=correct_fn)
    if not loc:
        return not_found_message(city, lang)
    return what_to_wear_periods_for_loc(loc, lang, hours=hours, date=date,
                                        advise_fn=advise_fn)
