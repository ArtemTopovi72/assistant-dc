"""The "what to wear" flow: ask for a city, look it up, describe the day.

Lifted out of tg_accounts.py, which had accumulated three unrelated user
flows (weather, songs, mashups) around the account code they had nothing to
do with.

The flow is deliberately spread over several turns and two threads, and the
reg_state contract between them is the subtle part: `wtw_city` and `wtw_date`
stay ARMED while the async lookup runs, because whether a city resolves is not
known when the user's message is handled. _send_weather is the only place that
clears them, and only once geocode_city() has actually succeeded -- so a failed
lookup leaves the user still in weather mode, and their next message is tried
as another city instead of falling through to the agent as an ordinary
question. The button handlers at the bottom keep the same contract.

Both LLM hooks live in weather.py so the desktop Weather tab uses the same
advisor and the same city corrector; a second copy of either prompt would be a
second place for the two surfaces to drift apart.

tg_bot is imported at the BOTTOM and read as ``tg_bot.<name>`` at call time,
the same convention the rest of the tg_* family uses.
"""
from __future__ import annotations

import _thread
import os as _os
import datetime as _dt
import html as _html_mod
import re as _re
import threading

import config as _config
import weather as _weather_mod


_DAYS = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}
_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _when(read: dict, today) -> tuple:
    """(date, hours) for the model's "when"."""
    when = read.get("when") or ""
    if when in _DAYS:
        return today + _dt.timedelta(days=_DAYS[when]), None
    if when == "now":
        return None, 0
    if when == "week":
        return None, 48
    if when in _WEEKDAY_NAMES:
        return today + _dt.timedelta(days=(_WEEKDAY_NAMES.index(when) - today.weekday()) % 7), None
    if when == "weekend":
        # Live 2026-09-28: "погода в Питере на выходных" got the next 24 hours.
        wd = today.weekday()
        return (today if wd >= 5 else today + _dt.timedelta(days=5 - wd)), (24 if wd == 6 else 48)
    return None, None


def _forecast(text: str, previous: str, today, followup: bool = False) -> dict:
    """{} or {"city", "date", "hours"} from the model's read (agent/intent.py).
    It replaced a weather-word list with its exclusions («погода в игре»,
    «вообще бывает осенью», several questions) and a city regex that named
    «на» and «сейчас» cities under IGNORECASE."""
    t = (text or "").strip()
    if not t or len(t) > 200:
        return {}
    import intent
    w = intent.read(None, t, previous)["weather"]
    # «а какая там погода» names no place: the chat knows it, the shortcut
    # does not -- unless a forecast was just sent, which is the place meant
    if not w or (w.get("by_reference") and not followup):
        return {}
    # a follow-up that names neither a place nor a day asks nothing new
    # («а в игре?» after a forecast)
    if followup and not w["city"] and not w["when"]:
        return {}
    date, hours = _when(w, today or _dt.date.today())
    return {"city": w["city"], "date": date, "hours": hours}


def weather_followup(text: str, today=None) -> dict:
    """A short "а в Сочи?" / "а завтра?" / «Питер» right after a forecast. {} otherwise."""
    return _forecast(text, "The assistant just sent a weather forecast.", today, followup=True)


def weather_request(text: str, today=None) -> dict:
    """{} when this is not a weather question; otherwise
    {"city": str|"", "date": date|None, "hours": int|None}."""
    return _forecast(text, "", today)


# The advice call queues for the ONE model slot behind whatever else is running
# (live: 25-80 s behind other chats), and the forecast waited with it. Past the
# deadline the forecast goes out with the fixed temperature-band advice; the
# late LLM answer is dropped.
ADVICE_WAIT_S = 8.0


def _deadline(fn, seconds):
    def wrapped(*a, **kw):
        box, done = {}, threading.Event()

        def run():
            try:
                box["r"] = fn(*a, **kw)
            finally:
                done.set()
        _thread.start_new_thread(run, ())   # not threading.Thread: tests stub that one
        done.wait(seconds)
        return box.get("r")        # None -> caller's fixed-table fallback
    return wrapped


import functools as _functools


@_functools.lru_cache(maxsize=256)
def city_tz(city: str) -> str:
    """IANA zone of a named city, "" when unknown (network miss is not cached
    as a real answer only for the process lifetime -- good enough)."""
    try:
        loc = _weather_mod.geocode_city(city, "ru") or {}
        return loc.get("timezone") or ""
    except Exception:
        return ""


def _city_of_fact(fact: str) -> str:
    """The city a saved fact says the user lives in or moved to, in the nominative
    («переехал в Самару» -> «Самара»: the geocoder reads "Самару" as Самар), or ""."""
    fact = (fact or "").strip()
    if not fact:
        return ""
    import intent
    if not intent.ask_yes("A saved fact about a user: {text}. Does it say which city the "
                          "user lives in or moved to?", fact):
        return ""
    if _os.getenv("F5_TEST_RUN") and not _os.getenv("INTENT_LIVE"):
        return (intent.CITY_STUB or (lambda t: ""))(fact)
    try:
        from llm import call_llm_simple
        c = (call_llm_simple(None, "Answer with the city name only.",
                             "Fact: «%s». Which city does the user live in now? The name in "
                             "the nominative case, in the same language and alphabet as the "
                             "fact (Самару -> Самара)." % fact[:500],
                             temperature=0.0, max_tokens=12) or "").strip(" .«»\"\n")
    except Exception:
        return ""
    return c if 1 < len(c) < 40 else ""


class WeatherMixin:
    def _start_weather_flow(self, chat_id: int, sess, lang: str) -> None:
        user = self._user_store.get(chat_id)
        default_city = ((user.prefs or {}).get("city") if user else "") or _config.WEATHER_DEFAULT_CITY
        sess.reg_state = "wtw_city"
        self._store.put(sess)
        self._send_text(chat_id,
            tg_bot._t("wtw_prompt", lang, city=_html_mod.escape(default_city)),
            parse_mode="HTML",
            keyboard={"inline_keyboard": [[
                {"text": tg_bot._t("wtw_default_btn", lang, city=default_city),
                 "callback_data": "wtw_default"}]]})

    def _remembered_city(self, chat_id: int) -> str:
        user = self._user_store.get(chat_id)
        return (((user.prefs or {}).get("city") if user else "")
                or self._fact_city(chat_id)
                or _config.WEATHER_DEFAULT_CITY)

    def _fact_city(self, chat_id: int) -> str:  # noqa: D401
        """Newest pinned 'lives in / moved to X'. Live: «переехал в Самару»
        was remembered, then «погода завтра» came back for St Petersburg."""
        try:
            facts = self._store.get(chat_id).get_tg_facts()
        except Exception:
            return ""
        for f in sorted(facts, key=lambda f: (f or {}).get("ts", 0) if isinstance(f, dict) else 0,
                        reverse=True):
            c = _city_of_fact(f.get("text", "") if isinstance(f, dict) else str(f))
            if c:
                return c
        return ""

    def _weather_quick(self, chat_id: int, sess, lang: str, hours: int) -> None:
        """A forecast window straight from a button, no questions asked.

        The whole point of the reworked keyboard: 🕗 48 ч used to be reachable
        only as a follow-up under a 24h result, so getting it meant waiting
        through a lookup nobody wanted. `hours=0` means current conditions.

        Any armed text capture is cleared first — pressing a forecast button
        while the bot was waiting for a city (or a date) previously left that
        state armed, so the NEXT ordinary message was silently eaten as the
        answer to a question the user had visibly moved on from.
        """
        if sess.reg_state in ("wtw_city", "wtw_date"):
            sess.reg_state = ""
            self._store.put(sess)
        city = self._remembered_city(chat_id)
        # A location resolved earlier is reused so a button press does not
        # re-geocode (and possibly re-run the LLM city correction) every time.
        loc = getattr(sess, "wtw_loc", None) or None
        if loc and (loc.get("query") or "").strip().lower() != city.strip().lower():
            loc = None
        self._start_weather_lookup(chat_id, city, lang,
                                   hours=(hours or None), loc=loc)

    def _weather_ask_date(self, chat_id: int, sess, lang: str) -> None:
        """Arm the date capture, offering today/tomorrow as one-tap buttons."""
        sess.reg_state = "wtw_date"
        self._store.put(sess)
        kb = {"inline_keyboard": [[
            {"text": tg_bot._t("wtw_today", lang), "callback_data": "wtwd:0"},
            {"text": tg_bot._t("wtw_tomorrow", lang), "callback_data": "wtwd:1"},
        ]]}
        self._send_text(chat_id, tg_bot._t("wtw_date_prompt", lang),
                        parse_mode="HTML", keyboard=kb)

    def _start_weather_lookup(self, chat_id: int, city: str, lang: str,
                              remember_for: int = 0, hours: int = None,
                              date=None, loc: dict = None) -> None:
        """remember_for: chat_id to save `city` as that user's new default for
        next time (0 = don't remember, e.g. when they just used the existing
        default). Runs the actual lookup off-thread -- it makes an HTTP call
        plus (normally) an LLM call, and this fires from the poll loop
        thread, which every other chat's updates are waiting behind.

        The remember-as-default write itself also happens on that background
        thread, and only once the city is confirmed to actually geocode (see
        _send_weather) -- saving it unconditionally here, before the lookup
        even ran, meant a typo'd/garbage city name got permanently baked in
        as the default the moment it was typed, even though the reply back
        said "couldn't find that city". Every future press of the default-
        city button would then retry the same unresolvable name forever,
        with no way back except typing a fresh city (there is no "clear
        city" control, unlike the admin badge's Clear button).

        `loc`: a location ALREADY resolved (see sess.wtw_loc) -- the 48h and
        pick-a-date follow-up buttons under a forecast reuse the city the
        user just typed rather than re-geocoding (and possibly re-running
        the LLM city-correction) from scratch."""
        self._send_text(chat_id, tg_bot._t("wtw_checking", lang))
        self._run_busy(chat_id, self._send_weather,
                       chat_id, city, lang, remember_for, hours, date, loc)

    # Both hooks moved into weather.py so the desktop Weather tab uses the SAME
    # advisor and the same city corrector -- a second copy of either prompt is
    # a second place for the two surfaces to diverge. Kept as methods because
    # this is where the flow reads, and one test drives them from the bot.
    def _weather_periods_advise_fn(self, ctx):
        return _deadline(_weather_mod.periods_advise_fn(ctx), ADVICE_WAIT_S)

    def _weather_correct_fn(self, ctx):
        return _weather_mod.city_correct_fn(ctx)

    def _send_weather(self, chat_id: int, city: str, lang: str,
                      remember_for: int = 0, hours: int = None, date=None,
                      loc: dict = None) -> None:
        try:
            ctx = self._get_ctx()
        except Exception:
            ctx = None
        advise_fn = self._weather_periods_advise_fn(ctx) if ctx is not None else None
        if loc is None:
            correct_fn = self._weather_correct_fn(ctx) if ctx is not None else None
            # Resolve first (with a shot at LLM correction) so a garbage/
            # unmatched city is never remembered as the new default -- see
            # _start_weather_lookup's docstring for why. Remember the
            # corrected, canonical name (loc["name"]) rather than the raw
            # typo, so future taps of the default-city button use the name
            # that actually geocodes.
            loc = _weather_mod.resolve_city(city, correct_fn=correct_fn, lang=lang)
            if remember_for and loc:
                user = self._user_store.get(remember_for)
                if user:
                    prefs = dict(user.prefs or {})
                    prefs["city"] = loc["name"]
                    user.prefs = prefs
                    self._user_store.put(user)
        if loc:
            # Remembered for the 48h/pick-a-date follow-up buttons, so they
            # can re-fetch without re-typing (or re-geocoding) the city.
            try:
                sess = self._get_session(chat_id)
                sess.wtw_loc = loc
                self._store.put(sess)
            except Exception:
                pass
        reply = (_weather_mod.what_to_wear_periods_for_loc(
                    loc, lang, hours=hours, date=date, advise_fn=advise_fn) if loc
                else _weather_mod.not_found_message(city, lang))
        # The follow-up buttons only make sense under an hours-based (rolling
        # window) forecast -- a date-based one is already a specific answer,
        # and "48 hours from THIS forecast" would be ambiguous once the user
        # has already picked a fixed calendar day.
        kb = None
        if loc and date is None:
            kb = {"inline_keyboard": [[
                {"text": tg_bot._t("wtw_48h_btn", lang), "callback_data": "wtw_range:48"},
                {"text": tg_bot._t("wtw_pickdate_btn", lang), "callback_data": "wtw_pickdate"},
            ]]}
        try:
            # parse_mode is NOT optional here: the forecast is a <pre> table
            # (see weather.format_forecast). Sent as plain text -- which is
            # what this call did at first, uniquely among the send sites in
            # this file -- Telegram printed the "<pre>" tags verbatim and drew
            # the table in a proportional font, so no column lined up.
            self._send_text(chat_id, reply, parse_mode="HTML", keyboard=kb)
        except Exception:
            pass
        # Only a resolved city ends the flow. On failure reg_state was left
        # armed by the callers (_user_gate's text path never cleared it;
        # wtw_default re-arms it before calling here) so the user's next
        # message is tried as another city instead of falling through to the
        # agent -- see _start_weather_lookup's docstring.
        if loc:
            try:
                sess = self._get_session(chat_id)
                if sess.reg_state in ("wtw_city", "wtw_date"):
                    sess.reg_state = ""
                # The model never saw this answer: «погода в Казани», then
                # «а там есть метро?» was answered about Moscow (live
                # 2026-09-29). The forecast joins the chat history.
                with self._history_lock(chat_id):
                    hist = sess.get_history()
                    hist += [{"role": "user", "content": f"[weather forecast asked for {loc['name']}]"},
                             {"role": "assistant",
                              "content": _re.sub(r"<[^>]+>", "", reply)[:700]}]
                    sess.set_history(hist)
                self._store.put(sess)
            except Exception:
                tg_bot.logger.debug("weather turn not recorded", exc_info=True)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
