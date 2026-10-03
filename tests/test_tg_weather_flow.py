"""The "What to wear" button end to end through the real bot: button press ->
prompt -> default-city button OR free-text city -> reply, plus /cancel, the
per-user remembered default, LLM-driven city-name correction on a failed
geocode, and graceful failure when weather.py itself fails. The network/LLM
call runs on a background thread in production (see
tg_accounts._start_weather_lookup's docstring for why); here
`threading.Thread` is patched to run synchronously so assertions don't need
to poll.

Run: venv/Scripts/python.exe tests/test_tg_weather_flow.py
"""
import sys, os, types, tempfile, threading
import datetime as _dt_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T
import weather as W
import config as CFG

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_weather_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


class _SyncThread:
    """Drop-in for threading.Thread that runs target() immediately, inline,
    instead of on a real thread -- so a background weather lookup finishes
    before the assertion right after it runs."""
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
    def start(self):
        self._target(*self._args, **self._kwargs)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot.sent_kb = []
    bot._send_text = lambda cid, text, **kw: (
        bot.sent.append(text), bot.sent_kb.append(kw.get("keyboard")), 1)[-1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    return bot


def cb_update(chat_id, data, msg_id=555):
    return {"callback_query": {
        "id": "cbq1", "data": data, "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}


def msg(cid, text, mid=1):
    return {"chat": {"id": cid}, "from": {"id": cid, "username": "u"},
            "message_id": mid, "text": text}


bot = make_bot()
threading.Thread = _SyncThread   # module-level patch: affects tg_accounts too,
                                 # since it does `import threading` and calls
                                 # threading.Thread(...) through that binding.

CID = 999901
bot._user_store.put(T._User(chat_id=CID, name="U", status="approved", is_admin=False))

print("=" * 66)
print("MAIN MENU -> WEATHER SUBMENU -> BACK")
print("=" * 66)

# Weather lives in its own submenu (not directly on the main keyboard) so the
# main menu doesn't get cluttered as more weather-adjacent buttons get added.
bot._dispatch(cb_update(CID, "n/a"))  # warm the session
weather_label = T._b("weather", "en")
check("the 'weather' key has a button label", bool(weather_label))

bot._resolve_and_push(CID, [{"type": "text", "text": weather_label}])
check("pressing 🌤 Weather opens the weather submenu",
      bot._get_session(CID).menu == "weather")
check("...and shows a submenu title", any("🌤" in s for s in bot.sent[-1:]), bot.sent[-1:])

back_label = T._b("back", "en")
bot._resolve_and_push(CID, [{"type": "text", "text": back_label}])
check("⬅ Back from the weather submenu returns to the main menu",
      bot._get_session(CID).menu == "")

print()
print("=" * 66)
print("BUTTON PRESS -> PROMPT")
print("=" * 66)

label = T._b("wear", "en")
check("the 'wear' key has a button label", bool(label))

bot._user_gate(CID, msg(CID, label))  # not used directly; go through resolve
bot._resolve_and_push(CID, [{"type": "text", "text": label}])
check("pressing What to Wear arms wtw_city",
      bot._get_session(CID).reg_state == "wtw_city")
check("...and puts the session in the weather submenu",
      bot._get_session(CID).menu == "weather")
check("...and sends a prompt naming the configured default city",
      any(CFG.WEATHER_DEFAULT_CITY in s or "🌤" in s for s in bot.sent[-1:]),
      extra=f"default={CFG.WEATHER_DEFAULT_CITY!r} sent={bot.sent[-1:]!r}")

print()
print("=" * 66)
print("DEFAULT-CITY BUTTON")
print("=" * 66)

# _send_weather resolves via weather.resolve_city() (which wraps geocode_city
# with an optional LLM-correction retry) and formats the reply via
# weather.what_to_wear_periods_for_loc() -- both stubbed here so no real
# network/LLM call happens.
import weather as _w
_orig_wtwfl = _w.what_to_wear_periods_for_loc
_orig_geocode = _w.geocode_city

def _stub_ok(loc, lang="en", hours=None, date=None, advise_fn=None):
    # Mirrors what_to_wear_periods_for_loc's own contract: hours=None (and no
    # date) means a 24h rolling window by default.
    tag = f"{hours or 24}h" if date is None else f"date={date}"
    return f"STUB WEATHER FOR {loc['name']} [{tag}]"

def _resolves(city, lang="en"):
    return {"lat": 1, "lon": 2, "name": city, "country": ""}

bot.sent.clear()
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves
try:
    bot._dispatch(cb_update(CID, "wtw_default"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode

check("default-city button clears reg_state once the city resolves",
      bot._get_session(CID).reg_state == "")
check("it fetches weather for the configured default city",
      any(f"STUB WEATHER FOR {CFG.WEATHER_DEFAULT_CITY}" in s for s in bot.sent),
      bot.sent)

print()
print("=" * 66)
print("FREE-TEXT CITY, AND IT BECOMES THE NEW DEFAULT")
print("=" * 66)

bot.sent.clear()
sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves
try:
    bot._user_gate(CID, msg(CID, "Paris"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
check("typing a resolvable city clears reg_state", bot._get_session(CID).reg_state == "")
check("it fetches weather for the typed city",
      any("STUB WEATHER FOR Paris" in s for s in bot.sent))
check("the typed city is remembered as the user's new default",
      bot._user_store.get(CID).prefs.get("city") == "Paris")

bot.sent.clear()
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves
try:
    bot._dispatch(cb_update(CID, "wtw_default"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
check("the default button now uses the remembered city, not the global fallback",
      any("STUB WEATHER FOR Paris" in s for s in bot.sent))

print()
print("=" * 66)
print("A CITY THE LLM CAN CORRECT IS RETRIED AND RESOLVED (live bug: Спб)")
print("=" * 66)

# The live report: "Спб" (an informal abbreviation) doesn't geocode as-is.
# weather.resolve_city() now gets one shot at asking the LLM to normalize it
# before giving up -- stub geocode_city to fail on the raw input but succeed
# on the "corrected" one, and stub the LLM call tg_accounts wires in as
# correct_fn so no real LM Studio call happens.
import llm as _llm
_orig_call_llm_simple = _llm.call_llm_simple

def _geocode_only_corrected(city, lang="en"):
    return {"lat": 1, "lon": 2, "name": "Saint Petersburg", "country": ""} \
        if city == "Saint Petersburg" else None

bot.sent.clear()
sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _geocode_only_corrected
_llm.call_llm_simple = lambda ctx, sys_p, user_p, **kw: "Saint Petersburg"
_orig_get_ctx = bot._get_ctx
bot._get_ctx = lambda: object()   # non-None so tg_accounts builds correct_fn
try:
    bot._user_gate(CID, msg(CID, "Спб"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
    _llm.call_llm_simple = _orig_call_llm_simple
    bot._get_ctx = _orig_get_ctx
check("an LLM-corrected city resolves and clears reg_state",
      bot._get_session(CID).reg_state == "",
      extra=bot._get_session(CID).reg_state)
check("the reply uses the corrected city's weather",
      any("STUB WEATHER FOR Saint Petersburg" in s for s in bot.sent), bot.sent)
check("the CORRECTED name is what's remembered as the new default, not the typo",
      bot._user_store.get(CID).prefs.get("city") == "Saint Petersburg",
      extra=bot._user_store.get(CID).prefs.get("city"))

print()
print("=" * 66)
print("A FAILED GEOCODE STAYS ARMED FOR A RETRY (live bug: Спб -> Питнр)")
print("=" * 66)

# Live bug report: typing an unresolvable city ("Спб") used to unconditionally
# clear reg_state, so the user's VERY NEXT message (another retry, "Питнр")
# fell through to the full agent pipeline as an ordinary question instead of
# being tried as a second city -- a real "Starting..." task got queued for
# what was obviously meant as another weather attempt. This section covers
# the case where even the LLM correction can't find a real city.

# Prime a real, working default first so we can tell if a failure overwrites it.
sess = bot._get_session(CID); sess.reg_state = ""; bot._store.put(sess)
u = bot._user_store.get(CID)
u.prefs = dict(u.prefs or {}); u.prefs["city"] = "Paris"
bot._user_store.put(u)

bot.sent.clear()
bot._backend = types.SimpleNamespace(
    push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
    drop_chat=lambda cid: 0, close=lambda: None)
sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
_w.geocode_city = lambda city, lang="en": None   # unresolvable, even after correction
_llm.call_llm_simple = lambda ctx, sys_p, user_p, **kw: "Nonexistentville"
try:
    allowed = bot._user_gate(CID, msg(CID, "Спб"))
finally:
    _w.geocode_city = _orig_geocode
    _llm.call_llm_simple = _orig_call_llm_simple
check("a city that fails to geocode does NOT overwrite the remembered default",
      bot._user_store.get(CID).prefs.get("city") == "Paris",
      extra=f"got {bot._user_store.get(CID).prefs.get('city')!r}")
check("...and reg_state STAYS armed for a retry, not cleared",
      bot._get_session(CID).reg_state == "wtw_city",
      extra=bot._get_session(CID).reg_state)
check("...and the failed attempt was NOT queued as an agent task either",
      bot._backend.depth() == 0)

# The retry itself must also be tried as a city, not routed to the agent --
# this is the exact step that broke live: "Питнр" spawned a real task.
bot.sent.clear()
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves
try:
    bot._user_gate(CID, msg(CID, "Питер"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
check("the retry after a failure is tried as a second city, not routed to the agent",
      any("STUB WEATHER FOR" in s for s in bot.sent), bot.sent)
check("a successful retry finally clears reg_state",
      bot._get_session(CID).reg_state == "")
check("...and nothing was ever queued as an ordinary agent task",
      bot._backend.depth() == 0)

print()
print("=" * 66)
print("wtw_default RE-ARMS TOO: A FAILING SAVED DEFAULT ALSO STAYS RETRYABLE")
print("=" * 66)

bot.sent.clear()
sess = bot._get_session(CID); sess.reg_state = ""; bot._store.put(sess)
_w.geocode_city = lambda city, lang="en": None
_llm.call_llm_simple = lambda ctx, sys_p, user_p, **kw: ""
try:
    bot._dispatch(cb_update(CID, "wtw_default"))
finally:
    _w.geocode_city = _orig_geocode
    _llm.call_llm_simple = _orig_call_llm_simple
check("a failing default-city lookup leaves wtw_city armed for a retry",
      bot._get_session(CID).reg_state == "wtw_city",
      extra=bot._get_session(CID).reg_state)

print()
print("=" * 66)
print("A RESOLVED FORECAST CARRIES 48H/PICK-A-DATE BUTTONS AND REMEMBERS THE LOC")
print("=" * 66)

bot.sent.clear(); bot.sent_kb.clear()
sess = bot._get_session(CID); sess.reg_state = "wtw_city"; sess.wtw_loc = {}
bot._store.put(sess)
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves
try:
    bot._user_gate(CID, msg(CID, "Berlin"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
check("the default 24h forecast is what gets sent",
      any("[24h]" in s for s in bot.sent), bot.sent)
_kb = bot.sent_kb[-1] or {}
_cb_data = [b["callback_data"] for row in _kb.get("inline_keyboard", []) for b in row]
check("the reply carries a 48h button", "wtw_range:48" in _cb_data, _cb_data)
check("...and a pick-a-date button", "wtw_pickdate" in _cb_data, _cb_data)
check("the resolved location is remembered on the session for the follow-ups",
      bot._get_session(CID).wtw_loc.get("name") == "Berlin",
      bot._get_session(CID).wtw_loc)

print()
print("=" * 66)
print("48H BUTTON REUSES THE REMEMBERED LOCATION -- NO RE-GEOCODE")
print("=" * 66)

bot.sent.clear()
_geocode_calls = []
def _resolves_tracked(city):
    _geocode_calls.append(city)
    return {"lat": 1, "lon": 2, "name": city, "country": ""}
_w.what_to_wear_periods_for_loc = _stub_ok
_w.geocode_city = _resolves_tracked
try:
    bot._dispatch(cb_update(CID, "wtw_range:48"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
    _w.geocode_city = _orig_geocode
check("the 48h button asks for 48 hours of Berlin (the remembered location)",
      any("STUB WEATHER FOR Berlin [48h]" in s for s in bot.sent), bot.sent)
check("...without re-geocoding the city at all",
      _geocode_calls == [], _geocode_calls)
check("reg_state clears once the 48h forecast resolves",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("PICK-A-DATE: PROMPT, PARSE, FORECAST FOR THAT DATE (NO FOLLOW-UP BUTTONS)")
print("=" * 66)

bot.sent.clear()
bot._dispatch(cb_update(CID, "wtw_pickdate"))
check("pick-a-date arms wtw_date", bot._get_session(CID).reg_state == "wtw_date")
check("...and sends a date prompt", any("🗓" in s for s in bot.sent), bot.sent)

bot.sent.clear(); bot.sent_kb.clear()
import datetime as _dtm
_seen_dates = []
def _stub_dated(loc, lang="en", hours=None, date=None, advise_fn=None):
    _seen_dates.append(date)
    return f"STUB WEATHER FOR {loc['name']} [date={date}]"
_w.what_to_wear_periods_for_loc = _stub_dated
try:
    bot._user_gate(CID, msg(CID, "tomorrow"))
finally:
    _w.what_to_wear_periods_for_loc = _orig_wtwfl
check("'tomorrow' parses to today+1 and drives the forecast",
      _seen_dates == [_dtm.date.today() + _dtm.timedelta(days=1)], _seen_dates)
check("a date-based forecast does NOT carry the 48h/pick-a-date buttons",
      not (bot.sent_kb[-1] or {}).get("inline_keyboard") if bot.sent_kb else True,
      bot.sent_kb)
check("reg_state clears after a successful date lookup",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("AN UNPARSEABLE DATE STAYS ARMED FOR A RETRY (same contract as wtw_city)")
print("=" * 66)

bot.sent.clear()
sess = bot._get_session(CID); sess.reg_state = "wtw_date"; bot._store.put(sess)
allowed = bot._user_gate(CID, msg(CID, "not a date at all"))
check("garbage date text is rejected, not silently accepted", allowed is False)
check("reg_state stays wtw_date for a retry",
      bot._get_session(CID).reg_state == "wtw_date")
check("an invalid-date notice was sent", any("Не разобрал" in s or "Couldn't parse" in s
                                             for s in bot.sent), bot.sent)

# Same abandonment contract as wtw_city/wtw_date's siblings.
sess = bot._get_session(CID); sess.reg_state = "wtw_date"; bot._store.put(sess)
bot._user_gate(CID, msg(CID, "/cancel"))
check("/cancel abandons the date prompt too", bot._get_session(CID).reg_state == "")

sess = bot._get_session(CID); sess.reg_state = "wtw_date"; bot._store.put(sess)
allowed = bot._user_gate(CID, msg(CID, "/broadcast hello"))
check("a non-/cancel command abandons wtw_date instead of becoming the date",
      bot._get_session(CID).reg_state == "" and allowed is True)

sess = bot._get_session(CID); sess.reg_state = "wtw_date"; bot._store.put(sess)
bot._dispatch(cb_update(CID, "acct_lang"))
check("an unrelated inline button abandons wtw_date too",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("/cancel AND UNRELATED-BUTTON ABANDONMENT")
print("=" * 66)

sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
bot._user_gate(CID, msg(CID, "/cancel"))
check("/cancel abandons the city prompt", bot._get_session(CID).reg_state == "")

sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
bot._dispatch(cb_update(CID, "acct_lang"))
check("an unrelated inline button also abandons the city prompt",
      bot._get_session(CID).reg_state == "")

print()
print("=" * 66)
print("A NON-/cancel COMMAND ABANDONS THE PROMPT, NOT BECOMES THE CITY")
print("=" * 66)

# The "/start became the password" bug shape: only "/cancel" was special-
# cased, so e.g. "/broadcast hello" while armed for wtw_city was silently
# treated as a literal (garbage) city name -- eating the command entirely.
sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
allowed = bot._user_gate(CID, msg(CID, "/broadcast hello everyone"))
check("a non-/cancel command abandons wtw_city instead of becoming the city",
      bot._get_session(CID).reg_state == "", bot._get_session(CID).reg_state)
check("...and is let through to run as a real command", allowed is True, repr(allowed))

print()
print("=" * 66)
print("EMPTY INPUT WHILE ARMED DOES NOT CRASH OR CLEAR SILENTLY")
print("=" * 66)

sess = bot._get_session(CID); sess.reg_state = "wtw_city"; bot._store.put(sess)
bot._user_gate(CID, msg(CID, ""))
check("empty text while armed stays armed (treated as noise, not a city)",
      bot._get_session(CID).reg_state == "wtw_city")

print()
print("=" * 66)
print("PERIODS-ADVISE LINE PARSING (live bug: the bucket key itself had a ':')")
print("=" * 66)

# Live bug: the bucket key format was "<date>:<period>" (e.g.
# "2026-08-15:afternoon"), and the LLM's reply lines were parsed with
# `line.partition(":")` -- which split on the colon INSIDE the key, not the
# one separating the key from the advice. Every reply silently fell back to
# the fixed temperature-band table because no key ever matched. Reproduced
# live against the real model, fixed by switching the key separator to "_".
#
# This has to round-trip through the REAL key-generation code, not just
# assert the parser handles underscore-separated input (which it always
# could) -- a stub that hardcodes the "already correct" key format would
# pass even if the generation side regressed back to ":" and never notice.
# So: capture the exact prompt the code sends the LLM, extract the KEYs IT
# put there, and have the fake LLM echo those same keys back (asking a
# model to "keep the same KEYs" and it doing so is the realistic case) --
# then confirm the parsed-out dict is keyed exactly like weather.py's own
# lookup (`f'{date.isoformat()}_{period}'`) expects.
import datetime as _dtm2, re as _re2
_orig_call_llm_simple2 = _llm.call_llm_simple
advisor = bot._weather_periods_advise_fn(object())
_buckets = [
    {"date": _dtm2.date(2026, 8, 15), "period": "afternoon",
     "temp_c": 17.0, "code": 61, "wind_kmh": 10},
    {"date": _dtm2.date(2026, 8, 16), "period": "morning",
     "temp_c": 9.0, "code": 0, "wind_kmh": 5},
]
_captured_prompt = {}
def _echo_llm(ctx, sys_p, user_p, **kw):
    _captured_prompt["text"] = user_p
    keys = _re2.findall(r"^(\S+) \|", user_p, flags=_re2.MULTILINE)
    return "\n".join(f"{k}: some advice for {k}" for k in keys)
_llm.call_llm_simple = _echo_llm
try:
    result = advisor(_buckets, "en")
finally:
    _llm.call_llm_simple = _orig_call_llm_simple2
_expected_keys = {f'{b["date"].isoformat()}_{b["period"]}' for b in _buckets}
check("the prompt's own keys contain no ':' for the parser to trip over",
      all(":" not in k for k in _re2.findall(r"^(\S+) \|", _captured_prompt.get("text", ""),
                                             flags=_re2.MULTILINE)),
      _captured_prompt.get("text"))
check("both bucket keys round-trip through generation AND parsing intact",
      set(result.keys()) == _expected_keys, result)

print()


print()
print("=" * 66)
print("THE REWORKED WEATHER KEYBOARD: every window in ONE tap")
print("=" * 66)
print("""
The submenu used to hold a single "what to wear" button, so reaching 48h meant
sitting through a 24h lookup and then pressing a follow-up under the result.
Each window is its own entry point now, and only the City button asks for text.
""")

_wkb = T._weather_kb("ru")
_wlabels = [b for row in _wkb["keyboard"] for b in row]
_wkeys = [T._LABEL2KEY.get(b, "") for b in _wlabels]
for _k in ("wtw_now", "wtw_24", "wtw_48", "wtw_date_btn", "wtw_city_btn"):
    check(f"the weather keyboard offers {_k}", _k in _wkeys, _wkeys)
check("...and still offers Back", "back" in _wkeys, _wkeys)
check("every weather button is wired to an action",
      all(T._DIRECT_KB.get(k) for k in _wkeys if k.startswith("wtw_")), _wkeys)
check("every weather button label is translated both ways",
      all((T._BTN.get(k, {}).get("ru") and T._BTN.get(k, {}).get("en"))
          for k in _wkeys if k.startswith("wtw_")), _wkeys)

# 48h must NOT require a 24h lookup first, and must not ask for a city when one
# is already remembered -- that was the whole complaint.
bot2 = make_bot()
CID2 = 999755
bot2._user_store.put(T._User(chat_id=CID2, name="U", status="approved", is_admin=False))
_asked = []
bot2._start_weather_lookup = lambda cid, city, lang, remember_for=0, hours=None, date=None, loc=None: (
    _asked.append({"city": city, "hours": hours, "date": date}), None)[-1]
bot2.sent.clear()
bot2._resolve_and_push(CID2, [{"type": "text", "text": T._b("wtw_48", "ru")}])
check("pressing 48 h goes straight to a 48h lookup",
      len(_asked) == 1 and _asked[0]["hours"] == 48, _asked)
check("...without asking which city (the remembered one is used)",
      bool(_asked and _asked[0]["city"]), _asked)
check("...and no text capture is left armed",
      bot2._get_session(CID2).reg_state == "", bot2._get_session(CID2).reg_state)

_asked.clear()
bot2._resolve_and_push(CID2, [{"type": "text", "text": T._b("wtw_now", "ru")}])
check("Now asks for current conditions (no hour window)",
      len(_asked) == 1 and not _asked[0]["hours"], _asked)

_asked.clear()
bot2._resolve_and_push(CID2, [{"type": "text", "text": T._b("wtw_24", "ru")}])
check("24 h asks for a 24h window", _asked and _asked[0]["hours"] == 24, _asked)

# The date button offers today/tomorrow as taps rather than demanding typing.
bot2.sent.clear(); bot2.sent_kb.clear()
bot2._resolve_and_push(CID2, [{"type": "text", "text": T._b("wtw_date_btn", "ru")}])
check("Date arms the date capture", bot2._get_session(CID2).reg_state == "wtw_date")
_dkb = (bot2.sent_kb[-1] or {}).get("inline_keyboard") or []
_ddata = [b["callback_data"] for row in _dkb for b in row]
check("...and offers today/tomorrow as one-tap buttons",
      "wtwd:0" in _ddata and "wtwd:1" in _ddata, _ddata)

_asked.clear()
bot2._dispatch(cb_update(CID2, "wtwd:1"))
check("tapping Tomorrow runs a date-based lookup",
      len(_asked) == 1 and _asked[0]["date"] is not None, _asked)
check("...for tomorrow specifically",
      _asked and _asked[0]["date"] == _dt_mod.date.today() + _dt_mod.timedelta(days=1),
      _asked)
check("...and clears the armed capture",
      bot2._get_session(CID2).reg_state == "", bot2._get_session(CID2).reg_state)

# A junk payload must not act on a guessed date.
_asked.clear()
bot2._dispatch(cb_update(CID2, "wtwd:99"))
check("an out-of-range date offset is refused", not _asked, _asked)
bot2._dispatch(cb_update(CID2, "wtwd:xyz"))
check("a non-numeric date offset does not crash", True)

# Pressing a forecast button while a city/date capture is armed must abandon it,
# not leave it to swallow the user's next ordinary message.
sess2 = bot2._get_session(CID2); sess2.reg_state = "wtw_city"; bot2._store.put(sess2)
_asked.clear()
bot2._resolve_and_push(CID2, [{"type": "text", "text": T._b("wtw_48", "ru")}])
check("a forecast button abandons an armed city capture",
      bot2._get_session(CID2).reg_state == "", bot2._get_session(CID2).reg_state)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
