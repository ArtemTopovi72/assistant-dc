"""The weather card names the city in the user's language.

Live 2026-09-12 (journey 7): a Russian user asked about Сочи and got a card
headed "Sochi" — the geocoder was always asked in English. `lang` now
reaches Open-Meteo's geocoder through resolve_city.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import weather as W

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

seen = []
class _R:
    def __init__(self, lang): self.lang = lang
    def raise_for_status(self): pass
    def json(self):
        name = {"ru": "Сочи"}.get(self.lang, "Sochi")
        return {"results": [{"latitude": 43.6, "longitude": 39.7, "name": name, "country": "Russia"}]}
W.requests.get = lambda url, params=None, timeout=None: (seen.append(params), _R(params.get("language")))[1]

loc = W.geocode_city("Сочи", "ru")
check("the geocoder is asked in the user's language", seen[-1]["language"] == "ru", seen[-1])
check("...and the card gets the local name", loc["name"] == "Сочи", loc)
check("English stays the default", W.geocode_city("Sochi")["name"] == "Sochi" and seen[-1]["language"] == "en")
check("a long locale code is trimmed", W.geocode_city("Сочи", "ru-RU") and seen[-1]["language"] == "ru")

loc = W.resolve_city("Сочи", lang="ru")
check("resolve_city passes the language through", seen[-1]["language"] == "ru" and loc["name"] == "Сочи")

# The spelling-correction retry keeps the language too.
class _None:
    def raise_for_status(self): pass
    def json(self): return {"results": []}
calls = {"n": 0}
def _get(url, params=None, timeout=None):
    calls["n"] += 1; seen.append(params)
    return _None() if calls["n"] == 1 else _R(params.get("language"))
W.requests.get = _get
loc = W.resolve_city("Сочы", correct_fn=lambda c: "Сочи", lang="ru")
check("the corrected retry is asked in the same language", seen[-1]["language"] == "ru" and loc["name"] == "Сочи", seen[-1])

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(W.__file__))), "bot/tg_weather.py"), encoding="utf-8").read()
check("the bot passes the session language", "resolve_city(city, correct_fn=correct_fn, lang=lang)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
