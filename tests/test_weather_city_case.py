"""«погода в Твери» must be Tver, not Tiberias. The geocoder matches name
prefixes, and the inflected «Твери» is a prefix of «Тверия» (Israel, +31°);
live persona run 2026-09-27. Network stubbed: the fake geocoder behaves like
Open-Meteo's prefix match."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weather as W

PLACES = [("Тверия", "Израиль"), ("Тверь", "Россия"), ("Нижний Новгород", "Россия")]
asked = []


class R:
    def __init__(self, res): self._r = res
    def raise_for_status(self): pass
    def json(self): return {"results": self._r}


def fake_get(url, params=None, **k):
    q = params["name"]; asked.append(q)
    return R([{"name": n, "country": c, "latitude": 1, "longitude": 2}
              for n, c in PLACES if n.lower().startswith(q.lower())])


W.requests.get = fake_get
assert W.geocode_city("Твери", "ru")["name"] == "Тверь", asked
assert W.geocode_city("Нижнем Новгороде", "ru")["name"] == "Нижний Новгород", asked
assert W.geocode_city("Тверь", "ru")["name"] == "Тверь"
assert W.geocode_city("Tver", "en") is None          # Latin input untouched by the Russian morphology
print("PASS weather city case")

# "на выходных" is Sat+Sun (live 2026-09-28). The place and day are the
# model's read (intent.weather); phrases: bench/intent_song_weather_live.py.
import datetime as _d
import intent
import tg_weather as TW
intent.STUB = lambda t: {"weather": {"city": "Питере", "when": "weekend"}}
_w = TW.weather_request("какая погода в Питере на выходных?", today=_d.date(2026, 9, 28))
intent.STUB = None
assert (_w["city"], _w["date"], _w["hours"]) == ("Питере", _d.date(2026, 10, 3), 48), _w
print("PASS weekend + city")
PLACES += [("Санкт-Петербург", "Россия"), ("Екатеринбург", "Россия"), ("Питерка", "Россия")]
assert W.geocode_city("Питере", "ru")["name"] == "Санкт-Петербург", asked
assert W.geocode_city("Екб", "ru")["name"] == "Екатеринбург", asked
print("PASS aliases")

