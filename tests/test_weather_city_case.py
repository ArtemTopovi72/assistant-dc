"""«погода в Твери» must be Tver, not Tiberias. The geocoder matches name
prefixes, and the inflected «Твери» is a prefix of «Тверия» (Israel, +31°).
Network stubbed: the fake geocoder behaves like
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


import types
HTTP = types.SimpleNamespace(get=fake_get)
assert W.geocode_city("Твери", "ru", http=HTTP)["name"] == "Тверь", asked
assert W.geocode_city("Нижнем Новгороде", "ru", http=HTTP)["name"] == "Нижний Новгород", asked
assert W.geocode_city("Тверь", "ru", http=HTTP)["name"] == "Тверь"
assert W.geocode_city("Tver", "en", http=HTTP) is None          # Latin input untouched by the Russian morphology
print("PASS weather city case")

# "на выходных" is Sat+Sun. The place and day are the
# model's read (intent.weather); phrases: bench/intent_song_weather_live.py.
import datetime as _d
import types
import tg_weather as TW
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import reads
_R = types.SimpleNamespace(**reads(read=lambda t: {"weather": {"city": "Питере", "when": "weekend"}}))
_w = TW.weather_request("какая погода в Питере на выходных?", today=_d.date(2026, 9, 28), ctx=_R)
assert (_w["city"], _w["date"], _w["hours"]) == ("Питере", _d.date(2026, 10, 3), 48), _w
print("PASS weekend + city")
PLACES += [("Санкт-Петербург", "Россия"), ("Екатеринбург", "Россия"), ("Питерка", "Россия")]
assert W.geocode_city("Питере", "ru", http=HTTP)["name"] == "Санкт-Петербург", asked
assert W.geocode_city("Екб", "ru", http=HTTP)["name"] == "Екатеринбург", asked
print("PASS aliases")

