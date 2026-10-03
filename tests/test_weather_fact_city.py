"""Weather uses the newest pinned 'lives in / moved to' city before the default."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_weather as t
# the model's read of a fact is stubbed; the phrases run live in bench/intent_sweep3_live.py
import intent
_CITIES = {"Пользователь живет в Казани": "Казань", "Пользователь переехал в Самару": "Самара",
           "Пользователь живет в Нижнем Новгороде": "Нижний Новгород"}
intent.YES_STUB = lambda q, t_: "which city" in q and t_ in _CITIES
intent.CITY_STUB = lambda f: _CITIES.get(f, "")


class _S:
    def __init__(self, facts): self.f = facts
    def get_tg_facts(self): return self.f


def city(facts):
    o = t.WeatherMixin()
    o._store = type("St", (), {"get": lambda s, c: _S(facts)})()
    o._user_store = type("U", (), {"get": lambda s, c: None})()
    return o._remembered_city(1)


assert city([{"ts": 1, "text": "Пользователь живет в Казани"},
             {"ts": 2, "text": "Пользователь переехал в Самару"}]) == "Самара"
assert city([{"ts": 1, "text": "Пользователь живет в Нижнем Новгороде"}]) == "Нижний Новгород"   # nominative, as the geocoder wants
assert city([{"ts": 1, "text": "Любит кофе"}]) == t._config.WEATHER_DEFAULT_CITY
print("PASS weather fact city")
