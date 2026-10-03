"""'Today' starts at the current hour, not midnight; >16 days is 'too far', not 'service down'.

Run: venv/Scripts/python.exe tests/test_weather_today_window.py
"""
import datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weather as W

utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
off = (18 - utc.hour) * 3600                        # local time is 18:xx
local = utc + dt.timedelta(seconds=off)
day0 = dt.datetime.combine(local.date(), dt.time(0, 0))
pts = [{"time": day0 + dt.timedelta(hours=h), "temp_c": float(h), "code": 0, "wind_kmh": 1.0, "humidity": 50}
       for h in range(0, 48)]


def fetch(lat, lon, days=3):
    return None if days > 16 else {"utc_offset_seconds": off, "points": pts}   # Open-Meteo caps at 16


W.fetch_hourly = fetch
bad = 0
b, _, err = W.forecast_data({"lat": 0, "lon": 0, "name": "X"}, "ru", date=local.date())
ok = not err and b and b[0]["temp_c"] >= 18         # temp_c == hour: nothing from before 18:00
print(("PASS" if ok else "FAIL") + "  today at 18:00 starts at 18:00", b[:1]); bad += not ok
b2, _, err2 = W.forecast_data({"lat": 0, "lon": 0, "name": "X"}, "ru", date=local.date() + dt.timedelta(days=30))
ok = not b2 and err2 == W.no_forecast_message("ru")
print(("PASS" if ok else "FAIL") + "  30 days out says 'too far', not 'service down'"); bad += not ok
sys.exit(1 if bad else 0)
