"""Live check of the song and weather reads (intent.song / song_seconds /
song_uses_previous_text / weather) through tg_songs and tg_weather. Phrases
from test_judge_exposure_and_song_intent, test_song_as_task,
test_weather_city_case and test_weather_fact_city, which tested the word lists.

    venv/Scripts/python bench/intent_song_weather_live.py
"""
import os, sys, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import tg_songs as S
import tg_weather as W

bad = 0
def check(name, ok, got=""):
    global bad
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {got}"))

for t in ["сочини короткую весёлую песню про кота, который ждёт ужина. секунд 30, не длиннее",
          "спой мне про лето", "сделай трек в стиле лоу-фай на 2 минуты", "write a song about rain, 1 minute",
          "песню про маму сочини", "Запиши песенку для дочки", "сгенерируй рэп про понедельник",
          "сочини гимн нашего отдела", "сделай колыбельную для сына", "а теперь сочини на него песню, 30 секунд"]:
    check("song: " + t, S.song_request(t)[0], S.song_request(t))
for t in ["напиши текст песни про кота", "нарисуй кота", "какая песня у битлз самая известная",
          "напиши слова для песни", "сочини стих про песню", "привет как дела",
          "напиши песню без музыки, только слова", "переведи песню на английский",
          "как научиться петь", "расскажи про гимн России"]:
    check("not a song: " + t, not S.song_request(t)[0], S.song_request(t))
check("30 seconds", S.song_request("сочини короткую весёлую песню про кота, секунд 30")[1] == 30)
check("2 minutes", S.song_request("сделай трек в стиле лоу-фай на 2 минуты")[1] == 120)
check("полторы минуты", S.request_seconds("песню на полторы минуты") == 90, S.request_seconds("песню на полторы минуты"))
check("'на него' uses the text", S.refers_to_previous_text("а теперь сочини на него песню, 30 секунд"))
check("a plain topic does not", not S.refers_to_previous_text("сочини песню про кота"))
check("style after a song = redo", S.song_followup("а теперь то же самое в стиле рок"))
check("a picture is not", not S.song_followup("нарисуй кота в стиле аниме"))
check("a new chorus rewrites", S.song_followup("замени припев на повеселее")
      and S.LYRICS_MARK not in S.followup_topic("замени припев на повеселее", "la"))
check("a new sound keeps the words", S.LYRICS_MARK in S.followup_topic("а теперь в стиле рок", "la"))

today = dt.date(2026, 9, 29)          # a Tuesday
w = W.weather_request("какая погода в Казани завтра?", today)
check("Kazan tomorrow", bool(w) and "Казан" in w["city"] and w["date"] == dt.date(2026, 9, 30), w)
check("now", W.weather_request("какая сейчас погода?", today).get("hours") == 0)
check("no city", W.weather_request("какая сейчас погода?", today).get("city") == "")
check("Погода Казань", "Казан" in W.weather_request("Погода Казань", today).get("city", ""))
check("hyphenated", "Ростов" in W.weather_request("погода в Ростове-на-Дону", today).get("city", ""))
check("London", W.weather_request("what's the weather in London today", today).get("city") == "London")
check("saturday", W.weather_request("какая погода будет в Казани в субботу?", today).get("date") == dt.date(2026, 10, 3))
check("weekend", W.weather_request("погода на выходных", today).get("hours") == 48)
check("погода на завтра: no city", W.weather_request("погода на завтра", today).get("city") == "")
for t in ("нарисуй погоду", "погода в игре тормозит", "привет", "спой песню о погоде",
          "какая погода вообще бывает осенью?", "а какая там погода сейчас?",
          "ответь на три вопроса: 1) столица Канады 2) 15% от 340 3) какая погода завтра в Сочи"):
    check("not weather: " + t, not W.weather_request(t, today), W.weather_request(t, today))
check("а в Сочи?", W.weather_followup("а в Сочи?").get("city") == "Сочи")
check("Питер", W.weather_followup("Питер").get("city") == "Питер")
check("а послезавтра?", W.weather_followup("а послезавтра?", today).get("date") == dt.date(2026, 10, 1))
for t in ("привет", "в целом нормально", "а в игре?", "в Москве живёт 13 миллионов", "Спасибо", "Мне нравится"):
    check("not a follow-up: " + t, W.weather_followup(t) == {}, W.weather_followup(t))
print("ALL OK" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
