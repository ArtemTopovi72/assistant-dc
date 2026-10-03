"""Live check of intent.must_call: the tool forced on the first round.

The phrases are the ones the old keyword forcers were tested on
(tests/test_forced_calculate.py, test_photo_math_force.py,
test_forced_deck_tool.py), plus paraphrases a word list misses.

    venv/Scripts/python bench/intent_force_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import intent

SEARCH = {"search"}
CASES = [
    # (text, attached, allowed must_call values)
    ("сколько будет 1234 * 5678?", "", {"calculate"}),
    ("посчитай 15% от 2480", "", {"calculate"}),
    ("посчитай 18432 * 977 + 15%", "", {"calculate"}),
    ("раздели 4096 на 7", "", {"calculate"}),
    ("а если умножить двести сорок на тысячу триста?", "", {"calculate"}),
    ("сколько будет 2*2", "", {"", "calculate"}),
    ("что было с 2020 по 2024 год", "", {"", "search"}),
    ("позвони на 8-800-555-3535", "", {""}),
    ("привет, как дела", "", {""}),
    ("расскажи анекдот", "", {""}),
    ("сколько я потратил без кофе и пакета?", "a photo", {"calculate"}),
    ("раздели чек поровну на троих", "a photo", {"calculate"}),
    ("сделай без фона", "a photo", {"", "inpaint_image", "redraw_image"}),
    ("what is the total profit for half a year?", "data with many numbers earlier in the chat", {"calculate"}),
    ("какая сейчас температура в Москве?", "", {"weather_forecast"}),
    ("кто сейчас президент Франции?", "", SEARCH),
    ("с каким счётом сыграл Зенит в последнем матче?", "", SEARCH),
    ("сколько стоит iPhone 17 Pro 256 ГБ?", "", SEARCH | {"ozon_search", "ozon_shop"}),
    ("где в Казани вкусно поесть? с адресами", "", SEARCH),
    ("какой сегодня день недели?", "", {"", "local_time"}),
    ("стоит ли учить python?", "", {""}),
    ("бита и мяч стоят 110 рублей, бита дороже мяча на 100. сколько стоит мяч?", "", {"", "calculate"}),
    ("сделай презентацию на 5 слайдов про сон", "", {"create_presentation"}),
    ("а теперь слайды о том, почему важно спать", "", {"create_presentation"}),
    ("напиши функцию на python, которая проверяет палиндром, и запусти её на 'шалаш'", "", {"run_code"}),
    ("напиши стих про осень", "", {""}),
]
bad = 0
for text, att, ok in CASES:
    got = intent.read(None, text, "", att)["must_call"]
    good = got in ok
    bad += not good
    print(("PASS " if good else "FAIL ") + f"{text!r} [{att}] -> {got!r} (want {sorted(ok)})")
print(f"{len(CASES) - bad}/{len(CASES)}")
sys.exit(1 if bad else 0)
